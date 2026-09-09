//! Poseidon2 shoe seed for the Blackjack payout defense-in-depth layer.
//!
//! Blackjack is a crash-style game: at the block level it is an ordinary
//! `in_house_bet` (tx_type 5). Unlike every other game, its win is NOT a
//! function of the fairness `random` — it depends on the 30 cards dealt from a
//! fair shoe, the packed action list, and the base stake.
//!
//! The GAME LOGIC lives once in `rolly_game_core::blackjack`: the deck layout
//! and fair shuffle ([`deal_shoe`]), the action (un)packing ([`unpack_actions`])
//! and the round replay ([`replay_full`]). This module only supplies the
//! Poseidon2 layer game-core deliberately omits — exactly as [`crate::hash`]
//! supplies the `random` the other games' payouts consume. The circuit mirrors
//! the same game-core logic in-circuit; a circuit-side differential test pins
//! this native shoe seed against the circuit's.

use rolly_game_core::blackjack::{
    deal_shoe, replay_full, unpack_actions, BlackjackResult, ACTION_HIT, ACTION_INSURANCE_DECLINE,
    MAX_CARDS,
};

use crate::hash::poseidon2;

/// Reconstruct the 30 dealt rank codes.
///
/// Computes `mix = Poseidon2(server_seed ‖ user_secret_random)` and the per-step
/// swap randomness `low32(Poseidon2(mix ‖ k))`, then defers the deterministic
/// shuffle to the pure `rolly_game_core::blackjack::deal_shoe`. Mixing the
/// player-chosen secret into the shoe seed makes the shuffle a mutual
/// commit–reveal so neither side can pre-grind a shoe.
pub fn unfold_shoe(server_seed: &[u64; 8], user_secret_random: &[u64; 4]) -> [u8; MAX_CARDS] {
    let mut mix_input = Vec::with_capacity(12);
    mix_input.extend_from_slice(server_seed);
    mix_input.extend_from_slice(user_secret_random);
    let mix = poseidon2(&mix_input);

    let swap_random: [u64; MAX_CARDS] = core::array::from_fn(|k| {
        poseidon2(&[mix[0], mix[1], mix[2], mix[3], k as u64])[0] & 0xFFFF_FFFF
    });
    deal_shoe(&swap_random)
}

/// Reproduce the blackjack round outcome from the primary slot inputs.
///
/// Returns `None` when the unpacked action list contains an invalid code
/// (outside `1..=6`), so callers surface a typed rejection instead of panicking
/// the engine's action decode. The dealt shoe is always 30 cards and every swap
/// index is in range, so the unfold itself cannot panic.
pub fn compute_payout(
    server_seed: &[u64; 8],
    user_secret_random: &[u64; 4],
    prediction_lo: u32,
    prediction_hi: u32,
    base_bet: u64,
) -> Option<BlackjackResult> {
    let actions = unpack_actions(prediction_lo, prediction_hi);
    if actions
        .iter()
        .any(|&a| a < ACTION_HIT || a > ACTION_INSURANCE_DECLINE)
    {
        return None;
    }
    let drawn = unfold_shoe(server_seed, user_secret_random);
    Some(replay_full(&drawn, &actions, base_bet))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rolly_game_core::blackjack::{pack_actions, ACTION_STAND, BLACKJACK_GAME_ID};

    #[test]
    fn unfold_shoe_is_deterministic_and_in_range() {
        let ss = [101u64, 202, 303, 404, 505, 606, 707, 808];
        let us = [11u64, 22, 33, 44];
        let a = unfold_shoe(&ss, &us);
        assert_eq!(a, unfold_shoe(&ss, &us), "same seed ⇒ same shoe");
        assert!(a.iter().all(|&c| (c as usize) < 13), "rank codes 0..12");
        // The player's secret changes the shuffle (mutual commit–reveal).
        assert_ne!(a, unfold_shoe(&ss, &[12u64, 22, 33, 44]));
    }

    #[test]
    fn compute_payout_rejects_invalid_action_code() {
        // A base-8 digit of 7 is a valid pack but not a valid action code.
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let us = [9u64, 10, 11, 12];
        assert!(
            compute_payout(&ss, &us, 7, 0, 1_000_000).is_none(),
            "action code 7 must be rejected, not panic replay",
        );
    }

    #[test]
    fn compute_payout_matches_direct_replay() {
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let us = [9u64, 10, 11, 12];
        let base_bet = 1_000_000u64;
        let (lo, hi) = pack_actions(&[ACTION_STAND]);
        let via = compute_payout(&ss, &us, lo, hi, base_bet).expect("valid round");
        let direct = replay_full(&unfold_shoe(&ss, &us), &[ACTION_STAND], base_bet);
        assert_eq!(via, direct);
        assert_eq!(BLACKJACK_GAME_ID, 7);
    }
}
