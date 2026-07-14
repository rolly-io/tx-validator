//! Fairness checks and payout verification for `in_house_bet` / `crash_settle`,
//! the native mirror of `build_fairness_constraints` (slot/fairness.rs) plus the
//! defense-in-depth payout layer that mirrors the per-game payout circuits
//! (`GenericPayoutCircuit<G>`).
//!
//! This module enforces:
//!   * C9  — seed-chain: `Poseidon2(server_seed)[0..3] == old_seed_hash` (IHB only).
//!   * C10 — user_seed binding: `user_seed == Poseidon2(game_id, bet, pred_hash,
//!     user_secret)` (IHB only).
//!   * Random derivation — IHB: `Poseidon2(server_seed || user_seed)`;
//!     crash: `server_seed[0..4]` (raw, no hashing — matches the circuit).
//!   * PayoutMismatch — `win_amount == compute_payout_<game>(random, bet, params)`
//!     and `prediction_hash == compute_prediction_hash(game_id, params)` via
//!     `rolly-game-core` (defense-in-depth; not a slot-circuit constraint but the
//!     payout-circuit constraint caught early).
//!   * `slot_h` — the 11-element multiset commitment.

use crate::flags::TxFlags;
use crate::{hash, SlotInput, SlotRejection};

use rolly_game_core::{crash, coinflip, dice, keno, limbo, plinko};

/// Post-fairness effects consumed by `compute_effects` to build `SlotEffects`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FairnessEffects {
    pub random: [u64; 4],
    pub slot_h: [u64; 4],
    /// IHB advances the seed; crash keeps the old one.
    pub new_seed_hash: [u64; 3],
}

/// Run the fairness + payout checks for `in_house_bet` / `crash_settle` and
/// produce the `random`, `slot_h`, and `new_seed_hash`.
///
/// Balance (C6, credit cap) and session auth (C11/C13/C14) are already checked
/// by the caller.
pub(crate) fn check_fairness(
    input: &SlotInput,
    flags: &TxFlags,
) -> Result<FairnessEffects, SlotRejection> {
    debug_assert!(flags.is_ihb_or_crash());

    // ── C9: seed-chain (IHB only) ──
    // Poseidon2(server_seed)[0..3] must reproduce old_seed_hash.
    if flags.is_in_house_bet {
        let computed = hash::seed_hash_truncated(input.server_seed);
        if computed != input.old_seed_hash {
            return Err(SlotRejection::SeedChainBreak);
        }
    }

    // ── crash reveal→commit (crash_settle only) ──
    // The revealed server_seed must hash to the seed committed in the
    // crash/crash house account (leaf 0) at round start. Mirrors the circuit's
    // per-block check; skipped when the caller did not supply the commitment.
    if flags.is_crash_settle {
        if let Some(seed_hash) = input.crash_seed_hash {
            let computed = hash::seed_hash_truncated(input.server_seed);
            if computed != seed_hash {
                return Err(SlotRejection::CrashSeedCommitMismatch);
            }
        }
    }

    // ── C10: user_seed binding (IHB only) ──
    // user_seed == Poseidon2(game_id, bet_full, prediction_hash[4], user_secret[4])
    if flags.is_in_house_bet {
        let computed = hash::user_seed_binding(
            input.game_id as u64,
            input.amount, // bet_full
            input.prediction_hash,
            input.user_secret_random,
        );
        if computed != input.user_seed {
            return Err(SlotRejection::UserSeedBindingMismatch);
        }
    }

    // ── Random derivation ──
    let random = if flags.is_in_house_bet {
        hash::random_ihb(input.server_seed, input.user_seed)
    } else {
        hash::random_crash(input.server_seed)
    };

    // ── Prediction hash verification (defense-in-depth) ──
    verify_prediction_hash(input, flags)?;

    // ── Payout verification (defense-in-depth) ──
    verify_payout(input, flags, &random)?;

    // ── slot_h: multiset commitment ──
    let computed_slot_h = hash::slot_h(
        random,
        input.amount,      // bet_full
        input.win_amount,  // win_full
        input.game_id as u64,
        input.prediction_hash,
    );

    // ── Seed advance ──
    let new_seed_hash = if flags.is_in_house_bet {
        input.next_server_seed_hash
    } else {
        input.old_seed_hash
    };

    Ok(FairnessEffects {
        random,
        slot_h: computed_slot_h,
        new_seed_hash,
    })
}

/// Verify that `prediction_hash` in the witness matches the raw game params.
fn verify_prediction_hash(
    input: &SlotInput,
    _flags: &TxFlags,
) -> Result<(), SlotRejection> {
    let gid = input.game_id as u64;
    let expected = if input.game_id == keno::KENO_GAME_ID {
        let mut sorted = input.keno_selected.clone();
        sorted.sort();
        hash::prediction_hash_keno(
            gid,
            input.game_mode as u64,  // risk
            sorted.len() as u64,     // pick_count
            &sorted,
        )
    } else {
        hash::prediction_hash_standard(
            gid,
            input.game_mode as u64,
            input.prediction_lo as u64,
            input.prediction_hi as u64,
        )
    };

    if expected != input.prediction_hash {
        return Err(SlotRejection::PayoutMismatch);
    }
    Ok(())
}

/// Verify `win_amount == compute_payout_<game>(random, bet, params)`.
fn verify_payout(
    input: &SlotInput,
    _flags: &TxFlags,
    random: &[u64; 4],
) -> Result<(), SlotRejection> {
    let expected_win = dispatch_payout(input, random)?;
    if expected_win != input.win_amount {
        return Err(SlotRejection::PayoutMismatch);
    }
    Ok(())
}

/// Dispatch to the correct game's payout function based on `game_id`.
/// Returns the expected win_amount, or `PayoutMismatch` on invalid game params.
fn dispatch_payout(
    input: &SlotInput,
    random: &[u64; 4],
) -> Result<u64, SlotRejection> {
    let bet = input.amount;

    if input.game_id == crash::CRASH_GAME_ID {
        // Crash: cashout_x100 = prediction_lo
        let cashout_x100 = input.prediction_lo;
        let payout = crash::compute_payout(random, bet, cashout_x100);
        return Ok(payout.win_amount);
    }

    // in_house_bet: dispatch by game_id
    match input.game_id {
        id if id == limbo::LIMBO_GAME_ID => {
            // p0=rtp (constant 98 in game-core), p1=prediction_x100
            let prediction_x100 = input.prediction_lo;
            let payout = limbo::compute_payout(random, bet, prediction_x100);
            Ok(payout.win_amount)
        }
        id if id == dice::DICE_GAME_ID => {
            let mode = dice::DiceMode::from_u8(input.game_mode as u8)
                .ok_or(SlotRejection::PayoutMismatch)?;
            let payout = dice::compute_payout(
                random,
                bet,
                mode,
                [input.prediction_lo, input.prediction_hi],
            );
            Ok(payout.win_amount)
        }
        id if id == keno::KENO_GAME_ID => {
            let risk = input.game_mode as u8;
            let mut sorted = input.keno_selected.clone();
            sorted.sort();
            let payout = keno::compute_payout(random, bet, risk, &sorted);
            Ok(payout.win_amount)
        }
        id if id == plinko::PLINKO_GAME_ID => {
            let sector = input.game_mode as u8;
            let rows = input.prediction_lo;
            let is_extreme = input.prediction_hi != 0;
            if !plinko::is_valid_config(sector, rows, is_extreme) {
                return Err(SlotRejection::PayoutMismatch);
            }
            let payout = plinko::compute_payout(random, bet, sector, rows, is_extreme);
            Ok(payout.win_amount)
        }
        id if id == coinflip::COINFLIP_GAME_ID => {
            let prediction = input.prediction_lo as u8;
            if prediction > 1 {
                return Err(SlotRejection::PayoutMismatch);
            }
            let payout = coinflip::compute_payout(random, bet, prediction);
            Ok(payout.win_amount)
        }
        _ => Err(SlotRejection::PayoutMismatch),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash;

    /// Build a minimal IHB input with valid fairness fields.
    /// Caller is responsible for committing the state tree (current_root etc.).
    fn ihb_input(
        game_id: u8,
        bet: u64,
        server_seed: [u64; 8],
        user_secret: [u64; 4],
        game_mode: u32,
        prediction_lo: u32,
        prediction_hi: u32,
    ) -> SlotInput {
        let old_seed_hash = hash::seed_hash_truncated(server_seed);
        let prediction_hash = hash::prediction_hash_standard(
            game_id as u64,
            game_mode as u64,
            prediction_lo as u64,
            prediction_hi as u64,
        );
        let user_seed = hash::user_seed_binding(
            game_id as u64,
            bet,
            prediction_hash,
            user_secret,
        );

        SlotInput {
            tx_type: crate::TX_IN_HOUSE_BET,
            game_id,
            amount: bet,
            server_seed,
            user_seed,
            old_seed_hash,
            next_server_seed_hash: [901, 902, 903],
            user_secret_random: user_secret,
            prediction_hash,
            game_mode,
            prediction_lo,
            prediction_hi,
            ..SlotInput::default()
        }
    }

    fn flags_ihb() -> TxFlags {
        TxFlags::from_tx_type(crate::TX_IN_HOUSE_BET)
    }

    fn flags_crash() -> TxFlags {
        TxFlags::from_tx_type(crate::TX_CRASH_SETTLE)
    }

    // ── C9: seed-chain ──

    #[test]
    fn ihb_valid_seed_chain_passes_c9() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0, // Under
            500,
            0,
        );
        // compute the correct win_amount
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = dice::compute_payout(&random, 1_000_000, dice::DiceMode::Under, [500, 0]);
        inp.win_amount = payout.win_amount;

        let result = check_fairness(&inp, &flags_ihb());
        assert!(result.is_ok(), "valid IHB must pass: {:?}", result);
        let eff = result.unwrap();
        assert_eq!(eff.random, random);
        assert_eq!(eff.new_seed_hash, [901, 902, 903]);
    }

    #[test]
    fn ihb_broken_seed_chain_fails_c9() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,
            500,
            0,
        );
        inp.old_seed_hash[0] ^= 1; // break the chain
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::SeedChainBreak)
        );
    }

    // ── C10: user_seed binding ──

    #[test]
    fn ihb_wrong_user_seed_fails_c10() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,
            500,
            0,
        );
        inp.user_seed[0] ^= 1; // tamper with user_seed
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::UserSeedBindingMismatch)
        );
    }

    // ── Crash: no C9/C10 ──

    #[test]
    fn crash_skips_seed_chain_and_user_seed_binding() {
        let ss = [11u64, 22, 33, 44, 55, 66, 77, 88];
        let random = hash::random_crash(ss);
        let cashout_x100 = 200u32; // 2.00x
        let prediction_hash = hash::prediction_hash_standard(
            crash::CRASH_GAME_ID as u64,
            0, // mode
            cashout_x100 as u64,
            0,
        );
        let payout = crash::compute_payout(&random, 500_000, cashout_x100);

        let inp = SlotInput {
            tx_type: crate::TX_CRASH_SETTLE,
            game_id: crash::CRASH_GAME_ID,
            amount: 500_000,
            win_amount: payout.win_amount,
            server_seed: ss,
            old_seed_hash: [999, 888, 777], // arbitrary — crash doesn't check
            prediction_hash,
            game_mode: 0,
            prediction_lo: cashout_x100,
            prediction_hi: 0,
            ..SlotInput::default()
        };

        let result = check_fairness(&inp, &flags_crash());
        assert!(result.is_ok(), "valid crash must pass: {:?}", result);
        let eff = result.unwrap();
        assert_eq!(eff.random, random);
        // Crash does NOT advance the seed
        assert_eq!(eff.new_seed_hash, inp.old_seed_hash);
    }

    // ── Crash reveal→commit ──

    #[test]
    fn crash_matching_seed_commit_passes() {
        let ss = [11u64, 22, 33, 44, 55, 66, 77, 88];
        let random = hash::random_crash(ss);
        let cashout_x100 = 200u32;
        let prediction_hash = hash::prediction_hash_standard(
            crash::CRASH_GAME_ID as u64,
            0,
            cashout_x100 as u64,
            0,
        );
        let payout = crash::compute_payout(&random, 500_000, cashout_x100);

        let inp = SlotInput {
            tx_type: crate::TX_CRASH_SETTLE,
            game_id: crash::CRASH_GAME_ID,
            amount: 500_000,
            win_amount: payout.win_amount,
            server_seed: ss,
            old_seed_hash: [999, 888, 777],
            prediction_hash,
            game_mode: 0,
            prediction_lo: cashout_x100,
            prediction_hi: 0,
            // commitment matches the revealed seed
            crash_seed_hash: Some(hash::seed_hash_truncated(ss)),
            ..SlotInput::default()
        };

        assert!(check_fairness(&inp, &flags_crash()).is_ok());
    }

    #[test]
    fn crash_wrong_seed_commit_rejected() {
        let ss = [11u64, 22, 33, 44, 55, 66, 77, 88];
        let random = hash::random_crash(ss);
        let cashout_x100 = 200u32;
        let prediction_hash = hash::prediction_hash_standard(
            crash::CRASH_GAME_ID as u64,
            0,
            cashout_x100 as u64,
            0,
        );
        let payout = crash::compute_payout(&random, 500_000, cashout_x100);

        let mut commit = hash::seed_hash_truncated(ss);
        commit[0] ^= 1; // operator-grindable seed would not match the commit

        let inp = SlotInput {
            tx_type: crate::TX_CRASH_SETTLE,
            game_id: crash::CRASH_GAME_ID,
            amount: 500_000,
            win_amount: payout.win_amount,
            server_seed: ss,
            old_seed_hash: [999, 888, 777],
            prediction_hash,
            game_mode: 0,
            prediction_lo: cashout_x100,
            prediction_hi: 0,
            crash_seed_hash: Some(commit),
            ..SlotInput::default()
        };

        assert_eq!(
            check_fairness(&inp, &flags_crash()),
            Err(SlotRejection::CrashSeedCommitMismatch)
        );
    }

    // ── PayoutMismatch ──

    #[test]
    fn ihb_wrong_win_amount_fails_payout() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,
            500,
            0,
        );
        inp.win_amount = 999_999_999; // wrong
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::PayoutMismatch)
        );
    }

    #[test]
    fn ihb_wrong_prediction_hash_fails() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,
            500,
            0,
        );
        inp.prediction_hash[0] ^= 1; // tamper
        // user_seed won't match either (it commits to prediction_hash), so
        // UserSeedBindingMismatch fires first.
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::UserSeedBindingMismatch)
        );
    }

    #[test]
    fn ihb_mismatched_game_mode_fails_prediction_hash() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        // Build with mode=0 (Under) but tell the validator mode=1 (Over)
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,     // build prediction_hash with mode=0
            500,
            0,
        );
        inp.game_mode = 1; // but claim mode=1
        // prediction_hash was built with mode=0, so the reconstructed hash
        // from mode=1 won't match → PayoutMismatch.
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::PayoutMismatch)
        );
    }

    // ── Coinflip ──

    #[test]
    fn ihb_coinflip_heads_payout_correct() {
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let secret = [10u64, 20, 30, 40];
        let bet = 2_000_000u64;

        // prediction=0 (heads) lives in prediction_lo; game_mode=0
        let mut inp = ihb_input(
            coinflip::COINFLIP_GAME_ID,
            bet,
            ss,
            secret,
            0, // game_mode
            0, // prediction_lo = heads
            0,
        );
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = coinflip::compute_payout(&random, bet, 0);
        inp.win_amount = payout.win_amount;

        assert!(check_fairness(&inp, &flags_ihb()).is_ok());
    }

    #[test]
    fn ihb_coinflip_tails_payout_correct() {
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let secret = [10u64, 20, 30, 40];
        let bet = 2_000_000u64;

        // prediction=1 (tails) lives in prediction_lo; game_mode=0
        let mut inp = ihb_input(
            coinflip::COINFLIP_GAME_ID,
            bet,
            ss,
            secret,
            0, // game_mode (always 0 for coinflip)
            1, // prediction_lo = tails
            0,
        );
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = coinflip::compute_payout(&random, bet, 1);
        inp.win_amount = payout.win_amount;

        assert!(check_fairness(&inp, &flags_ihb()).is_ok());
    }

    // ── Plinko ──

    #[test]
    fn ihb_plinko_payout_correct() {
        let ss = [5u64, 6, 7, 8, 9, 10, 11, 12];
        let secret = [50u64, 60, 70, 80];
        let sector = 0u32;
        let rows = 8u32;
        let is_extreme = 0u32;
        let bet = 1_000_000u64;

        let mut inp = ihb_input(
            plinko::PLINKO_GAME_ID,
            bet,
            ss,
            secret,
            sector,
            rows,
            is_extreme,
        );
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = plinko::compute_payout(&random, bet, sector as u8, rows, is_extreme != 0);
        inp.win_amount = payout.win_amount;

        assert!(check_fairness(&inp, &flags_ihb()).is_ok());
    }

    // ── Limbo ──

    #[test]
    fn ihb_limbo_payout_correct() {
        let ss = [3u64, 4, 5, 6, 7, 8, 9, 10];
        let secret = [30u64, 40, 50, 60];
        let rtp = limbo::RTP_PERCENT;
        let pred_x100 = 200u32; // 2.00x target
        let bet = 1_000_000u64;

        let mut inp = ihb_input(
            limbo::LIMBO_GAME_ID,
            bet,
            ss,
            secret,
            rtp,
            pred_x100,
            0,
        );
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = limbo::compute_payout(&random, bet, pred_x100);
        inp.win_amount = payout.win_amount;

        assert!(check_fairness(&inp, &flags_ihb()).is_ok());
    }

    // ── Keno ──

    #[test]
    fn ihb_keno_payout_correct() {
        let ss = [7u64, 8, 9, 10, 11, 12, 13, 14];
        let secret = [70u64, 80, 90, 100];
        let risk = 0u32; // low
        let selected: Vec<u8> = vec![2, 5, 10, 15, 20];
        let bet = 1_000_000u64;

        let mut sorted = selected.clone();
        sorted.sort();

        let prediction_hash = hash::prediction_hash_keno(
            keno::KENO_GAME_ID as u64,
            risk as u64,
            sorted.len() as u64,
            &sorted,
        );
        let user_seed = hash::user_seed_binding(
            keno::KENO_GAME_ID as u64,
            bet,
            prediction_hash,
            secret,
        );
        let old_seed_hash = hash::seed_hash_truncated(ss);

        let random = hash::random_ihb(ss, user_seed);
        let payout = keno::compute_payout(&random, bet, risk as u8, &sorted);

        let inp = SlotInput {
            tx_type: crate::TX_IN_HOUSE_BET,
            game_id: keno::KENO_GAME_ID,
            amount: bet,
            win_amount: payout.win_amount,
            server_seed: ss,
            user_seed,
            old_seed_hash,
            next_server_seed_hash: [801, 802, 803],
            user_secret_random: secret,
            prediction_hash,
            game_mode: risk,
            prediction_lo: 0,
            prediction_hi: 0,
            keno_selected: selected,
            ..SlotInput::default()
        };

        let result = check_fairness(&inp, &flags_ihb());
        assert!(result.is_ok(), "valid keno IHB must pass: {:?}", result);
    }

    // ── slot_h matches manual computation ──

    #[test]
    fn slot_h_matches_manual() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(
            dice::DICE_GAME_ID,
            1_000_000,
            ss,
            [100, 200, 300, 400],
            0,
            500,
            0,
        );
        let random = hash::random_ihb(ss, inp.user_seed);
        let payout = dice::compute_payout(&random, 1_000_000, dice::DiceMode::Under, [500, 0]);
        inp.win_amount = payout.win_amount;

        let eff = check_fairness(&inp, &flags_ihb()).unwrap();
        let manual_slot_h = hash::slot_h(
            random,
            inp.amount,
            inp.win_amount,
            inp.game_id as u64,
            inp.prediction_hash,
        );
        assert_eq!(eff.slot_h, manual_slot_h);
    }

    // ── Invalid game_id ──

    #[test]
    fn ihb_invalid_game_id_is_payout_mismatch() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut inp = ihb_input(99, 1_000_000, ss, [1, 2, 3, 4], 0, 0, 0);
        inp.win_amount = 0;
        assert_eq!(
            check_fairness(&inp, &flags_ihb()),
            Err(SlotRejection::PayoutMismatch)
        );
    }
}
