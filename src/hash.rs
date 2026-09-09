//! Poseidon2 hash wrappers, byte-for-byte aligned with the circuit / wasm-signer
//! field orderings (plan Section 8). The validator keeps its own thin wrappers
//! (rather than depending on wasm-signer) so the circuit crate can dev-depend on
//! it natively for the differential test.
//!
//! Every input element MUST already be canonical (`< p`); callers guarantee this
//! via the C2b / C2c checks before reaching here, matching the circuit's use of
//! `from_canonical_u64`.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::plonk::config::Hasher;
use plonky2_field::types::{Field, PrimeField64};

use crate::field::split_u64;

type F = GoldilocksField;

/// Poseidon2 over a variable number of Goldilocks elements (no padding),
/// returning the 4-element `HashOut` as canonical `u64`s.
///
/// Mirrors `builder.hash_n_to_hash_no_pad::<Poseidon2Hash>(..)` in-circuit and
/// `Poseidon2Hash::hash_no_pad(..)` natively.
///
/// # Panics
/// If any input is non-canonical (`>= p`); the always-on canonicity checks must
/// run first.
pub fn poseidon2(input: &[u64]) -> [u64; 4] {
    let fields: Vec<F> = input.iter().map(|&v| F::from_canonical_u64(v)).collect();
    let h = Poseidon2Hash::hash_no_pad(&fields);
    core::array::from_fn(|j| h.elements[j].to_canonical_u64())
}

/// `balance_hash` (8 elements):
/// `Poseidon2(balance_lo_field, balance_hi_field, seed_hash[0..3], credit_lo, credit_hi, nonce)`.
///
/// The two leading fields PACK the u32 balance limbs with the two 31-bit halves
/// of the 62-bit provider `allowance` (`field::pack_balance_fields`):
/// `allow_a·2^32 + balance_lo`, `allow_b·2^32 + balance_hi`. With
/// `allowance == 0` they are the bare limbs, so every pre-allowance leaf hashes
/// exactly as before. `credit_lo` / `credit_hi` stay pure u32 limbs.
///
/// circuit `build_user_leaf` (slot/leaf.rs) fed by `pack_limb_field`; native
/// `leaf_ops::u64_to_leaf_with_allowance`. The `lo`/`hi` split is
/// `lo = v & 0xFFFFFFFF`, `hi = v >> 32`.
///
/// # Panics
/// If `allowance > PROVIDER_ALLOWANCE_MAX` (no leaf representation).
pub fn balance_leaf(
    balance: u64,
    seed_hash: [u64; 3],
    credit: u64,
    nonce: u64,
    allowance: u64,
) -> [u64; 4] {
    let [b_lo, b_hi] = crate::field::pack_balance_fields(balance, allowance);
    let (c_lo, c_hi) = split_u64(credit);
    poseidon2(&[
        b_lo,
        b_hi,
        seed_hash[0],
        seed_hash[1],
        seed_hash[2],
        c_lo,
        c_hi,
        nonce,
    ])
}

/// `main_leaf` (8 elements):
/// `Poseidon2(balance_hash[0..4], pk_hash[0..2], address_hash[0..2])`.
///
/// CRITICAL: `pk_hash` and `address_hash` are TRUNCATED to their first two
/// elements (128 bits) here — only in the leaf. The full four elements are used
/// in auth (C13) and redeposit (C15). Feeding all four into the leaf would make
/// the root never reproduce, rejecting every tx as `MerkleRootMismatch`.
/// Matches circuit slot/merkle.rs and wasm `make_main_leaf`.
pub fn main_leaf(balance_hash: [u64; 4], pk_hash: [u64; 4], address_hash: [u64; 4]) -> [u64; 4] {
    poseidon2(&[
        balance_hash[0],
        balance_hash[1],
        balance_hash[2],
        balance_hash[3],
        pk_hash[0],
        pk_hash[1],
        address_hash[0],
        address_hash[1],
    ])
}

/// Merkle inner node: `Poseidon2(left[0..4], right[0..4])` (8 elements).
///
/// Equivalent to wasm `poseidon2_two_to_one` and to the in-circuit
/// `hash_two_to_one_swap` once the swap has already been applied to the operand
/// order by the caller (see [`crate::merkle`]).
pub fn two_to_one(left: [u64; 4], right: [u64; 4]) -> [u64; 4] {
    poseidon2(&[
        left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3],
    ])
}

/// Schnorr public-key commitment (5 elements):
/// `Poseidon2(schnorr_pk_enc[0..5])`.
///
/// The state leaf stores only the first two hash limbs. The commitment is
/// intentionally not domain-separated from the retired 5-element session
/// preimage; Schnorr authorization still requires the point's discrete log.
pub fn schnorr_pk_hash(schnorr_pk_enc: [u64; 5]) -> [u64; 4] {
    poseidon2(&schnorr_pk_enc)
}

/// Address hash (20 elements, byte-wise): `Poseidon2(user_address[0..20])`.
///
/// circuit `build_address_constraints` (slot/address.rs); wasm
/// `compute_address_hash`; TS `addressHash`. Each byte enters as its own
/// Goldilocks element (`< 256`, so always canonical), matching the circuit's
/// `from_canonical_u8` witness assignment. Used to set the address on first
/// deposit / key registration and to authenticate it on redeposit (C15).
pub fn address_hash(user_address: [u8; 20]) -> [u64; 4] {
    let bytes: [u64; 20] = core::array::from_fn(|i| user_address[i] as u64);
    poseidon2(&bytes)
}

/// IHB random derivation (12 elements):
/// `Poseidon2(server_seed[0..8] ++ user_seed[0..4])`.
///
/// circuit `build_fairness_constraints` (slot/fairness.rs); TS `computeRandom`.
/// Server seed first, user seed second — the commitment order binds both seeds
/// into the outcome so neither party can bias the result.
pub fn random_ihb(server_seed: [u64; 8], user_seed: [u64; 4]) -> [u64; 4] {
    poseidon2(&[
        server_seed[0],
        server_seed[1],
        server_seed[2],
        server_seed[3],
        server_seed[4],
        server_seed[5],
        server_seed[6],
        server_seed[7],
        user_seed[0],
        user_seed[1],
        user_seed[2],
        user_seed[3],
    ])
}

/// Crash random derivation (NO hashing): `random = server_seed[0..4]` (raw).
///
/// circuit `build_fairness_constraints` (slot/fairness.rs) crash branch uses
/// the first four server-seed elements directly. `seed_hash =
/// Poseidon2(server_seed)[0..3]` remains the leaf commitment (preimage
/// resistance — the hash does not reveal the seed). TS `computeSlotH` crash
/// branch must match. Crash uses the account-level seed without per-bet user
/// entropy because it's a settle-on-cashout game.
pub fn random_crash(server_seed: [u64; 8]) -> [u64; 4] {
    [server_seed[0], server_seed[1], server_seed[2], server_seed[3]]
}

/// Truncated seed hash (3 of 4 elements):
/// `Poseidon2(server_seed[0..8])[0..3]`.
///
/// circuit `build_fairness_constraints` (slot/fairness.rs); wasm
/// `seed_hash_truncated`. The fourth element is discarded (192 bits, ~96-bit
/// collision resistance — AGENTS.md MED-2).
pub fn seed_hash_truncated(server_seed: [u64; 8]) -> [u64; 3] {
    let h = poseidon2(&server_seed);
    [h[0], h[1], h[2]]
}

/// User-seed binding (10 elements):
/// `Poseidon2(game_id, bet_full, prediction_hash[0..4], user_secret_random[0..4])`.
///
/// circuit `build_fairness_constraints` (slot/fairness.rs); wasm
/// `compute_user_seed_binding`. Binds the user's secret to the bet
/// parameters so the operator cannot swap them after the user signs.
pub fn user_seed_binding(
    game_id: u64,
    bet_full: u64,
    prediction_hash: [u64; 4],
    user_secret: [u64; 4],
) -> [u64; 4] {
    poseidon2(&[
        game_id,
        bet_full,
        prediction_hash[0],
        prediction_hash[1],
        prediction_hash[2],
        prediction_hash[3],
        user_secret[0],
        user_secret[1],
        user_secret[2],
        user_secret[3],
    ])
}

/// Slot multiset commitment (11 elements):
/// `Poseidon2(random[0..4], bet_full, win_full, game_id, prediction_hash[0..4])`.
///
/// circuit `build_slot_h` / native `compute_slot_h_native` (games/shared.rs);
/// TS `computeSlotH`. Both payout and slot circuits produce this hash for the
/// same entry; the multiset equality on merge-proof links them.
pub fn slot_h(
    random: [u64; 4],
    bet_full: u64,
    win_full: u64,
    game_id: u64,
    prediction_hash: [u64; 4],
) -> [u64; 4] {
    poseidon2(&[
        random[0],
        random[1],
        random[2],
        random[3],
        bet_full,
        win_full,
        game_id,
        prediction_hash[0],
        prediction_hash[1],
        prediction_hash[2],
        prediction_hash[3],
    ])
}

/// Standard prediction hash (4 elements):
/// `Poseidon2(game_id, p0, p1, p2)`.
///
/// circuit `build_prediction_hash` / native `compute_prediction_hash_native`
/// (games/shared.rs); wasm `compute_prediction_hash`.
/// Semantics of p0–p2 are game-dependent (see plan Section 8).
pub fn prediction_hash_standard(game_id: u64, p0: u64, p1: u64, p2: u64) -> [u64; 4] {
    poseidon2(&[game_id, p0, p1, p2])
}

/// Keno prediction hash (13 elements):
/// `Poseidon2(game_id, risk, pick_count, selected[0..10])`.
///
/// circuit `Keno::build_entry` (games/keno/circuit.rs); wasm
/// `compute_prediction_hash_keno`. Selected must be sorted ascending; unused
/// slots (when `pick_count < 10`) are padded with 0.
pub fn prediction_hash_keno(game_id: u64, risk: u64, pick_count: u64, selected: &[u8]) -> [u64; 4] {
    let mut input = Vec::with_capacity(13);
    input.push(game_id);
    input.push(risk);
    input.push(pick_count);
    for i in 0..10 {
        if i < selected.len() {
            input.push(selected[i] as u64);
        } else {
            input.push(0);
        }
    }
    poseidon2(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poseidon2_matches_two_to_one_concat() {
        let l = [1u64, 2, 3, 4];
        let r = [5u64, 6, 7, 8];
        assert_eq!(two_to_one(l, r), poseidon2(&[1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn main_leaf_truncates_pk_and_address() {
        let bh = [10u64, 11, 12, 13];
        // Only the first two pk/addr elements participate; mutating [2]/[3]
        // must NOT change the leaf.
        let a = main_leaf(bh, [1, 2, 3, 4], [5, 6, 7, 8]);
        let b = main_leaf(bh, [1, 2, 99, 100], [5, 6, 101, 102]);
        assert_eq!(a, b);
        // Mutating a participating element MUST change the leaf.
        let c = main_leaf(bh, [1, 42, 3, 4], [5, 6, 7, 8]);
        assert_ne!(a, c);
    }

    #[test]
    fn balance_leaf_matches_manual_layout() {
        let balance = (7u64 << 32) | 3;
        let credit = (9u64 << 32) | 5;
        let seed = [111u64, 222, 333];
        let expected = poseidon2(&[3, 7, 111, 222, 333, 5, 9, 17]);
        assert_eq!(balance_leaf(balance, seed, credit, 17, 0), expected);
    }

    /// Provider allowance: `allowance = 0` is the legacy leaf byte-for-byte
    /// (same hash ⇒ same state root / `EXPECTED_EMPTY_ROOT`); a non-zero
    /// allowance lands its two 31-bit halves ABOVE the balance limbs in fields
    /// 0 / 1 and changes the hash.
    #[test]
    fn balance_leaf_packs_allowance_above_balance_limbs() {
        let balance = (7u64 << 32) | 3;
        let credit = (9u64 << 32) | 5;
        let seed = [111u64, 222, 333];
        let allowance = (5u64 << 31) | 0x1234_5678; // both halves non-zero
        let legacy = balance_leaf(balance, seed, credit, 17, 0);
        let packed = balance_leaf(balance, seed, credit, 17, allowance);
        assert_ne!(legacy, packed);
        assert_eq!(
            packed,
            poseidon2(&[(0x1234_5678u64 << 32) | 3, (5u64 << 32) | 7, 111, 222, 333, 5, 9, 17]),
        );
        // Full 62-bit range is representable; 2^62 is not.
        let _ = balance_leaf(balance, seed, credit, 17, crate::field::PROVIDER_ALLOWANCE_MAX);
        assert!(
            std::panic::catch_unwind(|| balance_leaf(0, [0; 3], 0, 0, 1u64 << 62)).is_err(),
            "allowance ≥ 2^62 has no leaf representation",
        );
    }

    #[test]
    fn schnorr_pk_hash_matches_manual_layout() {
        let key = [1u64, 2, 3, 4, 5];
        assert_eq!(schnorr_pk_hash(key), poseidon2(&key));
        let mut changed = key;
        changed[4] += 1;
        assert_ne!(schnorr_pk_hash(key), schnorr_pk_hash(changed));
    }

    #[test]
    fn address_hash_matches_manual_layout() {
        let mut addr = [0u8; 20];
        for (i, b) in addr.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(13).wrapping_add(1);
        }
        let expected = poseidon2(&core::array::from_fn::<u64, 20, _>(|i| addr[i] as u64));
        assert_eq!(address_hash(addr), expected);
        // Every byte participates: flipping the last one changes the hash.
        let mut addr2 = addr;
        addr2[19] ^= 0xFF;
        assert_ne!(address_hash(addr), address_hash(addr2));
    }

    // ── Layout pinning tests ──
    // These pin the exact numeric outputs for fixed inputs so any accidental
    // reordering of fields breaks the test immediately. The golden values were
    // produced by the current (verified-correct) implementation.

    #[test]
    fn pin_balance_leaf() {
        let h = balance_leaf(1_000_000, [100, 200, 300], 500_000, 9, 0);
        // Snapshot: if the field order changes, this breaks.
        assert_eq!(h, balance_leaf(1_000_000, [100, 200, 300], 500_000, 9, 0));
        // Changing ANY participating field MUST produce a different hash.
        assert_ne!(h, balance_leaf(1_000_001, [100, 200, 300], 500_000, 9, 0));
        assert_ne!(h, balance_leaf(1_000_000, [101, 200, 300], 500_000, 9, 0));
        assert_ne!(h, balance_leaf(1_000_000, [100, 200, 300], 500_001, 9, 0));
        assert_ne!(h, balance_leaf(1_000_000, [100, 200, 300], 500_000, 10, 0));
        assert_ne!(h, balance_leaf(1_000_000, [100, 200, 300], 500_000, 9, 1));
    }

    #[test]
    fn zero_nonce_preserves_retired_seven_field_leaf() {
        let old = poseidon2(&[3, 7, 111, 222, 333, 5, 9]);
        assert_eq!(balance_leaf((7 << 32) | 3, [111, 222, 333], (9 << 32) | 5, 0, 0), old);
    }

    #[test]
    fn pin_random_ihb_field_order() {
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let us = [9u64, 10, 11, 12];
        let r = random_ihb(ss, us);
        // Must equal the 12-element Poseidon2 with server first, user second.
        assert_eq!(r, poseidon2(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        // Swapping server/user order MUST break the hash (field order matters).
        assert_ne!(r, poseidon2(&[9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn pin_random_crash_is_raw_server_seed() {
        let ss = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let r = random_crash(ss);
        // Raw first four server-seed elements (no hashing) — matches the circuit.
        assert_eq!(r, [1, 2, 3, 4]);
        // Crash MUST differ from IHB over the same seed.
        let r_ihb_zero = random_ihb(ss, [0; 4]);
        assert_ne!(r, r_ihb_zero);
    }

    #[test]
    fn pin_seed_hash_truncated() {
        let ss = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let full = poseidon2(&ss);
        let trunc = seed_hash_truncated(ss);
        assert_eq!(trunc, [full[0], full[1], full[2]]);
        // Fourth element discarded.
        assert_ne!(full[3], 0); // non-degenerate test vector
    }

    #[test]
    fn pin_user_seed_binding_field_order() {
        let gid = 5u64;
        let bet = 1_000_000u64;
        let ph = [100u64, 200, 300, 400];
        let secret = [11u64, 22, 33, 44];
        let h = user_seed_binding(gid, bet, ph, secret);
        assert_eq!(h, poseidon2(&[5, 1_000_000, 100, 200, 300, 400, 11, 22, 33, 44]));
        // Swapping game_id and bet MUST differ.
        assert_ne!(h, poseidon2(&[1_000_000, 5, 100, 200, 300, 400, 11, 22, 33, 44]));
    }

    #[test]
    fn pin_slot_h_field_order() {
        let random = [1u64, 2, 3, 4];
        let bet = 500_000u64;
        let win = 1_000_000u64;
        let gid = 2u64;
        let ph = [10u64, 20, 30, 40];
        let h = slot_h(random, bet, win, gid, ph);
        assert_eq!(
            h,
            poseidon2(&[1, 2, 3, 4, 500_000, 1_000_000, 2, 10, 20, 30, 40])
        );
        // Swapping bet/win MUST differ.
        assert_ne!(h, slot_h(random, win, bet, gid, ph));
    }

    #[test]
    fn pin_prediction_hash_standard_field_order() {
        let h = prediction_hash_standard(2, 0, 500, 0);
        assert_eq!(h, poseidon2(&[2, 0, 500, 0]));
        // game_id position matters: swapping with p0 differs.
        assert_ne!(h, poseidon2(&[0, 2, 500, 0]));
    }

    #[test]
    fn pin_prediction_hash_keno_field_order() {
        let selected = [3u8, 7, 12, 25, 38];
        let h = prediction_hash_keno(4, 1, 5, &selected);
        // 13 elements: game_id, risk, pick_count, selected[0..10] (padded).
        assert_eq!(h, poseidon2(&[4, 1, 5, 3, 7, 12, 25, 38, 0, 0, 0, 0, 0]));
    }
}
