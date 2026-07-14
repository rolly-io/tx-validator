//! Goldilocks field canonicity helpers.
//!
//! The validator carries opaque field-element witness values as raw `u64`
//! (see [`crate::SlotInput`]) so it can reject non-canonical limbs itself
//! before anything is fed into Poseidon2. This is the whole point of the
//! C2b / C2c checks: the circuit's `from_canonical_u64` would otherwise panic
//! in debug or silently reduce `mod p` in release, desynchronizing the
//! TS/client commitments from the prover and bricking the block proof.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field64;

/// Goldilocks modulus `p = 2^64 - 2^32 + 1`. Any `u64` in `[p, 2^64)` is a
/// non-canonical field representative and is rejected by the validator.
pub const GOLDILOCKS_P: u64 = GoldilocksField::ORDER;

/// True iff `v` is a canonical Goldilocks representative (`v < p`).
#[inline]
pub fn is_canonical(v: u64) -> bool {
    v < GOLDILOCKS_P
}

/// True iff every element of `vals` is canonical (`< p`).
#[inline]
pub fn all_canonical(vals: &[u64]) -> bool {
    vals.iter().all(|&v| is_canonical(v))
}

/// True iff every element of `vals` is zero — the native test for an "empty"
/// hash slot. Mirrors the circuit's `and_all(is_equal(elem, 0))` used to detect
/// an unregistered pk (`old_pk_hash == 0`), an unset address (`old_address_hash
/// == 0`), or an uninitialized seed (`old_seed_hash == 0`).
#[inline]
pub fn all_zero(vals: &[u64]) -> bool {
    vals.iter().all(|&v| v == 0)
}

/// Split a full `u64` into circuit `(lo, hi)` u32 limbs, each `< 2^32`.
///
/// Mirrors the witness assignment in the circuit (`amount as u32`,
/// `(amount >> 32) as u32`). Because the source is a `u64`, both limbs are
/// `< 2^32` by construction, which is exactly what circuit constraint C2
/// range-checks — so the validator satisfies C2 structurally via this typed
/// representation rather than a runtime comparison that could never fail.
#[inline]
pub fn split_u64(v: u64) -> (u64, u64) {
    (v & 0xFFFF_FFFF, v >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_matches_known_value() {
        assert_eq!(GOLDILOCKS_P, 18446744069414584321);
    }

    #[test]
    fn canonicity_boundary() {
        assert!(is_canonical(0));
        assert!(is_canonical(GOLDILOCKS_P - 1));
        assert!(!is_canonical(GOLDILOCKS_P));
        assert!(!is_canonical(u64::MAX));
    }

    #[test]
    fn all_zero_detects_empty_slot() {
        assert!(all_zero(&[0u64; 4]));
        assert!(all_zero(&[] as &[u64]));
        assert!(!all_zero(&[0u64, 0, 1, 0]));
    }

    #[test]
    fn split_roundtrips() {
        for v in [0u64, 1, 0xFFFF_FFFF, 1 << 32, u64::MAX, GOLDILOCKS_P] {
            let (lo, hi) = split_u64(v);
            assert!(lo < (1 << 32));
            assert!(hi < (1 << 32));
            assert_eq!(lo | (hi << 32), v);
        }
    }
}
