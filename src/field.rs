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

// ── Provider-allowance packing (mirror of `leaf_ops::pack_balance_fields`) ──
//
// The two leading balance-leaf fields carry the u32 balance limb in bits
// 0..32 and one 31-bit half of the 62-bit provider allowance in bits 32..63:
//
//   field[0] = allow_a · 2^32 + balance_lo      allow_a = allowance mod 2^31
//   field[1] = allow_b · 2^32 + balance_hi      allow_b = allowance >> 31
//
// Each packed field is `< 2^63 < p`, so the (limb, half) split is canonical.
// `allowance = 0` reproduces the legacy pure-u32 limbs byte-for-byte.

/// Bit-width of the provider allowance stored in a balance leaf.
pub const PROVIDER_ALLOWANCE_BITS: u32 = 62;
/// Bit-width of each allowance half packed above a u32 balance limb.
pub const ALLOWANCE_HALF_BITS: u32 = 31;
/// Largest storable provider allowance (`2^62 − 1`).
pub const PROVIDER_ALLOWANCE_MAX: u64 = (1u64 << PROVIDER_ALLOWANCE_BITS) - 1;
const ALLOWANCE_HALF_MASK: u64 = (1u64 << ALLOWANCE_HALF_BITS) - 1;
const LIMB_MASK: u64 = 0xFFFF_FFFF;

/// Pack `balance` and the 62-bit provider `allowance` into the two leading
/// balance-leaf fields (see the module notes). Native twin of the circuit's
/// `pack_limb_field` / `split_allowance`; identical to
/// `rolly_circuit::helpers::leaf_ops::pack_balance_fields`.
///
/// # Panics
/// If `allowance > PROVIDER_ALLOWANCE_MAX` — such a value has no leaf
/// representation; callers must reject it first (`ProviderAllowanceOverflow`).
#[inline]
pub fn pack_balance_fields(balance: u64, allowance: u64) -> [u64; 2] {
    assert!(
        allowance <= PROVIDER_ALLOWANCE_MAX,
        "provider allowance {allowance} exceeds 2^62 - 1",
    );
    let (bal_lo, bal_hi) = split_u64(balance);
    let allow_a = allowance & ALLOWANCE_HALF_MASK;
    let allow_b = allowance >> ALLOWANCE_HALF_BITS;
    [(allow_a << 32) | bal_lo, (allow_b << 32) | bal_hi]
}

/// Inverse of [`pack_balance_fields`]: `(balance, allowance)` from the two
/// leading balance-leaf fields. Each field must be `< 2^63` (every
/// circuit-produced leaf satisfies this); the halves are read verbatim.
#[inline]
pub fn unpack_balance_fields(fields: [u64; 2]) -> (u64, u64) {
    debug_assert!(
        fields[0] < (1u64 << 63) && fields[1] < (1u64 << 63),
        "packed balance field out of range",
    );
    let balance = (fields[0] & LIMB_MASK) | ((fields[1] & LIMB_MASK) << 32);
    let allowance = ((fields[1] >> 32) << ALLOWANCE_HALF_BITS) | (fields[0] >> 32);
    (balance, allowance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_and_legacy_identity() {
        // allowance = 0 is exactly the legacy (lo, hi) limb pair.
        for balance in [0u64, 1, 0xFFFF_FFFF, 1 << 32, u64::MAX] {
            let (lo, hi) = split_u64(balance);
            assert_eq!(pack_balance_fields(balance, 0), [lo, hi]);
        }
        let cases: &[(u64, u64)] = &[
            (0, 0),
            (0, 1),
            (u64::MAX, PROVIDER_ALLOWANCE_MAX),
            (0xFFFF_FFFF, (1u64 << 31) - 1),
            (0xFFFF_FFFF_0000_0000, 1u64 << 31),
            (123_456_789, 0x2ABC_DEF0_1234_5678 & PROVIDER_ALLOWANCE_MAX),
        ];
        for &(balance, allowance) in cases {
            let f = pack_balance_fields(balance, allowance);
            assert!(f[0] < (1u64 << 63) && f[1] < (1u64 << 63), "packed field must stay canonical");
            assert!(is_canonical(f[0]) && is_canonical(f[1]));
            assert_eq!(f[0] & LIMB_MASK, balance & LIMB_MASK);
            assert_eq!(f[1] & LIMB_MASK, balance >> 32);
            assert_eq!(f[0] >> 32, allowance & ALLOWANCE_HALF_MASK);
            assert_eq!(f[1] >> 32, allowance >> 31);
            assert_eq!(unpack_balance_fields(f), (balance, allowance), "({balance}, {allowance})");
        }
    }

    #[test]
    #[should_panic(expected = "exceeds 2^62 - 1")]
    fn pack_rejects_allowance_above_62_bits() {
        let _ = pack_balance_fields(0, PROVIDER_ALLOWANCE_MAX + 1);
    }

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
