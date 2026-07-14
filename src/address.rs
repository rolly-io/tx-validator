//! Address / public-key setter logic and the redeposit address check (C15),
//! the native mirror of `build_address_constraints` (slot/address.rs).
//!
//! Two pieces of per-user identity live alongside the balance in the leaf:
//!   * `pk_hash` — the session authority. A `key_register_only` slot carrying a
//!     non-zero `new_pk_hash` is a *key setter* and installs it; every other
//!     slot keeps `old_pk_hash`.
//!   * `address_hash` — the on-chain withdrawal/deposit address, set once on the
//!     first deposit or first key registration (when the slot is `address-zero`)
//!     and thereafter immutable.
//!
//! C15 (redeposit): once an address is on record, a further `deposit` MUST carry
//! the same `user_address` — `Poseidon2(user_address) == old_address_hash` —
//! otherwise the deposit is rejected so funds cannot be misrouted to an aliased
//! address. (`user_address` byte-range C3 is enforced structurally by the
//! `[u8; 20]` representation; the circuit range-checks it in `build_slot_constraints`.)

use crate::field::all_zero;
use crate::flags::TxFlags;
use crate::{hash, SlotInput, SlotRejection};

/// Post-tx identity fields plus the `is_key_setter` predicate the auth layer
/// needs (a key rotation must be signed by the *old* key — C13).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddressEffects {
    pub new_pk_hash: [u64; 4],
    pub new_address_hash: [u64; 4],
    /// `is_key_register_only && new_pk_hash != 0` — a real key install/rotation.
    pub is_key_setter: bool,
}

/// Resolve the post-tx `pk_hash` / `address_hash` and enforce C15, mirroring
/// `build_address_constraints`. The full four-element hashes are returned (the
/// Merkle leaf truncates them to two elements; auth/redeposit use all four).
pub(crate) fn build_address(
    input: &SlotInput,
    flags: &TxFlags,
) -> Result<AddressEffects, SlotRejection> {
    // A key setter is a key_register_only slot that actually supplies a pk.
    let is_new_pk_provided = !all_zero(&input.new_pk_hash);
    let is_key_setter = flags.is_key_register_only && is_new_pk_provided;
    let new_pk_hash = if is_key_setter {
        input.new_pk_hash
    } else {
        input.old_pk_hash
    };

    // The address is set on the first deposit OR first key registration, i.e.
    // exactly when no address is yet on record.
    let is_addr_all_zero = all_zero(&input.old_address_hash);
    let is_deposit_first = flags.is_deposit && is_addr_all_zero;
    let is_key_setter_first = is_key_setter && is_addr_all_zero;
    let is_address_setter = is_deposit_first || is_key_setter_first;

    // A deposit onto an already-addressed account is a redeposit (C15). It is
    // mutually exclusive with `is_address_setter` (one needs addr==0, the other
    // addr!=0), so the branches below never overlap.
    let is_dep_redeposit = flags.is_deposit && !is_addr_all_zero;

    let new_address_hash = if is_address_setter {
        // First deposit / first key registration: stamp the address from the
        // supplied bytes. Hashed only here, where the result is actually used.
        hash::address_hash(input.user_address)
    } else if is_dep_redeposit {
        // C15: the redeposit address must match the one already committed.
        let computed = hash::address_hash(input.user_address);
        if computed != input.old_address_hash {
            return Err(SlotRejection::AddressHashMismatch);
        }
        input.old_address_hash
    } else {
        // Any other slot (incl. key_register_only onto an existing address, whose
        // user_address the circuit deliberately ignores) leaves the address as-is.
        input.old_address_hash
    };

    Ok(AddressEffects {
        new_pk_hash,
        new_address_hash,
        is_key_setter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TX_DEPOSIT, TX_KEY_REGISTER_ONLY, TX_NOOP, TX_WITHDRAWAL};

    const PK: [u64; 4] = [11, 22, 33, 44];
    const OLD_PK: [u64; 4] = [1, 2, 3, 4];

    fn addr_bytes(seed: u8) -> [u8; 20] {
        core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
    }

    fn input(tx_type: u8) -> SlotInput {
        SlotInput {
            tx_type,
            ..SlotInput::default()
        }
    }

    #[test]
    fn noop_keeps_identity() {
        let mut inp = input(TX_NOOP);
        inp.old_pk_hash = OLD_PK;
        inp.old_address_hash = [9, 9, 9, 9];
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_NOOP)).unwrap();
        assert_eq!(e.new_pk_hash, OLD_PK);
        assert_eq!(e.new_address_hash, [9, 9, 9, 9]);
        assert!(!e.is_key_setter);
    }

    #[test]
    fn first_deposit_sets_address() {
        let mut inp = input(TX_DEPOSIT);
        inp.user_address = addr_bytes(1);
        // old_address_hash defaults to zero ⇒ first deposit.
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_DEPOSIT)).unwrap();
        assert_eq!(e.new_address_hash, hash::address_hash(addr_bytes(1)));
        assert!(!e.is_key_setter);
    }

    #[test]
    fn redeposit_matching_address_passes() {
        let addr = addr_bytes(2);
        let mut inp = input(TX_DEPOSIT);
        inp.user_address = addr;
        inp.old_address_hash = hash::address_hash(addr);
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_DEPOSIT)).unwrap();
        // address unchanged on redeposit.
        assert_eq!(e.new_address_hash, hash::address_hash(addr));
    }

    #[test]
    fn redeposit_mismatching_address_is_rejected_c15() {
        let mut inp = input(TX_DEPOSIT);
        inp.user_address = addr_bytes(3);
        inp.old_address_hash = hash::address_hash(addr_bytes(99)); // different addr
        let err = build_address(&inp, &TxFlags::from_tx_type(TX_DEPOSIT)).unwrap_err();
        assert_eq!(err, SlotRejection::AddressHashMismatch);
    }

    #[test]
    fn key_register_first_sets_pk_and_address() {
        let mut inp = input(TX_KEY_REGISTER_ONLY);
        inp.new_pk_hash = PK;
        inp.old_pk_hash = [0; 4];
        inp.user_address = addr_bytes(4);
        // old_address_hash zero ⇒ key setter also stamps the address.
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_KEY_REGISTER_ONLY)).unwrap();
        assert!(e.is_key_setter);
        assert_eq!(e.new_pk_hash, PK);
        assert_eq!(e.new_address_hash, hash::address_hash(addr_bytes(4)));
    }

    #[test]
    fn key_rotation_keeps_existing_address_and_ignores_user_address() {
        let mut inp = input(TX_KEY_REGISTER_ONLY);
        inp.new_pk_hash = PK;
        inp.old_pk_hash = OLD_PK;
        inp.old_address_hash = [7, 7, 7, 7];
        inp.user_address = addr_bytes(5); // ignored: address already on record
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_KEY_REGISTER_ONLY)).unwrap();
        assert!(e.is_key_setter);
        assert_eq!(e.new_pk_hash, PK);
        assert_eq!(e.new_address_hash, [7, 7, 7, 7]);
    }

    #[test]
    fn key_register_without_pk_is_not_a_setter() {
        let mut inp = input(TX_KEY_REGISTER_ONLY);
        inp.new_pk_hash = [0; 4]; // no pk provided
        inp.old_pk_hash = OLD_PK;
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_KEY_REGISTER_ONLY)).unwrap();
        assert!(!e.is_key_setter);
        assert_eq!(e.new_pk_hash, OLD_PK);
    }

    #[test]
    fn non_deposit_never_runs_c15() {
        // A withdrawal carrying a wrong user_address must NOT trip the redeposit
        // check — C15 is deposit-only.
        let mut inp = input(TX_WITHDRAWAL);
        inp.old_address_hash = hash::address_hash(addr_bytes(6));
        inp.user_address = addr_bytes(123);
        let e = build_address(&inp, &TxFlags::from_tx_type(TX_WITHDRAWAL)).unwrap();
        assert_eq!(e.new_address_hash, hash::address_hash(addr_bytes(6)));
    }
}
