//! Session authorization, the native mirror of `build_auth_constraints`
//! (slot/auth.rs) and `add_session_expiry_check` (helpers/session.rs).
//!
//! The session model: a user's leaf stores `pk_hash = Poseidon2(session_key,
//! expiry)` (the four-element form; the leaf truncates to two). Any slot that
//! spends or moves user funds, rotates the key, or resets a live seed must be
//! "signed", i.e. carry the `(session_key, expiry)` pair that reproduces the
//! stored `pk_hash` (C13). Plain operator credits (win/bonus/referral),
//! deposits and the first seed/key initialization are unsigned.
//!
//! Checks (all gated on the same flags the circuit uses):
//!   * C11 — a base-signed type (bet | ihb | crash | withdrawal | transfer)
//!     requires a registered pk.
//!   * C12 — resetting a non-zero seed requires a registered pk.
//!   * C13 — when signing is required, `Poseidon2(session_key, expiry)` must
//!     equal `old_pk_hash`.
//!   * C14 — for regular signed ops and seed resets the session must be live:
//!     `block_ts <= expiry < block_ts + 2^40` (the 40-bit `split_le` window in
//!     session.rs). Key rotation is intentionally exempt from expiry so a user
//!     can rotate an expired key; the circuit still verifies its C13 hash.

use crate::field::all_zero;
use crate::flags::TxFlags;
use crate::{hash, SlotInput, SlotRejection};

/// Width of the session-expiry window enforced in `add_session_expiry_check`
/// (`split_le(expiry - block_timestamp, 40)`): the diff must be a 40-bit value,
/// so `expiry` lies in `[block_timestamp, block_timestamp + 2^40)`.
const SESSION_EXPIRY_BITS: u32 = 40;

/// Enforce the session-auth constraints for a candidate slot. `is_key_setter`
/// comes from the address layer and `is_seed_reset` from the (fairness) seed
/// layer — both feed the signed-set decode exactly as in the circuit.
pub(crate) fn check_auth(
    input: &SlotInput,
    flags: &TxFlags,
    is_key_setter: bool,
    is_seed_reset: bool,
) -> Result<(), SlotRejection> {
    let is_pk_registered = !all_zero(&input.old_pk_hash);

    // Base signed set: the value-moving / key-bearing types.
    let is_user_signed_base = flags.is_bet
        || flags.is_ihb_or_crash()
        || flags.is_withdrawal
        || flags.is_transfer;

    // C11: a value-moving tx without a registered key is unauthorized.
    if is_user_signed_base && !is_pk_registered {
        return Err(SlotRejection::SignedWithoutPk);
    }

    // C12: overwriting a live seed without a registered key is unauthorized.
    if is_seed_reset && !is_pk_registered {
        return Err(SlotRejection::SeedResetWithoutPk);
    }
    let seed_reset_needs_sig = is_seed_reset && is_pk_registered;

    // A key rotation (old pk present) must be authorized by the old key; the
    // first registration (no old pk) is unsigned.
    let is_key_setter_needs_sig = is_key_setter && is_pk_registered;
    let is_user_signed = is_user_signed_base || is_key_setter_needs_sig || seed_reset_needs_sig;

    // C13: when a signature is required, the supplied (session_key, expiry) must
    // reproduce the stored pk_hash. `should_check = is_user_signed &&
    // is_pk_registered`; for base-signed types C11 already guaranteed the pk is
    // registered, and the needs_sig predicates fold in `is_pk_registered` too.
    let should_check = is_user_signed && is_pk_registered;
    if should_check {
        let computed = hash::session_pk_hash(input.session_key, input.session_expiry);
        if computed != input.old_pk_hash {
            return Err(SlotRejection::SessionAuthMismatch);
        }
    }

    // C14: regular signed ops and seed resets need a live session. Key rotation
    // is excluded (matches `needs_expiry_check = is_user_signed_base ||
    // seed_reset_needs_sig`).
    let needs_expiry_check = is_user_signed_base || seed_reset_needs_sig;
    if needs_expiry_check && !session_is_live(input.session_expiry, input.max_block_timestamp) {
        return Err(SlotRejection::SessionExpired);
    }

    Ok(())
}

/// True iff `expiry - block_timestamp` is a canonical 40-bit value, i.e.
/// `block_timestamp <= expiry < block_timestamp + 2^40`. This is exactly the
/// window `split_le(expiry - block_timestamp, 40)` admits: an expired session
/// underflows in the field (huge, > 40 bits) and an absurd far-future expiry
/// exceeds 2^40 — both fail the circuit's range check.
fn session_is_live(expiry: u64, block_timestamp: u64) -> bool {
    match expiry.checked_sub(block_timestamp) {
        Some(diff) => diff < (1u64 << SESSION_EXPIRY_BITS),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TX_BET, TX_DEPOSIT, TX_KEY_REGISTER_ONLY, TX_NOOP, TX_SET_INIT_SEED_HASH, TX_WIN,
    };

    const KEY: [u64; 4] = [101, 202, 303, 404];
    const EXPIRY: u64 = 1_700_000_000;
    const BLOCK_TS: u64 = 1_699_000_000;

    /// Build a signed input whose (session_key, expiry) authenticates to the
    /// stored pk_hash, with a live session by default.
    fn signed_input(tx_type: u8) -> SlotInput {
        SlotInput {
            tx_type,
            session_key: KEY,
            session_expiry: EXPIRY,
            old_pk_hash: hash::session_pk_hash(KEY, EXPIRY),
            max_block_timestamp: BLOCK_TS,
            ..SlotInput::default()
        }
    }

    fn check(inp: &SlotInput, is_key_setter: bool, is_seed_reset: bool) -> Result<(), SlotRejection> {
        check_auth(inp, &TxFlags::from_tx_type(inp.tx_type), is_key_setter, is_seed_reset)
    }

    #[test]
    fn unsigned_types_skip_all_checks() {
        // noop / deposit / win carry no session and must pass regardless.
        for tx in [TX_NOOP, TX_DEPOSIT, TX_WIN] {
            let inp = SlotInput {
                tx_type: tx,
                ..SlotInput::default()
            };
            assert!(check(&inp, false, false).is_ok(), "tx {tx}");
        }
    }

    #[test]
    fn signed_bet_with_valid_session_passes() {
        assert!(check(&signed_input(TX_BET), false, false).is_ok());
    }

    #[test]
    fn signed_without_pk_is_rejected_c11() {
        let mut inp = signed_input(TX_BET);
        inp.old_pk_hash = [0; 4]; // no registered key
        assert_eq!(check(&inp, false, false), Err(SlotRejection::SignedWithoutPk));
    }

    #[test]
    fn wrong_session_key_is_rejected_c13() {
        let mut inp = signed_input(TX_BET);
        inp.session_key[0] ^= 1; // hash no longer matches old_pk_hash
        assert_eq!(
            check(&inp, false, false),
            Err(SlotRejection::SessionAuthMismatch)
        );
    }

    #[test]
    fn expired_session_is_rejected_c14() {
        let mut inp = signed_input(TX_BET);
        // expiry strictly before block timestamp ⇒ expired.
        inp.session_expiry = BLOCK_TS - 1;
        inp.old_pk_hash = hash::session_pk_hash(KEY, inp.session_expiry); // keep C13 valid
        assert_eq!(check(&inp, false, false), Err(SlotRejection::SessionExpired));
    }

    #[test]
    fn far_future_expiry_outside_40_bit_window_is_rejected() {
        let mut inp = signed_input(TX_BET);
        inp.session_expiry = BLOCK_TS + (1u64 << SESSION_EXPIRY_BITS); // diff == 2^40, out of window
        inp.old_pk_hash = hash::session_pk_hash(KEY, inp.session_expiry);
        assert_eq!(check(&inp, false, false), Err(SlotRejection::SessionExpired));
    }

    #[test]
    fn expiry_equal_to_block_ts_is_live() {
        let mut inp = signed_input(TX_BET);
        inp.session_expiry = BLOCK_TS; // diff == 0, in window
        inp.old_pk_hash = hash::session_pk_hash(KEY, inp.session_expiry);
        assert!(check(&inp, false, false).is_ok());
    }

    #[test]
    fn seed_reset_without_pk_is_rejected_c12() {
        // set_init over a live seed (is_seed_reset) needs a registered pk.
        let mut inp = signed_input(TX_SET_INIT_SEED_HASH);
        inp.old_pk_hash = [0; 4];
        assert_eq!(
            check(&inp, false, true),
            Err(SlotRejection::SeedResetWithoutPk)
        );
    }

    #[test]
    fn seed_reset_with_pk_checks_session_and_expiry() {
        // valid signature + live session ⇒ ok
        assert!(check(&signed_input(TX_SET_INIT_SEED_HASH), false, true).is_ok());
        // expired ⇒ rejected (seed reset is in needs_expiry_check)
        let mut inp = signed_input(TX_SET_INIT_SEED_HASH);
        inp.session_expiry = BLOCK_TS - 1;
        inp.old_pk_hash = hash::session_pk_hash(KEY, inp.session_expiry);
        assert_eq!(check(&inp, false, true), Err(SlotRejection::SessionExpired));
    }

    #[test]
    fn key_rotation_checks_session_but_not_expiry() {
        // Key rotation (is_key_setter + registered pk) must verify C13 but is
        // exempt from the expiry window: an expired key can still be rotated.
        let mut inp = signed_input(TX_KEY_REGISTER_ONLY);
        inp.session_expiry = BLOCK_TS - 5_000; // expired
        inp.old_pk_hash = hash::session_pk_hash(KEY, inp.session_expiry);
        assert!(
            check(&inp, true, false).is_ok(),
            "expired key rotation should still pass C14"
        );

        // ...but a wrong signature still fails C13.
        let mut bad = inp.clone();
        bad.session_key[1] ^= 0xFF;
        assert_eq!(check(&bad, true, false), Err(SlotRejection::SessionAuthMismatch));
    }

    #[test]
    fn first_key_registration_is_unsigned() {
        // is_key_setter but no old pk ⇒ first registration, no signature needed.
        let inp = SlotInput {
            tx_type: TX_KEY_REGISTER_ONLY,
            old_pk_hash: [0; 4],
            ..SlotInput::default()
        };
        assert!(check(&inp, true, false).is_ok());
    }
}
