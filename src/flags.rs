//! One-hot transaction-type flags, the native mirror of the circuit's
//! `build_tx_flags` (slot/tx_flags.rs).
//!
//! The circuit does NOT branch per tx type: it decodes `tx_type` into 13
//! boolean flags and drives every per-slot constraint (balance, auth, address,
//! fairness) by *selecting* on those flags. The validator mirrors that shape —
//! one set of flag-driven computations rather than a `match` — so each layer
//! reproduces the circuit's flag-selection exactly (e.g. a deposit carrying a
//! `session_key` still skips auth, because `is_user_signed` is flag-gated).
//!
//! C1 (`flag_sum == 1`, i.e. `tx_type in 0..=12`) is enforced by
//! [`crate::validate_slot`] before these flags are built, so exactly one flag
//! is set here.

use crate::{
    TX_CRASH_SETTLE, TX_BET, TX_BONUS, TX_DEPOSIT, TX_IN_HOUSE_BET, TX_KEY_REGISTER_ONLY, TX_NOOP,
    TX_REFERRAL, TX_RISK_REJECT, TX_SET_INIT_SEED_HASH, TX_TRANSFER, TX_WIN, TX_WITHDRAWAL,
};

/// Native one-hot decode of `tx_type`. Exactly one field is `true` for a valid
/// (`0..=12`) tx type; mirrors `TxFlags` in slot/tx_flags.rs field-for-field.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TxFlags {
    /// `tx_type == 0`. Present for a field-for-field mirror of the circuit's
    /// `TxFlags`; the validator's flag-driven flow expresses noop as the absence
    /// of every other effect, so this flag is not read outside tests.
    #[allow(dead_code)]
    pub is_noop: bool,
    pub is_deposit: bool,
    pub is_bet: bool,
    pub is_win: bool,
    pub is_bonus: bool,
    pub is_in_house_bet: bool,
    pub is_withdrawal: bool,
    pub is_risk_reject: bool,
    pub is_set_init_seed_hash: bool,
    pub is_key_register_only: bool,
    pub is_transfer: bool,
    pub is_referral: bool,
    pub is_crash_settle: bool,
}

impl TxFlags {
    /// Decode a validated `tx_type` (caller guarantees `0..=12`). Any other
    /// value yields all-`false`, but the C1 check rejects those before here.
    pub(crate) fn from_tx_type(tx_type: u8) -> Self {
        Self {
            is_noop: tx_type == TX_NOOP,
            is_deposit: tx_type == TX_DEPOSIT,
            is_bet: tx_type == TX_BET,
            is_win: tx_type == TX_WIN,
            is_bonus: tx_type == TX_BONUS,
            is_in_house_bet: tx_type == TX_IN_HOUSE_BET,
            is_withdrawal: tx_type == TX_WITHDRAWAL,
            is_risk_reject: tx_type == TX_RISK_REJECT,
            is_set_init_seed_hash: tx_type == TX_SET_INIT_SEED_HASH,
            is_key_register_only: tx_type == TX_KEY_REGISTER_ONLY,
            is_transfer: tx_type == TX_TRANSFER,
            is_referral: tx_type == TX_REFERRAL,
            is_crash_settle: tx_type == TX_CRASH_SETTLE,
        }
    }

    /// `in_house_bet | crash_settle`: the two fairness/payout slots that add
    /// `win_amount` to the balance and contribute `slot_h` to the multiset.
    #[inline]
    pub(crate) fn is_ihb_or_crash(&self) -> bool {
        self.is_in_house_bet || self.is_crash_settle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_TX_TYPE;

    #[test]
    fn exactly_one_flag_per_valid_type() {
        for tx in 0..=MAX_TX_TYPE {
            let f = TxFlags::from_tx_type(tx);
            let set = [
                f.is_noop,
                f.is_deposit,
                f.is_bet,
                f.is_win,
                f.is_bonus,
                f.is_in_house_bet,
                f.is_withdrawal,
                f.is_risk_reject,
                f.is_set_init_seed_hash,
                f.is_key_register_only,
                f.is_transfer,
                f.is_referral,
                f.is_crash_settle,
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            assert_eq!(set, 1, "tx_type {tx} must set exactly one flag");
        }
    }

    #[test]
    fn discriminants_line_up_with_constants() {
        assert!(TxFlags::from_tx_type(TX_NOOP).is_noop);
        assert!(TxFlags::from_tx_type(TX_DEPOSIT).is_deposit);
        assert!(TxFlags::from_tx_type(TX_CRASH_SETTLE).is_crash_settle);
        assert!(TxFlags::from_tx_type(TX_IN_HOUSE_BET).is_ihb_or_crash());
        assert!(TxFlags::from_tx_type(TX_CRASH_SETTLE).is_ihb_or_crash());
        assert!(!TxFlags::from_tx_type(TX_BET).is_ihb_or_crash());
    }
}
