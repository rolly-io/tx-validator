//! Provider allowance — the per-account cap on operator-authorized `bet`
//! debits, the native mirror of `build_allowance_constraints`
//! (slot/allowance.rs).
//!
//! A provider `bet` (tx 2) is placed by a game provider on the owner's behalf
//! and carries no owner signature. Without a cap the operator could drain any
//! account through provider bets; the allowance closes that hole:
//!
//! ```text
//! set_provider_allowance (13)  allowance := amount        (owner-signed, absolute)
//! bet                    (2)   require amount <= allowance; allowance -= amount
//! everything else              allowance unchanged
//! ```
//!
//! `win` (3) deliberately does NOT restore the allowance ("strict" mode): the
//! allowance is the total the owner lets providers debit, not a rolling
//! exposure. Revocation is `set_provider_allowance` with `amount = 0`.
//!
//! Storage: the 62-bit allowance is packed as two 31-bit halves above the u32
//! balance limbs in the two leading balance-leaf fields
//! (`field::pack_balance_fields`); `allowance = 0` leaves every pre-allowance
//! leaf byte-identical. The circuit range-checks the post-tx allowance to 62
//! bits in one `split_low_high`, which doubles as the bet borrow check; here
//! the same two bounds surface as typed rejections.

use crate::field::PROVIDER_ALLOWANCE_MAX;
use crate::flags::TxFlags;
use crate::{SlotInput, SlotRejection};

/// Compute the post-tx provider allowance for any tx type, mirroring
/// `build_allowance_constraints`:
///   * `set_provider_allowance` ⇒ `new = amount`, rejected with
///     [`SlotRejection::ProviderAllowanceOverflow`] when `amount >= 2^62`
///     (the circuit's `split_allowance` range check);
///   * `bet` ⇒ `new = old − amount`, rejected with
///     [`SlotRejection::ProviderAllowanceExceeded`] when `amount > old` (the
///     circuit's 62-bit check on the wrapped difference);
///   * otherwise `new = old`.
///
/// The stored `old_provider_allowance` must itself be `< 2^62`: a larger value
/// has no packed-leaf representation, so the circuit could never have produced
/// it — rejected as `ProviderAllowanceOverflow` before any arithmetic.
pub(crate) fn compute_new_allowance(
    input: &SlotInput,
    flags: &TxFlags,
) -> Result<u64, SlotRejection> {
    let old = input.old_provider_allowance;
    if old > PROVIDER_ALLOWANCE_MAX {
        return Err(SlotRejection::ProviderAllowanceOverflow);
    }
    if flags.is_set_provider_allowance {
        if input.amount > PROVIDER_ALLOWANCE_MAX {
            return Err(SlotRejection::ProviderAllowanceOverflow);
        }
        return Ok(input.amount);
    }
    if flags.is_bet {
        // With `old < 2^62`, `amount <= old` is exactly the circuit's
        // "difference fits in 62 bits" condition (its `bet_amount < 2^62`
        // guard only rules out the modular wrap, which `checked_sub` cannot hit).
        return old
            .checked_sub(input.amount)
            .ok_or(SlotRejection::ProviderAllowanceExceeded);
    }
    Ok(old)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TX_BET, TX_DEPOSIT, TX_IN_HOUSE_BET, TX_NOOP, TX_SET_PROVIDER_ALLOWANCE, TX_WIN,
        TX_WITHDRAWAL,
    };

    fn run(tx_type: u8, old_allowance: u64, amount: u64) -> Result<u64, SlotRejection> {
        let input = SlotInput {
            tx_type,
            amount,
            old_provider_allowance: old_allowance,
            ..SlotInput::default()
        };
        compute_new_allowance(&input, &TxFlags::from_tx_type(tx_type))
    }

    #[test]
    fn set_is_absolute_and_revokes_with_zero() {
        assert_eq!(run(TX_SET_PROVIDER_ALLOWANCE, 0, 5_000_000), Ok(5_000_000));
        assert_eq!(run(TX_SET_PROVIDER_ALLOWANCE, 5_000_000, 12), Ok(12), "set is not additive");
        assert_eq!(run(TX_SET_PROVIDER_ALLOWANCE, 12, 0), Ok(0), "revoke");
        assert_eq!(
            run(TX_SET_PROVIDER_ALLOWANCE, 7, PROVIDER_ALLOWANCE_MAX),
            Ok(PROVIDER_ALLOWANCE_MAX),
            "2^62 - 1 is the largest storable allowance",
        );
    }

    #[test]
    fn set_above_62_bits_is_overflow() {
        for amount in [1u64 << 62, u64::MAX >> 1, u64::MAX] {
            assert_eq!(
                run(TX_SET_PROVIDER_ALLOWANCE, 0, amount),
                Err(SlotRejection::ProviderAllowanceOverflow),
                "amount {amount}",
            );
        }
    }

    #[test]
    fn bet_decrements_within_allowance_and_rejects_above() {
        let old = 10_000_000u64;
        for amount in [0u64, 1, 4_999_999, old] {
            assert_eq!(run(TX_BET, old, amount), Ok(old - amount), "bet {amount}");
        }
        for amount in [old + 1, 2 * old, PROVIDER_ALLOWANCE_MAX, 1u64 << 62, u64::MAX] {
            assert_eq!(
                run(TX_BET, old, amount),
                Err(SlotRejection::ProviderAllowanceExceeded),
                "bet {amount} > {old}",
            );
        }
        // Zero allowance: any non-zero provider bet is rejected, a zero bet passes.
        assert_eq!(run(TX_BET, 0, 1), Err(SlotRejection::ProviderAllowanceExceeded));
        assert_eq!(run(TX_BET, 0, 0), Ok(0));
    }

    #[test]
    fn other_types_keep_the_allowance() {
        let old = 777_777u64;
        // `win` is strict: it does NOT credit the allowance back.
        for tx in [TX_NOOP, TX_DEPOSIT, TX_WIN, TX_IN_HOUSE_BET, TX_WITHDRAWAL] {
            assert_eq!(run(tx, old, 123_456), Ok(old), "tx {tx}");
            // A huge amount is irrelevant for non-provider types.
            assert_eq!(run(tx, old, u64::MAX), Ok(old), "tx {tx}");
        }
    }

    #[test]
    fn stored_allowance_above_62_bits_is_unrepresentable() {
        // No circuit-produced leaf can hold such a value, so it is rejected
        // before it could be packed (which would panic).
        for tx in [TX_NOOP, TX_BET, TX_WIN, TX_SET_PROVIDER_ALLOWANCE] {
            assert_eq!(
                run(tx, PROVIDER_ALLOWANCE_MAX + 1, 0),
                Err(SlotRejection::ProviderAllowanceOverflow),
                "tx {tx}",
            );
        }
    }
}
