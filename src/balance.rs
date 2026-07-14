//! Per-type balance `+/-` and deposit-credit cap, the native mirror of
//! `build_balance_changes` (slot/balance.rs).
//!
//! The circuit performs the balance transition with flag-selected limbs:
//! `is_user_add` (deposit | win | referral | bonus) adds `amount`;
//! `is_ihb_or_crash` (in_house_bet | crash_settle) adds `win_amount`;
//! `is_user_sub` (bet | ihb | crash | withdrawal | risk_reject | transfer)
//! subtracts `amount`. The result is
//! `new_balance = old_balance + total_add - user_sub`, computed via
//! `add_u64_in_circuit` (rejects 64-bit overflow) then `sub_u64_in_circuit`
//! (rejects underflow — this is C6). The deposit-credit is `old_credit` (plus
//! `amount` on deposit), auto-capped to `min(credit, new_balance)` for the
//! `is_balance_loss` set (bet | ihb | crash | withdrawal | transfer).
//!
//! The validator reproduces the identical add-then-sub ordering with checked
//! `u64` arithmetic so the C6 boundary matches bit-for-bit (note: for
//! IHB/crash C6 is therefore `old_balance + win >= bet`, not
//! `old_balance >= bet`).

use crate::flags::TxFlags;
use crate::{SlotInput, SlotRejection};

/// Post-tx balance/credit plus the full-`u64` add/sub deltas the TL-chain
/// consumes (`new_tl = old_tl + add_full - sub_full`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BalanceEffects {
    pub new_balance: u64,
    pub new_credit: u64,
    /// `user_add_full`: balance credited this slot (`amount` for user-add types,
    /// `win_amount` for IHB/crash, else 0) — the TL add delta.
    pub add_full: u64,
    /// `user_sub_full`: balance debited this slot (`amount` for the sub set,
    /// else 0) — the TL sub delta.
    pub sub_full: u64,
}

/// Compute the balance/credit transition for any tx type, mirroring
/// `build_balance_changes`. Returns [`SlotRejection::InsufficientBalance`] (C6)
/// on a debit underflow and [`SlotRejection::TlOverflow`] on a 64-bit add
/// overflow — the validator's u64 analogue of the circuit's `add_u64_in_circuit`
/// `overflow == 0` connect. (The plan folds monetary-accumulator overflow into
/// `TlOverflow`: a `deposit` moves balance, credit and TL by the same `amount`,
/// so it lists `TlOverflow` as the single reachable overflow; real bounded
/// casino balances never approach 2^64 regardless.)
pub(crate) fn build_balance_changes(
    input: &SlotInput,
    flags: &TxFlags,
) -> Result<BalanceEffects, SlotRejection> {
    let is_user_add = flags.is_deposit || flags.is_win || flags.is_referral || flags.is_bonus;
    let is_ihb_or_crash = flags.is_ihb_or_crash();

    // deposit/win/referral/bonus add `amount`; IHB/crash add `win_amount`.
    // The two predicates are disjoint by tx type, so at most one term is
    // non-zero; `checked_add` is purely defensive against a future overlap.
    let user_add = if is_user_add { input.amount } else { 0 };
    let ihb_add = if is_ihb_or_crash { input.win_amount } else { 0 };
    let total_add = user_add
        .checked_add(ihb_add)
        .ok_or(SlotRejection::TlOverflow)?;

    let is_user_sub = flags.is_bet
        || is_ihb_or_crash
        || flags.is_withdrawal
        || flags.is_risk_reject
        || flags.is_transfer;
    let user_sub = if is_user_sub { input.amount } else { 0 };

    // add-then-sub, matching the circuit's `add_u64_in_circuit` (overflow→reject)
    // followed by `sub_u64_in_circuit` (underflow→reject == C6).
    let after_add = input
        .old_balance
        .checked_add(total_add)
        .ok_or(SlotRejection::TlOverflow)?;
    let new_balance = after_add
        .checked_sub(user_sub)
        .ok_or(SlotRejection::InsufficientBalance)?;

    // deposit grows credit (uncapped); every other type leaves it at old_credit.
    let deposit_credit_add = if flags.is_deposit { input.amount } else { 0 };
    let credit_post_deposit = input
        .old_deposit_credit
        .checked_add(deposit_credit_add)
        .ok_or(SlotRejection::TlOverflow)?;

    // auto-cap to min(credit, new_balance) — the circuit selects the smaller via
    // the high bit of (new_balance - credit); `u64::min` is the native analogue.
    let capped_credit = credit_post_deposit.min(new_balance);
    let is_balance_loss = flags.is_bet
        || is_ihb_or_crash
        || flags.is_withdrawal
        || flags.is_transfer;
    let new_credit = if is_balance_loss {
        capped_credit
    } else {
        credit_post_deposit
    };

    Ok(BalanceEffects {
        new_balance,
        new_credit,
        add_full: total_add,
        sub_full: user_sub,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TX_BET, TX_BONUS, TX_DEPOSIT, TX_NOOP, TX_REFERRAL, TX_RISK_REJECT, TX_TRANSFER, TX_WIN,
        TX_WITHDRAWAL,
    };

    fn input_with(old_balance: u64, old_credit: u64, amount: u64, win_amount: u64) -> SlotInput {
        SlotInput {
            old_balance,
            old_deposit_credit: old_credit,
            amount,
            win_amount,
            ..SlotInput::default()
        }
    }

    fn run(tx_type: u8, old_balance: u64, old_credit: u64, amount: u64) -> BalanceEffects {
        let flags = TxFlags::from_tx_type(tx_type);
        build_balance_changes(&input_with(old_balance, old_credit, amount, 0), &flags)
            .expect("balance must succeed")
    }

    #[test]
    fn noop_keeps_balance_and_credit() {
        let e = run(TX_NOOP, 1000, 400, 0);
        assert_eq!((e.new_balance, e.new_credit), (1000, 400));
        assert_eq!((e.add_full, e.sub_full), (0, 0));
    }

    #[test]
    fn deposit_adds_uncapped_credit() {
        // balance and credit both grow by amount; credit is NOT capped.
        let e = run(TX_DEPOSIT, 1000, 400, 250);
        assert_eq!((e.new_balance, e.new_credit), (1250, 650));
        assert_eq!((e.add_full, e.sub_full), (250, 0));
    }

    #[test]
    fn win_bonus_referral_add_without_touching_credit() {
        for tx in [TX_WIN, TX_BONUS, TX_REFERRAL] {
            let e = run(tx, 1000, 400, 70);
            assert_eq!((e.new_balance, e.new_credit), (1070, 400), "tx {tx}");
            assert_eq!((e.add_full, e.sub_full), (70, 0), "tx {tx}");
        }
    }

    #[test]
    fn bet_subtracts_and_caps_credit_to_balance() {
        // new_balance = 300 < old_credit 400 ⇒ credit capped to 300.
        let e = run(TX_BET, 1000, 400, 700);
        assert_eq!((e.new_balance, e.new_credit), (300, 300));
        assert_eq!((e.add_full, e.sub_full), (0, 700));
    }

    #[test]
    fn loss_keeps_credit_when_balance_stays_above_it() {
        // new_balance = 600 >= old_credit 400 ⇒ credit unchanged.
        for tx in [TX_BET, TX_WITHDRAWAL, TX_TRANSFER] {
            let e = run(tx, 1000, 400, 400);
            assert_eq!((e.new_balance, e.new_credit), (600, 400), "tx {tx}");
        }
    }

    #[test]
    fn risk_reject_never_caps_credit() {
        // risk_reject debits the balance below old_credit but credit stays put
        // (it is excluded from the is_balance_loss cap set).
        let e = run(TX_RISK_REJECT, 1000, 900, 600);
        assert_eq!((e.new_balance, e.new_credit), (400, 900));
        assert_eq!((e.add_full, e.sub_full), (0, 600));
    }

    #[test]
    fn ihb_adds_win_then_subtracts_bet() {
        // in_house_bet: new_balance = old + win - bet, credit capped to result.
        let flags = TxFlags::from_tx_type(crate::TX_IN_HOUSE_BET);
        let e = build_balance_changes(&input_with(1000, 900, 200, 500), &flags).unwrap();
        assert_eq!(e.new_balance, 1300); // 1000 + 500 - 200
        assert_eq!(e.new_credit, 900); // 900 <= 1300 ⇒ unchanged
        assert_eq!((e.add_full, e.sub_full), (500, 200)); // TL: +win, -bet
    }

    #[test]
    fn ihb_c6_uses_balance_plus_win() {
        // old_balance 100 < bet 400, but old_balance + win 500 >= bet ⇒ OK.
        let flags = TxFlags::from_tx_type(crate::TX_IN_HOUSE_BET);
        let e = build_balance_changes(&input_with(100, 0, 400, 500), &flags).unwrap();
        assert_eq!(e.new_balance, 200); // 100 + 500 - 400
    }

    #[test]
    fn underflow_is_insufficient_balance() {
        let flags = TxFlags::from_tx_type(TX_BET);
        let err = build_balance_changes(&input_with(100, 0, 101, 0), &flags).unwrap_err();
        assert_eq!(err, SlotRejection::InsufficientBalance);
    }

    #[test]
    fn add_overflow_is_tl_overflow() {
        // Monetary-accumulator overflow folds into TlOverflow (see fn docs).
        let flags = TxFlags::from_tx_type(TX_WIN);
        let err = build_balance_changes(&input_with(u64::MAX, 0, 1, 0), &flags).unwrap_err();
        assert_eq!(err, SlotRejection::TlOverflow);
    }
}
