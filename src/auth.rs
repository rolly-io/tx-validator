//! Native mirror of MainCircuit's authenticated-transaction decode.
//!
//! Schnorr verification lives in a side-circuit and is connected to MainCircuit
//! through the auth chain. The slot validator therefore enforces the local
//! prerequisite only: every Schnorr-authenticated operation must reference an
//! account with a non-zero registered `pk_hash`.
//!
//! It also decodes WHICH nonce lane the signer committed to (see the two-lane
//! nonce in `lib.rs`): bets and `set_provider_allowance` sign the game lane,
//! withdrawals / transfers the account lane. The circuit's
//! `signed_nonce = select(is_acct_lane, acct, game)` is driven by the same
//! predicate.

use crate::field::all_zero;
use crate::flags::TxFlags;
use crate::{SlotInput, SlotRejection};

/// Flag-driven auth decode of one slot, mirroring the circuit's
/// `is_authenticated` / `is_acct_lane` / `is_game_bump` signals. At most one
/// lane flag is set (one-hot tx flags); both are `false` for unsigned types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AuthDecode {
    /// The slot is folded into the Schnorr auth chain.
    pub is_authenticated: bool,
    /// `withdrawal | transfer`: signed under, and bumps, the account lane.
    pub is_account_lane: bool,
    /// `in_house_bet | crash_settle | set_provider_allowance`: signed under,
    /// and bumps, the game lane.
    pub is_game_lane: bool,
}

/// Decode whether this slot contributes to the Schnorr auth chain and which
/// nonce lane its signature committed to.
///
/// Provider `bet` (2) is operator-authorized; what the owner signs is the
/// provider ALLOWANCE that caps it (`set_provider_allowance`, 13 — see
/// [`crate::allowance`]). An account without a registered Schnorr key can
/// therefore never hold a non-zero allowance, so provider bets against it are
/// rejected by the allowance check — by design. Key registration is authorized
/// by the separate EIP-712 registration chain and therefore is not part of
/// this set (its account-lane bump is gated by `is_key_setter` in the caller,
/// not by this decode).
pub(crate) fn check_auth(
    input: &SlotInput,
    flags: &TxFlags,
) -> Result<AuthDecode, SlotRejection> {
    let is_game_lane = flags.is_game_lane();
    let is_account_lane = flags.is_withdrawal || flags.is_transfer;
    let is_authenticated = is_game_lane || is_account_lane;

    if is_authenticated && all_zero(&input.old_pk_hash) {
        return Err(SlotRejection::SignedWithoutPk);
    }

    Ok(AuthDecode {
        is_authenticated,
        is_account_lane,
        is_game_lane,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TX_BET, TX_CRASH_SETTLE, TX_DEPOSIT, TX_IN_HOUSE_BET, TX_KEY_REGISTER_ONLY, TX_NOOP,
        TX_SET_INIT_SEED_HASH, TX_SET_PROVIDER_ALLOWANCE, TX_TRANSFER, TX_WITHDRAWAL, TX_WIN,
    };

    fn check(input: &SlotInput) -> Result<AuthDecode, SlotRejection> {
        check_auth(input, &TxFlags::from_tx_type(input.tx_type))
    }

    #[test]
    fn operator_and_registration_types_skip_schnorr_auth() {
        for tx_type in [
            TX_NOOP,
            TX_DEPOSIT,
            TX_BET,
            TX_WIN,
            TX_SET_INIT_SEED_HASH,
            TX_KEY_REGISTER_ONLY,
        ] {
            let input = SlotInput {
                tx_type,
                ..SlotInput::default()
            };
            assert_eq!(check(&input), Ok(AuthDecode::default()), "tx_type {tx_type}");
        }
    }

    #[test]
    fn authenticated_types_require_registered_pk() {
        for tx_type in [
            TX_IN_HOUSE_BET,
            TX_WITHDRAWAL,
            TX_TRANSFER,
            TX_CRASH_SETTLE,
            TX_SET_PROVIDER_ALLOWANCE,
        ] {
            let mut input = SlotInput {
                tx_type,
                ..SlotInput::default()
            };
            assert_eq!(
                check(&input),
                Err(SlotRejection::SignedWithoutPk),
                "tx_type {tx_type}",
            );

            input.old_pk_hash = [1, 2, 3, 4];
            let decoded = check(&input).unwrap_or_else(|e| panic!("tx_type {tx_type}: {e}"));
            assert!(decoded.is_authenticated, "tx_type {tx_type}");
            // Exactly one lane per authenticated type.
            assert_ne!(decoded.is_account_lane, decoded.is_game_lane, "tx_type {tx_type}");
        }
    }

    #[test]
    fn signed_lane_follows_tx_type() {
        let pk = [1u64, 2, 3, 4];
        for (tx_type, account_lane) in [
            (TX_IN_HOUSE_BET, false),
            (TX_CRASH_SETTLE, false),
            (TX_SET_PROVIDER_ALLOWANCE, false),
            (TX_WITHDRAWAL, true),
            (TX_TRANSFER, true),
        ] {
            let input = SlotInput {
                tx_type,
                old_pk_hash: pk,
                ..SlotInput::default()
            };
            assert_eq!(
                check(&input),
                Ok(AuthDecode {
                    is_authenticated: true,
                    is_account_lane: account_lane,
                    is_game_lane: !account_lane,
                }),
                "tx_type {tx_type}",
            );
        }
    }
}
