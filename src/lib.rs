//! Native, circuit-aligned per-tx-type slot validator for the Rolly ZK rollup.
//!
//! A single bad slot makes the whole 1500-tx block proof unverifiable. This
//! crate re-checks every per-slot constraint of the slot circuit (plus a
//! payout defense-in-depth layer) on a candidate witness BEFORE it mutates the
//! state tree / total-liability accumulator, returning a typed rejection
//! instead of poisoning the block.
//!
//! It is a plain rlib with no `wasm-bindgen` / `getrandom(js)` so the prover
//! circuit crate can dev-depend on it natively for the differential test that
//! pins this second implementation to the circuit. The backend consumes it
//! through a thin `#[wasm_bindgen] validate_slot` wrapper in wasm-signer.

use serde::{Deserialize, Serialize};

pub mod field;
pub mod hash;
pub mod merkle;

// Per-type effect layers (balance/credit, address/pk, session auth). Internal
// to the validator; the public surface stays `validate_slot` + the I/O types.
// Each mirrors one circuit module so the second implementation tracks the first.
mod address;
mod auth;
mod balance;
mod fairness;
mod flags;

/// Sparse Merkle tree depth — MUST stay in lockstep with the circuit's
/// `TREE_DEPTH`. The differential test (circuit dev-depends on this crate)
/// fails to compile if these drift, because `SlotInput` is constructed from
/// the circuit's `SlotWitness` and the `main_siblings` array length must match.
pub const TREE_DEPTH: usize = 28;

/// Highest valid transaction type. Types are `0..=MAX_TX_TYPE` inclusive
/// (noop, deposit, bet, win, bonus, in_house_bet, withdrawal, risk_reject,
/// set_init_seed_hash, key_register_only, transfer, referral, crash_settle).
pub const MAX_TX_TYPE: u8 = 12;

// Transaction type discriminants — mirror the circuit's one-hot decode in
// `slot/tx_flags.rs`. Named here so the per-type effect layer (follow-up
// to-dos) and callers can dispatch without bare magic numbers.
pub const TX_NOOP: u8 = 0;
pub const TX_DEPOSIT: u8 = 1;
pub const TX_BET: u8 = 2;
pub const TX_WIN: u8 = 3;
pub const TX_BONUS: u8 = 4;
pub const TX_IN_HOUSE_BET: u8 = 5;
pub const TX_WITHDRAWAL: u8 = 6;
pub const TX_RISK_REJECT: u8 = 7;
pub const TX_SET_INIT_SEED_HASH: u8 = 8;
pub const TX_KEY_REGISTER_ONLY: u8 = 9;
pub const TX_TRANSFER: u8 = 10;
pub const TX_REFERRAL: u8 = 11;
pub const TX_CRASH_SETTLE: u8 = 12;

/// Full candidate witness for one slot, mirroring the circuit's native
/// `SlotWitness`, plus the ambient block state the validator checks against
/// and the raw game parameters needed only by the payout layer.
///
/// Opaque field-element values are carried as raw `u64` (not reduced
/// `GoldilocksField`) precisely so the validator can enforce canonicity
/// (`< Goldilocks p`) itself — the point of the C2b/C2c checks. The circuit's
/// `from_canonical_u64` would otherwise panic (debug) or silently reduce mod p
/// (release), desynchronizing the TS/client commitments from the circuit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SlotInput {
    // --- mirror of the native SlotWitness ---
    pub tx_type: u8,
    pub user_id: u32,
    pub amount: u64,
    pub win_amount: u64,
    pub old_balance: u64,
    pub old_deposit_credit: u64,
    pub main_siblings: [[u64; 4]; TREE_DEPTH],

    pub server_seed: [u64; 8],
    pub user_seed: [u64; 4],
    pub old_seed_hash: [u64; 3],
    pub next_server_seed_hash: [u64; 3],

    pub session_key: [u64; 4],
    pub session_expiry: u64,

    pub old_pk_hash: [u64; 4],
    pub new_pk_hash: [u64; 4],

    pub user_address: [u8; 20],
    pub old_address_hash: [u64; 4],

    pub old_total_liability: u64,

    pub game_id: u8,
    pub prediction_hash: [u64; 4],
    pub user_secret_random: [u64; 4],

    // --- ambient block state the candidate is checked against ---
    /// Current Merkle root the old leaf must authenticate against (C16).
    pub current_root: [u64; 4],
    /// Current total-liability the slot's `old_total_liability` must equal (C18).
    pub current_tl: u64,
    /// `now + MAX_BLOCK_LATENCY`: session expiry must be `>=` this (C14).
    pub max_block_timestamp: u64,

    // --- raw game params (payout layer only: in_house_bet / crash_settle) ---
    // The witness carries only `prediction_hash` and `win_amount`, but
    // `compute_payout` / `compute_prediction_hash` need the raw parameters to
    // verify the payout and reproduce the prediction hash.
    #[serde(default)]
    pub game_mode: u32,
    #[serde(default)]
    pub prediction_lo: u32,
    #[serde(default)]
    pub prediction_hi: u32,

    /// Keno-only: the player's selected numbers (0-indexed, 1..=10 unique values
    /// in `[0, 39]`). Needed for the 13-element `prediction_hash_keno` and
    /// `keno::compute_payout`. Empty for all non-Keno games.
    #[serde(default)]
    pub keno_selected: Vec<u8>,

    /// crash_settle only: the 3-element fairness-seed commitment of the
    /// crash house account (leaf 0), fixed at round start. When `Some`,
    /// the validator enforces `Poseidon2(server_seed)[0..3] == crash_seed_hash`
    /// (the native mirror of the circuit's per-block reveal→commit check). When
    /// `None`, the check is skipped — callers without the commitment in hand
    /// (e.g. the differential test) still validate the rest of the slot.
    #[serde(default)]
    pub crash_seed_hash: Option<[u64; 3]>,
}

/// Authoritative post-tx state effects produced by a successful validation.
/// Returning these lets the validator be the single source of truth so the TS
/// layer can stop recomputing them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotEffects {
    pub new_balance: u64,
    pub new_credit: u64,
    pub new_seed_hash: [u64; 3],
    pub new_pk_hash: [u64; 4],
    pub new_address_hash: [u64; 4],
    pub new_root: [u64; 4],
    pub new_total_liability: u64,
    /// Fairness output: in_house_bet = `Poseidon2(server_seed || user_seed)`;
    /// crash_settle = `server_seed[0..4]` (raw); zero for all other tx types.
    pub random: [u64; 4],
    /// `Poseidon2(random, bet_full, win_full, game_id, prediction_hash)` — the
    /// multiset commitment. Meaningful only when `is_multiset_slot` is true.
    pub slot_h: [u64; 4],
    /// Whether `slot_h` is contributed to the multiset accumulator (true only
    /// for in_house_bet / crash_settle); any other slot must NOT add it.
    pub is_multiset_slot: bool,
}

/// Typed reason a candidate slot was rejected. Each variant maps 1:1 to an RPC
/// `{ ok: false, reason }`, so one bad slot is dropped cleanly instead of
/// bricking the block proof. Annotated with the constraint it guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotRejection {
    /// tx_type outside `0..=12` (C1: `flag_sum == 1`).
    BadTxType,
    /// An amount limb (amount/win/balance/credit, lo or hi) is `>= 2^32` (C2).
    AmountLimbOverflow,
    /// An opaque field limb is `>= Goldilocks p` (C2b).
    FieldNotCanonical,
    /// A full bet/win amount is `>= Goldilocks p` (C2c).
    AmountNotCanonical,
    /// A `user_address` byte is `>= 2^8` (C3).
    AddressByteOverflow,
    /// `user_id >= 2^TREE_DEPTH` (C17).
    UserIdOutOfRange,
    /// The Merkle proof of the old leaf does not reproduce `current_root` (C16).
    MerkleRootMismatch,
    /// Balance is below the bet/withdrawal/transfer amount (C6).
    InsufficientBalance,
    /// A u64 monetary accumulator over/underflowed: the TL update over/underflows
    /// u64, `old_total_liability != current_tl` (C5 / C18), or a balance/credit
    /// add overflows 64 bits (the validator's analogue of the circuit's
    /// `add_u64_in_circuit` `overflow == 0` connect — a `deposit` moves balance,
    /// credit and TL by the same amount, so the plan reports all three here).
    TlOverflow,
    /// `new_credit` does not equal `min(old_credit, new_balance)` where the
    /// credit cap applies.
    CreditCapMismatch,
    /// A signed tx type was submitted without a registered pk (C11).
    SignedWithoutPk,
    /// `set_init_seed_hash` reset an existing seed without a registered pk (C12).
    SeedResetWithoutPk,
    /// `Poseidon2(session_key, expiry) != old_pk_hash` (C13).
    SessionAuthMismatch,
    /// `session_expiry < max_block_timestamp` (C14).
    SessionExpired,
    /// `Poseidon2(server_seed)[0..3] != old_seed_hash` (C9, in_house_bet).
    SeedChainBreak,
    /// `user_seed != Poseidon2(game_id, bet, prediction_hash, user_secret)`
    /// (C10, in_house_bet).
    UserSeedBindingMismatch,
    /// `Poseidon2(user_address) != old_address_hash` on redeposit (C15).
    AddressHashMismatch,
    /// A `risk_reject` would drop the balance below `old_deposit_credit` (C4).
    RiskRejectBelowCredit,
    /// `win_amount != compute_payout(...)` or the `prediction_hash` does not
    /// match the raw game params (payout defense-in-depth layer).
    PayoutMismatch,
    /// `Poseidon2(server_seed)[0..3] != crash_seed_hash` on crash_settle: the
    /// revealed seed does not match the house-account (leaf 0) commitment fixed
    /// at round start (native mirror of the circuit's reveal→commit check).
    CrashSeedCommitMismatch,
}

impl core::fmt::Display for SlotRejection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SlotRejection {}

/// Re-check every per-slot constraint (and the payout defense-in-depth layer)
/// on a candidate witness, returning the authoritative post-tx effects on
/// success or a typed rejection the RPC layer maps to `{ ok: false, reason }`.
///
/// This pass implements the always-on layer that runs for EVERY tx type
/// (mirroring the circuit's unconditional `build_slot_constraints`):
///   * C1  — `tx_type` is in `0..=12`.
///   * C2b — every opaque field limb is canonical (`< p`).
///   * C2c — the full `amount` / `win_amount` are canonical (`< p`).
///   * C17 — `user_id < 2^TREE_DEPTH`.
///   * C16 — the old leaf authenticates against `current_root` (truncated
///     `balance_hash[0..4] || pk_hash[0..2] || address_hash[0..2]` layout).
///   * C18 — `old_total_liability == current_tl` (TL-chain).
///
/// C2 (u32 limbs) and C3 (address bytes) are enforced structurally by the
/// `u64` / `[u8; 20]` representation of [`SlotInput`] — both limbs of any `u64`
/// are `< 2^32` and any `u8` is `< 2^8`, so no runtime comparison can fail.
///
/// On top of the always-on layer, [`compute_effects`] applies the per-type
/// balance `+/-` and credit cap ([`balance`], C6), the risk_reject floor (C4),
/// the address/pk setter and redeposit check ([`address`], C15), session auth
/// ([`auth`], C11/C12/C13/C14), and — for `in_house_bet` / `crash_settle` —
/// the fairness layer ([`fairness`], C9 seed-chain, C10 user_seed binding,
/// random derivation, payout defense-in-depth via `rolly-game-core`, and the
/// `slot_h` multiset commitment). Then packs the new root / TL.
pub fn validate_slot(input: &SlotInput) -> Result<SlotEffects, SlotRejection> {
    // ── Always-on input validation ──
    check_tx_type(input)?; // C1
    check_field_canonicity(input)?; // C2b
    check_amount_canonicity(input)?; // C2c
    check_user_id_range(input)?; // C17

    // ── Always-on state consistency ──
    verify_current_root(input)?; // C16
    if input.old_total_liability != input.current_tl {
        return Err(SlotRejection::TlOverflow); // C18 (TL-chain)
    }

    // ── Per-type effects (balance/auth/fairness/payout) ──
    compute_effects(input)
}

/// C1: `tx_type` must decode to exactly one of the 13 one-hot flags.
fn check_tx_type(input: &SlotInput) -> Result<(), SlotRejection> {
    if input.tx_type > MAX_TX_TYPE {
        return Err(SlotRejection::BadTxType);
    }
    Ok(())
}

/// C2b: every opaque field-element limb carried as raw `u64` must be canonical.
/// Non-canonical limbs would make the prover's `from_canonical_u64` panic
/// (debug) or silently reduce `mod p` (release), desynchronizing the
/// TS/client commitments from the circuit.
fn check_field_canonicity(input: &SlotInput) -> Result<(), SlotRejection> {
    let ok = field::all_canonical(&input.server_seed)
        && field::all_canonical(&input.user_seed)
        && field::all_canonical(&input.old_seed_hash)
        && field::all_canonical(&input.next_server_seed_hash)
        && field::all_canonical(&input.session_key)
        && field::is_canonical(input.session_expiry)
        && field::all_canonical(&input.old_pk_hash)
        && field::all_canonical(&input.new_pk_hash)
        && field::all_canonical(&input.old_address_hash)
        && field::all_canonical(&input.prediction_hash)
        && field::all_canonical(&input.user_secret_random)
        && input.main_siblings.iter().all(|s| field::all_canonical(s));
    if ok {
        Ok(())
    } else {
        Err(SlotRejection::FieldNotCanonical)
    }
}

/// C2c: the full `amount` / `win_amount` are packed into a single field element
/// (`bet_full` / `win_full`) and fed into Poseidon2, so they must be canonical
/// (`< p`), not merely `< 2^64` as the u32 limbs alone would allow.
fn check_amount_canonicity(input: &SlotInput) -> Result<(), SlotRejection> {
    if field::is_canonical(input.amount) && field::is_canonical(input.win_amount) {
        Ok(())
    } else {
        Err(SlotRejection::AmountNotCanonical)
    }
}

/// C17: `user_id` indexes a `TREE_DEPTH`-deep tree, so it must be
/// `< 2^TREE_DEPTH` (the circuit enforces this via `split_le(user_id, DEPTH)`).
fn check_user_id_range(input: &SlotInput) -> Result<(), SlotRejection> {
    if (input.user_id as u64) < (1u64 << TREE_DEPTH) {
        Ok(())
    } else {
        Err(SlotRejection::UserIdOutOfRange)
    }
}

/// C16: rebuild the old leaf and authenticate it against `current_root`. The
/// key state-consistency check — proves the witness reflects the live tree.
///
/// Relies on C2b having already validated `old_seed_hash`, `old_pk_hash`,
/// `old_address_hash` and `main_siblings`, so the Poseidon2 calls cannot be fed
/// a non-canonical element.
fn verify_current_root(input: &SlotInput) -> Result<(), SlotRejection> {
    let old_balance_hash =
        hash::balance_leaf(input.old_balance, input.old_seed_hash, input.old_deposit_credit);
    let computed = merkle::compute_root(
        old_balance_hash,
        input.old_pk_hash,
        input.old_address_hash,
        &input.main_siblings,
        input.user_id,
    );
    if computed == input.current_root {
        Ok(())
    } else {
        Err(SlotRejection::MerkleRootMismatch)
    }
}

/// Pack the post-tx total liability into `u64`, rejecting under/overflow.
///
/// `new_tl = old_tl + add_full - sub_full`. Real liability is a non-wrapping
/// `u64` (bounded by deposits), so any add overflow or sub underflow is a
/// rejection rather than a silent field wrap — the validator catches it before
/// the value is chained into the next slot.
pub(crate) fn compute_new_tl(
    old_tl: u64,
    add_full: u64,
    sub_full: u64,
) -> Result<u64, SlotRejection> {
    old_tl
        .checked_add(add_full)
        .and_then(|after_add| after_add.checked_sub(sub_full))
        .ok_or(SlotRejection::TlOverflow)
}

/// Derive `new_root` from the post-tx user state, reusing the identical
/// truncated leaf layout as C16 (`balance_hash[0..4] || pk_hash[0..2] ||
/// address_hash[0..2]`). The per-type effect layer feeds the new balance /
/// credit / seed / pk / address it computes; here we just hash + lift.
pub(crate) fn compute_new_root(
    new_balance: u64,
    new_seed_hash: [u64; 3],
    new_credit: u64,
    new_pk_hash: [u64; 4],
    new_address_hash: [u64; 4],
    siblings: &[[u64; 4]; TREE_DEPTH],
    user_id: u32,
) -> [u64; 4] {
    let new_balance_hash = hash::balance_leaf(new_balance, new_seed_hash, new_credit);
    merkle::compute_root(new_balance_hash, new_pk_hash, new_address_hash, siblings, user_id)
}

/// Per-type effect computation, mirroring the flag-driven data flow of
/// `build_slot_constraints` (it does NOT branch per type — flags select).
///
/// The order matches the circuit: balance/credit → risk_reject floor →
/// address/pk → seed-reset signal → session auth → seed advance → new root / TL.
/// Every fully-specified type (`noop`, `deposit`, `bet`, `win`, `bonus`,
/// `withdrawal`, `risk_reject`, `set_init_seed_hash`, `key_register_only`,
/// `transfer`, `referral`) is completed here. `in_house_bet` / `crash_settle`
/// run their balance and auth here, then defer the fairness/payout assembly
/// (`random`, `slot_h`, multiset membership, C9/C10/PayoutMismatch) to the
/// fairness-payout to-do.
fn compute_effects(input: &SlotInput) -> Result<SlotEffects, SlotRejection> {
    let flags = flags::TxFlags::from_tx_type(input.tx_type);

    // Balance / credit transition (mirror balance.rs): includes C6 (underflow)
    // and the deposit-credit cap to `min(credit, new_balance)`.
    let bal = balance::build_balance_changes(input, &flags)?;

    // C4: a risk_reject (operator revert) must not push the balance below the
    // user's deposited credit. Mirrors the conditional `sub_u64` in slot/mod.rs.
    if flags.is_risk_reject && bal.new_balance < input.old_deposit_credit {
        return Err(SlotRejection::RiskRejectBelowCredit);
    }

    // Address / pk resolution (mirror address.rs): setter selection + redeposit
    // C15. Yields `is_key_setter`, which the auth signed-set decode needs.
    let addr = address::build_address(input, &flags)?;

    // is_seed_reset: `set_init_seed_hash` overwriting a live (non-zero) seed.
    // This is the single fairness-layer signal the auth set consumes; the rest
    // of fairness (seed-chain C9, user_seed binding C10, `random`) is the
    // fairness-payout to-do.
    let is_seed_reset = flags.is_set_init_seed_hash && !field::all_zero(&input.old_seed_hash);

    // Session auth (mirror auth.rs + session.rs): C11 / C12 / C13 / C14.
    auth::check_auth(input, &flags, addr.is_key_setter, is_seed_reset)?;

    // Fairness + payout layer for in_house_bet / crash_settle: C9 seed-chain,
    // C10 user_seed binding (IHB only), random derivation, payout verification
    // (defense-in-depth via rolly-game-core), and slot_h multiset commitment.
    let fairness_result = if flags.is_ihb_or_crash() {
        Some(fairness::check_fairness(input, &flags)?)
    } else {
        None
    };

    // Seed advance: IHB and set_init_seed_hash install `next_server_seed_hash`;
    // crash_settle and everything else keep the old seed. When IHB produced
    // a FairnessEffects, its `new_seed_hash` is already `next_server_seed_hash`.
    let new_seed_hash = if let Some(ref fe) = fairness_result {
        fe.new_seed_hash
    } else if flags.is_set_init_seed_hash {
        input.next_server_seed_hash
    } else {
        input.old_seed_hash
    };

    let new_total_liability =
        compute_new_tl(input.old_total_liability, bal.add_full, bal.sub_full)?;
    let new_root = compute_new_root(
        bal.new_balance,
        new_seed_hash,
        bal.new_credit,
        addr.new_pk_hash,
        addr.new_address_hash,
        &input.main_siblings,
        input.user_id,
    );

    Ok(SlotEffects {
        new_balance: bal.new_balance,
        new_credit: bal.new_credit,
        new_seed_hash,
        new_pk_hash: addr.new_pk_hash,
        new_address_hash: addr.new_address_hash,
        new_root,
        new_total_liability,
        random: fairness_result.as_ref().map_or([0; 4], |f| f.random),
        slot_h: fairness_result.as_ref().map_or([0; 4], |f| f.slot_h),
        is_multiset_slot: fairness_result.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fully-consistent `noop` candidate: the old leaf is committed into
    /// `current_root` and the TL-chain matches, so every always-on check passes.
    fn consistent_noop() -> SlotInput {
        let user_id = 0b1011u32;
        let old_balance = 1_500_000u64;
        let old_credit = 1_000_000u64;
        let old_seed_hash = [11u64, 22, 33];
        let old_pk_hash = [101u64, 102, 103, 104];
        let old_address_hash = [201u64, 202, 203, 204];
        let main_siblings: [[u64; 4]; TREE_DEPTH] =
            core::array::from_fn(|l| core::array::from_fn(|j| (l as u64) * 7 + j as u64 + 1));

        let old_balance_hash = hash::balance_leaf(old_balance, old_seed_hash, old_credit);
        let current_root = merkle::compute_root(
            old_balance_hash,
            old_pk_hash,
            old_address_hash,
            &main_siblings,
            user_id,
        );

        SlotInput {
            tx_type: TX_NOOP,
            user_id,
            amount: 0,
            win_amount: 0,
            old_balance,
            old_deposit_credit: old_credit,
            main_siblings,
            server_seed: [0; 8],
            user_seed: [0; 4],
            old_seed_hash,
            next_server_seed_hash: [0; 3],
            session_key: [0; 4],
            session_expiry: 0,
            old_pk_hash,
            new_pk_hash: [0; 4],
            user_address: [0; 20],
            old_address_hash,
            old_total_liability: 4_000_000,
            game_id: 0,
            prediction_hash: [0; 4],
            user_secret_random: [0; 4],
            current_root,
            current_tl: 4_000_000,
            max_block_timestamp: 0,
            game_mode: 0,
            prediction_lo: 0,
            prediction_hi: 0,
            keno_selected: Vec::new(),
            crash_seed_hash: None,
        }
    }

    #[test]
    fn noop_passes_and_preserves_state() {
        let inp = consistent_noop();
        let eff = validate_slot(&inp).expect("consistent noop must validate");
        assert_eq!(eff.new_balance, inp.old_balance);
        assert_eq!(eff.new_credit, inp.old_deposit_credit);
        assert_eq!(eff.new_seed_hash, inp.old_seed_hash);
        assert_eq!(eff.new_pk_hash, inp.old_pk_hash);
        assert_eq!(eff.new_address_hash, inp.old_address_hash);
        assert_eq!(eff.new_root, inp.current_root);
        assert_eq!(eff.new_total_liability, inp.current_tl);
        assert!(!eff.is_multiset_slot);
    }

    #[test]
    fn rejects_bad_tx_type() {
        let mut inp = consistent_noop();
        inp.tx_type = MAX_TX_TYPE + 1;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::BadTxType));
    }

    #[test]
    fn rejects_non_canonical_field_limb() {
        // Each opaque group is independently guarded; spot-check a few.
        let mut inp = consistent_noop();
        inp.old_pk_hash[2] = field::GOLDILOCKS_P;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::FieldNotCanonical));

        let mut inp = consistent_noop();
        inp.session_expiry = field::GOLDILOCKS_P;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::FieldNotCanonical));

        let mut inp = consistent_noop();
        inp.main_siblings[5][1] = u64::MAX;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::FieldNotCanonical));
    }

    #[test]
    fn rejects_non_canonical_amount() {
        let mut inp = consistent_noop();
        inp.amount = field::GOLDILOCKS_P;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::AmountNotCanonical));

        let mut inp = consistent_noop();
        inp.win_amount = u64::MAX;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::AmountNotCanonical));
    }

    #[test]
    fn rejects_user_id_out_of_range() {
        let mut inp = consistent_noop();
        inp.user_id = 1u32 << TREE_DEPTH; // == 2^28, just past the limit
                                          // The fabricated root no longer matches this id, but C17 fires first.
        assert_eq!(validate_slot(&inp), Err(SlotRejection::UserIdOutOfRange));
    }

    #[test]
    fn rejects_merkle_root_mismatch() {
        let mut inp = consistent_noop();
        inp.current_root[0] ^= 1;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::MerkleRootMismatch));

        // A tampered old-leaf field also breaks authentication.
        let mut inp = consistent_noop();
        inp.old_balance += 1;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::MerkleRootMismatch));
    }

    #[test]
    fn rejects_tl_chain_break() {
        let mut inp = consistent_noop();
        inp.old_total_liability = inp.current_tl + 1;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::TlOverflow));
    }

    #[test]
    fn always_on_checks_apply_to_every_tx_type() {
        // A non-noop type still gets the always-on checks; a broken root is
        // rejected before reaching the (unimplemented) per-type effect layer.
        let mut inp = consistent_noop();
        inp.tx_type = TX_IN_HOUSE_BET;
        inp.current_root[3] ^= 0xFF;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::MerkleRootMismatch));
    }

    #[test]
    fn compute_new_tl_detects_overflow_and_underflow() {
        assert_eq!(compute_new_tl(100, 50, 30), Ok(120));
        assert_eq!(compute_new_tl(0, 0, 0), Ok(0));
        assert_eq!(
            compute_new_tl(u64::MAX, 1, 0),
            Err(SlotRejection::TlOverflow)
        );
        assert_eq!(compute_new_tl(10, 0, 11), Err(SlotRejection::TlOverflow));
    }

    // ── End-to-end per-type effects through `validate_slot` ──
    //
    // These drive the full always-on + per-type pipeline (and independently
    // re-derive the post-tx root) so the layers are checked composed, not just
    // in their module unit tests.

    const SESSION_KEY: [u64; 4] = [1111, 2222, 3333, 4444];
    const SESSION_EXPIRY: u64 = 2_000_000_000;
    const MAX_BLOCK_TS: u64 = 1_900_000_000;
    const NEW_PK: [u64; 4] = [9001, 9002, 9003, 9004];

    /// Recompute `current_root` / `current_tl` from the old state so the
    /// always-on consistency layer passes and only the per-type effects are
    /// exercised.
    fn commit(mut inp: SlotInput) -> SlotInput {
        let bh = hash::balance_leaf(inp.old_balance, inp.old_seed_hash, inp.old_deposit_credit);
        inp.current_root = merkle::compute_root(
            bh,
            inp.old_pk_hash,
            inp.old_address_hash,
            &inp.main_siblings,
            inp.user_id,
        );
        inp.current_tl = inp.old_total_liability;
        inp
    }

    /// A registered, addressed, seeded user with a live session, committed into
    /// the tree. Address bytes `[7; 20]` hash to `old_address_hash`.
    fn registered_user(tx_type: u8) -> SlotInput {
        let main_siblings: [[u64; 4]; TREE_DEPTH] =
            core::array::from_fn(|l| core::array::from_fn(|j| (l as u64) * 5 + j as u64 + 3));
        commit(SlotInput {
            tx_type,
            user_id: 0b110101u32,
            old_balance: 2_000_000,
            old_deposit_credit: 1_200_000,
            main_siblings,
            old_seed_hash: [71, 82, 93],
            session_key: SESSION_KEY,
            session_expiry: SESSION_EXPIRY,
            old_pk_hash: hash::session_pk_hash(SESSION_KEY, SESSION_EXPIRY),
            old_address_hash: hash::address_hash([7u8; 20]),
            old_total_liability: 9_000_000,
            max_block_timestamp: MAX_BLOCK_TS,
            ..SlotInput::default()
        })
    }

    /// Independently rebuild the post-tx root from the returned effects. Equal
    /// to `eff.new_root` only if the right (balance, seed, credit, pk, address)
    /// were fed into the leaf — guards against argument-order regressions.
    fn root_of_effects(inp: &SlotInput, eff: &SlotEffects) -> [u64; 4] {
        let bh = hash::balance_leaf(eff.new_balance, eff.new_seed_hash, eff.new_credit);
        merkle::compute_root(
            bh,
            eff.new_pk_hash,
            eff.new_address_hash,
            &inp.main_siblings,
            inp.user_id,
        )
    }

    #[test]
    fn deposit_redeposit_grows_balance_and_credit() {
        let mut inp = registered_user(TX_DEPOSIT);
        inp.user_address = [7u8; 20]; // matches committed old_address_hash (C15)
        inp.amount = 500_000;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("redeposit must validate");
        assert_eq!(eff.new_balance, 2_500_000);
        assert_eq!(eff.new_credit, 1_700_000); // credit grows uncapped on deposit
        assert_eq!(eff.new_address_hash, inp.old_address_hash); // unchanged
        assert_eq!(eff.new_total_liability, 9_500_000); // TL += amount
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
        assert!(!eff.is_multiset_slot);
    }

    #[test]
    fn deposit_mismatched_address_is_rejected() {
        let mut inp = registered_user(TX_DEPOSIT);
        inp.user_address = [8u8; 20]; // does NOT match old_address_hash
        inp.amount = 100;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::AddressHashMismatch));
    }

    #[test]
    fn first_deposit_stamps_address() {
        let mut inp = registered_user(TX_DEPOSIT);
        inp.old_address_hash = [0; 4]; // no address yet ⇒ first deposit
        inp.user_address = [13u8; 20];
        inp.amount = 250_000;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("first deposit must validate");
        assert_eq!(eff.new_address_hash, hash::address_hash([13u8; 20]));
        assert_eq!(eff.new_balance, 2_250_000);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn signed_bet_caps_credit_and_debits_tl() {
        let mut inp = registered_user(TX_BET);
        inp.amount = 1_000_000; // new_balance 1_000_000 < credit 1_200_000
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("signed bet must validate");
        assert_eq!(eff.new_balance, 1_000_000);
        assert_eq!(eff.new_credit, 1_000_000); // capped to new balance
        assert_eq!(eff.new_total_liability, 8_000_000); // TL -= amount
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn bet_without_registered_pk_is_rejected() {
        let mut inp = registered_user(TX_BET);
        inp.old_pk_hash = [0; 4];
        inp.amount = 100_000;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SignedWithoutPk));
    }

    #[test]
    fn bet_with_expired_session_is_rejected() {
        let mut inp = registered_user(TX_BET);
        inp.amount = 100_000;
        inp.max_block_timestamp = SESSION_EXPIRY + 1; // session no longer live
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SessionExpired));
    }

    #[test]
    fn bet_with_wrong_session_key_is_rejected() {
        let mut inp = registered_user(TX_BET);
        inp.amount = 100_000;
        inp.session_key[0] ^= 1; // no longer reproduces old_pk_hash
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SessionAuthMismatch));
    }

    #[test]
    fn withdrawal_over_balance_is_insufficient() {
        let mut inp = registered_user(TX_WITHDRAWAL);
        inp.amount = 2_000_001; // > old_balance
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::InsufficientBalance));
    }

    #[test]
    fn win_adds_balance_without_signature_or_credit_change() {
        let mut inp = registered_user(TX_WIN);
        inp.old_pk_hash = [0; 4]; // unsigned: a win needs no session
        inp.amount = 333_333;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("win must validate unsigned");
        assert_eq!(eff.new_balance, 2_333_333);
        assert_eq!(eff.new_credit, 1_200_000); // unchanged
        assert_eq!(eff.new_total_liability, 9_333_333);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn risk_reject_below_credit_is_rejected_but_above_passes() {
        // Below the deposited credit ⇒ C4 fires.
        let mut low = registered_user(TX_RISK_REJECT);
        low.amount = 1_000_000; // new_balance 1_000_000 < credit 1_200_000
        let low = commit(low);
        assert_eq!(validate_slot(&low), Err(SlotRejection::RiskRejectBelowCredit));

        // Above the credit ⇒ passes, credit left untouched (never capped).
        let mut ok = registered_user(TX_RISK_REJECT);
        ok.amount = 500_000; // new_balance 1_500_000 >= credit 1_200_000
        let ok = commit(ok);
        let eff = validate_slot(&ok).expect("risk_reject above credit must validate");
        assert_eq!(eff.new_balance, 1_500_000);
        assert_eq!(eff.new_credit, 1_200_000); // unchanged
        assert_eq!(eff.new_total_liability, 8_500_000);
        assert_eq!(eff.new_root, root_of_effects(&ok, &eff));
    }

    #[test]
    fn set_init_seed_first_time_is_unsigned() {
        let mut inp = registered_user(TX_SET_INIT_SEED_HASH);
        inp.old_seed_hash = [0; 3]; // first initialization ⇒ no reset, no auth
        inp.old_pk_hash = [0; 4];
        inp.next_server_seed_hash = [501, 502, 503];
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("first seed init must validate");
        assert_eq!(eff.new_seed_hash, [501, 502, 503]);
        assert_eq!(eff.new_balance, inp.old_balance); // balance untouched
        assert_eq!(eff.new_total_liability, inp.old_total_liability);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn set_init_seed_reset_requires_session() {
        // Resetting a live seed with a valid session advances it.
        let mut inp = registered_user(TX_SET_INIT_SEED_HASH);
        inp.next_server_seed_hash = [601, 602, 603];
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("authorized seed reset must validate");
        assert_eq!(eff.new_seed_hash, [601, 602, 603]);

        // Same reset without a registered pk ⇒ C12.
        let mut no_pk = registered_user(TX_SET_INIT_SEED_HASH);
        no_pk.old_pk_hash = [0; 4];
        no_pk.next_server_seed_hash = [601, 602, 603];
        let no_pk = commit(no_pk);
        assert_eq!(validate_slot(&no_pk), Err(SlotRejection::SeedResetWithoutPk));
    }

    #[test]
    fn key_first_registration_is_unsigned() {
        let mut inp = registered_user(TX_KEY_REGISTER_ONLY);
        inp.old_pk_hash = [0; 4]; // first registration
        inp.old_address_hash = [0; 4]; // and first address
        inp.new_pk_hash = NEW_PK;
        inp.user_address = [21u8; 20];
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("first key registration must validate");
        assert_eq!(eff.new_pk_hash, NEW_PK);
        assert_eq!(eff.new_address_hash, hash::address_hash([21u8; 20]));
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn key_rotation_is_signed_but_expiry_exempt() {
        // Rotation with a valid signature but an EXPIRED session still passes
        // (key rotation is excluded from the expiry window).
        let mut inp = registered_user(TX_KEY_REGISTER_ONLY);
        inp.new_pk_hash = NEW_PK;
        inp.max_block_timestamp = SESSION_EXPIRY + 10_000; // expired
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("expired key rotation must still validate");
        assert_eq!(eff.new_pk_hash, NEW_PK);
        assert_eq!(eff.new_address_hash, inp.old_address_hash); // address kept

        // A wrong signature still fails C13.
        let mut bad = registered_user(TX_KEY_REGISTER_ONLY);
        bad.new_pk_hash = NEW_PK;
        bad.session_key[2] ^= 0xFF;
        let bad = commit(bad);
        assert_eq!(validate_slot(&bad), Err(SlotRejection::SessionAuthMismatch));
    }

    #[test]
    fn transfer_is_signed_and_debits_balance() {
        let mut inp = registered_user(TX_TRANSFER);
        inp.amount = 400_000;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("signed transfer must validate");
        assert_eq!(eff.new_balance, 1_600_000);
        assert_eq!(eff.new_credit, 1_200_000); // 1_600_000 >= credit ⇒ unchanged
        assert_eq!(eff.new_total_liability, 8_600_000);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    /// Build a fully-consistent IHB input: committed tree, valid session,
    /// valid seed-chain, correct user_seed binding, correct payout.
    fn consistent_ihb(
        game_id: u8,
        bet: u64,
        game_mode: u32,
        prediction_lo: u32,
        prediction_hi: u32,
    ) -> SlotInput {
        let server_seed = [101u64, 202, 303, 404, 505, 606, 707, 808];
        let user_secret = [11u64, 22, 33, 44];
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
        let random = hash::random_ihb(server_seed, user_seed);

        let win_amount = match game_id {
            2 => {
                let mode = rolly_game_core::dice::DiceMode::from_u8(game_mode as u8).unwrap();
                rolly_game_core::dice::compute_payout(
                    &random, bet, mode, [prediction_lo, prediction_hi],
                ).win_amount
            }
            5 => {
                rolly_game_core::coinflip::compute_payout(
                    &random, bet, prediction_lo as u8,
                ).win_amount
            }
            1 => {
                rolly_game_core::limbo::compute_payout(
                    &random, bet, prediction_lo,
                ).win_amount
            }
            _ => 0,
        };

        let mut inp = registered_user(TX_IN_HOUSE_BET);
        inp.game_id = game_id;
        inp.amount = bet;
        inp.win_amount = win_amount;
        inp.server_seed = server_seed;
        inp.user_seed = user_seed;
        inp.old_seed_hash = old_seed_hash;
        inp.next_server_seed_hash = [991, 992, 993];
        inp.user_secret_random = user_secret;
        inp.prediction_hash = prediction_hash;
        inp.game_mode = game_mode;
        inp.prediction_lo = prediction_lo;
        inp.prediction_hi = prediction_hi;
        commit(inp)
    }

    #[test]
    fn ihb_dice_full_pipeline() {
        let inp = consistent_ihb(
            2, // Dice
            1_000_000,
            0,   // Under
            500, // prediction_lo
            0,   // prediction_hi
        );
        let eff = validate_slot(&inp).expect("IHB Dice must validate");
        assert!(eff.is_multiset_slot);
        assert_ne!(eff.random, [0; 4]);
        assert_ne!(eff.slot_h, [0; 4]);
        assert_eq!(eff.new_seed_hash, inp.next_server_seed_hash);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn ihb_coinflip_heads_full_pipeline() {
        // CoinFlip: game_mode=0, prediction in prediction_lo
        let inp = consistent_ihb(
            5, // CoinFlip
            2_000_000,
            0, // game_mode (always 0)
            0, // prediction_lo = heads
            0,
        );
        let eff = validate_slot(&inp).expect("IHB CoinFlip heads must validate");
        assert!(eff.is_multiset_slot);
        assert_eq!(eff.new_seed_hash, inp.next_server_seed_hash);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn ihb_coinflip_tails_full_pipeline() {
        // CoinFlip: game_mode=0, prediction in prediction_lo
        let inp = consistent_ihb(
            5, // CoinFlip
            2_000_000,
            0, // game_mode (always 0)
            1, // prediction_lo = tails
            0,
        );
        let eff = validate_slot(&inp).expect("IHB CoinFlip tails must validate");
        assert!(eff.is_multiset_slot);
        assert_eq!(eff.new_seed_hash, inp.next_server_seed_hash);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    /// Reproduces real WE flow for CoinFlip tails (prediction=1):
    /// Backend sends game_mode=0, prediction_lo=1.
    /// WE computes win_amount using prediction_lo (correct).
    /// tx-validator dispatch_payout uses game_mode (BUG: always 0).
    #[test]
    fn ihb_coinflip_tails_real_we_flow() {
        let server_seed = [101u64, 202, 303, 404, 505, 606, 707, 808];
        let user_secret = [11u64, 22, 33, 44];
        let old_seed_hash = hash::seed_hash_truncated(server_seed);
        let bet = 2_000_000u64;
        let game_mode: u32 = 0;       // CoinFlip always sends game_mode=0
        let prediction_lo: u32 = 1;    // tails — the REAL prediction

        let prediction_hash = hash::prediction_hash_standard(
            5, game_mode as u64, prediction_lo as u64, 0,
        );
        let user_seed = hash::user_seed_binding(5, bet, prediction_hash, user_secret);
        let random = hash::random_ihb(server_seed, user_seed);

        // WE computes payout using prediction_lo (correct behavior)
        let win_amount = rolly_game_core::coinflip::compute_payout(
            &random, bet, prediction_lo as u8,
        ).win_amount;

        let mut inp = registered_user(TX_IN_HOUSE_BET);
        inp.game_id = 5;
        inp.amount = bet;
        inp.win_amount = win_amount;
        inp.server_seed = server_seed;
        inp.user_seed = user_seed;
        inp.old_seed_hash = old_seed_hash;
        inp.next_server_seed_hash = [991, 992, 993];
        inp.user_secret_random = user_secret;
        inp.prediction_hash = prediction_hash;
        inp.game_mode = game_mode;
        inp.prediction_lo = prediction_lo;
        inp.prediction_hi = 0;
        let inp = commit(inp);

        // This MUST pass — validator should use prediction_lo, not game_mode
        let result = validate_slot(&inp);
        assert!(
            result.is_ok(),
            "CoinFlip tails (prediction_lo=1, game_mode=0) must validate, got: {:?}",
            result,
        );
    }

    #[test]
    fn ihb_limbo_full_pipeline() {
        let inp = consistent_ihb(
            1, // Limbo
            1_000_000,
            98,  // rtp (constant)
            200, // prediction_x100 (2.00x target)
            0,
        );
        let eff = validate_slot(&inp).expect("IHB Limbo must validate");
        assert!(eff.is_multiset_slot);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn ihb_broken_seed_chain_rejected() {
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.old_seed_hash[0] ^= 1;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SeedChainBreak));
    }

    #[test]
    fn ihb_tampered_user_seed_rejected() {
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.user_seed[0] ^= 1;
        let inp = commit(inp);
        assert_eq!(
            validate_slot(&inp),
            Err(SlotRejection::UserSeedBindingMismatch)
        );
    }

    #[test]
    fn ihb_wrong_win_amount_rejected() {
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.win_amount = inp.win_amount.wrapping_add(1);
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::PayoutMismatch));
    }

    #[test]
    fn ihb_still_checks_auth() {
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.old_pk_hash = [0; 4]; // no registered pk → C11
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SignedWithoutPk));
    }

    #[test]
    fn ihb_still_checks_insufficient_balance() {
        let inp = consistent_ihb(5, 3_000_000, 0, 0, 0);
        // registered_user has old_balance = 2_000_000, bet = 3_000_000 > balance
        // even after adding win_amount (which may be 0 or less than 1M gap)
        assert_eq!(
            validate_slot(&inp),
            Err(SlotRejection::InsufficientBalance)
        );
    }

    /// Build a fully-consistent crash_settle input.
    fn consistent_crash(cashout_x100: u32) -> SlotInput {
        let server_seed = [201u64, 202, 203, 204, 205, 206, 207, 208];
        let random = hash::random_crash(server_seed);
        let bet = 1_000_000u64;

        let prediction_hash = hash::prediction_hash_standard(
            rolly_game_core::crash::CRASH_GAME_ID as u64,
            0, // mode
            cashout_x100 as u64,
            0,
        );
        let payout = rolly_game_core::crash::compute_payout(&random, bet, cashout_x100);

        let mut inp = registered_user(TX_CRASH_SETTLE);
        inp.game_id = rolly_game_core::crash::CRASH_GAME_ID;
        inp.amount = bet;
        inp.win_amount = payout.win_amount;
        inp.server_seed = server_seed;
        inp.prediction_hash = prediction_hash;
        inp.game_mode = 0;
        inp.prediction_lo = cashout_x100;
        inp.prediction_hi = 0;
        commit(inp)
    }

    #[test]
    fn crash_settle_full_pipeline() {
        let inp = consistent_crash(200); // 2.00x cashout
        let eff = validate_slot(&inp).expect("crash settle must validate");
        assert!(eff.is_multiset_slot);
        assert_ne!(eff.random, [0; 4]);
        assert_ne!(eff.slot_h, [0; 4]);
        // Crash does NOT advance the seed
        assert_eq!(eff.new_seed_hash, inp.old_seed_hash);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn crash_wrong_payout_rejected() {
        let mut inp = consistent_crash(200);
        inp.win_amount = inp.win_amount.wrapping_add(1);
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::PayoutMismatch));
    }

    #[test]
    fn crash_still_checks_auth() {
        let mut inp = consistent_crash(200);
        inp.old_pk_hash = [0; 4];
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SignedWithoutPk));
    }

    #[test]
    fn crash_seed_commit_enforced_when_supplied() {
        // With the matching commitment, the reveal→commit check passes.
        let mut ok = consistent_crash(200);
        ok.crash_seed_hash = Some(hash::seed_hash_truncated(ok.server_seed));
        let ok = commit(ok);
        assert!(validate_slot(&ok).is_ok(), "matching seed commit must validate");

        // A non-matching commitment (operator-grindable seed) is rejected.
        let mut bad = consistent_crash(200);
        let mut wrong = hash::seed_hash_truncated(bad.server_seed);
        wrong[0] ^= 1;
        bad.crash_seed_hash = Some(wrong);
        let bad = commit(bad);
        assert_eq!(
            validate_slot(&bad),
            Err(SlotRejection::CrashSeedCommitMismatch),
        );
    }
}
