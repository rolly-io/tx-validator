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

// Blackjack shoe unfold + native payout reproduction. Public so the witness
// engine can reuse it to GENERATE the win / total stake (blackjack's win is a
// function of the dealt shoe + actions, not the fairness `random`), the same
// value this crate's payout layer re-checks.
pub mod blackjack;

// Per-type effect layers (balance/credit, address/pk, auth-set decode). Internal
// to the validator; the public surface stays `validate_slot` + the I/O types.
// Each mirrors one circuit module so the second implementation tracks the first.
mod address;
pub use address::recipient_addr_hash_for;
mod allowance;
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
/// set_init_seed_hash, key_register_only, transfer, referral, crash_settle,
/// set_provider_allowance).
pub const MAX_TX_TYPE: u8 = 13;

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
/// The account owner sets the absolute provider allowance (`amount`) that caps
/// future provider `bet` debits; `amount = 0` revokes it. Schnorr-signed on the
/// game lane. See [`allowance`].
pub const TX_SET_PROVIDER_ALLOWANCE: u8 = 13;

// ── Two-lane anti-replay nonce (mirror of the circuit's `slot/nonce.rs`) ──
//
// `nonce = account_lane · 2^40 + game_lane`. in_house_bet / crash_settle /
// set_provider_allowance bump (and sign) the game lane; withdrawal / transfer /
// key_register bump (and sign) the account lane, so a bet landing while a
// withdrawal awaits approval cannot stale the withdrawal's signature.
// `account_lane < 2^23` keeps the packed value below 2^63 < p, so the split is
// canonical.

/// Width of the game lane (bits `0..40`).
pub const GAME_LANE_BITS: u32 = 40;
/// Width of the account lane (bits `40..63`).
pub const ACCOUNT_LANE_BITS: u32 = 23;
/// Highest storable game-lane value; a bump past it is rejected.
pub const GAME_LANE_MAX: u64 = (1u64 << GAME_LANE_BITS) - 1;
/// Highest storable account-lane value; a bump past it is rejected.
pub const ACCOUNT_LANE_MAX: u64 = (1u64 << ACCOUNT_LANE_BITS) - 1;
/// Weight of one account-lane step in the packed leaf nonce (`2^40`).
pub const ACCOUNT_LANE_UNIT: u64 = 1u64 << GAME_LANE_BITS;

/// Decompose a packed leaf nonce into `(account_lane, game_lane)`.
pub fn split_nonce(nonce: u64) -> (u64, u64) {
    (nonce >> GAME_LANE_BITS, nonce & GAME_LANE_MAX)
}

/// Inverse of [`split_nonce`]; lanes must be within their widths.
pub fn join_nonce(account_lane: u64, game_lane: u64) -> u64 {
    debug_assert!(account_lane <= ACCOUNT_LANE_MAX, "account lane overflows 23 bits");
    debug_assert!(game_lane <= GAME_LANE_MAX, "game lane overflows 40 bits");
    (account_lane << GAME_LANE_BITS) | game_lane
}

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
    pub old_nonce: u64,
    /// Provider allowance stored in the OLD leaf (62 bits; packed above the
    /// balance limbs — see [`allowance`] / `field::pack_balance_fields`). `0`
    /// for every pre-allowance leaf and for accounts that never granted one,
    /// hence `serde(default)`: a caller that predates the field describes
    /// exactly such a leaf.
    #[serde(default)]
    pub old_provider_allowance: u64,
    pub main_siblings: [[u64; 4]; TREE_DEPTH],

    pub server_seed: [u64; 8],
    pub user_seed: [u64; 4],
    pub old_seed_hash: [u64; 3],
    pub next_server_seed_hash: [u64; 3],

    pub old_pk_hash: [u64; 4],
    pub new_pk_hash: [u64; 4],

    pub user_address: [u8; 20],
    pub old_address_hash: [u64; 4],

    /// `Poseidon2(recipient_address)[0..2]` — payout counterparty (Rolly hot
    /// wallet). A signed input carried into `tx_hash`. F6: must equal
    /// [`recipient_addr_hash_for`]`(tx_type, user_address)` — the hash of the
    /// address a withdrawal pays, `[0, 0]` on every other type.
    #[serde(default)]
    pub recipient_addr_hash: [u64; 2],

    pub old_total_liability: u64,

    pub game_id: u8,
    pub prediction_hash: [u64; 4],
    pub user_secret_random: [u64; 4],

    // --- ambient block state the candidate is checked against ---
    /// Current Merkle root the old leaf must authenticate against (C16).
    pub current_root: [u64; 4],
    /// Current total-liability the slot's `old_total_liability` must equal (C18).
    pub current_tl: u64,
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

    /// Blackjack only: the base stake in atomic units. The rollup slot records
    /// the TOTAL staked (`amount` = base + doubles + splits + insurance); the
    /// payout layer re-derives both the total and the win by replaying the
    /// round from this base stake, because blackjack's win depends on the dealt
    /// shoe + actions rather than on `random`. Zero / unused for every other
    /// game (matching the circuit's blackjack entry witness `base_bet`).
    #[serde(default)]
    pub base_bet: u64,

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
    /// Full packed leaf nonce (`new_account_nonce · 2^40 + new_game_nonce`) —
    /// what is hashed into the new leaf and exported in pubdata.
    pub new_nonce: u64,
    /// Game lane of `new_nonce`: the counter the next `in_house_bet` /
    /// `crash_settle` signature must commit to.
    pub new_game_nonce: u64,
    /// Account lane of `new_nonce`: the counter the next `withdrawal` /
    /// `transfer` signature (and EIP-712 key registration) must commit to.
    pub new_account_nonce: u64,
    /// Absolute post-tx provider allowance (62 bits): `amount` after a
    /// `set_provider_allowance`, `old − amount` after a provider `bet`,
    /// unchanged otherwise. Packed into the new leaf and published in pubdata.
    pub new_provider_allowance: u64,
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
    /// tx_type outside `0..=13` (C1: `flag_sum == 1`).
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
    /// A `set_init_seed_hash` referenced a user leaf (`user_id != 0`). The op is
    /// operator-only and confined to the crash house account (leaf 0); a user
    /// account's seed is installed once by the first `key_register` and rotates
    /// only via `in_house_bet`.
    SetInitSeedNotAccountZero,
    /// An `in_house_bet` referenced the crash house account (`user_id == 0`)
    /// (F8). An IHB rotates the bettor's seed, which on leaf 0 would re-commit
    /// the crash seed outside the `set_init_seed_hash` path the block-level
    /// settle invariant tracks. Leaf 0 never bets.
    InHouseBetOnAccountZero,
    /// `Poseidon2(server_seed)[0..3] != old_seed_hash` (C9, in_house_bet).
    SeedChainBreak,
    /// A slot that (re)commits a seed (`in_house_bet`, `set_init_seed_hash`, or
    /// the first `key_register`) supplied an all-zero `next_server_seed_hash`.
    /// Zero has no known preimage, so it would freeze the account's game lane
    /// (or the crash house account) forever (mirror of `slot/fairness.rs`).
    ZeroSeedCommit,
    /// `user_seed != Poseidon2(game_id, bet, prediction_hash, user_secret)`
    /// (C10, in_house_bet).
    UserSeedBindingMismatch,
    /// `Poseidon2(user_address) != old_address_hash` on redeposit (C15).
    AddressHashMismatch,
    /// F6: `recipient_addr_hash` (the signed payout counterparty) does not
    /// equal `Poseidon2(user_address)[0..2]` on a withdrawal, or is non-zero on
    /// a slot that pays no one.
    RecipientAddrHashMismatch,
    /// A `risk_reject` would drop the balance below `old_deposit_credit` (C4).
    RiskRejectBelowCredit,
    /// `win_amount != compute_payout(...)` or the `prediction_hash` does not
    /// match the raw game params (payout defense-in-depth layer).
    PayoutMismatch,
    /// `Poseidon2(server_seed)[0..3] != crash_seed_hash` on crash_settle: the
    /// revealed seed does not match the house-account (leaf 0) commitment fixed
    /// at round start (native mirror of the circuit's reveal→commit check).
    CrashSeedCommitMismatch,
    /// The nonce lane this tx type bumps is already at its ceiling
    /// (`GAME_LANE_MAX` / `ACCOUNT_LANE_MAX`), or the packed leaf nonce is
    /// outside the 63-bit two-lane range. The circuit rejects instead of
    /// carrying into the other lane or wrapping (mirror of `slot/nonce.rs`).
    NonceLaneOverflow,
    /// A provider `bet` exceeds the account's provider allowance
    /// (`amount > old_provider_allowance`). The circuit's 62-bit check on
    /// `old − amount` fails for it (mirror of `slot/allowance.rs`). An account
    /// that never granted an allowance (or has no Schnorr key to sign one)
    /// rejects every non-zero provider bet here — by design.
    ProviderAllowanceExceeded,
    /// A `set_provider_allowance` amount, or a stored `old_provider_allowance`,
    /// is `>= 2^62` and cannot be packed into the leaf's two 31-bit halves
    /// (the circuit's `split_allowance` range check).
    ProviderAllowanceOverflow,
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
///   * C1  — `tx_type` is in `0..=13`.
///   * C2b — every opaque field limb is canonical (`< p`).
///   * C2c — the full `amount` / `win_amount` are canonical (`< p`).
///   * the stored provider allowance is `< 2^62` (the packed old-leaf fields
///     `unpack_limb_field` splits are `< 2^63`).
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
/// balance `+/-` and credit cap ([`balance`], C6), the provider-allowance
/// set / decrement ([`allowance`]), the risk_reject floor (C4), the address/pk
/// setter and redeposit check ([`address`], C15), auth-set prerequisite
/// ([`auth`], C11), and — for `in_house_bet` / `crash_settle` — the fairness
/// layer ([`fairness`], C9 seed-chain, C10 user_seed binding, random
/// derivation, payout defense-in-depth via `rolly-game-core`, and the `slot_h`
/// multiset commitment). Then packs the new root / TL.
pub fn validate_slot(input: &SlotInput) -> Result<SlotEffects, SlotRejection> {
    // ── Always-on input validation ──
    check_tx_type(input)?; // C1
    check_field_canonicity(input)?; // C2b
    check_amount_canonicity(input)?; // C2c
    check_stored_allowance_range(input)?;
    check_user_id_range(input)?; // C17

    // ── Always-on state consistency ──
    verify_current_root(input)?; // C16
    if input.old_total_liability != input.current_tl {
        return Err(SlotRejection::TlOverflow); // C18 (TL-chain)
    }

    // ── Per-type effects (balance/auth/fairness/payout) ──
    compute_effects(input)
}

/// C1: `tx_type` must decode to exactly one of the 14 one-hot flags.
fn check_tx_type(input: &SlotInput) -> Result<(), SlotRejection> {
    if input.tx_type > MAX_TX_TYPE {
        return Err(SlotRejection::BadTxType);
    }
    Ok(())
}

/// The stored provider allowance must fit its two 31-bit leaf halves. A value
/// `>= 2^62` has no packed representation (`unpack_limb_field` in the circuit
/// would not admit the field), so it can never have come from a proven leaf;
/// reject it before the old-leaf hash would try to pack it.
fn check_stored_allowance_range(input: &SlotInput) -> Result<(), SlotRejection> {
    if input.old_provider_allowance > field::PROVIDER_ALLOWANCE_MAX {
        return Err(SlotRejection::ProviderAllowanceOverflow);
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
        && field::is_canonical(input.old_nonce)
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
/// `old_address_hash` and `main_siblings` (and the allowance range check), so
/// the Poseidon2 calls cannot be fed a non-canonical element.
fn verify_current_root(input: &SlotInput) -> Result<(), SlotRejection> {
    let old_balance_hash =
        hash::balance_leaf(
            input.old_balance,
            input.old_seed_hash,
            input.old_deposit_credit,
            input.old_nonce,
            input.old_provider_allowance,
        );
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

/// Two-lane nonce advance for a slot of `tx_type` (see [`compute_new_nonce`]).
/// `is_key_setter` is `key_register_only && new_pk_hash != 0`. Public so the
/// witness engine's batch-delta mirror advances the nonce with this exact rule
/// instead of copying the arithmetic. Does NOT check the registered-pk
/// prerequisite — that is [`validate_slot`]'s job at accept time.
pub fn advance_nonce(old_nonce: u64, tx_type: u8, is_key_setter: bool) -> Result<u64, SlotRejection> {
    let flags = flags::TxFlags::from_tx_type(tx_type);
    let is_game_bump = flags.is_game_lane();
    let is_account_bump = flags.is_withdrawal || flags.is_transfer || is_key_setter;
    compute_new_nonce(old_nonce, is_game_bump, is_account_bump)
}

/// Two-lane nonce advance, the native mirror of `slot/nonce.rs`:
///   * `is_game_bump`    (`in_house_bet | crash_settle | set_provider_allowance`) → `+1`;
///   * `is_account_bump` (`withdrawal | transfer | is_key_setter`)               → `+2^40`;
///   * neither leaves the packed value untouched.
///
/// The two bump sets are disjoint (one-hot tx flags) and together cover exactly
/// `is_authenticated | is_key_setter`. A bump on a lane already at its ceiling
/// — or a leaf nonce the circuit's 63-bit range check would not admit — is a
/// typed rejection, never a carry or a wrap. (The old `p−1 → 0` field wrap is
/// unreachable: `account_lane < 2^23 ⇒ nonce < 2^63 < p`.)
pub(crate) fn compute_new_nonce(
    old_nonce: u64,
    is_game_bump: bool,
    is_account_bump: bool,
) -> Result<u64, SlotRejection> {
    if old_nonce >= (1u64 << (GAME_LANE_BITS + ACCOUNT_LANE_BITS)) {
        return Err(SlotRejection::NonceLaneOverflow);
    }
    let (account_lane, game_lane) = split_nonce(old_nonce);

    if is_game_bump && game_lane == GAME_LANE_MAX {
        return Err(SlotRejection::NonceLaneOverflow);
    }
    if is_account_bump && account_lane == ACCOUNT_LANE_MAX {
        return Err(SlotRejection::NonceLaneOverflow);
    }
    Ok(old_nonce + is_game_bump as u64 + is_account_bump as u64 * ACCOUNT_LANE_UNIT)
}

/// Derive `new_root` from the post-tx user state, reusing the identical
/// truncated leaf layout as C16 (`balance_hash[0..4] || pk_hash[0..2] ||
/// address_hash[0..2]`). The per-type effect layer feeds the new balance /
/// credit / seed / nonce / allowance / pk / address it computes; here we just
/// hash + lift.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_new_root(
    new_balance: u64,
    new_seed_hash: [u64; 3],
    new_credit: u64,
    new_nonce: u64,
    new_allowance: u64,
    new_pk_hash: [u64; 4],
    new_address_hash: [u64; 4],
    siblings: &[[u64; 4]; TREE_DEPTH],
    user_id: u32,
) -> [u64; 4] {
    let new_balance_hash =
        hash::balance_leaf(new_balance, new_seed_hash, new_credit, new_nonce, new_allowance);
    merkle::compute_root(new_balance_hash, new_pk_hash, new_address_hash, siblings, user_id)
}

/// Per-type effect computation, mirroring the flag-driven data flow of
/// `build_slot_constraints` (it does NOT branch per type — flags select).
///
/// The order matches the circuit: balance/credit → provider allowance →
/// risk_reject floor → address/pk → auth-set decode → nonce/seed advance →
/// new root / TL. Every fully-specified type (`noop`, `deposit`, `bet`, `win`,
/// `bonus`, `withdrawal`, `risk_reject`, `set_init_seed_hash`,
/// `key_register_only`, `transfer`, `referral`, `set_provider_allowance`) is
/// completed here. `in_house_bet` / `crash_settle` run their balance and auth
/// here, then defer the fairness/payout assembly (`random`, `slot_h`, multiset
/// membership, C9/C10/PayoutMismatch) to the fairness-payout to-do.
fn compute_effects(input: &SlotInput) -> Result<SlotEffects, SlotRejection> {
    let flags = flags::TxFlags::from_tx_type(input.tx_type);

    // set_init_seed_hash is operator-only on the crash house account (leaf 0).
    // Mirrors the circuit constraint `is_set_init_seed_hash → user_id == 0`.
    if flags.is_set_init_seed_hash && input.user_id != 0 {
        return Err(SlotRejection::SetInitSeedNotAccountZero);
    }

    // The crash house account (leaf 0) never places an in_house_bet (F8).
    // Mirrors the circuit constraint `is_in_house_bet AND user_id == 0 → 0`.
    if flags.is_in_house_bet && input.user_id == 0 {
        return Err(SlotRejection::InHouseBetOnAccountZero);
    }

    // Balance / credit transition (mirror balance.rs): includes C6 (underflow)
    // and the deposit-credit cap to `min(credit, new_balance)`.
    let bal = balance::build_balance_changes(input, &flags)?;

    // Provider allowance (mirror allowance.rs): `set_provider_allowance`
    // installs `amount`, a provider `bet` must fit in and decrements it,
    // everything else carries it forward.
    let new_provider_allowance = allowance::compute_new_allowance(input, &flags)?;

    // C4: a risk_reject (operator revert) must not push the balance below the
    // user's deposited credit. Mirrors the conditional `sub_u64` in slot/mod.rs.
    if flags.is_risk_reject && bal.new_balance < input.old_deposit_credit {
        return Err(SlotRejection::RiskRejectBelowCredit);
    }

    // Address / pk resolution (mirror address.rs): setter selection + redeposit
    // C15. Yields `is_key_setter`, which the auth signed-set decode needs.
    let addr = address::build_address(input, &flags)?;

    // F6: the signed payout counterparty must be the address the slot pays.
    address::check_recipient_binding(input, &flags)?;

    // The side-circuit verifies Schnorr signatures; locally we enforce that
    // every authenticated type has a registered key and decode which nonce
    // lane it signed. Same predicates as the circuit: `is_game_bump =
    // is_ihb | is_crash | is_set_provider_allowance`, `is_acct_bump =
    // is_acct_lane | is_key_setter`.
    let auth = auth::check_auth(input, &flags)?;
    let is_account_bump = auth.is_account_lane || addr.is_key_setter;
    let new_nonce = compute_new_nonce(input.old_nonce, auth.is_game_lane, is_account_bump)?;
    let (new_account_nonce, new_game_nonce) = split_nonce(new_nonce);

    // Fairness + payout layer for in_house_bet / crash_settle: C9 seed-chain,
    // C10 user_seed binding (IHB only), random derivation, payout verification
    // (defense-in-depth via rolly-game-core), and slot_h multiset commitment.
    let fairness_result = if flags.is_ihb_or_crash() {
        Some(fairness::check_fairness(input, &flags)?)
    } else {
        None
    };

    // Seed advance (mirror fairness.rs): in_house_bet rotates the seed (via its
    // FairnessEffects.new_seed_hash), set_init_seed_hash re-commits the crash
    // house seed (operator, leaf 0), and the FIRST key_register installs the
    // account's initial seed. A key rotation (old_seed_hash != 0) and every
    // other tx keep the old seed. crash_settle keeps the old seed too (its
    // FairnessEffects.new_seed_hash == old_seed_hash).
    let is_first_registration =
        flags.is_key_register_only && field::all_zero(&input.old_seed_hash);
    let is_needs_new_seed =
        flags.is_in_house_bet || flags.is_set_init_seed_hash || is_first_registration;
    if is_needs_new_seed && field::all_zero(&input.next_server_seed_hash) {
        return Err(SlotRejection::ZeroSeedCommit);
    }
    let new_seed_hash = if let Some(ref fe) = fairness_result {
        fe.new_seed_hash
    } else if flags.is_set_init_seed_hash || is_first_registration {
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
        new_nonce,
        new_provider_allowance,
        addr.new_pk_hash,
        addr.new_address_hash,
        &input.main_siblings,
        input.user_id,
    );

    Ok(SlotEffects {
        new_balance: bal.new_balance,
        new_credit: bal.new_credit,
        new_seed_hash,
        new_nonce,
        new_game_nonce,
        new_account_nonce,
        new_provider_allowance,
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
        let old_nonce = 17u64;
        let old_seed_hash = [11u64, 22, 33];
        let old_pk_hash = [101u64, 102, 103, 104];
        let old_address_hash = [201u64, 202, 203, 204];
        let main_siblings: [[u64; 4]; TREE_DEPTH] =
            core::array::from_fn(|l| core::array::from_fn(|j| (l as u64) * 7 + j as u64 + 1));

        let old_balance_hash =
            hash::balance_leaf(old_balance, old_seed_hash, old_credit, old_nonce, 0);
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
            old_nonce,
            old_provider_allowance: 0,
            main_siblings,
            server_seed: [0; 8],
            user_seed: [0; 4],
            old_seed_hash,
            next_server_seed_hash: [0; 3],
            old_pk_hash,
            new_pk_hash: [0; 4],
            user_address: [0; 20],
            old_address_hash,
            recipient_addr_hash: [0; 2],
            old_total_liability: 4_000_000,
            game_id: 0,
            prediction_hash: [0; 4],
            user_secret_random: [0; 4],
            current_root,
            current_tl: 4_000_000,
            game_mode: 0,
            prediction_lo: 0,
            prediction_hi: 0,
            base_bet: 0,
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
        assert_eq!(eff.new_nonce, inp.old_nonce);
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (0, inp.old_nonce));
        assert_eq!(eff.new_provider_allowance, 0);
        assert_eq!(eff.new_pk_hash, inp.old_pk_hash);
        assert_eq!(eff.new_address_hash, inp.old_address_hash);
        assert_eq!(eff.new_root, inp.current_root);
        assert_eq!(eff.new_total_liability, inp.current_tl);
        assert!(!eff.is_multiset_slot);
    }

    #[test]
    fn stored_allowance_above_62_bits_is_rejected_before_hashing() {
        // A leaf can never hold such a value; the always-on layer catches it
        // ahead of the old-leaf hash (which would otherwise panic on packing).
        let mut inp = consistent_noop();
        inp.old_provider_allowance = 1u64 << 62;
        assert_eq!(validate_slot(&inp), Err(SlotRejection::ProviderAllowanceOverflow));
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
        inp.old_nonce = field::GOLDILOCKS_P;
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

    // ── Two-lane nonce arithmetic (pure, no tree) ──

    #[test]
    fn split_and_join_nonce_round_trip() {
        assert_eq!(split_nonce(0), (0, 0));
        assert_eq!(split_nonce(41), (0, 41)); // legacy single-lane nonce
        assert_eq!(split_nonce(ACCOUNT_LANE_UNIT), (1, 0));
        assert_eq!(split_nonce(ACCOUNT_LANE_UNIT - 1), (0, GAME_LANE_MAX));
        let top = join_nonce(ACCOUNT_LANE_MAX, GAME_LANE_MAX);
        assert_eq!(top, (1u64 << 63) - 1);
        assert!(field::is_canonical(top), "a full two-lane nonce stays below p");
        for (a, g) in [(0, 0), (1, 0), (0, 1), (7, 123_456_789), (ACCOUNT_LANE_MAX, GAME_LANE_MAX)] {
            assert_eq!(split_nonce(join_nonce(a, g)), (a, g));
        }
    }

    #[test]
    fn compute_new_nonce_bumps_each_lane_in_isolation() {
        let n = join_nonce(5, 900);
        assert_eq!(compute_new_nonce(n, false, false), Ok(n));
        assert_eq!(compute_new_nonce(n, true, false), Ok(join_nonce(5, 901)));
        assert_eq!(compute_new_nonce(n, false, true), Ok(join_nonce(6, 900)));
        // Legacy nonce: the first account-lane bump lands in bits 40.., the
        // game lane is untouched.
        assert_eq!(compute_new_nonce(41, false, true), Ok(join_nonce(1, 41)));
        assert_eq!(compute_new_nonce(41, true, false), Ok(42));
    }

    #[test]
    fn compute_new_nonce_rejects_at_lane_ceiling_without_carry() {
        let game_full = join_nonce(2, GAME_LANE_MAX);
        assert_eq!(
            compute_new_nonce(game_full, true, false),
            Err(SlotRejection::NonceLaneOverflow)
        );
        // The other lane is still free to move.
        assert_eq!(
            compute_new_nonce(game_full, false, true),
            Ok(join_nonce(3, GAME_LANE_MAX))
        );

        let acct_full = join_nonce(ACCOUNT_LANE_MAX, 10);
        assert_eq!(
            compute_new_nonce(acct_full, false, true),
            Err(SlotRejection::NonceLaneOverflow)
        );
        assert_eq!(
            compute_new_nonce(acct_full, true, false),
            Ok(join_nonce(ACCOUNT_LANE_MAX, 11))
        );

        // Both lanes saturated: storable, but neither may bump.
        let both_full = join_nonce(ACCOUNT_LANE_MAX, GAME_LANE_MAX);
        assert_eq!(compute_new_nonce(both_full, false, false), Ok(both_full));
        assert_eq!(compute_new_nonce(both_full, true, false), Err(SlotRejection::NonceLaneOverflow));
        assert_eq!(compute_new_nonce(both_full, false, true), Err(SlotRejection::NonceLaneOverflow));

        // Outside the 63-bit two-lane range (still < p): the circuit's
        // `split_le(_, 63)` would not admit it, so neither do we — even unbumped.
        for bad in [1u64 << 63, field::GOLDILOCKS_P - 1] {
            assert_eq!(compute_new_nonce(bad, false, false), Err(SlotRejection::NonceLaneOverflow));
            assert_eq!(compute_new_nonce(bad, true, false), Err(SlotRejection::NonceLaneOverflow));
        }
    }

    #[test]
    fn advance_nonce_dispatches_lane_by_tx_type() {
        let n = join_nonce(3, 77);
        for tx_type in [TX_IN_HOUSE_BET, TX_CRASH_SETTLE, TX_SET_PROVIDER_ALLOWANCE] {
            assert_eq!(advance_nonce(n, tx_type, false), Ok(join_nonce(3, 78)), "tx_type {tx_type}");
        }
        for tx_type in [TX_WITHDRAWAL, TX_TRANSFER] {
            assert_eq!(advance_nonce(n, tx_type, false), Ok(join_nonce(4, 77)), "tx_type {tx_type}");
        }
        // key_register bumps the account lane only when it actually sets a key.
        assert_eq!(advance_nonce(n, TX_KEY_REGISTER_ONLY, true), Ok(join_nonce(4, 77)));
        assert_eq!(advance_nonce(n, TX_KEY_REGISTER_ONLY, false), Ok(n));
        for tx_type in [
            TX_NOOP, TX_DEPOSIT, TX_BET, TX_WIN, TX_BONUS, TX_RISK_REJECT,
            TX_SET_INIT_SEED_HASH, TX_REFERRAL,
        ] {
            assert_eq!(advance_nonce(n, tx_type, false), Ok(n), "tx_type {tx_type}");
        }
    }

    // ── End-to-end per-type effects through `validate_slot` ──
    //
    // These drive the full always-on + per-type pipeline (and independently
    // re-derive the post-tx root) so the layers are checked composed, not just
    // in their module unit tests.

    const SCHNORR_PK: [u64; 5] = [1111, 2222, 3333, 4444, 5555];
    const NEW_PK: [u64; 4] = [9001, 9002, 9003, 9004];

    /// Recompute `current_root` / `current_tl` from the old state so the
    /// always-on consistency layer passes and only the per-type effects are
    /// exercised.
    fn commit(mut inp: SlotInput) -> SlotInput {
        let bh = hash::balance_leaf(
            inp.old_balance,
            inp.old_seed_hash,
            inp.old_deposit_credit,
            inp.old_nonce,
            inp.old_provider_allowance,
        );
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

    /// A registered, addressed, seeded user committed into the tree. Address
    /// bytes `[7; 20]` hash to `old_address_hash`.
    fn registered_user(tx_type: u8) -> SlotInput {
        let main_siblings: [[u64; 4]; TREE_DEPTH] =
            core::array::from_fn(|l| core::array::from_fn(|j| (l as u64) * 5 + j as u64 + 3));
        commit(SlotInput {
            tx_type,
            user_id: 0b110101u32,
            old_balance: 2_000_000,
            old_deposit_credit: 1_200_000,
            old_nonce: 41,
            main_siblings,
            old_seed_hash: [71, 82, 93],
            old_pk_hash: hash::schnorr_pk_hash(SCHNORR_PK),
            old_address_hash: hash::address_hash([7u8; 20]),
            old_total_liability: 9_000_000,
            ..SlotInput::default()
        })
    }

    /// Independently rebuild the post-tx root from the returned effects. Equal
    /// to `eff.new_root` only if the right (balance, seed, credit, nonce,
    /// allowance, pk, address) were fed into the leaf — guards against
    /// argument-order regressions.
    fn root_of_effects(inp: &SlotInput, eff: &SlotEffects) -> [u64; 4] {
        let bh = hash::balance_leaf(
            eff.new_balance,
            eff.new_seed_hash,
            eff.new_credit,
            eff.new_nonce,
            eff.new_provider_allowance,
        );
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
    fn provider_bet_caps_credit_without_nonce_bump() {
        let mut inp = registered_user(TX_BET);
        inp.old_provider_allowance = 1_500_000; // the owner granted providers 1.5M
        inp.amount = 1_000_000; // new_balance 1_000_000 < credit 1_200_000
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("provider bet must validate");
        assert_eq!(eff.new_balance, 1_000_000);
        assert_eq!(eff.new_credit, 1_000_000); // capped to new balance
        assert_eq!(eff.new_total_liability, 8_000_000); // TL -= amount
        assert_eq!(eff.new_nonce, inp.old_nonce);
        assert_eq!(eff.new_provider_allowance, 500_000); // allowance -= amount
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn provider_bet_without_registered_pk_is_allowed_within_allowance() {
        // The bet itself needs no Schnorr key — only the allowance that caps it
        // was signed (an account with no key can never have set one, so in
        // practice its allowance is 0 and every non-zero provider bet rejects).
        let mut inp = registered_user(TX_BET);
        inp.old_pk_hash = [0; 4];
        inp.old_provider_allowance = 100_000;
        inp.amount = 100_000;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("provider bet does not require a Schnorr key");
        assert_eq!(eff.new_nonce, inp.old_nonce);
        assert_eq!(eff.new_provider_allowance, 0);
    }

    #[test]
    fn provider_bet_above_allowance_is_rejected() {
        // Balance would cover it (2M), but the owner only allowed 100k.
        let mut inp = registered_user(TX_BET);
        inp.old_provider_allowance = 100_000;
        inp.amount = 100_001;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::ProviderAllowanceExceeded));

        // No allowance ever granted (every legacy leaf): non-zero bets reject.
        let mut inp = registered_user(TX_BET);
        inp.amount = 1;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::ProviderAllowanceExceeded));

        // The allowance check is independent of C6: an over-balance bet within
        // the allowance still fails on the balance.
        let mut inp = registered_user(TX_BET);
        inp.old_provider_allowance = PROVIDER_ALLOWANCE_MAX_FOR_TEST;
        inp.amount = 2_000_001;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::InsufficientBalance));
    }

    const PROVIDER_ALLOWANCE_MAX_FOR_TEST: u64 = field::PROVIDER_ALLOWANCE_MAX;

    #[test]
    fn set_provider_allowance_is_signed_absolute_and_bumps_game_lane() {
        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.old_provider_allowance = 5;
        inp.amount = 7_500_000; // the new absolute allowance (may exceed balance)
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("set_provider_allowance must validate");
        assert_eq!(eff.new_provider_allowance, 7_500_000, "absolute, not additive");
        // Balance / credit / seed / TL untouched.
        assert_eq!(eff.new_balance, inp.old_balance);
        assert_eq!(eff.new_credit, inp.old_deposit_credit);
        assert_eq!(eff.new_seed_hash, inp.old_seed_hash);
        assert_eq!(eff.new_total_liability, inp.old_total_liability);
        // Game-lane bump (like a bet), account lane untouched.
        assert_eq!(eff.new_nonce, inp.old_nonce + 1);
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (0, 42));
        assert!(!eff.is_multiset_slot);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
        // The allowance actually lands in the leaf: a root built with 0 differs.
        let bh0 = hash::balance_leaf(eff.new_balance, eff.new_seed_hash, eff.new_credit, eff.new_nonce, 0);
        let root0 = merkle::compute_root(bh0, eff.new_pk_hash, eff.new_address_hash, &inp.main_siblings, inp.user_id);
        assert_ne!(eff.new_root, root0);

        // Revoke: amount = 0.
        let mut revoke = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        revoke.old_provider_allowance = 7_500_000;
        revoke.amount = 0;
        let revoke = commit(revoke);
        let eff = validate_slot(&revoke).expect("revoke must validate");
        assert_eq!(eff.new_provider_allowance, 0);
        assert_eq!(eff.new_root, root_of_effects(&revoke, &eff));
    }

    #[test]
    fn set_provider_allowance_requires_registered_pk() {
        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.old_pk_hash = [0; 4];
        inp.amount = 1;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::SignedWithoutPk));
    }

    #[test]
    fn set_provider_allowance_above_62_bits_is_rejected() {
        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.amount = 1u64 << 62;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::ProviderAllowanceOverflow));

        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.amount = PROVIDER_ALLOWANCE_MAX_FOR_TEST;
        let inp = commit(inp);
        assert_eq!(
            validate_slot(&inp).map(|e| e.new_provider_allowance),
            Ok(PROVIDER_ALLOWANCE_MAX_FOR_TEST),
        );
    }

    #[test]
    fn set_provider_allowance_rejects_recipient_and_game_lane_ceiling() {
        // F6: no payout counterparty on a set_provider_allowance.
        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.amount = 10;
        inp.recipient_addr_hash = [1, 2];
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::RecipientAddrHashMismatch));

        // Game lane saturated: reject, do not carry.
        let mut inp = registered_user(TX_SET_PROVIDER_ALLOWANCE);
        inp.old_nonce = join_nonce(2, GAME_LANE_MAX);
        inp.amount = 10;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::NonceLaneOverflow));
    }

    #[test]
    fn win_and_other_types_keep_the_allowance_strict() {
        // `win` after a provider bet does NOT restore the allowance.
        let mut inp = registered_user(TX_WIN);
        inp.old_pk_hash = [0; 4];
        inp.old_provider_allowance = 400_000;
        inp.amount = 333_333;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("win must validate");
        assert_eq!(eff.new_provider_allowance, 400_000);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));

        for tx in [TX_DEPOSIT, TX_WITHDRAWAL, TX_TRANSFER, TX_BONUS] {
            let mut inp = registered_user(tx);
            inp.old_provider_allowance = 400_000;
            inp.amount = 1_000;
            inp.user_address = [7u8; 20];
            inp.recipient_addr_hash = recipient_addr_hash_for(tx, inp.user_address);
            let inp = commit(inp);
            let eff = validate_slot(&inp).unwrap_or_else(|e| panic!("tx {tx}: {e}"));
            assert_eq!(eff.new_provider_allowance, 400_000, "tx {tx} must keep the allowance");
            assert_eq!(eff.new_root, root_of_effects(&inp, &eff), "tx {tx}");
        }
    }

    #[test]
    fn withdrawal_over_balance_is_insufficient() {
        let mut inp = registered_user(TX_WITHDRAWAL);
        inp.amount = 2_000_001; // > old_balance
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::InsufficientBalance));
    }

    #[test]
    fn withdrawal_pays_the_signed_counterparty_f6() {
        // The hot wallet the user signed for (any address — not necessarily the
        // account's registered one).
        let hot_wallet = [0xABu8; 20];
        let h = hash::address_hash(hot_wallet);

        let mut inp = registered_user(TX_WITHDRAWAL);
        inp.amount = 750_000;
        inp.user_address = hot_wallet;
        inp.recipient_addr_hash = [h[0], h[1]];
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("withdrawal to the signed counterparty must validate");
        assert_eq!(eff.new_balance, 1_250_000);
        assert_eq!(eff.new_nonce, join_nonce(1, 41)); // authenticated ⇒ account-lane bump
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (1, 41));
        assert_eq!(eff.new_address_hash, inp.old_address_hash); // leaf address untouched
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));

        // Same signed counterparty, but the slot pays a different address.
        let mut redirected = inp.clone();
        redirected.user_address = [0xCDu8; 20];
        assert_eq!(
            validate_slot(&redirected),
            Err(SlotRejection::RecipientAddrHashMismatch)
        );

        // Slot pays the signed address, but the recipient field was zeroed
        // (an unsigned destination).
        let mut unsigned_dest = inp.clone();
        unsigned_dest.recipient_addr_hash = [0; 2];
        assert_eq!(
            validate_slot(&unsigned_dest),
            Err(SlotRejection::RecipientAddrHashMismatch)
        );
    }

    #[test]
    fn non_payout_slot_with_recipient_is_rejected_f6() {
        let mut inp = registered_user(TX_TRANSFER);
        inp.amount = 100_000;
        inp.recipient_addr_hash = [11, 22];
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::RecipientAddrHashMismatch));
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
        // set_init_seed_hash is operator-only on the crash house account (leaf 0).
        let mut inp = registered_user(TX_SET_INIT_SEED_HASH);
        inp.user_id = 0;
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
    fn set_init_seed_reset_does_not_bump_nonce() {
        // Operator re-commit of the crash house seed on leaf 0.
        let mut inp = registered_user(TX_SET_INIT_SEED_HASH);
        inp.user_id = 0;
        inp.next_server_seed_hash = [601, 602, 603];
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("operator seed reset must validate");
        assert_eq!(eff.new_seed_hash, [601, 602, 603]);
        assert_eq!(eff.new_nonce, inp.old_nonce);

        // Seed reset is not a Schnorr-authenticated user operation.
        let mut no_pk = registered_user(TX_SET_INIT_SEED_HASH);
        no_pk.user_id = 0;
        no_pk.old_pk_hash = [0; 4];
        no_pk.next_server_seed_hash = [601, 602, 603];
        let no_pk = commit(no_pk);
        assert!(validate_slot(&no_pk).is_ok());
    }

    #[test]
    fn set_init_seed_on_user_leaf_is_rejected() {
        // A set_init_seed_hash on any non-zero leaf is operator-forbidden: a user
        // account's seed is installed by its first key_register and rotates only
        // via in_house_bet.
        let mut inp = registered_user(TX_SET_INIT_SEED_HASH);
        inp.user_id = 7; // not the crash house account
        inp.next_server_seed_hash = [601, 602, 603];
        let inp = commit(inp);
        assert_eq!(
            validate_slot(&inp),
            Err(SlotRejection::SetInitSeedNotAccountZero),
        );
    }

    #[test]
    fn first_key_registration_installs_initial_seed() {
        // The first key_register (old_seed_hash == 0) installs the account's
        // initial seed from next_server_seed_hash; a later rotation leaves it.
        let mut first = registered_user(TX_KEY_REGISTER_ONLY);
        first.old_pk_hash = [0; 4];
        first.old_address_hash = [0; 4];
        first.old_seed_hash = [0; 3];
        first.new_pk_hash = NEW_PK;
        first.user_address = [21u8; 20];
        first.next_server_seed_hash = [711, 712, 713];
        let first = commit(first);
        let eff = validate_slot(&first).expect("first key registration must validate");
        assert_eq!(eff.new_seed_hash, [711, 712, 713]);
        assert_eq!(eff.new_root, root_of_effects(&first, &eff));

        // Rotation: old_seed_hash != 0 ⇒ next_server_seed_hash is ignored.
        let mut rot = registered_user(TX_KEY_REGISTER_ONLY);
        rot.new_pk_hash = NEW_PK;
        rot.next_server_seed_hash = [711, 712, 713];
        let rot = commit(rot);
        let eff = validate_slot(&rot).expect("key rotation must validate");
        assert_eq!(eff.new_seed_hash, rot.old_seed_hash);
        assert_eq!(eff.new_root, root_of_effects(&rot, &eff));
    }

    #[test]
    fn zero_seed_commit_is_rejected_on_every_seed_writing_path() {
        // set_init_seed_hash on leaf 0 with an all-zero commitment.
        let mut init = registered_user(TX_SET_INIT_SEED_HASH);
        init.user_id = 0;
        init.next_server_seed_hash = [0; 3];
        let init = commit(init);
        assert_eq!(validate_slot(&init), Err(SlotRejection::ZeroSeedCommit));

        // First key_register (old_seed_hash == 0) installing a zero seed.
        let mut first = registered_user(TX_KEY_REGISTER_ONLY);
        first.old_pk_hash = [0; 4];
        first.old_address_hash = [0; 4];
        first.old_seed_hash = [0; 3];
        first.new_pk_hash = NEW_PK;
        first.user_address = [21u8; 20];
        first.next_server_seed_hash = [0; 3];
        let first = commit(first);
        assert_eq!(validate_slot(&first), Err(SlotRejection::ZeroSeedCommit));

        // A key ROTATION does not write the seed, so a zero
        // next_server_seed_hash is irrelevant there.
        let mut rot = registered_user(TX_KEY_REGISTER_ONLY);
        rot.new_pk_hash = NEW_PK;
        rot.next_server_seed_hash = [0; 3];
        let rot = commit(rot);
        assert!(validate_slot(&rot).is_ok(), "rotation keeps the old seed");

        // Nor does a plain transfer.
        let mut xfer = registered_user(TX_TRANSFER);
        xfer.amount = 1;
        xfer.next_server_seed_hash = [0; 3];
        let xfer = commit(xfer);
        assert!(validate_slot(&xfer).is_ok(), "non-seed tx ignores next_server_seed_hash");
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
        assert_eq!(eff.new_nonce, inp.old_nonce + ACCOUNT_LANE_UNIT);
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn key_rotation_bumps_account_lane() {
        let mut inp = registered_user(TX_KEY_REGISTER_ONLY);
        inp.new_pk_hash = NEW_PK;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("registration side-circuit authorizes rotation");
        assert_eq!(eff.new_pk_hash, NEW_PK);
        assert_eq!(eff.new_address_hash, inp.old_address_hash); // address kept
        assert_eq!(split_nonce(eff.new_nonce), (1, 41));
    }

    #[test]
    fn nonce_lanes_bump_independently() {
        // Account lane: withdrawal / transfer / key_register add 2^40 and leave
        // the low 40 bits alone; game lane: IHB / crash add 1 and leave the
        // high 23 bits alone. Legacy nonces (< 2^40) decode as (0, n).
        assert_eq!(split_nonce(41), (0, 41));
        let packed = join_nonce(5, 123_456_789);
        assert_eq!(split_nonce(packed), (5, 123_456_789));

        let mut inp = registered_user(TX_TRANSFER);
        inp.old_nonce = packed;
        inp.amount = 1;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("transfer on a two-lane nonce must validate");
        assert_eq!(split_nonce(eff.new_nonce), (6, 123_456_789));
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (6, 123_456_789));

        let mut inp = registered_user(TX_KEY_REGISTER_ONLY);
        inp.old_nonce = packed;
        inp.new_pk_hash = NEW_PK;
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("rotation on a two-lane nonce must validate");
        assert_eq!(split_nonce(eff.new_nonce), (6, 123_456_789));
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (6, 123_456_789));

        // Unsigned slots leave both lanes untouched.
        let mut inp = registered_user(TX_NOOP);
        inp.old_nonce = packed;
        let inp = commit(inp);
        let eff = validate_slot(&inp).unwrap();
        assert_eq!(eff.new_nonce, packed);
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (5, 123_456_789));
    }

    #[test]
    fn withdrawal_signed_lane_survives_interleaved_bets() {
        // The scenario the two lanes exist for: a withdrawal is signed under
        // account lane N; bets land before it is applied and move only the game
        // lane, so the account lane the signature committed to is still N.
        let hot_wallet = [0xABu8; 20];
        let h = hash::address_hash(hot_wallet);
        let start = join_nonce(4, 41);

        let mut bet = consistent_ihb(2, 1_000_000, 0, 500, 0);
        bet.old_nonce = start;
        let bet = commit(bet);
        let after_bet = validate_slot(&bet).expect("bet must validate");
        assert_eq!((after_bet.new_account_nonce, after_bet.new_game_nonce), (4, 42));

        let mut wd = registered_user(TX_WITHDRAWAL);
        wd.old_nonce = after_bet.new_nonce; // leaf after the bet
        wd.amount = 750_000;
        wd.user_address = hot_wallet;
        wd.recipient_addr_hash = [h[0], h[1]];
        let wd = commit(wd);
        let eff = validate_slot(&wd).expect("withdrawal after a bet must validate");
        // Account lane was 4 when signed and is still 4 at apply time; it bumps
        // to 5 and the game lane keeps the bet's advance.
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (5, 42));
        assert_eq!(eff.new_nonce, join_nonce(5, 42));
    }

    #[test]
    fn nonce_lane_ceiling_is_typed_rejection() {
        // Account lane full: account-lane types reject, nothing carries.
        let full_acct = join_nonce(ACCOUNT_LANE_MAX, 10);
        for tx_type in [TX_WITHDRAWAL, TX_TRANSFER, TX_KEY_REGISTER_ONLY] {
            let mut inp = registered_user(tx_type);
            inp.old_nonce = full_acct;
            inp.amount = 1;
            if tx_type == TX_WITHDRAWAL {
                inp.user_address = [7u8; 20];
                inp.recipient_addr_hash = recipient_addr_hash_for(tx_type, inp.user_address);
            }
            if tx_type == TX_KEY_REGISTER_ONLY {
                inp.new_pk_hash = NEW_PK;
            }
            let inp = commit(inp);
            assert_eq!(
                validate_slot(&inp),
                Err(SlotRejection::NonceLaneOverflow),
                "tx_type {tx_type} at the account-lane ceiling",
            );
        }
        // A saturated lane that is not bumped is still a storable leaf.
        let mut inp = registered_user(TX_NOOP);
        inp.old_nonce = join_nonce(ACCOUNT_LANE_MAX, GAME_LANE_MAX);
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp).unwrap().new_nonce, inp.old_nonce);

        // Game lane full: an account-lane bump still goes through.
        let mut inp = registered_user(TX_KEY_REGISTER_ONLY);
        inp.old_nonce = join_nonce(3, GAME_LANE_MAX);
        inp.new_pk_hash = NEW_PK;
        let inp = commit(inp);
        assert_eq!(split_nonce(validate_slot(&inp).unwrap().new_nonce), (4, GAME_LANE_MAX));

        // A canonical leaf nonce outside the 63-bit two-lane range can never
        // have been produced by the circuit (its `split_le(_, 63)` rejects it).
        let mut inp = registered_user(TX_NOOP);
        inp.old_nonce = 1u64 << 63;
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::NonceLaneOverflow));
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
        assert_eq!(eff.new_nonce, inp.old_nonce + ACCOUNT_LANE_UNIT);
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
        assert_eq!(eff.new_nonce, inp.old_nonce + 1); // game-lane bump
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn ihb_on_crash_house_account_is_rejected() {
        // F8: leaf 0 holds the crash seed commitment and never bets. An
        // otherwise fully consistent IHB retargeted at user_id 0 is a typed
        // rejection before any fairness / payout work.
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.user_id = 0;
        let inp = commit(inp);
        assert_eq!(
            validate_slot(&inp),
            Err(SlotRejection::InHouseBetOnAccountZero),
        );
    }

    #[test]
    fn ihb_bumps_game_lane_only_and_rejects_at_its_ceiling() {
        // An account that has withdrawn before (account lane 2): a bet moves the
        // game lane and leaves the account lane alone.
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.old_nonce = join_nonce(2, 41);
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("IHB on a two-lane nonce must validate");
        assert_eq!(split_nonce(eff.new_nonce), (2, 42));
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (2, 42));

        // Game lane at its ceiling: reject, do not carry into the account lane.
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.old_nonce = join_nonce(2, GAME_LANE_MAX);
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::NonceLaneOverflow));

        // Account lane at its ceiling does not block a bet.
        let mut inp = consistent_ihb(2, 1_000_000, 0, 500, 0);
        inp.old_nonce = join_nonce(ACCOUNT_LANE_MAX, 41);
        let inp = commit(inp);
        assert_eq!(split_nonce(validate_slot(&inp).unwrap().new_nonce), (ACCOUNT_LANE_MAX, 42));
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
        // ...but it is a game-lane bump like in_house_bet.
        assert_eq!(eff.new_nonce, inp.old_nonce + 1);
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (0, inp.old_nonce + 1));
        assert_eq!(eff.new_root, root_of_effects(&inp, &eff));
    }

    #[test]
    fn crash_settle_on_two_lane_nonce_keeps_account_lane() {
        let mut inp = consistent_crash(200);
        inp.old_nonce = join_nonce(9, 41);
        let inp = commit(inp);
        let eff = validate_slot(&inp).expect("crash settle on a two-lane nonce must validate");
        assert_eq!((eff.new_account_nonce, eff.new_game_nonce), (9, 42));

        let mut inp = consistent_crash(200);
        inp.old_nonce = join_nonce(9, GAME_LANE_MAX);
        let inp = commit(inp);
        assert_eq!(validate_slot(&inp), Err(SlotRejection::NonceLaneOverflow));
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
