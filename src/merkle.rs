//! Native sparse-Merkle root recomputation, mirroring the in-circuit
//! `build_merkle_proof_in_circuit` (helpers/merkle_proof.rs) and the truncated
//! main-leaf layout (slot/merkle.rs).

use crate::hash;
use crate::TREE_DEPTH;

/// Lift a precomputed main leaf to the root along `user_id`'s authentication
/// path. Bits are consumed LSB-first (`split_le`), and at each level the node /
/// sibling order is swapped on a set bit — exactly matching the circuit's
/// `hash_two_to_one_swap(current, sibling, bit)`.
pub fn root_from_leaf(
    leaf: [u64; 4],
    siblings: &[[u64; 4]; TREE_DEPTH],
    user_id: u32,
) -> [u64; 4] {
    let mut current = leaf;
    for (level, sibling) in siblings.iter().enumerate() {
        let bit = (user_id >> level) & 1 == 1;
        let (left, right) = if bit {
            (*sibling, current)
        } else {
            (current, *sibling)
        };
        current = hash::two_to_one(left, right);
    }
    current
}

/// Recompute the Merkle root for a user whose leaf is
/// `Poseidon2(balance_hash[0..4], pk_hash[0..2], address_hash[0..2])`.
///
/// Used both to authenticate the old leaf against `current_root` (C16) and to
/// derive `new_root` for the post-tx state — the identical truncated layout.
pub fn compute_root(
    balance_hash: [u64; 4],
    pk_hash: [u64; 4],
    address_hash: [u64; 4],
    siblings: &[[u64; 4]; TREE_DEPTH],
    user_id: u32,
) -> [u64; 4] {
    let leaf = hash::main_leaf(balance_hash, pk_hash, address_hash);
    root_from_leaf(leaf, siblings, user_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_siblings() -> [[u64; 4]; TREE_DEPTH] {
        core::array::from_fn(|l| core::array::from_fn(|j| (l * 4 + j + 1) as u64))
    }

    #[test]
    fn root_changes_with_path_index() {
        let leaf = [9u64, 8, 7, 6];
        let sibs = sample_siblings();
        // Distinct user_ids generically authenticate to distinct roots.
        assert_ne!(
            root_from_leaf(leaf, &sibs, 0),
            root_from_leaf(leaf, &sibs, 1)
        );
    }

    #[test]
    fn first_level_swap_matches_manual() {
        let leaf = [1u64, 2, 3, 4];
        let mut sibs = sample_siblings();
        for s in sibs.iter_mut().skip(1) {
            *s = [0; 4];
        }
        // bit0 = 0: node is left, sibling is right.
        let even = root_from_leaf(leaf, &sibs, 0b0);
        // bit0 = 1: node is right, sibling is left.
        let odd = root_from_leaf(leaf, &sibs, 0b1);

        // Recompute level 0 by hand, then lift through the zeroed tail.
        let mut up_even = hash::two_to_one(leaf, sibs[0]);
        let mut up_odd = hash::two_to_one(sibs[0], leaf);
        for sib in sibs.iter().skip(1) {
            up_even = hash::two_to_one(up_even, *sib);
            up_odd = hash::two_to_one(up_odd, *sib);
        }
        assert_eq!(even, up_even);
        assert_eq!(odd, up_odd);
    }
}
