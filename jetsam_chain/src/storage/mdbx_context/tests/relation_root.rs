// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Review I1 (v1.5.0, inherited from the v1.3 boundary): a HistoryStep
//! relation that starts at an activation height is rooted at the header of
//! the block before it, read off the branch it judges. Its terminal proves
//! the blocks above that root and nothing at or below it. A suffix (or a
//! reorg) whose base lies below that root carried blocks no terminal it
//! brought proves — their spend authorizations attested by nothing.
//!
//! Driven on a relation schedule installed on the test thread (a relation
//! starting at 4, rooted at 3), every chain at most 5 blocks long: a real
//! boundary is 17 750 blocks away on the public profile, and past the v1.4
//! height (6) of the test network, whose anchor target a debug build cannot
//! mine in a test.

use super::*;
use crate::consensus::params::history_step_relation_root_height_with;

const ACTIVATION: u64 = 4;
const ROOT: u64 = ACTIVATION - 1;
const TIP: u64 = ACTIVATION + 1;

#[test]
fn a_relations_root_is_the_block_before_its_activation() {
    // The public schedule of today, and the one v1.5 arms.
    for (v1_3, v1_5) in [(Some(17_750), None), (Some(17_750), Some(37_440))] {
        assert_eq!(history_step_relation_root_height_with(1, v1_3, v1_5), 0);
        assert_eq!(history_step_relation_root_height_with(17_749, v1_3, v1_5), 0);
        assert_eq!(history_step_relation_root_height_with(17_750, v1_3, v1_5), 17_749);
        assert_eq!(history_step_relation_root_height_with(37_439, v1_3, v1_5), 17_749);
    }
    assert_eq!(
        history_step_relation_root_height_with(37_440, Some(17_750), Some(37_440)),
        37_439
    );
    assert_eq!(
        history_step_relation_root_height_with(40_000, Some(17_750), Some(37_440)),
        37_439
    );
    // A relation that starts at genesis is rooted there.
    assert_eq!(history_step_relation_root_height_with(5, Some(0), None), 0);
}

fn header_of(bundle: &crate::AcceptedBlockBundle) -> BlockHeader {
    crate::Block::from_bytes(bundle.block_bytes()).unwrap().header
}

/// A node whose tip is `blocks[..height]`'s.
fn node_at(path: &Path, blocks: &[crate::AcceptedBlockBundle], height: u64) -> MdbxChainContext {
    let mut node = easy_block_context(path);
    for bundle in &blocks[..height as usize] {
        accept_test_bundle(&mut node, bundle);
    }
    assert_eq!(node.tip_height(), height);
    node
}

/// The terminal of `bundle`, through a verifier that accepts it (the tests
/// are about which blocks a terminal may authorize, not about the proof).
fn verified(
    node: &MdbxChainContext,
    bundle: &crate::AcceptedBlockBundle,
) -> VerifiedHistoryStepTerminal {
    let header = header_of(bundle);
    let anchor = node
        .get_header_from_store(tx_epoch_anchor_height_for_child(header.height))
        .unwrap()
        .unwrap();
    verify_history_step_terminal_candidate(
        header,
        anchor,
        bundle.history_step_terminal_bytes().to_vec(),
        |_| Ok(Default::default()),
    )
    .unwrap()
}

fn apply_bodies(
    node: &mut MdbxChainContext,
    authority: &mut VerifiedRecursiveSuffix,
    bodies: &[crate::AcceptedBlockBundle],
) -> Result<(), MdbxContextError> {
    for bundle in bodies {
        node.apply_verified_recursive_suffix_block(
            authority,
            bundle.block_bytes(),
            header_of(bundle).timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
        )?;
    }
    Ok(())
}

/// I1: a node two blocks below the root of the tip's relation is handed a
/// suffix to a tip above the activation with the tip's terminal only: the
/// blocks up to the root are proved by nothing, so it is not authorized.
/// From the root on, the tip's terminal is enough (unchanged).
#[test]
fn a_suffix_below_the_root_of_its_tips_relation_is_not_authorized_by_its_tip_alone() {
    let _schedule = test_relation_schedule::install(Some(ACTIVATION), None);
    let blocks = block_sequence(TIP as usize);
    for base in [ROOT - 2, ROOT - 1] {
        let directory = tempfile::tempdir().unwrap();
        let mut node = node_at(directory.path(), &blocks, base);
        let tip = verified(&node, &blocks[(TIP - 1) as usize]);
        let refused = node.begin_preverified_recursive_suffix(tip);
        let error = refused.expect_err("blocks up to the root would be proved by nothing");
        assert!(
            error.to_string().contains("starts below the root"),
            "{error}"
        );
        assert_eq!(node.tip_height(), base, "nothing committed");
    }
    let directory = tempfile::tempdir().unwrap();
    let mut node = node_at(directory.path(), &blocks, ROOT);
    let tip = verified(&node, &blocks[(TIP - 1) as usize]);
    let mut authority = node.begin_preverified_recursive_suffix(tip).unwrap();
    apply_bodies(&mut node, &mut authority, &blocks[ROOT as usize..]).unwrap();
    assert_eq!(node.tip_height(), TIP);
}

/// I1: with the root block's own verified terminal, the same suffix is
/// authorized and applied; a root terminal of another block at the root
/// height (another branch) stops the suffix at that body, and one at
/// another height is refused outright.
#[test]
fn with_its_roots_terminal_a_suffix_below_the_root_is_authorized() {
    let _schedule = test_relation_schedule::install(Some(ACTIVATION), None);
    let blocks = block_sequence(TIP as usize);
    let directory = tempfile::tempdir().unwrap();
    let mut node = node_at(directory.path(), &blocks, ROOT - 2);
    let tip = verified(&node, &blocks[(TIP - 1) as usize]);
    let root = verified(&node, &blocks[(ROOT - 1) as usize]);
    let mut authority = node
        .begin_preverified_recursive_suffix_with_root(tip, Some(root))
        .unwrap();
    apply_bodies(&mut node, &mut authority, &blocks[(ROOT - 2) as usize..]).unwrap();
    assert!(authority.is_complete());
    assert_eq!(node.tip_height(), TIP);

    // The root terminal of another branch's block at the root height.
    let other = {
        let producer_dir = tempfile::tempdir().unwrap();
        let mut producer = node_at(producer_dir.path(), &blocks, ROOT - 1);
        let other = test_next_bundle_for_miner(&producer, 0x62);
        accept_test_bundle(&mut producer, &other);
        other
    };
    assert_ne!(header_of(&other), header_of(&blocks[(ROOT - 1) as usize]));
    let directory = tempfile::tempdir().unwrap();
    let mut node = node_at(directory.path(), &blocks, ROOT - 2);
    let tip = verified(&node, &blocks[(TIP - 1) as usize]);
    let foreign_root = verified(&node, &other);
    let mut authority = node
        .begin_preverified_recursive_suffix_with_root(tip, Some(foreign_root))
        .unwrap();
    let error = apply_bodies(&mut node, &mut authority, &blocks[(ROOT - 2) as usize..])
        .expect_err("the suffix's root is not the one its terminal proves");
    assert!(error.to_string().contains("relation root"), "{error}");
    assert_eq!(node.tip_height(), ROOT - 1, "stopped before the root body");

    // A "root" terminal at another height.
    let directory = tempfile::tempdir().unwrap();
    let mut node = node_at(directory.path(), &blocks, ROOT - 2);
    let tip = verified(&node, &blocks[(TIP - 1) as usize]);
    let not_the_root = verified(&node, &blocks[(ROOT - 2) as usize]);
    assert!(node
        .begin_preverified_recursive_suffix_with_root(tip, Some(not_the_root))
        .is_err());
}

/// I1, reorg: a heavier replacement forking below the root of its tip's
/// relation is not authorized by its tip's terminal alone; with the root
/// terminal of the replacement branch it is, and replaces the chain.
#[test]
fn a_reorg_below_the_root_of_its_tips_relation_needs_the_roots_terminal() {
    let _schedule = test_relation_schedule::install(Some(ACTIVATION), None);
    let ancestor = ROOT - 2;
    let honest = block_sequence(TIP as usize - 1);
    let replacement = {
        let producer_dir = tempfile::tempdir().unwrap();
        let mut producer = node_at(producer_dir.path(), &honest, ancestor);
        let mut branch = honest[..ancestor as usize].to_vec();
        // Up to TIP: one block more than the honest branch, and below the
        // test network's v1.4 height (6), whose anchor target a debug build
        // cannot mine in a test.
        for _ in ancestor..TIP {
            let bundle = test_next_bundle_for_miner(&producer, 0x63);
            accept_test_bundle(&mut producer, &bundle);
            branch.push(bundle);
        }
        branch
    };
    let tip_index = replacement.len() - 1;
    let directory = tempfile::tempdir().unwrap();
    let mut node = node_at(directory.path(), &honest, honest.len() as u64);
    let tip = verified(&node, &replacement[tip_index]);
    assert!(node.authorize_preverified_reorg_suffix(ancestor, tip).is_err());

    let tip = verified(&node, &replacement[tip_index]);
    let root = verified(&node, &replacement[(ROOT - 1) as usize]);
    let authority = node
        .authorize_preverified_reorg_suffix_with_root(ancestor, tip, Some(root))
        .unwrap();
    let bodies: Vec<Vec<u8>> = replacement[ancestor as usize..]
        .iter()
        .map(|bundle| bundle.block_bytes().to_vec())
        .collect();
    node.apply_verified_reorg_suffix_with_applier(
        authority,
        &bodies,
        header_of(&replacement[tip_index]).timestamp,
        |block, state| {
            crate::materialize_accepted_block_state(state, block)
                .map_err(|error| format!("{error:?}"))
        },
    )
    .unwrap();
    assert_eq!(node.tip_hash(), replacement[tip_index].block_hash());
}
