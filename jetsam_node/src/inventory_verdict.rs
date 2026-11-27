// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! What one header batch does to its sender's divergence verdict, as the
//! `HeaderInventoryBatch` arm reads it (T3.10b, audit A3 §4).
//!
//! The batch is planned against the canonical chain and the header DAG
//! ([`plan_header_inventory`]); the plan then settles the sender's verdict,
//! before anything else in the arm reads it:
//! * confirmed through the tip, or a candidate through its target: a verdict
//!   at or below that height ends (the peer is on this chain through it),
//!   one above it does not — headers below a fork say nothing past it;
//! * no common ancestor inside the non-final window: refused at the
//!   finalized height;
//! * a header refused by a rule only a fork explains: refused at its height;
//! * behind, older headers wanted, or a local failure: nothing.
//!
//! The answers to a branch probe (`branch_observer`) go through the same
//! reading. Going down from the divergence height they carry the canonical
//! headers below it, which confirm nothing at the verdict's height; going up
//! they carry the refused branch, which is refused again or wants older
//! headers. The verdict only moves toward refusal: the test below drives both
//! through this function, the one the arm calls.

use std::sync::Arc;
use std::time::Instant;

use jetsam_chain::storage::{MdbxChainContext, MdbxStore};
use jetsam_node::networking::header_dag::HeaderDag;
use jetsam_p2p::header_protocol::HeaderInventoryRecord;
use libp2p::PeerId;
use tokio::sync::RwLock;

use crate::peer_chain_evidence::PeerChainEvidence;
use crate::HeaderInventoryPlan;

/// Plan `records` from `from`, then settle `from`'s verdict from the plan
/// (module comment). The plan is returned for the rest of the arm.
pub(crate) async fn plan_and_settle<H>(
    chain: &Arc<RwLock<MdbxChainContext>>,
    store: &MdbxStore,
    header_dag: &mut HeaderDag,
    evidence: &mut PeerChainEvidence<H>,
    from: PeerId,
    records: Vec<HeaderInventoryRecord>,
    now: Instant,
) -> Result<HeaderInventoryPlan, String> {
    let plan = crate::plan_header_inventory(chain, store, header_dag, records).await;
    match &plan {
        Ok(HeaderInventoryPlan::Confirmed { tip }) => {
            evidence.confirm_compatible_through(from, tip.height);
        }
        Ok(HeaderInventoryPlan::Candidate { target, .. }) => {
            evidence.confirm_compatible_through(from, target.height);
        }
        Ok(HeaderInventoryPlan::FinalizedDivergence) => {
            let finalized_height = header_dag.finalized().height;
            crate::refuse_peer_branch(header_dag, evidence, from, finalized_height, now);
        }
        Ok(HeaderInventoryPlan::ConsensusFork { height, .. }) => {
            crate::refuse_peer_branch(header_dag, evidence, from, *height, now);
        }
        Ok(HeaderInventoryPlan::Behind | HeaderInventoryPlan::NeedOlder { .. }) | Err(_) => {}
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::block_header::{block_id, BlockHeader};
    use jetsam_chain::consensus::params::block_time_at;
    use jetsam_chain::consensus::{
        asert_anchor_height, expected_target, genesis_header, AnchorInfo,
    };
    use jetsam_node::branch_observer::{BranchObserver, BranchProbe, CanonicalView, DivergentPeer};
    use jetsam_node::branch_observer_node::StoreCanonicalView;
    use std::collections::HashSet;

    /// The pre-mined nonce of `snapshot_header_staging`'s fixture header at
    /// height 1 (`FIXTURE_CHAIN_NONCE`, one per network; re-mined there by
    /// `print_new_fixture_nonce` after a layout or genesis change).
    const FIXTURE_CHAIN_NONCE: u128 = if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        611_542
    } else {
        162_878
    };

    /// `snapshot_header_staging`'s fixture header 1: a child of genesis that
    /// passes every native rule, proof of work included.
    fn valid_child_of_genesis() -> BlockHeader {
        let genesis = genesis_header();
        let anchor_height = asert_anchor_height(0);
        assert_eq!(anchor_height, 0);
        BlockHeader {
            prev_block_hash: block_id(&genesis),
            state_root: [1; 32],
            tx_root: [0x41; 32],
            timestamp: genesis.timestamp + block_time_at(1),
            height: 1,
            miner_address: genesis.miner_address,
            nonce: FIXTURE_CHAIN_NONCE,
            difficulty_target: expected_target(
                &AnchorInfo {
                    anchor_height,
                    anchor_timestamp: genesis.timestamp,
                    anchor_parent_timestamp: genesis.timestamp,
                    anchor_target: genesis.difficulty_target,
                },
                1,
                genesis.timestamp,
            ),
            log_slots: genesis.log_slots,
            active_slot_count: genesis.active_slot_count,
            alloc_counter: genesis.alloc_counter,
        }
    }

    /// A child of `parent` under another rule set's target: refused by a
    /// rule only a fork explains (`BadDifficultyTarget`).
    fn refused_child_of(parent: &BlockHeader) -> BlockHeader {
        let mut header = *parent;
        header.prev_block_hash = block_id(parent);
        header.height = parent.height + 1;
        header.timestamp = parent.timestamp + block_time_at(header.height);
        header.state_root = [0xF2; 32];
        header.nonce = 0;
        header.difficulty_target[0] ^= 0x01;
        header
    }

    fn records(headers: &[BlockHeader]) -> Vec<HeaderInventoryRecord> {
        headers
            .iter()
            .map(|header| HeaderInventoryRecord::header_only(*header))
            .collect()
    }

    fn unix_now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// A3 §4, required of T3.10b. A node at genesis; a peer whose branch is
    /// 1 (valid, this node does not hold it yet), then 2' refused for its
    /// target: refused at 2. The branch observer probes it, and each answer
    /// goes through the arm's own reading (`plan_and_settle`), as on the
    /// node. Going down (canonical headers below the divergence height) the
    /// answers confirm or offer a candidate below 2 and end nothing; going up
    /// (the refused branch) they want older headers and end nothing; the
    /// whole range refuses the peer again. The verdict only moves toward
    /// refusal. A peer refused at 1 that now serves the valid 1 is the
    /// contrast: there the same reading ends the verdict, as it must.
    #[tokio::test]
    async fn a_probe_answer_moves_a_verdict_only_toward_refusal() {
        let directory = tempfile::tempdir().unwrap();
        let context = MdbxChainContext::open_or_create(directory.path()).unwrap();
        let store = context.store.clone();
        let mut dag = crate::canonical_header_dag(&context).unwrap();
        let chain = Arc::new(RwLock::new(context));
        let genesis = genesis_header();
        let one = valid_child_of_genesis();
        let two = refused_child_of(&one);
        let peer = PeerId::random();
        let mut evidence = PeerChainEvidence::<()>::default();
        let now = Instant::now();

        // The batch that refuses it.
        let plan = plan_and_settle(
            &chain,
            &store,
            &mut dag,
            &mut evidence,
            peer,
            records(&[genesis, one, two]),
            now,
        )
        .await;
        assert!(
            matches!(
                plan,
                Ok(HeaderInventoryPlan::ConsensusFork { height: 2, .. })
            ),
            "{:?}",
            plan.as_ref().err()
        );
        assert_eq!(evidence.divergence_height(&peer, now), Some(2));

        // The observer probes it, under the production rules.
        let unix = unix_now();
        let canonical = StoreCanonicalView::at_tip(store.clone()).unwrap();
        let mut observer = BranchObserver::new(unix);
        let only_peer = HashSet::from([peer]);
        observer.on_divergent_peers(
            &[DivergentPeer {
                peer,
                divergence_height: 2,
                announced_tip: None,
            }],
            unix,
        );
        assert_eq!(
            observer.probes_due(&only_peer, &HashSet::new(), unix),
            [BranchProbe {
                peer,
                start_height: 0,
                count: 3
            }]
        );

        // Down: what it serves below 2. A candidate through 1 ends no
        // verdict at 2...
        let down = records(&[genesis, one]);
        observer.on_headers(peer, &down, &canonical, unix);
        let plan = plan_and_settle(&chain, &store, &mut dag, &mut evidence, peer, down, now).await;
        assert!(
            matches!(&plan, Ok(HeaderInventoryPlan::Candidate { target, .. }) if target.height == 1),
            "{:?}",
            plan.as_ref().err()
        );
        assert_eq!(
            evidence.divergence_height(&peer, now),
            Some(2),
            "down: no verdict ends"
        );
        // ...nor does a confirmation of this node's tip.
        let plan = plan_and_settle(
            &chain,
            &store,
            &mut dag,
            &mut evidence,
            peer,
            records(&[genesis]),
            now,
        )
        .await;
        assert!(matches!(&plan, Ok(HeaderInventoryPlan::Confirmed { tip }) if tip.height == 0));
        assert_eq!(
            evidence.divergence_height(&peer, now),
            Some(2),
            "down: no verdict ends"
        );

        // Up: the refused branch, above the fork the observer found.
        assert_eq!(
            observer.probes_due(&only_peer, &HashSet::new(), unix),
            [BranchProbe {
                peer,
                start_height: 2,
                count: 512
            }]
        );
        let up = records(&[two]);
        observer.on_headers(peer, &up, &canonical, unix);
        let plan = plan_and_settle(&chain, &store, &mut dag, &mut evidence, peer, up, now).await;
        assert!(matches!(plan, Ok(HeaderInventoryPlan::NeedOlder { .. })));
        assert_eq!(
            evidence.divergence_height(&peer, now),
            Some(2),
            "up: no verdict ends"
        );

        // The whole range: refused again.
        let later = now + std::time::Duration::from_secs(60);
        let plan = plan_and_settle(
            &chain,
            &store,
            &mut dag,
            &mut evidence,
            peer,
            records(&[genesis, one, two]),
            later,
        )
        .await;
        assert!(matches!(
            plan,
            Ok(HeaderInventoryPlan::ConsensusFork { height: 2, .. })
        ));
        assert_eq!(evidence.divergence_height(&peer, later), Some(2));
        assert!(
            evidence.is_divergent(
                &peer,
                now + crate::peer_chain_evidence::PEER_CHAIN_EVIDENCE_TTL
            ),
            "the verdict was renewed"
        );

        // What the observer made of the answers: forked at genesis, 1 weighed
        // (its proof of work checked), refused at 2.
        let response = observer.observed_branches(0, canonical.tip(), [0; 32], unix);
        assert_eq!(response.branches.len(), 1);
        let branch = &response.branches[0];
        assert_eq!((branch.fork_height, branch.tip_height), (0, 2));
        assert_eq!(branch.rule.as_deref(), Some("h=2: BadDifficultyTarget"));
        assert_eq!(
            branch.work_hex,
            hex::encode(jetsam_chain::add_work(
                &canonical.work_at(0).unwrap(),
                &jetsam_chain::block_work(&one.difficulty_target)
            ))
        );

        // The contrast: refused at 1, it now serves the valid 1.
        let moved = PeerId::random();
        evidence.mark_divergent(moved, 1, now);
        plan_and_settle(
            &chain,
            &store,
            &mut dag,
            &mut evidence,
            moved,
            records(&[genesis, one]),
            now,
        )
        .await
        .unwrap();
        assert!(
            !evidence.is_divergent(&moved, now),
            "validated through 1: it ends"
        );

        // No common ancestor inside the non-final window (another genesis):
        // refused at the finalized height.
        let mut other_genesis = genesis;
        other_genesis.state_root = [0xEE; 32];
        let stranger = PeerId::random();
        let plan = plan_and_settle(
            &chain,
            &store,
            &mut dag,
            &mut evidence,
            stranger,
            records(&[other_genesis]),
            now,
        )
        .await;
        assert!(matches!(plan, Ok(HeaderInventoryPlan::FinalizedDivergence)));
        assert_eq!(
            evidence.divergence_height(&stranger, now),
            Some(dag.finalized().height)
        );
    }
}
