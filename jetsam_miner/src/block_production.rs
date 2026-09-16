// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The sole node-owned block-production path.
//!
//! Both the internal miner and `submitBlock(template_id, nonce)` prepare this
//! exact nonce-independent witness before PoW, then consume it once to prove
//! and atomically commit one [`jetsam_chain::AcceptedBlockBundle`].

use jetsam_chain::block::Block;
use jetsam_chain::consensus::params::HistoryStepPackGeneration;
use jetsam_chain::consensus::pow::{block_id, validate_pow};
use jetsam_chain::storage::{MdbxChainContext, MdbxContextError};

use crate::template::BlockTemplate;

type HistoryStepRuntime = jetsam_recursive::acceptance::history_step::HistoryStepRuntime;
type PreparedGhost =
    jetsam_recursive::acceptance::history_step::PreparedHistoryStepGhostAuthorization;
type PreparedStateCommit = jetsam_chain::consensus::template::PreparedBlockStateCommit;
type LocallyProvedStateCommit = jetsam_chain::consensus::template::LocallyProvedBlockCommit;

/// The witness, at the page-position tier of the class *and generation* that
/// proves it. The small class holds 25 positions under the launch relation
/// and 24 under v1.3; the large class holds 255 in both, so it needs no
/// second arm.
enum PreparedWitness {
    B24(jetsam_block::PreparedHistoryStepWitness<24>),
    B25(jetsam_block::PreparedHistoryStepWitness<25>),
    B255(jetsam_block::PreparedHistoryStepWitness<255>),
}

/// A single-use, nonce-independent block witness prepared entirely by the
/// node. The external PoW worker never receives it and can change only nonce.
pub struct PreparedBlockAttempt {
    /// The relation this attempt was prepared under — the generation of the
    /// block's own height. Every class question asked of the block later is
    /// answered by it, so an attempt cannot be read under one ladder and
    /// proved under the other.
    generation: HistoryStepPackGeneration,
    block: Block,
    terminal_bytes: Vec<u8>,
    end_accumulator: jetsam_recursive::ChainAccumulator,
    start_accumulator: jetsam_recursive::ChainAccumulator,
    parent_header: jetsam_chain::BlockHeader,
    expected_parent_id: [u8; 32],
    expected_parent_height: u64,
    payload_weight: usize,
    retained_bytes: usize,
    state_commit: PreparedStateCommit,
}

/// A private-capability carrier created only after the HistoryStep prover has
/// successfully produced the exact terminal bundled with `block`.
pub struct ProvedBlock {
    local_commit: LocallyProvedStateCommit,
}

/// The same complete block after its bundle and public state transition have
/// been committed atomically to the canonical chain.
pub struct CommittedBlock {
    block: Block,
    bundle: jetsam_chain::AcceptedBlockBundle,
}

fn decode_template_authorizations(
    authorization_bytes: Vec<Option<Vec<u8>>>,
) -> Result<Vec<jetsam_gkr::zk_authorization::ZkAuthorizationProof>, String> {
    authorization_bytes
        .into_iter()
        .enumerate()
        .map(|(index, encoded)| {
            let encoded = encoded.ok_or_else(|| {
                format!("missing wallet authorization for user transaction {index}")
            })?;
            jetsam_gkr::WalletAuthorizationBundle::from_bytes(&encoded)
                .map(|bundle| bundle.proof)
                .map_err(|error| format!("wallet authorization {index} is not canonical: {error}"))
        })
        .collect()
}

/// The launch-generation boundary at `header`: one anchor, carried in
/// canonical form, the older lane conflated with it.
///
/// This is [`jetsam_recursive::ChainAccumulator::from_canonical_headers`]
/// under the launch generation, and exists so that the tests below read as
/// the one-anchor statement they are.
#[cfg(test)]
fn accumulator_from_header_boundary(
    header: &jetsam_chain::BlockHeader,
    epoch_anchor_header: &jetsam_chain::BlockHeader,
) -> jetsam_recursive::ChainAccumulator {
    jetsam_recursive::ChainAccumulator::from_canonical_headers(
        HistoryStepPackGeneration::V1,
        header,
        epoch_anchor_header,
        None,
    )
}

impl PreparedBlockAttempt {
    /// Prepare every nonce-independent part of the next HistoryStep.
    pub fn prepare(
        template: BlockTemplate,
        runtime: &HistoryStepRuntime,
        ghost: &PreparedGhost,
        local_time: u64,
    ) -> Result<Self, String> {
        Self::prepare_inner(template, runtime, ghost, local_time, None)?.ok_or_else(|| {
            "uncancellable HistoryStep preparation was unexpectedly cancelled".to_string()
        })
    }

    /// Prepare a block while allowing a canonical-tip change to stop work at
    /// transcript-safe phase boundaries. `Ok(None)` is an expected local
    /// cancellation, not a proving failure.
    pub fn prepare_cancellable(
        template: BlockTemplate,
        runtime: &HistoryStepRuntime,
        ghost: &PreparedGhost,
        local_time: u64,
        cancellation: &std::sync::atomic::AtomicBool,
    ) -> Result<Option<Self>, String> {
        Self::prepare_inner(template, runtime, ghost, local_time, Some(cancellation))
    }

    fn prepare_inner(
        template: BlockTemplate,
        runtime: &HistoryStepRuntime,
        ghost: &PreparedGhost,
        local_time: u64,
        cancellation: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Option<Self>, String> {
        let cancelled =
            || cancellation.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire));
        if cancelled() {
            return Ok(None);
        }
        let BlockTemplate {
            inner,
            parent,
            authorization_bytes,
            parent_state,
            finalized_active_counts,
            previous_timestamps,
            asert_anchor,
            parent_tx_epoch_anchor_header,
            parent_previous_tx_epoch_anchor_header,
            parent_history_step_terminal_bytes,
            prepared_state_commit,
            ..
        } = template;

        let expected_parent_id = block_id(&parent);
        let expected_parent_height = parent.height;
        // The relation this block is proved under is the one its own height
        // selects — never the tip's, never the compiled ladder's. The runtime
        // was resolved from the same height by the caller's selector; if the
        // two disagree the wrong pack is in hand, and refusing here costs
        // nothing while proving against it would waste the whole attempt.
        let child_height = expected_parent_height.saturating_add(1);
        let generation = HistoryStepPackGeneration::at_height(child_height);
        if runtime.bank().generation() != generation {
            return Err(format!(
                "height {child_height} is proved by {generation:?} but the supplied HistoryStep \
                 runtime carries {:?}",
                runtime.bank().generation()
            ));
        }
        let authorization_weight = authorization_bytes
            .iter()
            .filter_map(|bytes| bytes.as_ref())
            .try_fold(0usize, |total, bytes| total.checked_add(bytes.len()))
            .ok_or_else(|| "prepared authorization byte weight overflow".to_string())?;
        let authorization_proofs = decode_template_authorizations(authorization_bytes)?;
        if cancelled() {
            return Ok(None);
        }
        let start_accumulator = jetsam_recursive::ChainAccumulator::from_canonical_headers(
            generation,
            &parent,
            &parent_tx_epoch_anchor_header,
            parent_previous_tx_epoch_anchor_header.as_ref(),
        );
        let parent_terminal = match (parent.height, parent_history_step_terminal_bytes) {
            (0, None) => None,
            (0, Some(_)) => {
                return Err("genesis parent unexpectedly has a HistoryStep terminal".into());
            }
            (_, Some(bytes)) => Some(
                jetsam_recursive::acceptance::history_step::decode_history_step_terminal(
                    runtime, &bytes,
                )
                .map_err(|error| format!("parent HistoryStep terminal: {error}"))?,
            ),
            (_, None) => return Err("non-genesis parent HistoryStep terminal is missing".into()),
        };
        if cancelled() {
            return Ok(None);
        }

        let block = inner.into_block(0);
        let payload_weight = block
            .to_bytes()
            .len()
            .checked_add(authorization_weight)
            .ok_or_else(|| "prepared block byte weight overflow".to_string())?;
        let stream = jetsam_chain::validate_block_page_stream_in(&block.transactions, generation)
            .map_err(|error| format!("prepared block body is non-canonical: {error}"))?;
        let proof_class = stream.proof_class;
        let tier = proof_class.page_capacity_in_generation(generation);
        // Last line of defence, before any proof of work is spent on this
        // template. A terminal too large for the wire cap is refused by every
        // node including the one that built it, and the refusal only arrives at
        // submitBlock — by which point the transactions are frozen into the
        // hash, so nothing can be dropped to rescue the solution. Refuse here,
        // where refusing costs nothing.
        let class_id = jetsam_recursive::canonical_history_step_class_id_in(generation, tier)
            .ok_or_else(|| format!("no proof class registered for the {tier}-page tier"))?;
        let terminal_bytes = jetsam_recursive::history_step_terminal_wire_bytes(runtime, class_id)
            .map_err(|error| format!("terminal size for {proof_class:?} is unknown: {error:?}"))?;
        // The cap this block will be judged by is the one active at its own
        // height — the child of this parent — not at the tip we happen to see.
        let terminal_cap = jetsam_chain::consensus::wire_limits::history_step_terminal_bytes_limit(child_height);
        if terminal_bytes > terminal_cap {
            return Err(format!(
                "refusing to prepare a {proof_class:?} template at height {child_height}: its \
                 terminal is {terminal_bytes} bytes and the consensus cap is {terminal_cap}"
            ));
        }
        let context = jetsam_block::HistoryStepPreparationContext {
            parent_header: &parent,
            tx_epoch_anchor_header: &parent_tx_epoch_anchor_header,
            parent_state: &parent_state,
            start_accumulator: &start_accumulator,
            previous_timestamps: &previous_timestamps,
            finalized_active_counts: &finalized_active_counts,
            asert_anchor: &asert_anchor,
            local_time,
            generation,
            previous_tx_epoch_anchor_header: parent_previous_tx_epoch_anchor_header.as_ref(),
        };
        // Dispatch on the tier, not on the class name: the small class is
        // `<25>` under the launch relation and `<24>` under v1.3, and the
        // tier is the only value that names the right one in both.
        let witness = match tier {
            24 => jetsam_block::prepare_history_step_witness::<24>(
                block,
                context,
                authorization_proofs,
                ghost,
                runtime,
                parent_terminal.as_ref(),
            )
            .map(PreparedWitness::B24),
            25 => jetsam_block::prepare_history_step_witness::<25>(
                block,
                context,
                authorization_proofs,
                ghost,
                runtime,
                parent_terminal.as_ref(),
            )
            .map(PreparedWitness::B25),
            255 => jetsam_block::prepare_history_step_witness::<255>(
                block,
                context,
                authorization_proofs,
                ghost,
                runtime,
                parent_terminal.as_ref(),
            )
            .map(PreparedWitness::B255),
            other => {
                return Err(format!(
                    "{generation:?} selected a {other}-page tier for {proof_class:?}, which this \
                     binary has no witness for"
                ));
            }
        }
        .map_err(|error| error.to_string())?;
        if cancelled() {
            return Ok(None);
        }

        // The relation is nonce-free: finish the exact template (every native
        // check except PoW) and prove the complete HistoryStep before any
        // nonce search. Post-nonce work is one native PoW check + atomic
        // commit.
        macro_rules! finish_and_prove {
            ($witness:expr) => {{
                let (block, built, end) = $witness
                    .finish_template(runtime)
                    .map_err(|error| error.to_string())?;
                if cancelled() {
                    return Ok(None);
                }
                let terminal = match cancellation {
                    Some(cancellation) => {
                        match jetsam_recursive::acceptance::history_step::prove_built_history_step_terminal_cancellable(
                            runtime,
                            &built,
                            cancellation,
                        ) {
                            Ok(terminal) => terminal,
                            Err(jetsam_recursive::acceptance::history_step::HistoryStepError::Cancelled) => {
                                return Ok(None);
                            }
                            Err(error) => return Err(error.to_string()),
                        }
                    }
                    None => jetsam_recursive::acceptance::history_step::prove_built_history_step_terminal(
                        runtime,
                        &built,
                    )
                    .map_err(|error| error.to_string())?,
                };
                if cancelled() {
                    return Ok(None);
                }
                let terminal_bytes =
                    jetsam_recursive::acceptance::history_step::encode_history_step_terminal(
                        runtime, &terminal,
                    )
                    .map_err(|error| error.to_string())?;
                if cancelled() {
                    return Ok(None);
                }
                (block, terminal_bytes, end)
            }};
        }
        let (block, terminal_bytes, end_accumulator) = match witness {
            PreparedWitness::B24(witness) => finish_and_prove!(witness),
            PreparedWitness::B25(witness) => finish_and_prove!(witness),
            PreparedWitness::B255(witness) => finish_and_prove!(witness),
        };

        let retained_bytes = payload_weight
            .checked_add(terminal_bytes.len())
            .ok_or_else(|| "prepared HistoryStep retained-byte weight overflow".to_string())?;

        Ok(Some(Self {
            generation,
            block,
            terminal_bytes,
            end_accumulator,
            start_accumulator,
            parent_header: parent,
            expected_parent_id,
            expected_parent_height,
            payload_weight,
            retained_bytes,
            state_commit: prepared_state_commit,
        }))
    }

    pub fn pow_header(&self, nonce: u128) -> jetsam_chain::BlockHeader {
        let mut header = self.block.header;
        header.nonce = nonce;
        header
    }

    pub fn user_page_count(&self) -> usize {
        usize::from(
            jetsam_chain::validate_block_page_stream_in(&self.block.transactions, self.generation)
                .expect("prepared block has a canonical body")
                .page_count,
        )
    }

    /// The relation this attempt was prepared and proved under.
    pub const fn generation(&self) -> HistoryStepPackGeneration {
        self.generation
    }

    pub fn proof_class(&self) -> jetsam_chain::consensus::paged_spend::BlockProofClass {
        jetsam_chain::validate_block_page_stream_in(&self.block.transactions, self.generation)
            .expect("prepared block has a canonical body")
            .proof_class
    }

    pub const fn expected_parent_id(&self) -> [u8; 32] {
        self.expected_parent_id
    }

    pub const fn expected_parent_height(&self) -> u64 {
        self.expected_parent_height
    }

    /// Consensus-bounded serialized block and authorization payload weight.
    pub const fn payload_weight(&self) -> usize {
        self.payload_weight
    }

    /// Exact block/auth byte weight plus the staged builder's allocated
    /// witness buffer. External lifecycle admission uses this independently
    /// of the consensus payload cap.
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Seal the winning nonce with one native PoW check and create the only
    /// complete network/storage block object.
    ///
    /// The complete HistoryStep terminal was already proven at preparation:
    /// the relation is nonce-free and binds the semantic projection, so it
    /// covers every nonce of this exact template. `runtime` stays in the
    /// signature as the proving-authority witness of the caller's phase.
    pub fn prove(self, _runtime: &HistoryStepRuntime, nonce: u128) -> Result<ProvedBlock, String> {
        let sealed_header = self.pow_header(nonce);
        validate_pow(&sealed_header).map_err(|error| format!("proof of work: {error}"))?;
        let end = self
            .start_accumulator
            .advance(&self.parent_header, &sealed_header)
            .map_err(|error| format!("sealed accumulator transition failed: {error:?}"))?;
        if end != self.end_accumulator {
            return Err("sealed boundary drifted from the proven template".to_string());
        }
        let mut block = self.block;
        block.header.nonce = nonce;
        // SAFETY: `finish_template` performed the complete native consensus
        // checks for this exact template, `validate_pow` above checked the
        // only nonce-dependent rule, and `terminal_bytes` is the canonical
        // encoding of the terminal returned directly by the pinned prover at
        // preparation. The local commit intentionally does not verify its own
        // freshly-authored proof a second time.
        let local_commit = unsafe {
            self.state_commit
                .seal_after_trusted_history_step_proof_unchecked(block, self.terminal_bytes)
        }?;
        Ok(ProvedBlock { local_commit })
    }
}

impl ProvedBlock {
    pub const fn block(&self) -> &Block {
        self.local_commit.block()
    }

    pub const fn bundle(&self) -> &jetsam_chain::AcceptedBlockBundle {
        self.local_commit.bundle()
    }

    /// Consume the post-prover capability and commit without replaying the
    /// proof that this process has just generated successfully.
    pub fn commit(self, chain: &mut MdbxChainContext) -> Result<CommittedBlock, MdbxContextError> {
        let (block, bundle) = chain.commit_locally_proved_next_block(self.local_commit)?;
        Ok(CommittedBlock { block, bundle })
    }
}

impl CommittedBlock {
    pub const fn block(&self) -> &Block {
        &self.block
    }

    pub const fn bundle(&self) -> &jetsam_chain::AcceptedBlockBundle {
        &self.bundle
    }

    pub fn into_parts(self) -> (Block, jetsam_chain::AcceptedBlockBundle) {
        (self.block, self.bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::params::GENESIS_TARGET;
    use jetsam_poseidon2b::primitives::Address;

    fn header(height: u64, prev_block_hash: [u8; 32]) -> jetsam_chain::BlockHeader {
        jetsam_chain::BlockHeader {
            prev_block_hash,
            state_root: [height as u8; 32],
            tx_root: [0x33; 32],
            timestamp: 1_000 + height,
            height,
            miner_address: Address([0x44; 32]),
            nonce: height as u128,
            difficulty_target: GENESIS_TARGET,
            log_slots: 24,
            active_slot_count: height,
            alloc_counter: height,
        }
    }

    #[test]
    fn production_boundary_uses_parent_anchor_before_advancing_child_anchor() {
        // JETSAM CHANGE: the epoch boundary is derived from TX_EPOCH_BLOCKS
        // rather than the literal 144.
        let epoch = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let genesis = jetsam_chain::consensus::genesis_header();
        let boundary = header(epoch, [0x11; 32]);
        let child = header(epoch + 1, block_id(&boundary));

        let start = accumulator_from_header_boundary(&boundary, &genesis);
        start
            .validate_local_header_boundary(&boundary, &genesis)
            .expect("the boundary block's terminal still carries the genesis anchor");

        let end = start
            .advance(&boundary, &child)
            .expect("crossing the boundary advances the recursive anchor");
        end.validate_local_header_boundary(&child, &boundary)
            .expect("the next block's terminal carries the derived boundary anchor");

        let conflated = accumulator_from_header_boundary(&boundary, &boundary);
        assert!(matches!(
            conflated.validate_local_header_boundary(&boundary, &boundary),
            Err(
                jetsam_recursive::ChainAccumulatorLocalBoundaryError::EpochAnchorHeight {
                    expected: 0,
                    actual,
                }
            ) if actual == epoch
        ));
    }

    /// The start boundary the miner hands the witness has the shape of the
    /// generation proving the child, and of nothing else: one anchor with the
    /// older lane conflated at launch, two distinct anchors from v1.3. Under
    /// the launch generation the previous-anchor header is not consulted even
    /// when one is supplied, which is what keeps the pre-fork path fixed.
    #[test]
    fn the_start_boundary_has_the_shape_of_the_childs_generation() {
        use jetsam_chain::consensus::params::HistoryStepPackGeneration::{V1, V1_3};

        let epoch = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let parent = header(3 * epoch + 1, [0x11; 32]);
        let current = header(3 * epoch, [0x22; 32]);
        let previous = header(2 * epoch, [0x33; 32]);

        let launch =
            jetsam_recursive::ChainAccumulator::from_canonical_headers(V1, &parent, &current, None);
        assert_eq!(launch, accumulator_from_header_boundary(&parent, &current));
        assert_eq!(launch.previous_epoch_anchor_id, launch.epoch_anchor_id);
        assert_eq!(
            jetsam_recursive::ChainAccumulator::from_canonical_headers(
                V1,
                &parent,
                &current,
                Some(&previous)
            ),
            launch,
            "the launch relation has no older lane to fill"
        );

        let v1_3 = jetsam_recursive::ChainAccumulator::from_canonical_headers(
            V1_3,
            &parent,
            &current,
            Some(&previous),
        );
        assert_eq!(v1_3.epoch_anchor_id, launch.epoch_anchor_id);
        assert_eq!(v1_3.previous_epoch_anchor_id, block_id(&previous));
        assert_ne!(v1_3.previous_epoch_anchor_id, v1_3.epoch_anchor_id);
        v1_3.validate_local_header_boundary_in(V1_3, &parent, &current, Some(&previous))
            .expect("both anchors are bound to their canonical headers");
        launch
            .validate_local_header_boundary_in(V1, &parent, &current, None)
            .expect("the launch boundary binds the one anchor it has");
    }

    /// The witness tier the miner dispatches on is the generation's page
    /// capacity for the class it selected, and this binary carries an arm and
    /// a registered proof class for every tier either generation can name.
    /// The twenty-fifth page is the small class at launch and the large class
    /// under v1.3 — the same block, sorted by the relation proving it.
    #[test]
    fn the_witness_tier_is_the_generations_and_every_tier_has_an_arm() {
        use jetsam_chain::consensus::paged_spend::BlockProofClass;
        use jetsam_chain::consensus::params::HistoryStepPackGeneration::{V1, V1_3};

        for (pages, generation, expected_tier) in [
            (0, V1, 25),
            (24, V1, 25),
            (25, V1, 25),
            (26, V1, 255),
            (255, V1, 255),
            (0, V1_3, 24),
            (24, V1_3, 24),
            (25, V1_3, 255),
            (255, V1_3, 255),
        ] {
            let class = BlockProofClass::for_page_count_in_generation(pages, generation)
                .expect("both ladders hold 255 pages");
            let tier = class.page_capacity_in_generation(generation);
            assert_eq!(tier, expected_tier, "{pages} pages under {generation:?}");
            // Exactly the three arms of `PreparedWitness`.
            assert!(
                matches!(tier, 24 | 25 | 255),
                "no witness arm for the {tier}-page tier"
            );
            assert!(
                jetsam_recursive::canonical_history_step_class_id_in(generation, tier).is_some(),
                "no proof class registered for the {tier}-page tier of {generation:?}"
            );
        }
        // A block too large for either ladder has no class, and so no tier.
        for generation in [V1, V1_3] {
            assert_eq!(
                BlockProofClass::for_page_count_in_generation(256, generation),
                None
            );
        }
    }
}
