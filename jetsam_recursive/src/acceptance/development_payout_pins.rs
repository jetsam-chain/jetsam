// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The development-payout recipients a frozen class matrix carries.
//!
//! [`bind_development_payout_action`] names the two fund addresses of the
//! profile it was compiled under, so a matrix pack is not profile-neutral: it
//! freezes one network's recipients for ever. A pack generated on the wrong
//! profile is nonetheless accepted by every existing check, because the gate
//! that reads those constants is armed only once per target-time day:
//!
//! ```text
//! difference = actual_wire + const_block(expected)   // char 2: + is XOR
//! gated      = mul(builder, live, difference)        // one R1CS row r
//! pin_zero(gated)
//! ```
//!
//! `live` is `development_payout_due(child_height)`, false on every height
//! that is not a multiple of `TARGET_BLOCKS_PER_DAY`. On those heights
//! `A[r]·z = 0` and the row is satisfied whatever constant `B[r]` holds. On
//! the 960th it is not, and every node stops at the same block. The test chain
//! did exactly that at block 1920 on 2026-09-17.
//!
//! What a matrix stores in the constant-pinned column of those rows is not the
//! address alone. The payout spine is
//! `select_spine_inputs_trace(live, raw, ghost)` and a select is
//! `when_zero + mul(selector, when_one + when_zero)`, so the dead branch — the
//! constant canonical protocol ghost body — contributes a constant term that
//! rides along into the gate:
//!
//! ```text
//! B[r][const_pin] = ghost.leaves[owner_leaf][lane] + expected_address_lane
//! ```
//!
//! On the lab leaf the ghost lane happens to be zero and the constant is the
//! address alone; on the network leaf it is not. Reading the column naively
//! therefore matches one fund and not the other, which is exactly the kind of
//! half-passing check that would have let 2026-09-17 through a second time.
//!
//! [`DevelopmentPayoutPinScan`] is the reader. It is a streaming scan so that
//! a caller can feed it a constant column straight off a multi-hundred-megabyte
//! embedded relation without materializing it.
//!
//! [`bind_development_payout_action`]: super::trace::action_surface::bind_development_payout_action

use jetsam_ivc_core::field::F128;
use jetsam_poseidon2b::primitives::Address;

use super::trace::action_surface::{LEAF_OUTPUT0_OWNER, LEAF_OUTPUT1_OWNER};
use super::trace::flat_of;

/// Two recipients, two lanes each.
pub const DEVELOPMENT_PAYOUT_PIN_COUNT: usize = 4;

/// What each pin is, in the order `bind_development_payout_action` emits them.
pub const DEVELOPMENT_PAYOUT_PIN_LABELS: [&str; DEVELOPMENT_PAYOUT_PIN_COUNT] = [
    "network fund lane 0",
    "network fund lane 1",
    "lab fund lane 0",
    "lab fund lane 1",
];

const _: () = assert!(LEAF_OUTPUT0_OWNER + 2 == LEAF_OUTPUT1_OWNER);

/// The four constants the payout gate freezes for `network_fund`/`lab_fund`.
pub fn development_payout_owner_pins_for(
    network_fund: Address,
    lab_fund: Address,
) -> [F128; DEVELOPMENT_PAYOUT_PIN_COUNT] {
    let ghost =
        jetsam_gkr::spine_statement::spine_inputs_from_body(&jetsam_gkr::ghost_tx::ghost_tx_body());
    let mut pins = [F128::ZERO; DEVELOPMENT_PAYOUT_PIN_COUNT];
    for (recipient, address) in [network_fund, lab_fund].into_iter().enumerate() {
        let owner_leaf = LEAF_OUTPUT0_OWNER + 2 * recipient;
        for (lane, expected) in address.as_fields().into_iter().enumerate() {
            pins[2 * recipient + lane] =
                flat_of(ghost.leaves[owner_leaf][lane]) + flat_of(expected);
        }
    }
    pins
}

/// The four constants this build's own consensus recipients produce.
pub fn development_payout_owner_pins() -> [F128; DEVELOPMENT_PAYOUT_PIN_COUNT] {
    development_payout_owner_pins_for(
        jetsam_chain::consensus::development_allocation::NETWORK_FUND_ADDRESS,
        jetsam_chain::consensus::development_allocation::LAB_FUND_ADDRESS,
    )
}

/// A matrix whose constant column is missing one of this build's payout pins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevelopmentPayoutPinError {
    /// Which of the four constants was absent.
    pub label: &'static str,
    /// The recipient it belongs to.
    pub address: Address,
}

impl core::fmt::Display for DevelopmentPayoutPinError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "the relation does not pin the {} of this build (",
            self.label
        )?;
        for byte in self.address.0 {
            write!(formatter, "{byte:02x}")?;
        }
        write!(
            formatter,
            "): its matrices were generated under a different network profile"
        )
    }
}

impl std::error::Error for DevelopmentPayoutPinError {}

/// Streaming reader for the four payout constants of one class matrix.
///
/// Feed it every coefficient of the relation's constant-pinned column, in any
/// order, then [`finish`](Self::finish).
#[derive(Debug, Clone)]
pub struct DevelopmentPayoutPinScan {
    network_fund: Address,
    lab_fund: Address,
    pins: [F128; DEVELOPMENT_PAYOUT_PIN_COUNT],
    seen: [usize; DEVELOPMENT_PAYOUT_PIN_COUNT],
}

impl DevelopmentPayoutPinScan {
    /// A scan for the recipients this binary is compiled to pay.
    pub fn for_this_build() -> Self {
        Self::for_recipients(
            jetsam_chain::consensus::development_allocation::NETWORK_FUND_ADDRESS,
            jetsam_chain::consensus::development_allocation::LAB_FUND_ADDRESS,
        )
    }

    pub fn for_recipients(network_fund: Address, lab_fund: Address) -> Self {
        Self {
            network_fund,
            lab_fund,
            pins: development_payout_owner_pins_for(network_fund, lab_fund),
            seen: [0; DEVELOPMENT_PAYOUT_PIN_COUNT],
        }
    }

    /// The constants this scan is looking for.
    pub fn pins(&self) -> &[F128; DEVELOPMENT_PAYOUT_PIN_COUNT] {
        &self.pins
    }

    /// How many constant-column coefficients matched each pin so far.
    pub fn occurrences(&self) -> &[usize; DEVELOPMENT_PAYOUT_PIN_COUNT] {
        &self.seen
    }

    /// Offer one coefficient of the constant-pinned column.
    #[inline]
    pub fn observe(&mut self, constant: F128) {
        for (index, pin) in self.pins.iter().enumerate() {
            if *pin == constant {
                self.seen[index] += 1;
            }
        }
    }

    /// Fold in a scan of a disjoint part of the same column.
    pub fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.pins, other.pins, "two scans of different recipients");
        for (total, part) in self.seen.iter_mut().zip(other.seen.iter()) {
            *total += *part;
        }
    }

    /// `Ok` when every one of the four constants was somewhere in the column.
    ///
    /// Absence is the whole signal: a relation frozen under another profile
    /// carries that profile's four constants and none of these. A count above
    /// one is not an error here — a relation that grew a second payout gate
    /// would still pay these recipients, and refusing to start over it would
    /// cost exactly the availability this guard exists to protect. The
    /// per-emission count is asserted where it belongs, against the gate
    /// itself, in this module's tests.
    pub fn finish(&self) -> Result<(), DevelopmentPayoutPinError> {
        for (index, &count) in self.seen.iter().enumerate() {
            if count == 0 {
                return Err(DevelopmentPayoutPinError {
                    label: DEVELOPMENT_PAYOUT_PIN_LABELS[index],
                    address: if index < 2 {
                        self.network_fund
                    } else {
                        self.lab_fund
                    },
                });
            }
        }
        Ok(())
    }
}

/// Why a frozen relation could not be shown to pay this build's funds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevelopmentPayoutRelationError {
    /// The relation pins no constant column, so it holds no readable literal
    /// and this check would pass over every pack without reading one.
    NoConstantColumn,
    /// One of this build's recipients is not frozen in that column.
    MissingPin(DevelopmentPayoutPinError),
}

impl core::fmt::Display for DevelopmentPayoutRelationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoConstantColumn => write!(
                formatter,
                "the relation pins no constant column: its frozen constants cannot be read"
            ),
            Self::MissingPin(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DevelopmentPayoutRelationError {}

/// Read one frozen class relation's constant column and refuse it unless it
/// freezes this build's own development-payout recipients.
///
/// The column is walked one authenticated row group per task. A block-bearing
/// class is tens of millions of nonzeros and this runs on the startup path of
/// every node: a serial walk of it is seconds, and a fan-out over the groups
/// the artifact is already cut into is the same answer for a fraction of the
/// wall clock.
pub fn verify_relation_development_payout_pins(
    relation: &jetsam_ivc_core::field_r1cs::CompactFieldR1cs,
) -> Result<(), DevelopmentPayoutRelationError> {
    use rayon::prelude::*;

    let column = u32::try_from(
        relation
            .shape()
            .const_pin
            .ok_or(DevelopmentPayoutRelationError::NoConstantColumn)?,
    )
    .map_err(|_| DevelopmentPayoutRelationError::NoConstantColumn)?;
    let scan = (0..relation.b_group_count())
        .into_par_iter()
        .fold(
            DevelopmentPayoutPinScan::for_this_build,
            |mut scan, group| {
                relation.for_each_b_group_column_entry(group, column, |_row, coefficient| {
                    scan.observe(coefficient);
                });
                scan
            },
        )
        .reduce(
            DevelopmentPayoutPinScan::for_this_build,
            |mut left, right| {
                left.merge(&right);
                left
            },
        );
    scan.finish()
        .map_err(DevelopmentPayoutRelationError::MissingPin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::block_slots::{constant_spine_inputs_trace, select_spine_inputs_trace};
    use crate::acceptance::trace::action_surface::bind_development_payout_action;
    use crate::acceptance::trace::alloc_block;
    use crate::acceptance::trace::tx_body_spine::SpineInputsTrace;
    use jetsam_core::{Block128, TowerField};
    use jetsam_ivc_core::field_circuit::FieldR1csBuilder;
    use jetsam_tx::{output_bitmap_bit, TxBody, TxInput, TxOutput, TX_INPUTS};

    fn payout_body(amount: u64) -> TxBody {
        TxBody {
            epoch_anchor: [0x72; 32],
            fee: 0,
            input_owner: Address([0u8; 32]),
            inputs: [TxInput::dummy(); TX_INPUTS],
            outputs: [
                TxOutput {
                    slot_index: 24,
                    amount,
                    owner: jetsam_chain::consensus::NETWORK_FUND_ADDRESS,
                },
                TxOutput {
                    slot_index: 25,
                    amount,
                    owner: jetsam_chain::consensus::LAB_FUND_ADDRESS,
                },
            ],
            validity_bitmap: output_bitmap_bit(0) | output_bitmap_bit(1),
            is_coinbase: true,
        }
    }

    /// Build the payout gate with the production wiring: a real `live` wire,
    /// and the spine selected between the raw body and the constant ghost.
    fn payout_relation() -> jetsam_ivc_core::field_r1cs::FieldR1cs {
        let native = jetsam_gkr::spine_statement::spine_inputs_from_body(&payout_body(123));
        let ghost_native = jetsam_gkr::spine_statement::spine_inputs_from_body(
            &jetsam_gkr::ghost_tx::ghost_tx_body(),
        );
        let mut b = FieldR1csBuilder::new();
        let raw = SpineInputsTrace::alloc(&mut b, &native);
        let ghost = constant_spine_inputs_trace(&ghost_native);
        // `live` must be a wire. `bind_development_allocation` derives it from
        // the constrained child height; a literal constant would let the
        // builder fold the gate away and the rows under test would not exist.
        let live = alloc_block(&mut b, Block128::ONE);
        let spine = select_spine_inputs_trace(&mut b, &live, &raw, &ghost);
        let _ = bind_development_payout_action(&mut b, &spine, &live);
        let (r1cs, z) = b.build();
        assert!(r1cs.satisfies(&z), "the honest payout witness must satisfy");
        r1cs
    }

    fn constant_column(r1cs: &jetsam_ivc_core::field_r1cs::FieldR1cs) -> Vec<F128> {
        let column = r1cs
            .const_pin
            .expect("the payout relation pins a constant column");
        let mut constants = Vec::new();
        for row in 0..r1cs.b_0.num_rows {
            for (at, coefficient) in r1cs.b_0.row(row) {
                if at as usize == column {
                    constants.push(coefficient);
                }
            }
        }
        constants
    }

    /// The constants this module computes are the constants the gate emits.
    ///
    /// This is the join between the guard and the relation it guards. If the
    /// gate ever stops putting these four values in the constant column — a
    /// different spine, a different select, a third recipient — the guard
    /// would be reading a column that no longer holds them and would pass a
    /// pack it never checked. Measuring the gate is what keeps them in step,
    /// and it needs no matrix pack to do it.
    #[test]
    fn the_payout_gate_freezes_this_build_recipients_in_the_constant_column() {
        let r1cs = payout_relation();
        let constants = constant_column(&r1cs);
        let mut scan = DevelopmentPayoutPinScan::for_this_build();
        for constant in &constants {
            scan.observe(*constant);
        }
        assert_eq!(
            scan.occurrences(),
            &[1usize; DEVELOPMENT_PAYOUT_PIN_COUNT],
            "each payout recipient lane is gated exactly once per block relation",
        );
        scan.finish().expect("this build's own recipients");
    }

    /// The same measurement through the artifact the node actually reads: the
    /// relation is written as a canonical artifact, reopened as the compact
    /// view an embedded pack decodes to, and its constant column walked from
    /// there.
    ///
    /// And the 2026-09-17 defect itself, at the scale of one gate: flipping
    /// the coefficient the network fund's first lane sits on is precisely what
    /// a pack frozen under another profile carries, and the node refuses it by
    /// name.
    #[test]
    fn a_frozen_relation_is_read_through_its_artifact_and_refused_when_it_pays_elsewhere() {
        use jetsam_ivc_core::field_r1cs::CompactFieldR1cs;
        use jetsam_ivc_core::proof::FieldShape;

        let honest = payout_relation();
        let pins = development_payout_owner_pins();

        let compact = |r1cs: &jetsam_ivc_core::field_r1cs::FieldR1cs| {
            let shape = FieldShape::of(r1cs);
            let digest = r1cs.structural_statement_digest();
            let mut bytes = Vec::new();
            r1cs.write_artifact(&mut bytes).expect("write artifact");
            CompactFieldR1cs::open(bytes.into_boxed_slice(), shape, digest)
                .expect("canonical artifact opens")
        };

        verify_relation_development_payout_pins(&compact(&honest))
            .expect("a relation built here pays the recipients this build is compiled with");

        // Repoint the coefficient the network fund's lane 0 rides on. Every
        // other row is untouched, exactly as between two packs that differ
        // only by profile.
        let mut foreign = honest.clone();
        let slot = foreign
            .b_0
            .value_table
            .iter()
            .position(|value| *value == pins[0])
            .expect("the honest relation holds the network fund's first lane");
        foreign.b_0.value_table[slot] += F128::ONE;
        let error = verify_relation_development_payout_pins(&compact(&foreign))
            .expect_err("a relation that no longer pays the network fund must be refused");
        assert_eq!(
            error,
            DevelopmentPayoutRelationError::MissingPin(DevelopmentPayoutPinError {
                label: "network fund lane 0",
                address: jetsam_chain::consensus::NETWORK_FUND_ADDRESS,
            }),
        );
    }

    /// The 2026-09-17 defect, reproduced at the scale of one gate: a relation
    /// frozen for another network's recipients is refused, and the refusal
    /// names the fund it could not find.
    #[test]
    fn a_relation_frozen_for_other_recipients_is_refused_by_name() {
        let r1cs = payout_relation();
        let constants = constant_column(&r1cs);

        let foreign_network = Address([0x11; 32]);
        let foreign_lab = Address([0x22; 32]);
        assert_ne!(
            foreign_network,
            jetsam_chain::consensus::NETWORK_FUND_ADDRESS
        );
        let mut scan = DevelopmentPayoutPinScan::for_recipients(foreign_network, foreign_lab);
        for constant in &constants {
            scan.observe(*constant);
        }
        let error = scan
            .finish()
            .expect_err("a relation that pays other recipients must be refused");
        assert_eq!(error.label, "network fund lane 0");
        assert_eq!(error.address, foreign_network);
        let rendered = error.to_string();
        assert!(
            rendered.contains("network fund lane 0")
                && rendered.contains("different network profile"),
            "the refusal must say what is missing and why: {rendered}",
        );

        // And the half-passing read the ghost term causes: taking the address
        // alone matches the lab fund, whose ghost lanes are zero, and misses
        // the network fund. A guard written that way would pass the very pack
        // that stopped the test chain.
        let naive: Vec<F128> = [
            jetsam_chain::consensus::NETWORK_FUND_ADDRESS,
            jetsam_chain::consensus::LAB_FUND_ADDRESS,
        ]
        .into_iter()
        .flat_map(|address| address.as_fields().into_iter().map(flat_of))
        .collect();
        let matched = naive.iter().filter(|lane| constants.contains(lane)).count();
        assert!(
            matched < naive.len(),
            "the ghost term must not cancel, or this module is checking nothing",
        );
    }
}
