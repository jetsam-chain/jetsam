// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `agent-batch-v1`: a computed batch of catalogue entry 1. **Off-chain
//! prototype** (step 1 of the proof-aggregation design): nothing here is
//! registered, listed in a catalogue or read by the consensus.
//!
//! One client circuit of the imposed form proves `N` statements of catalogue
//! entry 1 ("an AI agent's trace respected its tool policy and its budget")
//! instead of one. `N`, the *capacity*, is fixed in the matrix, so in `D`.
//!
//! **Statements.** Slot `i` runs entry 1's own circuit,
//! [`jetsam_client_agent::enforce_agent_policy`], verbatim, over eight wires
//! holding the statement's entry-1 public IO `io_i`. Policies are per
//! statement: a batcher serves several clients.
//!
//! **Leaves.** `L_i = hash_leaf(io_i)` (fixed-length mode over the eight
//! lanes), computed in circuit. It is the IO commitment a unitary proof of
//! entry 1 carries on chain for the same statement (`client_io_commitment`),
//! so a statement names one leaf whether it was proved alone or in a batch.
//! Slots from `count` on are idle: a suffix, with the zero digest as leaf and
//! no calls.
//!
//! **Root.** `R` is the binary Merkle root of the leaves under the PCS node
//! compression (`merkle::hash_pair`), over `N.next_power_of_two()` leaves.
//! Leaves past `N` are zero constants and fold natively: they depend on `N`
//! only, never on the values.
//!
//! **Public IO** (the form's eight lanes): `R` (2), `count`, the calls summed
//! over the active statements, a period identifier the batcher chooses (free),
//! two reserved lanes pinned to zero, and the tag `JTMBAT01`.
//!
//! A root rather than a list: eight IO lanes cannot carry `N` leaves, and a
//! Merkle path is the receipt the SDK plans for a batch root. A receipt is
//! `(io, statement, index, path)`; [`verify_receipt`] is its native check.

use jetsam_client_agent::{
    enforce_agent_policy, hash_leaf_trace, lanes as agent_lanes, pad_to_form, public_io,
    AgentPolicy, AgentPolicyError, ToolCall, ALLOWED_TOOLS, PUBLIC_IO_LANES,
};
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_circuit::{
    f128_from_u128, f128_to_u128, poseidon2b_permute, FieldR1csBuilder, LinExpr, Wire,
};
use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_ivc_core::merkle::{self, Hash};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::WitnessSlice;
use jetsam_poseidon2b::native::{capacity_iv_flat, DomainTag};

/// The tag in the last public lane: `JTMBAT01`.
pub const BATCH_TAG: u64 = u64::from_le_bytes(*b"JTMBAT01");
/// Width of one statement's call count (at most 32).
pub const CALLS_BITS: usize = 6;
/// Width of the batch's call sum.
pub const CALLS_SUM_BITS: usize = 16;
/// Largest capacity whose call sum fits [`CALLS_SUM_BITS`]: 2047 · 32 < 2^16.
pub const MAX_CAPACITY: usize = 2047;
/// The inner-node domain of `merkle::hash_pair`.
const NODE_TAG: DomainTag = DomainTag::new(b"IVCPCSN_");

/// The public-IO layout, lane by lane.
pub mod lanes {
    pub const ROOT: usize = 0;
    pub const COUNT: usize = 2;
    pub const CALLS_SUM: usize = 3;
    pub const PERIOD: usize = 4;
    pub const RESERVED: usize = 5;
    pub const TAG: usize = 7;
}

/// One statement of catalogue entry 1: a policy and a trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub policy: AgentPolicy,
    pub calls: Vec<ToolCall>,
}

impl Statement {
    /// What an idle slot proves: the empty trace under the empty policy,
    /// which complies.
    pub fn idle() -> Self {
        Self {
            policy: AgentPolicy {
                allowed_tools: [0; ALLOWED_TOOLS],
                budget_cap: 0,
            },
            calls: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchError {
    /// A capacity of zero, or above [`MAX_CAPACITY`].
    BadCapacity { capacity: usize },
    /// More statements than the capacity.
    TooManyStatements { statements: usize, capacity: usize },
    /// Statement `index` cannot be laid out (too many calls).
    Statement {
        index: usize,
        error: AgentPolicyError,
    },
    /// No statement at `index`: the batch holds `count`.
    NoSuchStatement { index: usize, count: usize },
    /// The form is not one this circuit can be laid out in.
    UnsupportedForm,
    /// The circuit does not fit the form's `2^m` rows.
    FormTooSmall { rows: usize, capacity: usize },
}

impl core::fmt::Display for BatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadCapacity { capacity } => {
                write!(f, "batch capacity {capacity} is not in 1..={MAX_CAPACITY}")
            }
            Self::TooManyStatements {
                statements,
                capacity,
            } => write!(f, "{statements} statements, the batch holds {capacity}"),
            Self::Statement { index, error } => write!(f, "statement {index}: {error}"),
            Self::NoSuchStatement { index, count } => {
                write!(f, "no statement {index}: the batch holds {count}")
            }
            Self::UnsupportedForm => f.write_str("agent batch circuit cannot take this form"),
            Self::FormTooSmall { rows, capacity } => {
                write!(
                    f,
                    "agent batch circuit needs {rows} rows, the form has {capacity}"
                )
            }
        }
    }
}

impl std::error::Error for BatchError {}

/// The batch instance: the matrix (its digest would be `D_batch`), a
/// witness, and the public IO the witness carries in the form's IO slice.
pub struct BatchInstance {
    pub r1cs: FieldR1cs,
    pub witness: Vec<F128>,
    pub io: Vec<F128>,
    /// Rows the circuit itself uses (before padding to the form).
    pub circuit_rows: usize,
}

fn check_batch(capacity: usize, statements: &[Statement]) -> Result<(), BatchError> {
    if capacity == 0 || capacity > MAX_CAPACITY {
        return Err(BatchError::BadCapacity { capacity });
    }
    if statements.len() > capacity {
        return Err(BatchError::TooManyStatements {
            statements: statements.len(),
            capacity,
        });
    }
    Ok(())
}

/// The leaf of a statement: `hash_leaf` of its entry-1 public IO, the IO
/// commitment a unitary proof of the same statement carries.
pub fn statement_leaf(statement: &Statement) -> Result<Hash, AgentPolicyError> {
    let io = public_io(&statement.policy, &statement.calls)?;
    Ok(merkle::hash_leaf(&lanes_bytes(&io)))
}

/// The `capacity` leaves of a batch: the statements' leaves in order, then
/// zero digests for the idle slots.
pub fn batch_leaves(capacity: usize, statements: &[Statement]) -> Result<Vec<Hash>, BatchError> {
    check_batch(capacity, statements)?;
    let mut leaves = statements
        .iter()
        .enumerate()
        .map(|(index, statement)| {
            statement_leaf(statement).map_err(|error| BatchError::Statement { index, error })
        })
        .collect::<Result<Vec<_>, _>>()?;
    leaves.resize(capacity, [0u8; 32]);
    Ok(leaves)
}

/// Depth of the tree of a batch of `capacity`: the length of a receipt path.
pub fn tree_depth(capacity: usize) -> usize {
    capacity.next_power_of_two().trailing_zeros() as usize
}

/// Every level of the tree, leaves (padded with zero digests to a power of
/// two) first, the root last.
fn tree_levels(leaves: &[Hash]) -> Vec<Vec<Hash>> {
    let mut level = leaves.to_vec();
    level.resize(leaves.len().next_power_of_two(), [0u8; 32]);
    let mut levels = vec![level];
    while levels.last().map_or(0, Vec::len) > 1 {
        let next = levels
            .last()
            .expect("a level")
            .chunks_exact(2)
            .map(|pair| merkle::hash_pair(&pair[0], &pair[1]))
            .collect();
        levels.push(next);
    }
    levels
}

/// `R`, natively.
pub fn batch_root(capacity: usize, statements: &[Statement]) -> Result<Hash, BatchError> {
    let levels = tree_levels(&batch_leaves(capacity, statements)?);
    Ok(levels.last().expect("a root")[0])
}

/// The receipt path of statement `index` (siblings, leaf level first).
pub fn batch_path(
    capacity: usize,
    statements: &[Statement],
    index: usize,
) -> Result<Vec<Hash>, BatchError> {
    if index >= statements.len() {
        return Err(BatchError::NoSuchStatement {
            index,
            count: statements.len(),
        });
    }
    let levels = tree_levels(&batch_leaves(capacity, statements)?);
    let mut position = index;
    let mut path = Vec::with_capacity(levels.len() - 1);
    for level in &levels[..levels.len() - 1] {
        path.push(level[position ^ 1]);
        position >>= 1;
    }
    Ok(path)
}

/// The eight public lanes of a batch.
pub fn batch_public_io(
    capacity: usize,
    statements: &[Statement],
    period: u64,
) -> Result<[F128; PUBLIC_IO_LANES], BatchError> {
    let root = digest_lanes(&batch_root(capacity, statements)?);
    let calls: usize = statements
        .iter()
        .map(|statement| statement.calls.len())
        .sum();
    Ok([
        root[0],
        root[1],
        lane(statements.len() as u128),
        lane(calls as u128),
        lane(u128::from(period)),
        F128::ZERO,
        F128::ZERO,
        lane(u128::from(BATCH_TAG)),
    ])
}

/// The native check of a receipt: the batch IO `io` (as published by a
/// proof of capacity `capacity`) holds `statement` at `index`, `path`
/// leading from its leaf to `R`. It says nothing about the proof: the reader
/// checks that `io` is the IO of a proof the chain accepted.
pub fn verify_receipt(
    capacity: usize,
    io: &[F128],
    statement: &Statement,
    index: usize,
    path: &[Hash],
) -> bool {
    if io.len() != PUBLIC_IO_LANES
        || io[lanes::TAG] != lane(u128::from(BATCH_TAG))
        || path.len() != tree_depth(capacity)
        || (index as u128) >= f128_to_u128(io[lanes::COUNT])
    {
        return false;
    }
    let Ok(leaf) = statement_leaf(statement) else {
        return false;
    };
    let mut root = [0u8; 32];
    root[..16].copy_from_slice(&f128_to_u128(io[lanes::ROOT]).to_le_bytes());
    root[16..].copy_from_slice(&f128_to_u128(io[lanes::ROOT + 1]).to_le_bytes());
    merkle::verify_merkle_proof(&root, &leaf, index, path)
}

/// Build the batch circuit of capacity `capacity`, laid out in `shape` with
/// its public IO at `io_slice`, and its witness. The matrix depends on the
/// capacity only; the witness satisfies it exactly when every statement
/// complies.
pub fn agent_batch_instance(
    capacity: usize,
    statements: &[Statement],
    period: u64,
    shape: FieldShape,
    io_slice: WitnessSlice,
) -> Result<BatchInstance, BatchError> {
    check_form(shape, io_slice)?;
    let io = batch_public_io(capacity, statements, period)?;
    let mut b = FieldR1csBuilder::new();
    batch_circuit(&mut b, capacity, statements, &io, io_slice)?;
    let circuit_rows = b.num_wires();
    check_rows(circuit_rows, shape)?;
    let (small, witness) = b.build();
    let (r1cs, witness) = pad_to_form(small, witness, shape.k_log);
    Ok(BatchInstance {
        r1cs,
        witness,
        io: io.to_vec(),
        circuit_rows,
    })
}

/// The witness alone, for a batcher that keeps the matrix of its capacity
/// resident (it never changes): the same wires and values as
/// [`agent_batch_instance`], without recording a single matrix row.
/// Returns the witness padded to the form, and the public IO.
pub fn agent_batch_witness(
    capacity: usize,
    statements: &[Statement],
    period: u64,
    shape: FieldShape,
    io_slice: WitnessSlice,
) -> Result<(Vec<F128>, Vec<F128>), BatchError> {
    check_form(shape, io_slice)?;
    let io = batch_public_io(capacity, statements, period)?;
    let mut b = FieldR1csBuilder::new_witness_only();
    batch_circuit(&mut b, capacity, statements, &io, io_slice)?;
    check_rows(b.num_wires(), shape)?;
    let (_, mut witness) = b.build_witness_only();
    witness.resize(1usize << shape.k_log, F128::ZERO);
    Ok((witness, io.to_vec()))
}

fn check_form(shape: FieldShape, io_slice: WitnessSlice) -> Result<(), BatchError> {
    if shape.k_log != shape.m
        || shape.const_pin != Some(0)
        || shape.k_skip != jetsam_ivc_core::zerocheck::K_SKIP
        || io_slice.len() != PUBLIC_IO_LANES
        || io_slice.start() == 0
    {
        return Err(BatchError::UnsupportedForm);
    }
    Ok(())
}

fn check_rows(rows: usize, shape: FieldShape) -> Result<(), BatchError> {
    let form_rows = 1usize << shape.m;
    if rows > form_rows {
        return Err(BatchError::FormTooSmall {
            rows,
            capacity: form_rows,
        });
    }
    Ok(())
}

/// The circuit, appended to a fresh builder: the public IO `io` at
/// `io_slice`, then one slot per statement, the count, the call sum and `R`.
fn batch_circuit(
    b: &mut FieldR1csBuilder,
    capacity: usize,
    statements: &[Statement],
    io: &[F128; PUBLIC_IO_LANES],
    io_slice: WitnessSlice,
) -> Result<(), BatchError> {
    // Wire 0 is the constant one; the public IO sits in the form's slice.
    while b.num_wires() < io_slice.start() {
        b.alloc_f128(F128::ZERO);
    }
    let public = io
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect::<Vec<_>>();

    let idle = Statement::idle();
    let mut actives: Vec<LinExpr> = Vec::with_capacity(capacity);
    let mut leaves: Vec<[LinExpr; 2]> = Vec::with_capacity(capacity);
    let mut calls_sum = vec![LinExpr::zero(); CALLS_SUM_BITS];
    for slot in 0..capacity {
        let statement = statements.get(slot).unwrap_or(&idle);
        let statement_io = public_io(&statement.policy, &statement.calls)
            .map_err(|error| BatchError::Statement { index: slot, error })?;
        let active = LinExpr::from_wire(b.alloc_bool(slot < statements.len()));
        // Active slots are a prefix: a statement only follows a statement.
        if let Some(previous) = actives.last() {
            let gap = LinExpr::from_wire(b.mul(&active, &previous.add_const(F128::ONE)));
            b.pin_f128(&gap, F128::ZERO);
        }
        // Catalogue entry 1, verbatim, over the statement's own IO wires.
        let statement_public = statement_io
            .iter()
            .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
            .collect::<Vec<_>>();
        enforce_agent_policy(b, &statement.policy, &statement.calls, &statement_public)
            .map_err(|error| BatchError::Statement { index: slot, error })?;
        // The leaf: entry 1's IO commitment, the zero digest on an idle slot.
        let commitment = hash_leaf_trace(b, &statement_public);
        leaves.push([
            LinExpr::from_wire(b.mul(&active, &commitment[0])),
            LinExpr::from_wire(b.mul(&active, &commitment[1])),
        ]);
        // Calls: none on an idle slot, summed over the batch.
        let calls = &statement_public[agent_lanes::CALLS];
        let idle_calls = LinExpr::from_wire(b.mul(&active.add_const(F128::ONE), calls));
        b.pin_f128(&idle_calls, F128::ZERO);
        let call_bits = bit_exprs(b.decompose_bits_le(calls, CALLS_BITS));
        calls_sum = add_bits(b, &calls_sum, &call_bits);
        actives.push(active);
    }

    // The number of statements: the one slot where the prefix ends.
    let mut count = LinExpr::zero();
    for (slot, active) in actives.iter().enumerate() {
        let next = actives.get(slot + 1).cloned().unwrap_or_else(LinExpr::zero);
        count = count.add(&active.add(&next).scale(lane(slot as u128 + 1)));
    }
    pin_eq(b, &count, &public[lanes::COUNT]);
    pin_eq(b, &pack_bits(&calls_sum), &public[lanes::CALLS_SUM]);

    // `R`, over the leaves (zero constants past the capacity).
    let root = root_trace(b, leaves);
    pin_eq(b, &root[0], &public[lanes::ROOT]);
    pin_eq(b, &root[1], &public[lanes::ROOT + 1]);
    b.pin_f128(&public[lanes::RESERVED], F128::ZERO);
    b.pin_f128(&public[lanes::RESERVED + 1], F128::ZERO);
    b.pin_f128(&public[lanes::TAG], lane(u128::from(BATCH_TAG)));
    // The period lane is free: the batcher's label, bound by the proof only.
    Ok(())
}

fn lane(value: u128) -> F128 {
    f128_from_u128(value)
}

fn digest_lanes(digest: &Hash) -> [F128; 2] {
    let half = |bytes: &[u8]| {
        let mut buffer = [0u8; 16];
        buffer.copy_from_slice(bytes);
        lane(u128::from_le_bytes(buffer))
    };
    [half(&digest[..16]), half(&digest[16..])]
}

fn lanes_bytes(lanes: &[F128]) -> Vec<u8> {
    lanes
        .iter()
        .flat_map(|lane| f128_to_u128(*lane).to_le_bytes())
        .collect()
}

fn pin_eq(b: &mut FieldR1csBuilder, left: &LinExpr, right: &LinExpr) {
    b.pin_f128(&left.add(right), F128::ZERO);
}

fn bit_exprs(bits: Vec<Wire>) -> Vec<LinExpr> {
    bits.into_iter().map(LinExpr::from_wire).collect()
}

/// `Σ bit_k · 2^k` in the flat encoding of integers.
fn pack_bits(bits: &[LinExpr]) -> LinExpr {
    bits.iter()
        .enumerate()
        .fold(LinExpr::zero(), |sum, (index, bit)| {
            sum.add(&bit.scale(lane(1u128 << index)))
        })
}

/// Ripple-carry `x + y` over `x.len()` bits, `y` zero-extended (a half adder
/// past `y`'s width); the final carry is pinned to zero.
fn add_bits(b: &mut FieldR1csBuilder, x: &[LinExpr], y: &[LinExpr]) -> Vec<LinExpr> {
    let mut carry = LinExpr::zero();
    let mut sum = Vec::with_capacity(x.len());
    for (bit, x_bit) in x.iter().enumerate() {
        match y.get(bit) {
            Some(y_bit) => {
                let half = x_bit.add(y_bit);
                let both = LinExpr::from_wire(b.mul(x_bit, y_bit));
                let propagated = LinExpr::from_wire(b.mul(&carry, &half));
                sum.push(LinExpr::from_wire(b.materialize(&half.add(&carry))));
                carry = both.add(&propagated);
            }
            None => {
                sum.push(LinExpr::from_wire(b.materialize(&x_bit.add(&carry))));
                carry = LinExpr::from_wire(b.mul(x_bit, &carry));
            }
        }
    }
    b.pin_f128(&carry, F128::ZERO);
    sum
}

/// In-circuit `merkle::hash_pair`: one feed-forward permutation over
/// `[l0, l1, r0 ⊕ IV_hi, r1 ⊕ IV_lo]`. Two constant children fold to the
/// native digest (a structural fact: it depends on the capacity only).
fn hash_pair_trace(b: &mut FieldR1csBuilder, l: &[LinExpr; 2], r: &[LinExpr; 2]) -> [LinExpr; 2] {
    if l.iter().chain(r.iter()).all(LinExpr::is_const) {
        let bytes = |digest: &[LinExpr; 2]| {
            let mut out = [0u8; 32];
            out[..16].copy_from_slice(&f128_to_u128(digest[0].constant).to_le_bytes());
            out[16..].copy_from_slice(&f128_to_u128(digest[1].constant).to_le_bytes());
            out
        };
        let folded = digest_lanes(&merkle::hash_pair(&bytes(l), &bytes(r)));
        return [LinExpr::constant(folded[0]), LinExpr::constant(folded[1])];
    }
    let [iv_hi, iv_lo] = capacity_iv_flat(NODE_TAG);
    let state = [
        l[0].clone(),
        l[1].clone(),
        r[0].add_const(lane(iv_hi)),
        r[1].add_const(lane(iv_lo)),
    ];
    let out = poseidon2b_permute(b, state);
    [out[0].add(&l[0]), out[1].add(&l[1])]
}

/// The root of `leaves`, padded with zero constants to a power of two.
fn root_trace(b: &mut FieldR1csBuilder, mut level: Vec<[LinExpr; 2]>) -> [LinExpr; 2] {
    level.resize(
        level.len().next_power_of_two(),
        [LinExpr::zero(), LinExpr::zero()],
    );
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| hash_pair_trace(b, &pair[0], &pair[1]))
            .collect();
    }
    level.pop().expect("a root")
}

#[cfg(test)]
mod tests;
