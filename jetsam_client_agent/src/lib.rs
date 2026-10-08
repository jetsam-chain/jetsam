// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Catalogue entry 1 — an AI agent's trace respected its tool policy and its
//! budget (v1.5 prototype, M2 task 2.6).
//!
//! A client of the HistoryStep client slot is a C1 field proof of the imposed
//! form (the small-class form: m = k_log = 22, rate 1/4, batch 2^5) whose
//! matrix `D` is registered on chain. This crate builds such a matrix and its
//! witness for one deliberately simple, honest statement:
//!
//! > the committed trace `T` holds `n` tool calls; every call names a tool of
//! > the allowed list committed in `P`, and the calls' cumulative cost is at
//! > most the budget cap committed in `P`.
//!
//! **Trace.** Up to [`MAX_CALLS`] records `(tool, cost, receipt_hash)`, padded
//! with zero records. `T = hash_leaf` of the padded records (four lanes per
//! record), computed in circuit. `receipt_hash` is a slot for the provider's
//! signed receipt of the call: it is committed in `T` but **not verified** in
//! the circuit (a receipt is checked off circuit, by whoever relies on the
//! proof; binding the signature itself is left for later).
//!
//! **Policy.** [`ALLOWED_TOOLS`] tool identifiers and a 32-bit budget cap;
//! `P = hash_leaf` of those eight lanes.
//!
//! **Public IO** (eight lanes, the form's IO slice): `T` (2), `P` (2), the
//! cap, the total spent, the number of calls, and a catalogue tag.
//!
//! **Circuit.** Data-independent: every trace, compliant or not, yields the
//! same matrix — the one registered as `D`. A violating trace yields a
//! witness that does not satisfy it, so no valid proof exists. Costs are
//! 32-bit integers, the running total a 40-bit ripple-carry sum (the field
//! has characteristic 2: integer arithmetic is done on bits), and the cap
//! comparison a 40-bit borrow chain whose final borrow must be zero.

use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_circuit::{
    f128_from_u128, f128_to_u128, poseidon2b_permute, FieldR1csBuilder, LinExpr,
};
use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
use jetsam_ivc_core::merkle;
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::WitnessSlice;

/// Tool calls a trace may hold.
pub const MAX_CALLS: usize = 32;
/// Tools a policy may allow (with the cap: eight policy lanes).
pub const ALLOWED_TOOLS: usize = 7;
/// Width of a cost and of the cap.
pub const AMOUNT_BITS: usize = 32;
/// Width of the running total: 32 calls of 32-bit costs fit in 37 bits.
pub const TOTAL_BITS: usize = 40;
/// Public-IO lanes of the statement.
pub const PUBLIC_IO_LANES: usize = 8;
/// Lanes of one committed record: tool, cost, receipt hash (two lanes).
pub const RECORD_LANES: usize = 4;
/// The catalogue tag in the last public lane: `JTMAGT01`.
pub const CATALOG_TAG: u64 = u64::from_le_bytes(*b"JTMAGT01");

/// One tool call of the agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub tool: u32,
    pub cost: u32,
    /// Hash of the provider's signed receipt for this call (not verified in
    /// the circuit; committed in the trace).
    pub receipt_hash: [u8; 32],
}

/// What the agent was allowed: a tool list and a budget cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgentPolicy {
    pub allowed_tools: [u32; ALLOWED_TOOLS],
    pub budget_cap: u32,
}

/// The public-IO layout, lane by lane.
pub mod lanes {
    pub const TRACE_COMMITMENT: usize = 0;
    pub const POLICY_COMMITMENT: usize = 2;
    pub const BUDGET_CAP: usize = 4;
    pub const TOTAL_SPENT: usize = 5;
    pub const CALLS: usize = 6;
    pub const CATALOG_TAG: usize = 7;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentPolicyError {
    /// More calls than [`MAX_CALLS`].
    TooManyCalls { calls: usize },
    /// The form is not one this circuit can be laid out in (`k_log = m`, a
    /// constant pin on column 0, an eight-lane IO slice past column 0).
    UnsupportedForm,
    /// The circuit does not fit the form's `2^m` rows.
    FormTooSmall { rows: usize, capacity: usize },
}

impl core::fmt::Display for AgentPolicyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyCalls { calls } => {
                write!(f, "agent trace has {calls} calls, at most {MAX_CALLS} fit")
            }
            Self::UnsupportedForm => f.write_str("agent policy circuit cannot take this form"),
            Self::FormTooSmall { rows, capacity } => {
                write!(f, "agent policy circuit needs {rows} rows, the form has {capacity}")
            }
        }
    }
}

impl std::error::Error for AgentPolicyError {}

/// The client instance: the matrix (registered as `D`), a witness, and the
/// public IO the witness carries in the form's IO slice.
pub struct AgentPolicyInstance {
    pub r1cs: FieldR1cs,
    pub witness: Vec<F128>,
    pub io: Vec<F128>,
    /// Rows the circuit itself uses (before padding to the form).
    pub circuit_rows: usize,
}

/// The native predicate the circuit enforces.
pub fn complies(policy: &AgentPolicy, calls: &[ToolCall]) -> bool {
    calls.len() <= MAX_CALLS
        && calls
            .iter()
            .all(|call| policy.allowed_tools.contains(&call.tool))
        && calls.iter().map(|call| u64::from(call.cost)).sum::<u64>()
            <= u64::from(policy.budget_cap)
}

/// `P`: `hash_leaf` of the allowed tools and the cap.
pub fn policy_commitment(policy: &AgentPolicy) -> [u8; 32] {
    merkle::hash_leaf(&lanes_bytes(&policy_lanes(policy)))
}

/// `T`: `hash_leaf` of the records, padded with zero records to
/// [`MAX_CALLS`].
pub fn trace_commitment(calls: &[ToolCall]) -> Result<[u8; 32], AgentPolicyError> {
    let lanes = padded_records(calls)?.concat();
    Ok(merkle::hash_leaf(&lanes_bytes(&lanes)))
}

/// The eight public lanes of `(policy, calls)`.
pub fn public_io(
    policy: &AgentPolicy,
    calls: &[ToolCall],
) -> Result<[F128; PUBLIC_IO_LANES], AgentPolicyError> {
    let trace = digest_lanes(&trace_commitment(calls)?);
    let policy_digest = digest_lanes(&policy_commitment(policy));
    let total: u64 = calls.iter().map(|call| u64::from(call.cost)).sum();
    Ok([
        trace[0],
        trace[1],
        policy_digest[0],
        policy_digest[1],
        lane(u128::from(policy.budget_cap)),
        lane(u128::from(total)),
        lane(calls.len() as u128),
        lane(u128::from(CATALOG_TAG)),
    ])
}

/// Build the circuit, laid out in `shape` with its public IO at `io_slice`,
/// and its witness for `(policy, calls)`. The matrix does not depend on the
/// values; the witness satisfies it exactly when the trace complies.
pub fn agent_policy_instance(
    policy: &AgentPolicy,
    calls: &[ToolCall],
    shape: FieldShape,
    io_slice: WitnessSlice,
) -> Result<AgentPolicyInstance, AgentPolicyError> {
    if calls.len() > MAX_CALLS {
        return Err(AgentPolicyError::TooManyCalls { calls: calls.len() });
    }
    if shape.k_log != shape.m
        || shape.const_pin != Some(0)
        || shape.k_skip != jetsam_ivc_core::zerocheck::K_SKIP
        || io_slice.len() != PUBLIC_IO_LANES
        || io_slice.start() == 0
    {
        return Err(AgentPolicyError::UnsupportedForm);
    }
    let io = public_io(policy, calls)?;

    // Wire 0 is the constant one; the public IO sits in the form's slice.
    let mut b = FieldR1csBuilder::new();
    while b.num_wires() < io_slice.start() {
        b.alloc_f128(F128::ZERO);
    }
    let public = io
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect::<Vec<_>>();
    enforce_agent_policy(&mut b, policy, calls, &public)?;

    let circuit_rows = b.num_wires();
    let capacity = 1usize << shape.m;
    if circuit_rows > capacity {
        return Err(AgentPolicyError::FormTooSmall {
            rows: circuit_rows,
            capacity,
        });
    }
    let (small, witness) = b.build();
    let (r1cs, witness) = pad_to_form(small, witness, shape.k_log);
    Ok(AgentPolicyInstance {
        r1cs,
        witness,
        io: io.to_vec(),
        circuit_rows,
    })
}

/// The statement's constraints, appended to `b` over the eight lanes
/// `public` (laid out as [`lanes`]): the whole circuit of catalogue entry 1
/// after its public-IO wires. [`agent_policy_instance`] is this gadget over
/// the form's IO slice; a batch circuit runs it once per statement, over
/// wires of its own. The rows it appends never depend on the values.
pub fn enforce_agent_policy(
    b: &mut FieldR1csBuilder,
    policy: &AgentPolicy,
    calls: &[ToolCall],
    public: &[LinExpr],
) -> Result<(), AgentPolicyError> {
    if public.len() != PUBLIC_IO_LANES {
        return Err(AgentPolicyError::UnsupportedForm);
    }
    let records = padded_records(calls)?;

    // The policy: allowed tools, the cap (a 32-bit integer), and `P`.
    let tools = policy
        .allowed_tools
        .iter()
        .map(|tool| LinExpr::from_wire(b.alloc_f128(lane(u128::from(*tool)))))
        .collect::<Vec<_>>();
    let cap = LinExpr::from_wire(b.alloc_f128(lane(u128::from(policy.budget_cap))));
    let cap_bits = bit_exprs(b.decompose_bits_le(&cap, AMOUNT_BITS));
    let mut committed_policy = tools.clone();
    committed_policy.push(cap.clone());
    let policy_digest = hash_leaf_trace(b, &committed_policy);
    pin_eq(b, &policy_digest[0], &public[lanes::POLICY_COMMITMENT]);
    pin_eq(b, &policy_digest[1], &public[lanes::POLICY_COMMITMENT + 1]);
    pin_eq(b, &cap, &public[lanes::BUDGET_CAP]);

    // The records, one slot per possible call.
    let mut actives: Vec<LinExpr> = Vec::with_capacity(MAX_CALLS);
    let mut committed_records = Vec::with_capacity(MAX_CALLS * RECORD_LANES);
    let mut total_bits = vec![LinExpr::zero(); TOTAL_BITS];
    for (slot, record) in records.iter().enumerate() {
        let active = LinExpr::from_wire(b.alloc_bool(slot < calls.len()));
        let cells = record
            .iter()
            .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
            .collect::<Vec<_>>();
        let (tool, cost) = (&cells[0], &cells[1]);
        let cost_bits = bit_exprs(b.decompose_bits_le(cost, AMOUNT_BITS));
        // An idle slot is a zero record.
        let idle = active.add_const(F128::ONE);
        for cell in &cells {
            let masked = LinExpr::from_wire(b.mul(&idle, cell));
            b.pin_f128(&masked, F128::ZERO);
        }
        // Calls are a prefix of the slots: a call only follows a call.
        if let Some(previous) = actives.last() {
            let gap = LinExpr::from_wire(b.mul(&active, &previous.add_const(F128::ONE)));
            b.pin_f128(&gap, F128::ZERO);
        }
        // The tool is allowed: Π_j (tool − a_j) vanishes on a call.
        let mut product = tool.add(&tools[0]);
        for allowed in &tools[1..] {
            product = LinExpr::from_wire(b.mul(&product, &tool.add(allowed)));
        }
        let violation = LinExpr::from_wire(b.mul(&active, &product));
        b.pin_f128(&violation, F128::ZERO);
        // The running total.
        total_bits = add_bits(b, &total_bits, &cost_bits);
        actives.push(active);
        committed_records.extend(cells);
    }

    // Total spent, and total ≤ cap: the borrow of cap − total is zero.
    pin_eq(b, &pack_bits(&total_bits), &public[lanes::TOTAL_SPENT]);
    let mut borrow = LinExpr::zero();
    for (bit, total) in total_bits.iter().enumerate() {
        let cap_bit = cap_bits.get(bit).cloned().unwrap_or_else(LinExpr::zero);
        let owed = LinExpr::from_wire(b.mul(&cap_bit.add_const(F128::ONE), total));
        let carried = LinExpr::from_wire(b.mul(
            &cap_bit.add(total).add_const(F128::ONE),
            &borrow,
        ));
        borrow = LinExpr::from_wire(b.materialize(&owed.add(&carried)));
    }
    b.pin_f128(&borrow, F128::ZERO);

    // The number of calls: the one slot where the prefix ends.
    let mut count = LinExpr::zero();
    for (slot, active) in actives.iter().enumerate() {
        let next = actives.get(slot + 1).cloned().unwrap_or_else(LinExpr::zero);
        count = count.add(&active.add(&next).scale(lane(slot as u128 + 1)));
    }
    pin_eq(b, &count, &public[lanes::CALLS]);
    b.pin_f128(&public[lanes::CATALOG_TAG], lane(u128::from(CATALOG_TAG)));

    // `T`, over the padded records.
    let trace_digest = hash_leaf_trace(b, &committed_records);
    pin_eq(b, &trace_digest[0], &public[lanes::TRACE_COMMITMENT]);
    pin_eq(b, &trace_digest[1], &public[lanes::TRACE_COMMITMENT + 1]);
    Ok(())
}

fn lane(value: u128) -> F128 {
    f128_from_u128(value)
}

fn digest_lanes(digest: &[u8; 32]) -> [F128; 2] {
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

fn policy_lanes(policy: &AgentPolicy) -> Vec<F128> {
    policy
        .allowed_tools
        .iter()
        .map(|tool| lane(u128::from(*tool)))
        .chain([lane(u128::from(policy.budget_cap))])
        .collect()
}

fn padded_records(calls: &[ToolCall]) -> Result<Vec<[F128; RECORD_LANES]>, AgentPolicyError> {
    if calls.len() > MAX_CALLS {
        return Err(AgentPolicyError::TooManyCalls { calls: calls.len() });
    }
    let mut records = calls
        .iter()
        .map(|call| {
            let receipt = digest_lanes(&call.receipt_hash);
            [
                lane(u128::from(call.tool)),
                lane(u128::from(call.cost)),
                receipt[0],
                receipt[1],
            ]
        })
        .collect::<Vec<_>>();
    records.resize(MAX_CALLS, [F128::ZERO; RECORD_LANES]);
    Ok(records)
}

fn pin_eq(b: &mut FieldR1csBuilder, left: &LinExpr, right: &LinExpr) {
    b.pin_f128(&left.add(right), F128::ZERO);
}

fn bit_exprs(bits: Vec<jetsam_ivc_core::field_circuit::Wire>) -> Vec<LinExpr> {
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

/// Ripple-carry `x + y` over [`TOTAL_BITS`] (`y` zero-extended); the final
/// carry is pinned to zero.
fn add_bits(b: &mut FieldR1csBuilder, x: &[LinExpr], y: &[LinExpr]) -> Vec<LinExpr> {
    let mut carry = LinExpr::zero();
    let mut sum = Vec::with_capacity(TOTAL_BITS);
    for (bit, x_bit) in x.iter().enumerate() {
        let y_bit = y.get(bit).cloned().unwrap_or_else(LinExpr::zero);
        let half = x_bit.add(&y_bit);
        let both = LinExpr::from_wire(b.mul(x_bit, &y_bit));
        let propagated = LinExpr::from_wire(b.mul(&carry, &half));
        sum.push(LinExpr::from_wire(b.materialize(&half.add(&carry))));
        carry = both.add(&propagated);
    }
    b.pin_f128(&carry, F128::ZERO);
    sum
}

/// In-circuit `merkle::hash_leaf` of an even number of lanes (the
/// fixed-length, no-pad mode), with no constant folding: the rows never
/// depend on the values.
pub fn hash_leaf_trace(b: &mut FieldR1csBuilder, lanes: &[LinExpr]) -> [LinExpr; 2] {
    assert!(!lanes.is_empty() && lanes.len() % 2 == 0, "even lane count");
    let [iv_hi, iv_lo] = merkle::leaf_fixed_iv_flat(lanes.len() * 16);
    let mut state = [
        LinExpr::zero(),
        LinExpr::zero(),
        LinExpr::constant(lane(iv_hi)),
        LinExpr::constant(lane(iv_lo)),
    ];
    for pair in lanes.chunks_exact(2) {
        state[0] = state[0].add(&pair[0]);
        state[1] = state[1].add(&pair[1]);
        state = poseidon2b_permute(b, state);
    }
    [state[0].clone(), state[1].clone()]
}

/// Lay a built circuit out in the form's `2^k_log` rows: empty rows, zero
/// witness, `useful_rows` unchanged.
pub fn pad_to_form(small: FieldR1cs, mut witness: Vec<F128>, k_log: usize) -> (FieldR1cs, Vec<F128>) {
    let k = 1usize << k_log;
    let pad = |mut matrix: SparseFieldMatrix| {
        let entries = matrix.col_indices.len();
        matrix.row_offsets.resize(k + 1, entries);
        matrix.num_rows = k;
        matrix.num_cols = k;
        matrix
    };
    witness.resize(k, F128::ZERO);
    let r1cs = FieldR1cs {
        m: k_log,
        k_log,
        k_skip: small.k_skip,
        useful_rows: small.useful_rows,
        a_0: pad(small.a_0),
        b_0: pad(small.b_0),
        const_pin: small.const_pin,
        digest_cache: Default::default(),
        csc_cache: Default::default(),
    };
    r1cs.validate_shape();
    (r1cs, witness)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small form for tests: the circuit is ~30 k rows, so 2^16 fits.
    const TEST_M: usize = 16;

    fn test_form() -> (FieldShape, WitnessSlice) {
        (
            FieldShape {
                m: TEST_M,
                k_log: TEST_M,
                k_skip: jetsam_ivc_core::zerocheck::K_SKIP,
                const_pin: Some(0),
            },
            WitnessSlice {
                log2_len: 3,
                index: 1,
            },
        )
    }

    fn policy() -> AgentPolicy {
        AgentPolicy {
            allowed_tools: [101, 102, 103, 205, 206, 300, 777],
            budget_cap: 50_000,
        }
    }

    fn call(tool: u32, cost: u32, receipt: u8) -> ToolCall {
        ToolCall {
            tool,
            cost,
            receipt_hash: [receipt; 32],
        }
    }

    fn honest_trace() -> Vec<ToolCall> {
        vec![
            call(101, 1_200, 1),
            call(205, 9_000, 2),
            call(777, 30_000, 3),
            call(102, 800, 4),
        ]
    }

    fn instance(calls: &[ToolCall]) -> AgentPolicyInstance {
        let (shape, slice) = test_form();
        agent_policy_instance(&policy(), calls, shape, slice).expect("agent policy instance")
    }

    #[test]
    fn a_compliant_trace_satisfies_the_circuit_and_carries_its_statement() {
        let calls = honest_trace();
        assert!(complies(&policy(), &calls));
        let built = instance(&calls);
        let (shape, slice) = test_form();
        assert_eq!(FieldShape::of(&built.r1cs), shape);
        assert!(built.r1cs.satisfies(&built.witness));
        assert_eq!(
            built.io,
            public_io(&policy(), &calls).unwrap().to_vec(),
            "the witness carries the native statement"
        );
        assert_eq!(
            built.witness[slice.start()..slice.start() + PUBLIC_IO_LANES],
            built.io[..]
        );
        let io = &built.io;
        let digest = |lane: usize| {
            let mut bytes = [0u8; 32];
            bytes[..16].copy_from_slice(&f128_to_u128(io[lane]).to_le_bytes());
            bytes[16..].copy_from_slice(&f128_to_u128(io[lane + 1]).to_le_bytes());
            bytes
        };
        assert_eq!(digest(lanes::TRACE_COMMITMENT), trace_commitment(&calls).unwrap());
        assert_eq!(digest(lanes::POLICY_COMMITMENT), policy_commitment(&policy()));
        assert_eq!(f128_to_u128(io[lanes::BUDGET_CAP]), 50_000);
        assert_eq!(f128_to_u128(io[lanes::TOTAL_SPENT]), 41_000);
        assert_eq!(f128_to_u128(io[lanes::CALLS]), 4);
        assert_eq!(f128_to_u128(io[lanes::CATALOG_TAG]), CATALOG_TAG as u128);
        assert!(built.circuit_rows < 1 << TEST_M);
    }

    #[test]
    fn edge_traces_comply() {
        // No call at all, a full trace, and a total exactly at the cap.
        let full = (0..MAX_CALLS)
            .map(|index| call(policy().allowed_tools[index % ALLOWED_TOOLS], 1_000, index as u8))
            .collect::<Vec<_>>();
        let at_cap = vec![call(300, 49_999, 9), call(300, 1, 10)];
        for calls in [Vec::new(), full, at_cap] {
            assert!(complies(&policy(), &calls));
            let built = instance(&calls);
            assert!(built.r1cs.satisfies(&built.witness), "{} calls", calls.len());
        }
    }

    #[test]
    fn a_violating_trace_has_no_satisfying_witness() {
        let mut forbidden = honest_trace();
        forbidden[2].tool = 778;
        let mut over_budget = honest_trace();
        over_budget[1].cost = 18_001; // total 50 001 > 50 000
        let mut forbidden_idle_tool = honest_trace();
        forbidden_idle_tool.push(call(0, 0, 0)); // tool 0 is not allowed
        for (name, calls) in [
            ("forbidden tool", forbidden),
            ("budget exceeded by one", over_budget),
            ("tool zero", forbidden_idle_tool),
        ] {
            assert!(!complies(&policy(), &calls), "{name}");
            let built = instance(&calls);
            assert!(!built.r1cs.satisfies(&built.witness), "{name} satisfied the circuit");
        }
    }

    /// `D` is one matrix: it does not move with the trace, compliant or not.
    #[test]
    fn the_matrix_does_not_depend_on_the_trace() {
        let mut forbidden = honest_trace();
        forbidden[0].tool = 5;
        let digests = [honest_trace(), Vec::new(), forbidden, vec![call(103, u32::MAX, 0)]]
            .iter()
            .map(|calls| instance(calls).r1cs.structural_statement_digest())
            .collect::<Vec<_>>();
        assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn malformed_inputs_are_refused_not_panicked() {
        let (shape, slice) = test_form();
        let too_many = vec![call(101, 1, 0); MAX_CALLS + 1];
        assert_eq!(
            agent_policy_instance(&policy(), &too_many, shape, slice).err(),
            Some(AgentPolicyError::TooManyCalls { calls: MAX_CALLS + 1 })
        );
        let small = FieldShape {
            m: 10,
            k_log: 10,
            ..shape
        };
        assert!(matches!(
            agent_policy_instance(&policy(), &honest_trace(), small, slice),
            Err(AgentPolicyError::FormTooSmall { .. })
        ));
        let skewed = FieldShape {
            k_log: TEST_M - 1,
            ..shape
        };
        assert_eq!(
            agent_policy_instance(&policy(), &honest_trace(), skewed, slice).err(),
            Some(AgentPolicyError::UnsupportedForm)
        );
        let wide = WitnessSlice {
            log2_len: 4,
            index: 1,
        };
        assert_eq!(
            agent_policy_instance(&policy(), &honest_trace(), shape, wide).err(),
            Some(AgentPolicyError::UnsupportedForm)
        );
    }
}
