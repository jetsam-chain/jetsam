// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Form-Z twin of the deferred C1 verifier (v1.6): the client arm of
//! a batch step verifies a hiding client proof (`jetsam_ivc_core::zk_form`).
//!
//! Line-for-line replay of
//! `jetsam_ivc_core::verifier::verify_field_c1_zk_deferred_matrix_with_post_commit_context`
//! on a lane channel, every native refusal a pin:
//!
//! - the statement binds the layout, then the physical matrix digest `D`
//!   (wires), the PCS parameters and the commitment;
//! - the masked zerocheck absorbs `κ` before `λ`, assumes `P' = λκ` on the
//!   skip set and derives the mask's final value `(running + a·b)/λ`;
//! - the masked lincheck draws `λ'` after `α` (and `β`), absorbs `M(r)` after
//!   `z_partial` and adds `λ'·M(r)` to the fresh claim;
//! - the tautology rows are removed from the fresh claim in closed form
//!   (Lagrange basis of the skip): the returned claim concerns the physical
//!   catalogue matrix only, the one the fold and the decision evaluate;
//! - the two masks' final values are sparse linear claims of the batched
//!   opening: they enter the batching target, and `final_b` carries their
//!   weights on the reserved cells.
//!
//! Every weight is a function of public challenges and of the layout; no
//! prover value enters a weight.

use std::collections::BTreeMap;

use jetsam_ivc_core::field::{F128, F256, PHI_8_TABLE};
use jetsam_ivc_core::field_circuit::{ExtExpr, FieldR1csBuilder, FsChannelOps, LinExpr};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::{C1ZkFieldR1csProof, FieldShape};
use jetsam_ivc_core::public_io::{PublicIoSpec, POST_COMMIT_CLASS_BINDING_LABEL};
use jetsam_ivc_core::zerocheck::K_SKIP;
use jetsam_ivc_core::zk_form::{
    ZkFormLayout, F256_X, ZK_COMPANION_LANES, ZK_FORM_K_SKIP, ZK_SKIP_MASK_COEFFICIENTS,
    ZK_TYPE_B_CELLS,
};

use super::matrix_fold::C1FreshLincheckClaimTrace;
use super::self_verify::{
    alloc_f256, basefold_verify_c1_trace, bind_public_io_c1_trace,
    bind_statement_field_parts_c1_trace, dot_ext_trace, flatten_ext_lanes,
    interpolate_combined_ext_trace, interpolate_lambda_ext_trace, inverse_ext_trace,
    lagrange_weights_window_ext_trace, observe_flat_digest, pin_transcript_constant_coordinates,
    C1BaseFoldProofTrace, C1FieldR1csProofTrace, C1LincheckClaimTrace, C1LincheckProofTrace,
    C1QuirkyDirectClaimTrace, C1QuirkyPointTrace, C1R1csClaimTrace, C1ZClaimTrace,
    C1ZerocheckClaimTrace, C1ZerocheckProofTrace, FieldPostCommitTraceContext, FlatDigestExpr,
    PcsWalkObligations,
};
use super::{eq_ind_partial_eval_ext_trace, mul_ext, pin_eq_ext};

/// A form-Z client proof as wires: the v1.5 body, `κ` and `M(r)`.
pub struct C1ZkFieldR1csProofTrace {
    pub base: C1FieldR1csProofTrace,
    pub kappa: ExtExpr,
    pub mask_value: ExtExpr,
}

impl C1ZkFieldR1csProofTrace {
    /// `base` is the v1.5 body already allocated (shape mode); `κ` and
    /// `M(r)` follow it.
    pub fn alloc(b: &mut FieldR1csBuilder, native: &C1ZkFieldR1csProof, base: C1FieldR1csProofTrace) -> Self {
        Self {
            base,
            kappa: alloc_f256(b, native.zerocheck_kappa),
            mask_value: alloc_f256(b, native.lincheck_mask_value),
        }
    }
}

/// Twin of `jetsam_ivc_core::proof::bind_statement_zk_form_c1`.
pub fn bind_statement_zk_form_c1_trace(
    b: &mut FieldR1csBuilder,
    channel: &mut impl FsChannelOps,
    layout: &ZkFormLayout,
    statement_digest: &FlatDigestExpr,
    pcs_params: &PcsParams,
    root: &FlatDigestExpr,
) {
    channel.observe_label(b, b"history-zk-client-form-v1");
    let mut bytes = Vec::with_capacity(24);
    for value in [layout.m, layout.coin_coefficients, layout.reserved_coefficients] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    channel.observe_bytes_const(b, &bytes);
    bind_statement_field_parts_c1_trace(b, channel, statement_digest, pcs_params, root);
}

/// The masked zerocheck's claim and the derived final value of its mask.
pub struct C1ZkZerocheckClaimTrace {
    pub claim: C1ZerocheckClaimTrace,
    pub lambda: ExtExpr,
    pub mask_value: ExtExpr,
}

/// Twin of `jetsam_ivc_core::zerocheck::field_c1::verify_zk`.
pub fn zerocheck_field_verify_c1_zk_trace(
    b: &mut FieldR1csBuilder,
    channel: &mut impl FsChannelOps,
    log_n: usize,
    proof: &C1ZerocheckProofTrace,
    kappa: &ExtExpr,
) -> C1ZkZerocheckClaimTrace {
    let multilinear_rounds = log_n - K_SKIP;
    assert_eq!(proof.round1_ab.len(), 1 << K_SKIP);
    assert_eq!(proof.round1_c.len(), 1 << K_SKIP);
    assert_eq!(proof.multilinear_rounds.len(), multilinear_rounds);
    channel.observe_label(b, b"history-field-zerocheck-c1");
    let r_rest = channel.sample_f256_vec(b, multilinear_rounds);
    channel.observe_label(b, b"history-field-zerocheck-c1-zk");
    channel.observe_f256(b, kappa);
    let lambda = channel.sample_f256(b);
    channel.observe_f256_slice(b, &proof.round1_ab);
    channel.observe_f256_slice(b, &proof.round1_c);
    let z = channel.sample_f256(b);

    let computed_c = interpolate_lambda_ext_trace(b, &proof.round1_c, K_SKIP, &z);
    pin_eq_ext(b, &computed_c, &proof.final_c_eval);
    // R' - lambda kappa vanishes on S.
    let shift = mul_ext(b, &lambda, kappa);
    let combined = proof
        .round1_ab
        .iter()
        .zip(&proof.round1_c)
        .map(|(ab, c)| ab.add(c).add(&shift))
        .collect::<Vec<_>>();
    let mut running = interpolate_combined_ext_trace(b, &combined, K_SKIP, &z)
        .add(&shift)
        .add(&computed_c);

    let mut challenges = Vec::with_capacity(multilinear_rounds);
    for (round, (message_one, message_infinity)) in proof.multilinear_rounds.iter().enumerate() {
        let eq_challenge = &r_rest[round];
        let one_plus_eq = eq_challenge.add_const(F256::ONE);
        let inverse = inverse_ext_trace(b, &one_plus_eq);
        let eq_times_one = mul_ext(b, eq_challenge, message_one);
        let value_zero = mul_ext(b, &running.add(&eq_times_one), &inverse);

        channel.observe_f256(b, message_one);
        channel.observe_f256(b, message_infinity);
        let challenge = channel.sample_f256(b);
        challenges.push(challenge.clone());

        let one_plus_challenge = challenge.add_const(F256::ONE);
        let term_zero = mul_ext(b, &value_zero, &one_plus_challenge);
        let term_one = mul_ext(b, message_one, &challenge);
        let term_infinity = mul_ext(b, message_infinity, &challenge);
        let term_infinity = mul_ext(b, &term_infinity, &one_plus_challenge);
        running = term_zero.add(&term_one).add(&term_infinity);
    }

    channel.observe_f256(b, &proof.final_a_eval);
    channel.observe_f256(b, &proof.final_b_eval);
    // Native refuses lambda = 0; here the inverse's pin is unsatisfiable.
    let lambda_inverse = inverse_ext_trace(b, &lambda);
    let product = mul_ext(b, &proof.final_a_eval, &proof.final_b_eval);
    let mask_value = mul_ext(b, &running.add(&product), &lambda_inverse);
    C1ZkZerocheckClaimTrace {
        claim: C1ZerocheckClaimTrace {
            z,
            mlv_challenges: challenges,
            r_rest,
            a_eval: proof.final_a_eval.clone(),
            b_eval: proof.final_b_eval.clone(),
            c_eval: proof.final_c_eval.clone(),
        },
        lambda,
        mask_value,
    }
}

/// Twin of `jetsam_ivc_core::lincheck::c1::verify_deferred_zk` (square form:
/// no outer variables). `mask_value` is the proof's `M(r)`.
#[allow(clippy::too_many_arguments)]
pub fn lincheck_verify_c1_zk_trace_deferred(
    b: &mut FieldR1csBuilder,
    channel: &mut impl FsChannelOps,
    k_log: usize,
    k_skip: usize,
    const_pin: Option<usize>,
    point: &C1QuirkyPointTrace,
    a_value: &ExtExpr,
    b_value: &ExtExpr,
    proof: &C1LincheckProofTrace,
    mask_value: &ExtExpr,
) -> (C1LincheckClaimTrace, C1FreshLincheckClaimTrace) {
    let count = 1usize << k_skip;
    let inner_rest_len = k_log - k_skip;
    assert_eq!(point.x_inner_rest.len(), inner_rest_len);
    assert!(point.x_outer.is_empty(), "form Z is square");
    assert_eq!(proof.rounds.len(), inner_rest_len);
    assert_eq!(proof.z_partial.len(), count);

    channel.observe_label(b, b"history-lincheck-c1");
    let alpha = channel.sample_f256(b);
    let mut running = mul_ext(b, &alpha, a_value).add(b_value);
    let beta = if const_pin.is_some() {
        let beta = channel.sample_f256(b);
        running = running.add(&beta);
        Some(beta)
    } else {
        None
    };
    channel.observe_label(b, b"history-lincheck-c1-zk");
    let lambda = channel.sample_f256(b);
    // The mask sums to zero: no lambda kappa' term.
    let mut sampled = Vec::with_capacity(inner_rest_len);
    for (at_one, at_infinity) in &proof.rounds {
        channel.observe_f256(b, at_one);
        channel.observe_f256(b, at_infinity);
        let challenge = channel.sample_f256(b);
        let at_zero = running.add(at_one);
        let linear = at_zero.add(at_one).add(at_infinity);
        let square = mul_ext(b, &challenge, &challenge);
        running = mul_ext(b, at_infinity, &square)
            .add(&mul_ext(b, &linear, &challenge))
            .add(&at_zero);
        sampled.push(challenge);
    }
    channel.observe_f256_slice(b, &proof.z_partial);
    let r_inner_rest = sampled.iter().rev().cloned().collect::<Vec<_>>();
    channel.observe_f256(b, mask_value);

    let mut fresh_value = running.add(&mul_ext(b, &lambda, mask_value));
    if let (Some(beta), Some(column)) = (beta, const_pin) {
        let mut pin_weight = ExtExpr::one();
        for (index, challenge) in r_inner_rest.iter().enumerate() {
            let factor = if (column >> k_skip >> index) & 1 == 1 {
                challenge.clone()
            } else {
                challenge.add_const(F256::ONE)
            };
            pin_weight = mul_ext(b, &pin_weight, &factor);
        }
        let term = mul_ext(b, &beta, &proof.z_partial[column & (count - 1)]);
        fresh_value = fresh_value.add(&mul_ext(b, &term, &pin_weight));
    }
    let fresh = C1FreshLincheckClaimTrace {
        alpha,
        z_skip: point.z_skip.clone(),
        x_inner_rest: point.x_inner_rest.clone(),
        r_inner_rest: r_inner_rest.clone(),
        z_partial: proof.z_partial.clone(),
        value: fresh_value,
    };
    let r_inner_skip = channel.sample_f256(b);
    let weights = lagrange_weights_window_ext_trace(b, &r_inner_skip, 0, count, 0);
    let w = dot_ext_trace(b, &weights, &proof.z_partial);
    (
        C1LincheckClaimTrace {
            r_inner_skip,
            r_inner_rest,
            w,
        },
        fresh,
    )
}

fn product_trace(b: &mut FieldR1csBuilder, factors: impl IntoIterator<Item = ExtExpr>) -> ExtExpr {
    factors
        .into_iter()
        .fold(ExtExpr::one(), |product, factor| mul_ext(b, &product, &factor))
}

/// `eq(point, x)` for a constant integer x (bit i against point[i]).
fn eq_at_trace(b: &mut FieldR1csBuilder, point: &[ExtExpr], x: usize) -> ExtExpr {
    let factors = point
        .iter()
        .enumerate()
        .map(|(i, c)| if (x >> i) & 1 == 1 { c.clone() } else { c.add_const(F256::ONE) })
        .collect::<Vec<_>>();
    product_trace(b, factors)
}

/// Twin of `ZkFormLayout::tautology_fresh_value`: the tautology part of a
/// fresh lincheck claim, `Σ_s T[s]·z_partial[s]`, in closed form.
pub fn tautology_fresh_value_trace(
    b: &mut FieldR1csBuilder,
    layout: &ZkFormLayout,
    alpha: &ExtExpr,
    z_skip: &ExtExpr,
    rho: &[ExtExpr],
    r_inner: &[ExtExpr],
    z_partial: &[ExtExpr],
) -> ExtExpr {
    let rounds = layout.rounds();
    assert_eq!(rho.len(), rounds);
    assert_eq!(r_inner.len(), rounds);
    assert_eq!(z_partial.len(), 1 << ZK_FORM_K_SKIP);
    let skip = lagrange_weights_window_ext_trace(b, z_skip, 0, 1 << ZK_FORM_K_SKIP, 0);
    let c = layout.coin_x_end().trailing_zeros() as usize;
    let eq1 = |x: &ExtExpr, y: &ExtExpr| x.add(y).add_const(F256::ONE);
    let mut both_zero = Vec::with_capacity(rounds);
    for j in 0..rounds {
        both_zero.push(mul_ext(b, &rho[j].add_const(F256::ONE), &r_inner[j].add_const(F256::ONE)));
    }
    let p_all = product_trace(b, (0..rounds).map(|j| eq1(&rho[j], &r_inner[j])));
    let p_coin = product_trace(
        b,
        (0..rounds).map(|j| if j < c { eq1(&rho[j], &r_inner[j]) } else { both_zero[j].clone() }),
    );
    let e0 = product_trace(b, both_zero.iter().cloned());
    let rho_coin = product_trace(b, (c..rounds).map(|j| rho[j].add_const(F256::ONE)));
    let rho_zero = product_trace(b, rho.iter().map(|v| v.add_const(F256::ONE)));
    let r_zero = product_trace(b, r_inner.iter().map(|v| v.add_const(F256::ONE)));

    let public = |s: usize| s == 0 || (8..16).contains(&s);
    let alpha_coin = mul_ext(b, alpha, &p_coin);
    let alpha_public = mul_ext(b, alpha, &p_coin.add(&e0));
    let alpha_all = mul_ext(b, alpha, &p_all);
    let mut out = Vec::with_capacity(64);
    // sum_a = rho_coin·Σ_private + (rho_coin + rho_zero)·Σ_public + Σ_companion
    let mut skip_private = ExtExpr::zero();
    let mut skip_public = ExtExpr::zero();
    let mut skip_companion = ExtExpr::zero();
    for (s, weight) in skip.iter().enumerate() {
        let lane = s & 31;
        if lane < ZK_COMPANION_LANES.start {
            if public(s) {
                out.push(mul_ext(b, weight, &alpha_public));
                skip_public = skip_public.add(weight);
            } else {
                out.push(mul_ext(b, weight, &alpha_coin));
                skip_private = skip_private.add(weight);
            }
        } else {
            out.push(mul_ext(b, weight, &alpha_all));
            skip_companion = skip_companion.add(weight);
        }
    }
    let mut sum_a = mul_ext(b, &rho_coin, &skip_private)
        .add(&mul_ext(b, &rho_coin.add(&rho_zero), &skip_public))
        .add(&skip_companion);
    let mut sum_b = ExtExpr::zero();
    // Type-B cells: (s, x) = (cell & 63, cell >> 6); x repeats, eq memoised.
    let mut eq_rho = BTreeMap::new();
    let mut eq_r = BTreeMap::new();
    let alpha_plus_one = alpha.add_const(F256::ONE);
    for j in 0..ZK_TYPE_B_CELLS {
        let cell = layout.type_b_cell(j);
        let (s, x) = (cell & 63, cell >> 6);
        if !eq_rho.contains_key(&x) {
            let value = eq_at_trace(b, rho, x);
            eq_rho.insert(x, value);
            let value = eq_at_trace(b, r_inner, x);
            eq_r.insert(x, value);
        }
        let rw = mul_ext(b, &skip[s], &eq_rho[&x]);
        let term = mul_ext(b, &alpha_plus_one, &rw);
        out[s] = out[s].add(&mul_ext(b, &term, &eq_r[&x]));
        sum_a = sum_a.add(&rw);
        sum_b = sum_b.add(&rw);
    }
    let inner = sum_a.add(&mul_ext(b, alpha, &sum_b));
    out[0] = out[0].add(&mul_ext(b, &r_zero, &inner));
    dot_ext_trace(b, &out, z_partial)
}

/// A sparse linear claim on constant cells: weights are wires (functions of
/// public challenges), the value is the proof's or the verifier's.
pub struct ZkSparseClaimTrace {
    pub cells: Vec<usize>,
    pub weights: Vec<ExtExpr>,
    pub value: ExtExpr,
}

/// Two cells per F256 coefficient: `(lo, w)`, `(hi, w·X)` (`ZkFormLayout::sparse`).
fn sparse_claim_trace(
    coefficient_weights: Vec<ExtExpr>,
    cell: impl Fn(usize, usize) -> usize,
    value: ExtExpr,
) -> ZkSparseClaimTrace {
    let mut cells = Vec::with_capacity(2 * coefficient_weights.len());
    let mut weights = Vec::with_capacity(2 * coefficient_weights.len());
    for (j, weight) in coefficient_weights.into_iter().enumerate() {
        cells.push(cell(j, 0));
        weights.push(weight.clone());
        cells.push(cell(j, 1));
        weights.push(weight.scale_ext(F256_X));
    }
    ZkSparseClaimTrace { cells, weights, value }
}

/// Twin of `ZkFormLayout::zerocheck_mask_claim`.
pub fn zerocheck_mask_claim_trace(
    b: &mut FieldR1csBuilder,
    layout: &ZkFormLayout,
    z_skip: &ExtExpr,
    rho: &[ExtExpr],
    value: ExtExpr,
) -> ZkSparseClaimTrace {
    assert_eq!(rho.len(), layout.rounds());
    let zs = product_trace(
        b,
        PHI_8_TABLE[..1 << ZK_FORM_K_SKIP]
            .iter()
            .map(|&node| z_skip.add_const(F256::from_base(node))),
    );
    let mut weights = Vec::with_capacity(layout.zerocheck_mask_coefficients());
    let mut power = ExtExpr::one();
    for index in 0..ZK_SKIP_MASK_COEFFICIENTS {
        weights.push(mul_ext(b, &zs, &power));
        if index + 1 < ZK_SKIP_MASK_COEFFICIENTS {
            power = mul_ext(b, &power, z_skip);
        }
    }
    for r in rho {
        let square = mul_ext(b, r, r);
        weights.extend([ExtExpr::one(), r.clone(), square]);
    }
    sparse_claim_trace(weights, |j, half| layout.zerocheck_mask_cell(j, half), value)
}

/// Twin of `ZkFormLayout::lincheck_mask_claim` (`ZkLincheckMask::weights_at`
/// at the layout's public point `u`).
pub fn lincheck_mask_claim_trace(
    b: &mut FieldR1csBuilder,
    layout: &ZkFormLayout,
    r_inner: &[ExtExpr],
    value: ExtExpr,
) -> ZkSparseClaimTrace {
    let rounds = layout.rounds();
    assert_eq!(r_inner.len(), rounds);
    let u = layout.lincheck_mask_point();
    let eqs = r_inner
        .iter()
        .zip(&u)
        .map(|(p, &u)| p.add_const(F256::ONE + u))
        .collect::<Vec<_>>();
    // others_t = prod_{k != t} eqs_k, from prefix and suffix products.
    let mut prefix = vec![ExtExpr::one()];
    for factor in &eqs[..rounds - 1] {
        let last = prefix.last().expect("prefix").clone();
        prefix.push(mul_ext(b, &last, factor));
    }
    let mut suffix = vec![ExtExpr::one(); rounds + 1];
    for t in (1..rounds).rev() {
        suffix[t] = mul_ext(b, &suffix[t + 1].clone(), &eqs[t]);
    }
    let mut weights = Vec::with_capacity(3 * rounds);
    for t in 0..rounds {
        let others = mul_ext(b, &prefix[t], &suffix[t + 1]);
        let linear = mul_ext(b, &others, &r_inner[t]);
        let square = mul_ext(b, &linear, &r_inner[t]);
        weights.extend([others, linear, square]);
    }
    sparse_claim_trace(weights, |j, half| layout.lincheck_mask_cell(j, half), value)
}

/// `Σ_j w_j·eq(challenges, cell_j)` over constant cells: eq factors through
/// the lane (5 low bits), the low 8 coefficient bits and the remaining high
/// bits, each part memoised (the reserved cells share a few coefficients and
/// one high pattern).
fn sparse_final_b_trace(
    b: &mut FieldR1csBuilder,
    challenges: &[ExtExpr],
    claim: &ZkSparseClaimTrace,
) -> ExtExpr {
    const LANE_BITS: usize = 5;
    const LOW_BITS: usize = 8;
    let bits = challenges.len();
    assert!(bits > LANE_BITS);
    let low_end = bits.min(LANE_BITS + LOW_BITS);
    let part = |b: &mut FieldR1csBuilder, from: usize, to: usize, value: usize| {
        eq_at_trace(b, &challenges[from..to], value)
    };
    let mut lanes = BTreeMap::new();
    let mut lows = BTreeMap::new();
    let mut highs = BTreeMap::new();
    // coefficient -> Σ w·E_lane over its cells
    let mut by_coefficient: BTreeMap<usize, ExtExpr> = BTreeMap::new();
    for (&cell, weight) in claim.cells.iter().zip(&claim.weights) {
        assert!(cell >> bits == 0, "sparse cell outside the committed vector");
        let lane = cell & ((1 << LANE_BITS) - 1);
        if !lanes.contains_key(&lane) {
            let value = part(b, 0, LANE_BITS, lane);
            lanes.insert(lane, value);
        }
        let term = mul_ext(b, weight, &lanes[&lane]);
        let entry = by_coefficient.entry(cell >> LANE_BITS).or_insert_with(ExtExpr::zero);
        *entry = entry.add(&term);
    }
    let mut total = ExtExpr::zero();
    for (coefficient, inner) in by_coefficient {
        let low = coefficient & ((1 << (low_end - LANE_BITS)) - 1);
        let high = coefficient >> (low_end - LANE_BITS);
        if !lows.contains_key(&low) {
            let value = part(b, LANE_BITS, low_end, low);
            lows.insert(low, value);
        }
        if !highs.contains_key(&high) {
            let value = part(b, low_end, bits, high);
            highs.insert(high, value);
        }
        let eq = mul_ext(b, &lows[&low], &highs[&high]);
        total = total.add(&mul_ext(b, &eq, &inner));
    }
    total
}

/// Twin of `pcs::verify_opening_batch_quirky_direct_c1_with_sparse`
/// (region mode: PCS hashing left as obligations).
#[allow(clippy::too_many_arguments)]
pub fn verify_opening_batch_quirky_direct_c1_with_sparse_trace_region(
    b: &mut FieldR1csBuilder,
    channel: &mut impl FsChannelOps,
    commitment_root: &FlatDigestExpr,
    claims: &[C1QuirkyDirectClaimTrace],
    sparse: &[ZkSparseClaimTrace],
    proof: &C1BaseFoldProofTrace,
    params: &PcsParams,
    region: Option<&mut PcsWalkObligations>,
) -> Vec<std::ops::Range<usize>> {
    assert!(!claims.is_empty(), "need at least one claim");
    let log_length = params.m - LOG_PACKING;
    for claim in claims {
        assert_eq!(claim.x_rest.len() + claim.k_skip, log_length);
    }
    for claim in sparse {
        assert_eq!(claim.cells.len(), claim.weights.len());
        assert!(claim.cells.iter().all(|&cell| cell >> log_length == 0));
    }

    channel.observe_label(b, b"history-pcs-open-field-c1");
    for claim in claims {
        channel.observe_label(b, b"history-pcs-quirky-direct-c1");
        channel.observe_f256(b, &claim.z_skip);
        channel.observe_f256(
            b,
            &ExtExpr::constant(F256::from_base(F128::new(claim.k_skip as u64, 0))),
        );
        let flattened = flatten_ext_lanes(&claim.x_rest);
        let pinned = pin_transcript_constant_coordinates(b, &flattened);
        let transcript_rest = pinned
            .chunks_exact(2)
            .map(|coordinates| ExtExpr::new(coordinates[0].clone(), coordinates[1].clone()))
            .collect::<Vec<_>>();
        channel.observe_f256_slice(b, &transcript_rest);
        channel.observe_f256(b, &claim.value);
    }
    channel.observe_label(b, b"history-pcs-sparse-c1");
    for claim in sparse {
        channel.observe_f256(
            b,
            &ExtExpr::constant(F256::from_base(F128::new(claim.cells.len() as u64, 0))),
        );
        channel.observe_f256(b, &claim.value);
    }
    let gammas = (0..claims.len() + sparse.len())
        .map(|_| channel.sample_f256(b))
        .collect::<Vec<_>>();
    let mut target = ExtExpr::zero();
    for (claim, gamma) in claims.iter().zip(&gammas) {
        target = target.add(&mul_ext(b, gamma, &claim.value));
    }
    for (claim, gamma) in sparse.iter().zip(&gammas[claims.len()..]) {
        target = target.add(&mul_ext(b, gamma, &claim.value));
    }

    let (challenges, query_bit_ranges) =
        basefold_verify_c1_trace(b, channel, &target, proof, commitment_root, params, region);
    assert_eq!(challenges.len(), log_length);

    let mut expected_final_b = ExtExpr::zero();
    for (claim, gamma) in claims.iter().zip(&gammas) {
        let count = 1usize << claim.k_skip;
        let weights = lagrange_weights_window_ext_trace(b, &claim.z_skip, 0, count, 0);
        let eq_skip = eq_ind_partial_eval_ext_trace(b, &challenges[..claim.k_skip]);
        let skip_factor = dot_ext_trace(b, &weights, &eq_skip);
        let eq_rest = claim
            .x_rest
            .iter()
            .zip(&challenges[claim.k_skip..])
            .fold(ExtExpr::one(), |product, (left, right)| {
                mul_ext(b, &product, &left.add(right).add_const(F256::ONE))
            });
        let term = mul_ext(b, gamma, &skip_factor);
        expected_final_b = expected_final_b.add(&mul_ext(b, &term, &eq_rest));
    }
    for (claim, gamma) in sparse.iter().zip(&gammas[claims.len()..]) {
        let sum = sparse_final_b_trace(b, &challenges, claim);
        expected_final_b = expected_final_b.add(&mul_ext(b, gamma, &sum));
    }
    pin_eq_ext(b, &expected_final_b, &proof.final_b);
    query_bit_ranges
}

/// Twin of `verify_field_c1_zk_deferred_matrix_with_post_commit_context`:
/// the deferred form-Z verifier in region mode, `D` read from wires. The
/// fresh claim returned has the tautologies removed: it concerns the
/// physical (catalogue) matrix.
#[allow(clippy::too_many_arguments)]
pub fn verify_field_c1_zk_trace_deferred_region_with_post_commit_context_expr<C, PostCommit>(
    b: &mut FieldR1csBuilder,
    channel: &mut C,
    layout: &ZkFormLayout,
    shape: &FieldShape,
    pcs_params: &PcsParams,
    statement_digest: &FlatDigestExpr,
    commitment_root: &FlatDigestExpr,
    proof: &C1ZkFieldR1csProofTrace,
    spec: &PublicIoSpec,
    io: &[LinExpr],
    post_commit_class_digest: &FlatDigestExpr,
    region: Option<&mut PcsWalkObligations>,
    post_commit: PostCommit,
) -> (C1R1csClaimTrace, C1FreshLincheckClaimTrace)
where
    C: FsChannelOps,
    PostCommit: FnOnce(&mut FieldR1csBuilder, &mut FieldPostCommitTraceContext<'_, C>),
{
    // Native: ParamsMismatch. Here a class constant: the arm is built for
    // this layout, so a mismatch is a construction bug.
    assert!(layout.validate().is_ok(), "form-Z layout");
    assert_eq!(shape.m, layout.m);
    assert_eq!(shape.k_log, layout.m);
    assert_eq!(shape.k_skip, ZK_FORM_K_SKIP);
    assert_eq!(pcs_params.m, shape.m + LOG_PACKING);
    assert_eq!(pcs_params.log_batch_size, 5);

    let mut ledger = b.num_wires();
    bind_statement_zk_form_c1_trace(b, channel, layout, statement_digest, pcs_params, commitment_root);
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: statement bind");
    let io_claims = bind_public_io_c1_trace(b, channel, spec, io, shape.m);
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: public-IO bind");
    channel.observe_label(b, POST_COMMIT_CLASS_BINDING_LABEL);
    observe_flat_digest(b, channel, post_commit_class_digest);
    let auxiliary_claims = {
        let mut context = FieldPostCommitTraceContext::new(commitment_root, shape.m, channel);
        post_commit(b, &mut context);
        context.finish()
    };

    let zerocheck = zerocheck_field_verify_c1_zk_trace(
        b,
        channel,
        shape.m,
        &proof.base.zerocheck,
        &proof.kappa,
    );
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: masked zerocheck");
    let rounds = layout.rounds();
    let lincheck_point = C1QuirkyPointTrace {
        z_skip: zerocheck.claim.z.clone(),
        x_inner_rest: zerocheck.claim.mlv_challenges[..rounds].to_vec(),
        x_outer: Vec::new(),
    };
    let (lincheck, mut fresh) = lincheck_verify_c1_zk_trace_deferred(
        b,
        channel,
        shape.k_log,
        shape.k_skip,
        shape.const_pin,
        &lincheck_point,
        &zerocheck.claim.a_eval,
        &zerocheck.claim.b_eval,
        &proof.base.lincheck,
        &proof.mask_value,
    );
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: masked lincheck");
    let tautologies = tautology_fresh_value_trace(
        b,
        layout,
        &fresh.alpha,
        &fresh.z_skip,
        &fresh.x_inner_rest,
        &fresh.r_inner_rest,
        &fresh.z_partial,
    );
    fresh.value = fresh.value.add(&tautologies);
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: tautologies (closed form)");

    let ab = C1ZClaimTrace {
        z_skip: lincheck.r_inner_skip.clone(),
        x_inner_rest: lincheck.r_inner_rest.clone(),
        x_outer: Vec::new(),
        value: lincheck.w.clone(),
    };
    let c = C1ZClaimTrace {
        z_skip: zerocheck.claim.z.clone(),
        x_inner_rest: zerocheck.claim.r_rest[..rounds].to_vec(),
        x_outer: Vec::new(),
        value: zerocheck.claim.c_eval.clone(),
    };
    let mut claims = vec![
        C1QuirkyDirectClaimTrace {
            z_skip: ab.z_skip.clone(),
            k_skip: shape.k_skip,
            x_rest: ab.x_inner_rest.clone(),
            value: ab.value.clone(),
        },
        C1QuirkyDirectClaimTrace {
            z_skip: c.z_skip.clone(),
            k_skip: shape.k_skip,
            x_rest: c.x_inner_rest.clone(),
            value: c.value.clone(),
        },
    ];
    claims.extend(io_claims);
    claims.extend(
        auxiliary_claims
            .base
            .into_iter()
            .map(|claim| C1QuirkyDirectClaimTrace {
                z_skip: ExtExpr::from_base(claim.z_skip),
                k_skip: claim.k_skip,
                x_rest: claim.x_rest.into_iter().map(ExtExpr::from_base).collect(),
                value: ExtExpr::from_base(claim.value),
            }),
    );
    claims.extend(auxiliary_claims.c1);
    let sparse = [
        zerocheck_mask_claim_trace(
            b,
            layout,
            &zerocheck.claim.z,
            &zerocheck.claim.mlv_challenges,
            zerocheck.mask_value.clone(),
        ),
        lincheck_mask_claim_trace(b, layout, &lincheck.r_inner_rest, proof.mask_value.clone()),
    ];
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: sparse claim weights");
    verify_opening_batch_quirky_direct_c1_with_sparse_trace_region(
        b,
        channel,
        commitment_root,
        &claims,
        &sparse,
        &proof.base.pcs_open,
        pcs_params,
        region,
    );
    crate::acceptance::row_ledger_mark(b, &mut ledger, "Z-c1: PCS opening batch (sparse)");
    (C1R1csClaimTrace { ab, c }, fresh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::trace::self_verify::{
        alloc_flat_digest, const_flat_digest, C1FieldR1csProofTrace, PcsWalkObligations,
    };
    use jetsam_ivc_core::challenger::{Challenger, FsLaneChallenger};
    use jetsam_ivc_core::field::{F128, F256};
    use jetsam_ivc_core::field_circuit::{FieldR1csBuilder, FsChannelTrace, LinExpr};
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
    use jetsam_ivc_core::proof::{C1ZkFieldR1csProof, FieldShape};
    use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
    use jetsam_ivc_core::zk_form::{draw_zk_form_randomness, ZkFormLayout, ZkFormRelation};

    const DOMAIN: &[u8] = b"e9-zk-trace";
    const CLASS: [u8; 32] = [0xE9; 32];

    struct Rng(u64);
    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f128(&mut self) -> F128 {
            F128::new(self.u64(), self.u64())
        }
        fn below(&mut self, n: usize) -> usize {
            (self.u64() % n as u64) as usize
        }
    }

    /// A small catalogue relation of the client conventions (constant pinned
    /// at 0, IO at 8..16, then `gates` gates `z_c = (k z_a + z_b) z_d`),
    /// renamed into form Z at the reduced layout.
    fn instance(m: usize, gates: usize, seed: u64) -> (ZkFormRelation, Vec<F128>) {
        let n = 1usize << m;
        let mut rng = Rng(seed | 1);
        let mut z = vec![F128::ZERO; n];
        let mut a_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
        let mut b_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
        z[0] = F128::ONE;
        a_rows[0].push((0, F128::ONE));
        b_rows[0].push((0, F128::ONE));
        for io in 8..16 {
            z[io] = rng.f128();
            a_rows[io].push((io as u32, F128::ONE));
            b_rows[io].push((0, F128::ONE));
        }
        for c in 16..16 + gates {
            let (x, y, w) = (rng.below(c), rng.below(c), rng.below(c));
            let k = rng.f128();
            if x == y {
                a_rows[c].push((x as u32, k + F128::ONE));
            } else {
                let mut row = vec![(x as u32, k), (y as u32, F128::ONE)];
                row.sort_by_key(|e| e.0);
                a_rows[c] = row;
            }
            b_rows[c].push((w as u32, F128::ONE));
            let left = if x == y { (k + F128::ONE) * z[x] } else { k * z[x] + z[y] };
            z[c] = left * z[w];
        }
        let logical = FieldR1cs {
            m,
            k_log: m,
            k_skip: 6,
            useful_rows: 16 + gates,
            a_0: SparseFieldMatrix::from_rows(n, a_rows),
            b_0: SparseFieldMatrix::from_rows(n, b_rows),
            const_pin: Some(0),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        assert!(logical.satisfies(&z));
        let relation = ZkFormRelation::from_logical(ZkFormLayout::reduced(m), &logical).unwrap();
        (relation, z)
    }

    fn io_spec() -> PublicIoSpec {
        PublicIoSpec {
            io_slice: WitnessSlice { log2_len: 3, index: 1 },
            io_len: 8,
            claims: Vec::new(),
        }
    }

    fn pcs_params(m: usize) -> PcsParams {
        PcsParams {
            m: m + LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 5,
            profile: Default::default(),
        }
    }

    struct Proved {
        relation: ZkFormRelation,
        proof: C1ZkFieldR1csProof,
        commitment: jetsam_ivc_core::pcs::Commitment,
        io: Vec<F128>,
    }

    fn prove(seed: u64) -> Proved {
        let (relation, witness) = instance(12, 700, seed);
        let mut rng = Rng(seed ^ 0x5EED);
        let randomness = draw_zk_form_randomness(*relation.layout(), &mut |bytes: &mut [u8]| {
            for chunk in bytes.chunks_mut(8) {
                let v = rng.u64().to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
        });
        let io = witness[8..16].to_vec();
        let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                &relation,
                &witness,
                randomness,
                &pcs_params(12),
                &io_spec(),
                &io,
                &CLASS,
                &mut challenger,
                |_| (),
            )
            .expect("form-Z proof");
        Proved { relation, proof, commitment, io }
    }

    /// The trace verifier on `proof`; returns the builder (to check
    /// satisfiability), the fresh claim's evaluated fields, the two claim
    /// values and the next transcript sample.
    fn run_trace(
        proved: &Proved,
        proof: &C1ZkFieldR1csProof,
    ) -> (bool, Vec<F256>, F256, F256, F256, usize) {
        let layout = *proved.relation.layout();
        let physical = proved.relation.physical();
        let shape = FieldShape::of(physical);
        let params = pcs_params(layout.m);
        let mut b = FieldR1csBuilder::new();
        let mut channel = FsChannelTrace::new_c1(&mut b, DOMAIN);
        let digest = alloc_flat_digest(&mut b, &physical.statement_digest());
        let root = alloc_flat_digest(&mut b, &proved.commitment.root);
        let io = proved
            .io
            .iter()
            .map(|v| LinExpr::from_wire(b.alloc_f128(*v)))
            .collect::<Vec<_>>();
        let base = C1FieldR1csProofTrace::alloc_shape_mode(&mut b, &proof.base, &shape, &params, false);
        let proof_trace = C1ZkFieldR1csProofTrace::alloc(&mut b, proof, base);
        let mut obligations = PcsWalkObligations::default();
        let before = b.num_wires();
        let (claim, fresh) = verify_field_c1_zk_trace_deferred_region_with_post_commit_context_expr(
            &mut b,
            &mut channel,
            &layout,
            &shape,
            &params,
            &digest,
            &root,
            &proof_trace,
            &io_spec(),
            &io,
            &const_flat_digest(&CLASS),
            Some(&mut obligations),
            |_, _| {},
        );
        let rows = b.num_wires() - before;
        assert!(!obligations.leaves.is_empty());
        let values = b.values().to_vec();
        let mut fields = vec![fresh.alpha.eval(&values), fresh.z_skip.eval(&values)];
        fields.extend(fresh.x_inner_rest.iter().map(|v| v.eval(&values)));
        fields.extend(fresh.r_inner_rest.iter().map(|v| v.eval(&values)));
        fields.extend(fresh.z_partial.iter().map(|v| v.eval(&values)));
        let fresh_value = fresh.value.eval(&values);
        let ab = claim.ab.value.eval(&values);
        let next = channel.sample_f256(&mut b).eval(b.values());
        let (relation, witness) = b.build();
        (relation.satisfies(&witness), fields, fresh_value, ab, next, rows)
    }

    #[test]
    fn the_form_z_trace_verifier_replays_the_native_one() {
        let proved = prove(0xE901);
        let layout = *proved.relation.layout();
        let physical = proved.relation.physical();
        let shape = FieldShape::of(physical);
        let mut native = FsLaneChallenger::new_c1(DOMAIN);
        let (claim, fresh) =
            jetsam_ivc_core::verifier::verify_field_c1_zk_deferred_matrix_with_post_commit_context(
                &layout,
                &shape,
                &physical.statement_digest(),
                &proved.commitment,
                &proved.proof,
                &io_spec(),
                &proved.io,
                &CLASS,
                &(),
                &mut native,
                |_, _| Ok(()),
            )
            .expect("native form-Z verifier accepts");
        let native_next = native.sample_f256();
        let mut expected = vec![fresh.alpha, fresh.z_skip];
        expected.extend(&fresh.x_inner_rest);
        expected.extend(&fresh.r_inner_rest);
        expected.extend(&fresh.z_partial);

        let (satisfied, fields, value, ab, next, rows) = run_trace(&proved, &proved.proof);
        eprintln!("[e9 trace] form-Z verifier at m = {}: {rows} rows", layout.m);
        assert_eq!(fields, expected, "fresh claim point");
        assert_eq!(value, fresh.value, "stripped fresh value");
        assert_eq!(ab, claim.ab.value, "ab claim");
        assert_eq!(next, native_next, "transcript in lockstep");
        assert!(satisfied, "an honest form-Z proof satisfies the trace");
    }

    #[test]
    fn a_wrong_kappa_or_mask_value_leaves_the_trace_unsatisfiable() {
        let proved = prove(0xE902);
        for alter in 0..4 {
            let mut proof = proved.proof.clone();
            match alter {
                0 => proof.zerocheck_kappa += F256::ONE,
                1 => proof.zerocheck_kappa = F256::ZERO,
                2 => proof.lincheck_mask_value += F256::ONE,
                _ => proof.base.zerocheck.round1_ab[5] += F256::ONE,
            }
            let (satisfied, ..) = run_trace(&proved, &proof);
            assert!(!satisfied, "alteration {alter} accepted");
        }
    }

    #[test]
    fn the_tautology_closed_form_trace_equals_the_native_one() {
        let layout = ZkFormLayout::reduced(12);
        let mut rng = Rng(0xE903);
        let mut f256 = || F256::new(rng.f128(), rng.f128());
        let alpha = f256();
        let z_skip = f256();
        let rho = (0..layout.rounds()).map(|_| f256()).collect::<Vec<_>>();
        let r_inner = (0..layout.rounds()).map(|_| f256()).collect::<Vec<_>>();
        let z_partial = (0..64).map(|_| f256()).collect::<Vec<_>>();
        let native = layout.tautology_fresh_value(alpha, z_skip, &rho, &r_inner, &z_partial);
        let mut b = FieldR1csBuilder::new();
        let alloc = |value: F256, b: &mut FieldR1csBuilder| {
            jetsam_ivc_core::field_circuit::ExtExpr::new(
                LinExpr::from_wire(b.alloc_f128(value.lo)),
                LinExpr::from_wire(b.alloc_f128(value.hi)),
            )
        };
        let alpha_t = alloc(alpha, &mut b);
        let z_t = alloc(z_skip, &mut b);
        let rho_t = rho.iter().map(|v| alloc(*v, &mut b)).collect::<Vec<_>>();
        let r_t = r_inner.iter().map(|v| alloc(*v, &mut b)).collect::<Vec<_>>();
        let zp_t = z_partial.iter().map(|v| alloc(*v, &mut b)).collect::<Vec<_>>();
        let value = tautology_fresh_value_trace(&mut b, &layout, &alpha_t, &z_t, &rho_t, &r_t, &zp_t);
        assert_eq!(value.eval(b.values()), native);
        let (relation, witness) = b.build();
        assert!(relation.satisfies(&witness));
    }

    /// The sparse claims' weights are the verifier's own and are not
    /// observed: only the `final_b` pin binds them (and through them `M(r)`
    /// and the derived zerocheck mask value) to the commitment. A weight
    /// altered on the verifier side, the opening honest, is refused by that
    /// pin alone (no other check catches it).
    #[test]
    fn a_sparse_weight_is_bound_by_the_final_b_pin() {
        use crate::acceptance::trace::self_verify::C1BaseFoldProofTrace;
        use jetsam_ivc_core::field_circuit::ExtExpr;
        use jetsam_ivc_core::pcs::{self, C1QuirkyDirectClaim};
        use jetsam_ivc_core::zk_form::ZkSparseClaim;
        let m = 12usize;
        let mut rng = Rng(0xE9D1);
        let witness: Vec<F128> = (0..1usize << m).map(|_| rng.f128()).collect();
        let params = pcs_params(m);
        let (commitment, data) = pcs::commit(&witness, &params);
        let mut f256 = || F256::new(rng.f128(), rng.f128());
        let x_rest = (0..m - 6).map(|_| f256()).collect::<Vec<_>>();
        let z_skip = f256();
        let weights = (0..3).map(|_| f256()).collect::<Vec<_>>();
        // Skip Lagrange weights and eq table as constants (the trace helpers
        // fold constant operands without a wire).
        let (lag, eq) = {
            let mut scratch = FieldR1csBuilder::new();
            let lag = lagrange_weights_window_ext_trace(&mut scratch, &ExtExpr::constant(z_skip), 0, 64, 0);
            let point = x_rest.iter().map(|v| ExtExpr::constant(*v)).collect::<Vec<_>>();
            let eq = crate::acceptance::trace::eq_ind_partial_eval_ext_trace(&mut scratch, &point);
            let values = scratch.values().to_vec();
            (
                lag.iter().map(|v| v.eval(&values)).collect::<Vec<_>>(),
                eq.iter().map(|v| v.eval(&values)).collect::<Vec<_>>(),
            )
        };
        let value = (0..1usize << m).fold(F256::ZERO, |sum, i| sum + (lag[i & 63] * eq[i >> 6]).scale_base(witness[i]));
        let claim = C1QuirkyDirectClaim { z_skip, k_skip: 6, x_rest: x_rest.clone(), value };
        let mut sparse = ZkSparseClaim { cells: vec![5, 700, 4000], weights: weights.clone(), value: F256::ZERO };
        sparse.value = sparse.evaluate(&witness);
        let mut prover = FsLaneChallenger::new_c1(b"e9-sparse-pin");
        let opening = pcs::open_batch_quirky_direct_c1_with_sparse(
            &witness,
            &data,
            &commitment,
            std::slice::from_ref(&claim),
            std::slice::from_ref(&sparse),
            &mut prover,
        );
        let run = |delta: F256| {
            let mut b = FieldR1csBuilder::new();
            let mut channel = FsChannelTrace::new_c1(&mut b, b"e9-sparse-pin");
            let root = alloc_flat_digest(&mut b, &commitment.root);
            let proof = C1BaseFoldProofTrace::alloc_mode(&mut b, &opening, &params, false);
            let ext = |v: F256, b: &mut FieldR1csBuilder| {
                ExtExpr::new(LinExpr::from_wire(b.alloc_f128(v.lo)), LinExpr::from_wire(b.alloc_f128(v.hi)))
            };
            let claims = [C1QuirkyDirectClaimTrace {
                z_skip: ext(z_skip, &mut b),
                k_skip: 6,
                x_rest: x_rest.iter().map(|v| ext(*v, &mut b)).collect(),
                value: ext(value, &mut b),
            }];
            let mut trace_weights = weights.iter().map(|w| ext(*w, &mut b)).collect::<Vec<_>>();
            trace_weights[1] = trace_weights[1].add_const(delta);
            let sparse_trace = [ZkSparseClaimTrace {
                cells: sparse.cells.clone(),
                weights: trace_weights,
                value: ext(sparse.value, &mut b),
            }];
            let mut obligations = PcsWalkObligations::default();
            verify_opening_batch_quirky_direct_c1_with_sparse_trace_region(
                &mut b,
                &mut channel,
                &root,
                &claims,
                &sparse_trace,
                &proof,
                &params,
                Some(&mut obligations),
            );
            let (relation, witness) = b.build();
            relation.satisfies(&witness)
        };
        assert!(run(F256::ZERO), "the honest sparse opening satisfies the trace");
        assert!(!run(F256::ONE), "an altered sparse weight is refused");
    }
}
