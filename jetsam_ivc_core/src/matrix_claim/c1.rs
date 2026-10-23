//! Extension-field matrix accumulator for the C1 History profile.
//!
//! Matrix coefficients remain in F128. Claims, sumcheck messages, transcript
//! challenges, folded tables, and the outgoing accumulator live in F256.

use super::{MatrixFoldError, MatrixFoldRows};
use crate::challenger::Challenger;
#[cfg(test)]
use crate::field::F128;
use crate::field::F256;
use crate::field_r1cs::{
    CompactFieldR1cs, FieldR1cs, FieldR1csArtifactError, FieldR1csArtifactMatrix,
    parallel_compact_group_fold_c1,
};
use crate::proof::FieldShape;
use crate::zerocheck::field_c1::{build_eq_table, lagrange_weights};
use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;
use rayon::prelude::*;

const C1_MATRIX_CLAIM_REQUEST_DOMAIN: &[u8] = b"JTM/IVC/MATRIX-CLAIM-REQUEST/C1/V1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct C1MatrixAccClaim {
    pub point: Vec<F256>,
    pub value: F256,
}

impl C1MatrixAccClaim {
    pub fn zero(k_log: usize) -> Self {
        Self {
            point: vec![F256::ZERO; 2 * k_log + 1],
            value: F256::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct C1FreshLincheckClaim {
    pub alpha: F256,
    pub z_skip: F256,
    pub x_inner_rest: Vec<F256>,
    pub r_inner_rest: Vec<F256>,
    pub z_partial: Vec<F256>,
    pub value: F256,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct C1MatrixFoldProof {
    pub phase1_rounds: Vec<[F256; 2]>,
    pub g_v: F256,
    pub g_e: F256,
    pub phase2_rounds: Vec<[F256; 2]>,
    pub final_matrix_eval: F256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthenticatedC1MatrixClaimEvaluations {
    structural_digest: [u8; 32],
    request_binding: [u8; 32],
    fresh_value: Option<F256>,
    accumulated_value: Option<F256>,
}

impl AuthenticatedC1MatrixClaimEvaluations {
    pub const fn structural_digest(&self) -> [u8; 32] {
        self.structural_digest
    }

    pub const fn fresh_value(&self) -> Option<F256> {
        self.fresh_value
    }

    pub const fn accumulated_value(&self) -> Option<F256> {
        self.accumulated_value
    }

    pub fn is_bound_to(
        &self,
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
    ) -> bool {
        self.request_binding == c1_matrix_claim_request_binding(fresh, accumulated)
    }

    fn new(
        structural_digest: [u8; 32],
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
        fresh_value: Option<F256>,
        accumulated_value: Option<F256>,
    ) -> Self {
        Self {
            structural_digest,
            request_binding: c1_matrix_claim_request_binding(fresh, accumulated),
            fresh_value,
            accumulated_value,
        }
    }
}

fn c1_matrix_claim_request_binding(
    fresh: Option<&C1FreshLincheckClaim>,
    accumulated: Option<&C1MatrixAccClaim>,
) -> [u8; 32] {
    fn push_field(bytes: &mut Vec<u8>, value: F256) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn push_fields(bytes: &mut Vec<u8>, values: &[F256]) {
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for &value in values {
            push_field(bytes, value);
        }
    }

    let mut bytes = Vec::new();
    bytes.push(u8::from(fresh.is_some()));
    if let Some(claim) = fresh {
        push_field(&mut bytes, claim.alpha);
        push_field(&mut bytes, claim.z_skip);
        push_fields(&mut bytes, &claim.x_inner_rest);
        push_fields(&mut bytes, &claim.r_inner_rest);
        push_fields(&mut bytes, &claim.z_partial);
        push_field(&mut bytes, claim.value);
    }
    bytes.push(u8::from(accumulated.is_some()));
    if let Some(claim) = accumulated {
        push_fields(&mut bytes, &claim.point);
        push_field(&mut bytes, claim.value);
    }
    poseidon2b_hash_byte_slices(C1_MATRIX_CLAIM_REQUEST_DOMAIN, &[&bytes])
}

pub trait C1MatrixClaimEvaluator {
    fn field_shape(&self) -> FieldShape;

    fn evaluate_matrix_claims_c1(
        &mut self,
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
    ) -> Result<AuthenticatedC1MatrixClaimEvaluations, FieldR1csArtifactError>;
}

fn validate_claim_shapes(
    shape: FieldShape,
    fresh: Option<&C1FreshLincheckClaim>,
    accumulated: Option<&C1MatrixAccClaim>,
) -> Result<(), FieldR1csArtifactError> {
    if let Some(claim) = fresh {
        let rest = shape.k_log - shape.k_skip;
        if claim.x_inner_rest.len() != rest || claim.r_inner_rest.len() != rest {
            return Err(FieldR1csArtifactError::MatrixClaimShape(
                "C1 fresh inner-rest width",
            ));
        }
        if claim.z_partial.len() != 1usize << shape.k_skip {
            return Err(FieldR1csArtifactError::MatrixClaimShape(
                "C1 fresh partial window",
            ));
        }
    }
    if accumulated.is_some_and(|claim| claim.point.len() != 2 * shape.k_log + 1) {
        return Err(FieldR1csArtifactError::MatrixClaimShape(
            "C1 accumulated point width",
        ));
    }
    Ok(())
}

impl C1MatrixClaimEvaluator for FieldR1cs {
    fn field_shape(&self) -> FieldShape {
        FieldShape::of(self)
    }

    fn evaluate_matrix_claims_c1(
        &mut self,
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
    ) -> Result<AuthenticatedC1MatrixClaimEvaluations, FieldR1csArtifactError> {
        let shape = FieldShape::of(&*self);
        validate_claim_shapes(shape, fresh, accumulated)?;
        Ok(AuthenticatedC1MatrixClaimEvaluations::new(
            self.structural_statement_digest(),
            fresh,
            accumulated,
            fresh.map(|claim| fresh_claim_value_c1(self, claim)),
            accumulated.map(|claim| stacked_matrix_mle_eval_c1(self, claim)),
        ))
    }
}

pub fn fresh_claim_value_c1(r1cs: &FieldR1cs, claim: &C1FreshLincheckClaim) -> F256 {
    let low_count = 1usize << r1cs.k_skip;
    let mask = low_count - 1;
    let lambda = lagrange_weights(r1cs.k_skip, claim.z_skip, 0);
    let row_rest = build_eq_table(&claim.x_inner_rest);
    let column_rest = build_eq_table(&claim.r_inner_rest);
    [(&r1cs.a_0, claim.alpha), (&r1cs.b_0, F256::ONE)]
        .into_par_iter()
        .map(|(matrix, stack_weight)| {
            (0..matrix.num_rows)
                .into_par_iter()
                .map(|row| {
                    let row_weight =
                        lambda[row & mask] * row_rest[row >> r1cs.k_skip] * stack_weight;
                    matrix
                        .row(row)
                        .fold(F256::ZERO, |sum, (column, coefficient)| {
                            let column = column as usize;
                            let column_weight =
                                claim.z_partial[column & mask] * column_rest[column >> r1cs.k_skip];
                            sum + (row_weight * column_weight).scale_base(coefficient)
                        })
                })
                .reduce(|| F256::ZERO, |left, right| left + right)
        })
        .reduce(|| F256::ZERO, |left, right| left + right)
}

pub fn stacked_matrix_mle_eval_c1(r1cs: &FieldR1cs, claim: &C1MatrixAccClaim) -> F256 {
    assert_eq!(claim.point.len(), 2 * r1cs.k_log + 1);
    let (row_point, column_point) = claim.point.split_at(r1cs.k_log + 1);
    let row_weights = build_eq_table(row_point);
    let column_weights = build_eq_table(column_point);
    let width = 1usize << r1cs.k_log;
    [(&r1cs.a_0, 0usize), (&r1cs.b_0, width)]
        .into_par_iter()
        .map(|(matrix, offset)| {
            (0..matrix.num_rows)
                .into_par_iter()
                .map(|row| {
                    let row_weight = row_weights[offset + row];
                    matrix
                        .row(row)
                        .fold(F256::ZERO, |sum, (column, coefficient)| {
                            sum + (row_weight * column_weights[column as usize])
                                .scale_base(coefficient)
                        })
                })
                .reduce(|| F256::ZERO, |left, right| left + right)
        })
        .reduce(|| F256::ZERO, |left, right| left + right)
}

fn eq_points(left: &[F256], right: &[F256]) -> F256 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .fold(F256::ONE, |product, (&left, &right)| {
            product * (F256::ONE + left + right)
        })
}

fn small_mle_eval(values: &[F256], point: &[F256]) -> F256 {
    assert_eq!(values.len(), 1usize << point.len());
    values
        .iter()
        .zip(build_eq_table(point))
        .fold(F256::ZERO, |sum, (&value, weight)| sum + value * weight)
}

fn alpha_weight(alpha: F256, stack_point: F256) -> F256 {
    alpha + stack_point * (alpha + F256::ONE)
}

fn u_weight_eval(fresh: &C1FreshLincheckClaim, k_skip: usize, rho: &[F256]) -> F256 {
    let lambda = lagrange_weights(k_skip, fresh.z_skip, 0);
    let low = small_mle_eval(&lambda, &rho[..k_skip]);
    let rest = eq_points(&fresh.x_inner_rest, &rho[k_skip..rho.len() - 1]);
    let stack = alpha_weight(fresh.alpha, rho[rho.len() - 1]);
    low * rest * stack
}

fn v_weight_eval(fresh: &C1FreshLincheckClaim, k_skip: usize, sigma: &[F256]) -> F256 {
    let partial = small_mle_eval(&fresh.z_partial, &sigma[..k_skip]);
    let rest = eq_points(&fresh.r_inner_rest, &sigma[k_skip..]);
    partial * rest
}

fn absorb_fold_header<Ch: Challenger>(
    channel: &mut Ch,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: F256,
) {
    channel.observe_label(b"history-matrix-claim-fold-c1");
    channel.observe_f256(fresh.alpha);
    channel.observe_f256(fresh.z_skip);
    channel.observe_f256_slice(&fresh.x_inner_rest);
    channel.observe_f256_slice(&fresh.r_inner_rest);
    channel.observe_f256_slice(&fresh.z_partial);
    channel.observe_f256(fresh.value);
    channel.observe_f256_slice(&incoming.point);
    channel.observe_f256(incoming.value);
    channel.observe_f256(gate);
}

fn fold_table_pairs_reusing(
    mut table: Vec<F256>,
    mut spare: Vec<F256>,
    challenge: F256,
) -> (Vec<F256>, Vec<F256>) {
    let half = table.len() / 2;
    if half >= 1024 {
        if spare.len() < half {
            let folded = (0..half)
                .into_par_iter()
                .map(|index| {
                    let low = table[2 * index];
                    let high = table[2 * index + 1];
                    low + challenge * (low + high)
                })
                .collect();
            return (folded, table);
        }
        spare.truncate(half);
        spare
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let low = table[2 * index];
                let high = table[2 * index + 1];
                *output = low + challenge * (low + high);
            });
        (spare, table)
    } else {
        for index in 0..half {
            let low = table[2 * index];
            let high = table[2 * index + 1];
            table[index] = low + challenge * (low + high);
        }
        table.truncate(half);
        (table, spare)
    }
}

#[cfg(test)]
fn round_coefficients_two_products(
    w1: &[F256],
    g1: &[F256],
    w2: &[F256],
    g2: &[F256],
) -> [F256; 2] {
    let half = w1.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|index| {
            let mut output = [F256::ZERO; 2];
            for (weights, values) in [(w1, g1), (w2, g2)] {
                let weight_low = weights[2 * index];
                let weight_high = weights[2 * index + 1];
                let value_low = values[2 * index];
                let value_high = values[2 * index + 1];
                output[0] += weight_low * value_low;
                output[1] += (weight_low + weight_high) * (value_low + value_high);
            }
            output
        })
        .reduce(
            || [F256::ZERO; 2],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        )
}

/// Phase-one coefficient kernel with the final stacked A/B coordinate kept
/// factored out of both weight tables. The fresh weights are
/// `[alpha * base, base]`; the incoming equality weights are
/// `[(1 + point) * base, point * base]`. Folding those two affine factors
/// into the corresponding A/B value halves produces exactly the same two
/// coefficients without materializing either doubled weight table.
fn round_coefficients_two_stacked_products(
    u_base: &[F256],
    alpha: F256,
    g_v: &[F256],
    incoming_base: &[F256],
    incoming_stack_point: F256,
    g_e: &[F256],
) -> [F256; 2] {
    let side_len = u_base.len();
    debug_assert!(side_len > 1);
    debug_assert_eq!(incoming_base.len(), side_len);
    debug_assert_eq!(g_v.len(), 2 * side_len);
    debug_assert_eq!(g_e.len(), 2 * side_len);
    let half = side_len / 2;
    (0..half)
        .into_par_iter()
        .map(|index| {
            let low = 2 * index;
            let high = low + 1;

            let fresh_value_low = alpha * g_v[low] + g_v[side_len + low];
            let fresh_value_delta =
                alpha * (g_v[low] + g_v[high]) + g_v[side_len + low] + g_v[side_len + high];
            let incoming_value_low =
                g_e[low] + incoming_stack_point * (g_e[low] + g_e[side_len + low]);
            let incoming_value_delta = g_e[low]
                + g_e[high]
                + incoming_stack_point
                    * (g_e[low] + g_e[high] + g_e[side_len + low] + g_e[side_len + high]);

            [
                u_base[low] * fresh_value_low + incoming_base[low] * incoming_value_low,
                (u_base[low] + u_base[high]) * fresh_value_delta
                    + (incoming_base[low] + incoming_base[high]) * incoming_value_delta,
            ]
        })
        .reduce(
            || [F256::ZERO; 2],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        )
}

fn final_stacked_round_coefficients(
    u_base: F256,
    alpha: F256,
    g_v: &[F256],
    incoming_base: F256,
    incoming_stack_point: F256,
    g_e: &[F256],
) -> [F256; 2] {
    debug_assert_eq!(g_v.len(), 2);
    debug_assert_eq!(g_e.len(), 2);
    let u_low = u_base * alpha;
    let u_high = u_base;
    let incoming_low = incoming_base * (F256::ONE + incoming_stack_point);
    let incoming_high = incoming_base * incoming_stack_point;
    [
        u_low * g_v[0] + incoming_low * g_e[0],
        (u_low + u_high) * (g_v[0] + g_v[1]) + (incoming_low + incoming_high) * (g_e[0] + g_e[1]),
    ]
}

fn round_coefficients_one_product(weights: &[F256], values: &[F256]) -> [F256; 2] {
    let half = weights.len() / 2;
    (0..half)
        .into_par_iter()
        .map(|index| {
            let weight_low = weights[2 * index];
            let weight_high = weights[2 * index + 1];
            let value_low = values[2 * index];
            let value_high = values[2 * index + 1];
            [
                weight_low * value_low,
                (weight_low + weight_high) * (value_low + value_high),
            ]
        })
        .reduce(
            || [F256::ZERO; 2],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        )
}

fn mix_and_round_coefficients_one_product(
    weights: &mut [F256],
    equality: &[F256],
    values: &[F256],
    scale: F256,
) -> [F256; 2] {
    weights
        .par_chunks_exact_mut(2)
        .zip(equality.par_chunks_exact(2))
        .zip(values.par_chunks_exact(2))
        .map(|((weights, equality), values)| {
            let low = weights[0] + scale * equality[0];
            let high = weights[1] + scale * equality[1];
            weights[0] = low;
            weights[1] = high;
            [low * values[0], (low + high) * (values[0] + values[1])]
        })
        .reduce(
            || [F256::ZERO; 2],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        )
}

#[cfg(test)]
fn run_phase<Ch: Challenger>(
    mut claim: F256,
    mut w1: Vec<F256>,
    mut g1: Vec<F256>,
    mut w2: Vec<F256>,
    mut g2: Vec<F256>,
    channel: &mut Ch,
) -> (Vec<[F256; 2]>, Vec<F256>, F256, [F256; 4]) {
    let rounds_count = w1.len().trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(rounds_count);
    let mut point = Vec::with_capacity(rounds_count);
    let mut spare = Vec::new();
    for _ in 0..rounds_count {
        let [constant, quadratic] = round_coefficients_two_products(&w1, &g1, &w2, &g2);
        let linear = claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = (quadratic * challenge + linear) * challenge + constant;
        rounds.push(wire);
        point.push(challenge);
        (w1, spare) = fold_table_pairs_reusing(w1, spare, challenge);
        (g1, spare) = fold_table_pairs_reusing(g1, spare, challenge);
        (w2, spare) = fold_table_pairs_reusing(w2, spare, challenge);
        (g2, spare) = fold_table_pairs_reusing(g2, spare, challenge);
    }
    (rounds, point, claim, [w1[0], g1[0], w2[0], g2[0]])
}

fn run_phase_stacked<Ch: Challenger>(
    mut claim: F256,
    mut u_base: Vec<F256>,
    alpha: F256,
    mut g_v: Vec<F256>,
    mut incoming_base: Vec<F256>,
    incoming_stack_point: F256,
    mut g_e: Vec<F256>,
    channel: &mut Ch,
) -> (Vec<[F256; 2]>, Vec<F256>, F256, [F256; 4]) {
    debug_assert_eq!(u_base.len(), incoming_base.len());
    debug_assert_eq!(g_v.len(), 2 * u_base.len());
    debug_assert_eq!(g_e.len(), 2 * u_base.len());
    let rounds_count = u_base.len().trailing_zeros() as usize + 1;
    let mut rounds = Vec::with_capacity(rounds_count);
    let mut point = Vec::with_capacity(rounds_count);
    let mut spare = Vec::new();

    while u_base.len() > 1 {
        let [constant, quadratic] = round_coefficients_two_stacked_products(
            &u_base,
            alpha,
            &g_v,
            &incoming_base,
            incoming_stack_point,
            &g_e,
        );
        let linear = claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = (quadratic * challenge + linear) * challenge + constant;
        rounds.push(wire);
        point.push(challenge);
        (u_base, spare) = fold_table_pairs_reusing(u_base, spare, challenge);
        (g_v, spare) = fold_table_pairs_reusing(g_v, spare, challenge);
        (incoming_base, spare) = fold_table_pairs_reusing(incoming_base, spare, challenge);
        (g_e, spare) = fold_table_pairs_reusing(g_e, spare, challenge);
    }

    let [constant, quadratic] = final_stacked_round_coefficients(
        u_base[0],
        alpha,
        &g_v,
        incoming_base[0],
        incoming_stack_point,
        &g_e,
    );
    let linear = claim + quadratic;
    let wire = [constant, quadratic];
    channel.observe_f256_slice(&wire);
    let challenge = channel.sample_f256();
    claim = (quadratic * challenge + linear) * challenge + constant;
    rounds.push(wire);
    point.push(challenge);

    let u_final = u_base[0] * (alpha + challenge * (alpha + F256::ONE));
    let incoming_low = F256::ONE + incoming_stack_point;
    let incoming_final =
        incoming_base[0] * (incoming_low + challenge * (incoming_low + incoming_stack_point));
    (g_v, spare) = fold_table_pairs_reusing(g_v, spare, challenge);
    (g_e, _) = fold_table_pairs_reusing(g_e, spare, challenge);
    (
        rounds,
        point,
        claim,
        [u_final, g_v[0], incoming_final, g_e[0]],
    )
}

fn run_phase_one_product<Ch: Challenger>(
    mut claim: F256,
    mut weights: Vec<F256>,
    mut values: Vec<F256>,
    mut first_round: Option<[F256; 2]>,
    channel: &mut Ch,
) -> (Vec<[F256; 2]>, Vec<F256>, F256, [F256; 2]) {
    let rounds_count = weights.len().trailing_zeros() as usize;
    let mut rounds = Vec::with_capacity(rounds_count);
    let mut point = Vec::with_capacity(rounds_count);
    let mut spare = Vec::new();
    for _ in 0..rounds_count {
        let [constant, quadratic] = first_round
            .take()
            .unwrap_or_else(|| round_coefficients_one_product(&weights, &values));
        let linear = claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        claim = (quadratic * challenge + linear) * challenge + constant;
        rounds.push(wire);
        point.push(challenge);
        (weights, spare) = fold_table_pairs_reusing(weights, spare, challenge);
        (values, spare) = fold_table_pairs_reusing(values, spare, challenge);
    }
    (rounds, point, claim, [weights[0], values[0]])
}

// ---- Merge tables on the matrix's support ----
//
// A merge's row images vanish on rows past the last row with an entry, its
// column image on columns past the largest column: below the width, both are
// a prefix `2^u` of their table. The two-sided merge prover works on those
// prefixes only — the weights' `eq` factors of the zero high bits are applied
// round by round — and yields exactly the rounds of the full-width tables.

/// The least `u` with `2^u ≥ bound`.
fn support_bits(bound: usize) -> usize {
    if bound <= 1 {
        0
    } else {
        (bound - 1).ilog2() as usize + 1
    }
}

/// One past the last row with an entry and one past the largest column, over
/// both base matrices of `r1cs`.
fn matrix_support(r1cs: &FieldR1cs) -> (usize, usize) {
    let mut rows = 0;
    let mut columns = 0;
    for matrix in [&r1cs.a_0, &r1cs.b_0] {
        let entries = matrix.col_indices.len();
        if entries > 0 {
            // The first row offset that reaches the end closes the last row
            // with an entry.
            rows = rows.max(matrix.row_offsets.partition_point(|&offset| offset < entries));
        }
        if let Some(&largest) = matrix.col_indices.par_iter().max() {
            columns = columns.max(largest as usize + 1);
        }
    }
    (rows, columns)
}

/// `Π (1 + c)`: the `eq` factor of `coordinates` against all-zero bits.
fn eq_at_zero(coordinates: &[F256]) -> F256 {
    coordinates
        .iter()
        .fold(F256::ONE, |product, &coordinate| product * (F256::ONE + coordinate))
}

/// Row images on the support: rows below `rows` of A, then of B (`2·rows`).
fn fill_row_images_on_support(
    r1cs: &FieldR1cs,
    left_weights: &[F256],
    right_weights: &[F256],
    rows: usize,
    g_left: &mut [F256],
    g_right: &mut [F256],
) {
    for (matrix, offset) in [(&r1cs.a_0, 0usize), (&r1cs.b_0, rows)] {
        g_left[offset..offset + rows]
            .par_iter_mut()
            .zip(g_right[offset..offset + rows].par_iter_mut())
            .enumerate()
            .for_each(|(row, (left, right))| {
                if row >= matrix.num_rows {
                    return;
                }
                for (column, coefficient) in matrix.row(row) {
                    *left += left_weights[column as usize].scale_base(coefficient);
                    *right += right_weights[column as usize].scale_base(coefficient);
                }
            });
    }
}

/// The column image of the row weights `equality` (A rows at `0`, B rows at
/// `width`) on the support: columns below `columns`, rows below `rows`. Each
/// worker owns a slice of the columns and scans the rows for it, so the image
/// is written once, in parallel, without private copies.
fn column_image_on_support(
    r1cs: &FieldR1cs,
    equality: &FactoredEqTable,
    width: usize,
    rows: usize,
    columns: usize,
) -> Vec<F256> {
    let mut image = vec![F256::ZERO; columns];
    let span = columns.div_ceil(rayon::current_num_threads().max(1)).max(1);
    image
        .par_chunks_mut(span)
        .enumerate()
        .for_each(|(part, slice)| {
            let first = part * span;
            let end = first + slice.len();
            for (matrix, offset) in [(&r1cs.a_0, 0usize), (&r1cs.b_0, width)] {
                for row in 0..rows.min(matrix.num_rows) {
                    let mut weight = None;
                    for at in matrix.row_range(row) {
                        let column = matrix.col_indices[at] as usize;
                        if column < first || column >= end {
                            continue;
                        }
                        let weight = *weight.get_or_insert_with(|| equality.value(offset + row));
                        let coefficient = matrix.value_table[matrix.value_indices[at] as usize];
                        slice[column - first] += weight.scale_base(coefficient);
                    }
                }
            }
        });
    image
}

/// Both products' coefficients, each scaled (`scale_i · Σ w_i·g_i`).
fn round_coefficients_two_products_scaled(
    w1: &[F256],
    g1: &[F256],
    scale1: F256,
    w2: &[F256],
    g2: &[F256],
    scale2: F256,
) -> [F256; 2] {
    let half = w1.len() / 2;
    let [first_constant, first_quadratic, second_constant, second_quadratic] = (0..half)
        .into_par_iter()
        .map(|index| {
            let (low, high) = (2 * index, 2 * index + 1);
            [
                w1[low] * g1[low],
                (w1[low] + w1[high]) * (g1[low] + g1[high]),
                w2[low] * g2[low],
                (w2[low] + w2[high]) * (g2[low] + g2[high]),
            ]
        })
        .reduce(
            || [F256::ZERO; 4],
            |left, right| {
                [
                    left[0] + right[0],
                    left[1] + right[1],
                    left[2] + right[2],
                    left[3] + right[3],
                ]
            },
        );
    [
        scale1 * first_constant + scale2 * second_constant,
        scale1 * first_quadratic + scale2 * second_quadratic,
    ]
}

/// The two-product phase `Σ w1·g1 + w2·g2` on tables restricted to a
/// support. Variables, low first: `prefix_bits` held in the tables' low
/// bits; then the tail variables (`tail1`, `tail2`: the weights' point
/// coordinates there), on which `g1`, `g2` vanish once set — left implicit;
/// then the tables' high bits. `w1`, `w2` are `eq` tables over the prefix
/// and high variables; the tail factor `eq(tail_i, ·)` is applied round by
/// round (`Π (1 + tail)` while the tail bits are all zero). `g2 = None`
/// shares `g1`. Same rounds, challenges and final values as the phase on
/// the full tables.
#[allow(clippy::too_many_arguments)]
fn run_phase_on_support<Ch: Challenger>(
    mut claim: F256,
    prefix_bits: usize,
    mut w1: Vec<F256>,
    tail1: &[F256],
    mut g1: Vec<F256>,
    mut w2: Vec<F256>,
    tail2: &[F256],
    g2: Option<Vec<F256>>,
    channel: &mut Ch,
) -> (Vec<[F256; 2]>, Vec<F256>, F256, [F256; 4]) {
    assert_eq!(tail1.len(), tail2.len(), "one tail for both products");
    assert!(w1.len() >= 1usize << prefix_bits);
    let shared = g2.is_none();
    let mut g2 = g2.unwrap_or_default();
    // above_i[j] = Π_{l ≥ j} (1 + tail_i[l]).
    let above = |tail: &[F256]| {
        let mut products = vec![F256::ONE; tail.len() + 1];
        for at in (0..tail.len()).rev() {
            products[at] = products[at + 1] * (F256::ONE + tail[at]);
        }
        products
    };
    let (above1, above2) = (above(tail1), above(tail2));
    let rounds_count = w1.len().trailing_zeros() as usize + tail1.len();
    let mut rounds = Vec::with_capacity(rounds_count);
    let mut point = Vec::with_capacity(rounds_count);
    let mut spare = Vec::new();
    let mut round = |claim: &mut F256, [constant, quadratic]: [F256; 2], channel: &mut Ch| {
        let linear = *claim + quadratic;
        let wire = [constant, quadratic];
        channel.observe_f256_slice(&wire);
        let challenge = channel.sample_f256();
        *claim = (quadratic * challenge + linear) * challenge + constant;
        rounds.push(wire);
        point.push(challenge);
        challenge
    };
    for variable in 0..rounds_count {
        let held = variable < prefix_bits || variable >= prefix_bits + tail1.len();
        let g2_values = if shared { &g1 } else { &g2 };
        if held {
            // A variable held in the tables: under the prefix the whole tail
            // is still zero (its factor scales each side), above it the tail
            // is bound into the weights.
            let (scale1, scale2) = if variable < prefix_bits {
                (above1[0], above2[0])
            } else {
                (F256::ONE, F256::ONE)
            };
            let coefficients = round_coefficients_two_products_scaled(
                &w1, &g1, scale1, &w2, g2_values, scale2,
            );
            let challenge = round(&mut claim, coefficients, channel);
            (w1, spare) = fold_table_pairs_reusing(w1, spare, challenge);
            (g1, spare) = fold_table_pairs_reusing(g1, spare, challenge);
            (w2, spare) = fold_table_pairs_reusing(w2, spare, challenge);
            if !shared {
                (g2, spare) = fold_table_pairs_reusing(g2, spare, challenge);
            }
        } else {
            // A tail variable: the weights are `(1 + p)·rest` with the bit
            // clear and `p·rest` with it set; `g` is zero with it set.
            let at = variable - prefix_bits;
            let sum1 = w1.iter().zip(&g1).fold(F256::ZERO, |sum, (&w, &g)| sum + w * g);
            let sum2 = w2.iter().zip(g2_values).fold(F256::ZERO, |sum, (&w, &g)| sum + w * g);
            let constant = (F256::ONE + tail1[at]) * above1[at + 1] * sum1
                + (F256::ONE + tail2[at]) * above2[at + 1] * sum2;
            let quadratic = above1[at + 1] * sum1 + above2[at + 1] * sum2;
            let challenge = round(&mut claim, [constant, quadratic], channel);
            let factor1 = F256::ONE + tail1[at] + challenge;
            let factor2 = F256::ONE + tail2[at] + challenge;
            let keep = F256::ONE + challenge;
            w1.iter_mut().for_each(|w| *w *= factor1);
            w2.iter_mut().for_each(|w| *w *= factor2);
            g1.iter_mut().for_each(|g| *g *= keep);
            if !shared {
                g2.iter_mut().for_each(|g| *g *= keep);
            }
        }
    }
    let last2 = if shared { g1[0] } else { g2[0] };
    (rounds, point, claim, [w1[0], g1[0], w2[0], last2])
}

fn build_eq_table_scaled(point: &[F256], scale: F256) -> Vec<F256> {
    if scale.is_zero() {
        return vec![F256::ZERO; 1usize << point.len()];
    }
    let mut output = vec![scale];
    for (round, &challenge) in point.iter().enumerate() {
        let length = 1usize << round;
        output.resize(2 * length, F256::ZERO);
        let one_plus = F256::ONE + challenge;
        let (low, high) = output.split_at_mut(length);
        if length >= 4096 {
            low.par_iter_mut()
                .zip(high.par_iter_mut())
                .for_each(|(low, high)| {
                    let value = *low;
                    *high = value * challenge;
                    *low = value * one_plus;
                });
        } else {
            for (low, high) in low.iter_mut().zip(high.iter_mut()) {
                let value = *low;
                *high = value * challenge;
                *low = value * one_plus;
            }
        }
    }
    output
}

fn build_u_base_weights(k_log: usize, k_skip: usize, lambda: &[F256], rest: &[F256]) -> Vec<F256> {
    let width = 1usize << k_log;
    let low_count = 1usize << k_skip;
    let mut output = vec![F256::ZERO; width];
    output
        .par_chunks_mut(low_count)
        .zip(rest.par_iter())
        .for_each(|(chunk, &rest_weight)| {
            for (output, &low_weight) in chunk.iter_mut().zip(lambda) {
                *output = low_weight * rest_weight;
            }
        });
    output
}

struct FactoredEqTable {
    low: Vec<F256>,
    high: Vec<F256>,
    low_mask: usize,
    low_bits: usize,
}

impl FactoredEqTable {
    fn new(point: &[F256]) -> Self {
        let low_bits = point.len() / 2;
        Self {
            low: build_eq_table(&point[..low_bits]),
            high: build_eq_table(&point[low_bits..]),
            low_mask: (1usize << low_bits) - 1,
            low_bits,
        }
    }

    fn value(&self, index: usize) -> F256 {
        self.low[index & self.low_mask] * self.high[index >> self.low_bits]
    }
}

impl MatrixFoldRows<'_> {
    fn fill_row_images_c1(
        &self,
        v_table: &[F256],
        e_c: &[F256],
        g_v: &mut [F256],
        g_e: &mut [F256],
    ) {
        let width = 1usize << self.k_log();
        match self {
            Self::Resident(r1cs) => {
                for (matrix, offset) in [(&r1cs.a_0, 0usize), (&r1cs.b_0, width)] {
                    g_v[offset..offset + width]
                        .par_iter_mut()
                        .zip(g_e[offset..offset + width].par_iter_mut())
                        .enumerate()
                        .for_each(|(row, (g_v, g_e))| {
                            if row >= matrix.num_rows {
                                return;
                            }
                            for (column, coefficient) in matrix.row(row) {
                                *g_v += v_table[column as usize].scale_base(coefficient);
                                *g_e += e_c[column as usize].scale_base(coefficient);
                            }
                        });
                }
            }
            Self::Compact(r1cs) => {
                const GROUP_ROWS: usize = 2048;
                for (side, offset) in [
                    (FieldR1csArtifactMatrix::A, 0usize),
                    (FieldR1csArtifactMatrix::B, width),
                ] {
                    g_v[offset..offset + width]
                        .par_chunks_mut(GROUP_ROWS)
                        .zip(g_e[offset..offset + width].par_chunks_mut(GROUP_ROWS))
                        .enumerate()
                        .for_each(|(group, (g_v_group, g_e_group))| {
                            if group >= r1cs.matrix_group_count(side) {
                                return;
                            }
                            let first_row = group * GROUP_ROWS;
                            let visited = r1cs.for_each_matrix_group_entry(
                                side,
                                group,
                                |row, column, coefficient| {
                                    let local = row - first_row;
                                    g_v_group[local] +=
                                        v_table[column as usize].scale_base(coefficient);
                                    g_e_group[local] +=
                                        e_c[column as usize].scale_base(coefficient);
                                },
                            );
                            assert!(visited);
                        });
                }
            }
        }
    }

    fn weighted_column_image_c1(&self, equality: &FactoredEqTable) -> Vec<F256> {
        let width = 1usize << self.k_log();
        match self {
            Self::Resident(r1cs) => {
                let parts = [(&r1cs.a_0, 0usize), (&r1cs.b_0, width)]
                    .into_par_iter()
                    .map(|(matrix, offset)| {
                        let mut output = vec![F256::ZERO; width];
                        for row in 0..matrix.num_rows {
                            let weight = equality.value(offset + row);
                            if weight.is_zero() {
                                continue;
                            }
                            for (column, coefficient) in matrix.row(row) {
                                output[column as usize] += weight.scale_base(coefficient);
                            }
                        }
                        output
                    })
                    .collect::<Vec<_>>();
                let mut parts = parts.into_iter();
                let mut output = parts.next().expect("A matrix image");
                for part in parts {
                    output
                        .par_iter_mut()
                        .zip(part)
                        .for_each(|(left, right)| *left += right);
                }
                output
            }
            Self::Compact(r1cs) => {
                let group_count = r1cs
                    .matrix_group_count(FieldR1csArtifactMatrix::A)
                    .max(r1cs.matrix_group_count(FieldR1csArtifactMatrix::B));
                parallel_compact_group_fold_c1(width, group_count, |groups, output| {
                    for group in groups {
                        for (side, offset) in [
                            (FieldR1csArtifactMatrix::A, 0usize),
                            (FieldR1csArtifactMatrix::B, width),
                        ] {
                            if group >= r1cs.matrix_group_count(side) {
                                continue;
                            }
                            let mut cached_row = usize::MAX;
                            let mut weight = F256::ZERO;
                            let visited = r1cs.for_each_matrix_group_entry(
                                side,
                                group,
                                |row, column, coefficient| {
                                    if row != cached_row {
                                        cached_row = row;
                                        weight = equality.value(offset + row);
                                    }
                                    if !weight.is_zero() {
                                        output[column as usize] += weight.scale_base(coefficient);
                                    }
                                },
                            );
                            assert!(visited);
                        }
                    }
                })
            }
        }
    }
}

fn prove_from_rows<Ch: Challenger>(
    rows: MatrixFoldRows<'_>,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
    let timing = std::env::var_os("NOIDH_MATRIX_FOLD_TIMING").is_some();
    let total_started = std::time::Instant::now();
    let storage = rows.storage_name();
    let gate_enabled = gate;
    let k_log = rows.k_log();
    let k_skip = rows.k_skip();
    let width = 1usize << k_log;
    assert_eq!(fresh.x_inner_rest.len(), k_log - k_skip);
    assert_eq!(fresh.r_inner_rest.len(), k_log - k_skip);
    assert_eq!(fresh.z_partial.len(), 1usize << k_skip);
    assert_eq!(incoming.point.len(), 2 * k_log + 1);

    let gate = if gate { F256::ONE } else { F256::ZERO };
    absorb_fold_header(channel, fresh, incoming, gate);
    let gamma = channel.sample_f256();
    let (incoming_row, incoming_column) = incoming.point.split_at(k_log + 1);

    let tables_started = std::time::Instant::now();
    let rest = build_eq_table(&fresh.r_inner_rest);
    let low_count = 1usize << k_skip;
    let mut v_table = vec![F256::ZERO; width];
    v_table
        .par_chunks_mut(low_count)
        .zip(rest)
        .for_each(|(chunk, rest_weight)| {
            for (output, &partial) in chunk.iter_mut().zip(&fresh.z_partial) {
                *output = partial * rest_weight;
            }
        });
    let e_c = build_eq_table(incoming_column);
    let tables_ms = tables_started.elapsed().as_millis();

    let row_images_started = std::time::Instant::now();
    let mut g_v = vec![F256::ZERO; 2 * width];
    let mut g_e = vec![F256::ZERO; 2 * width];
    rows.fill_row_images_c1(&v_table, &e_c, &mut g_v, &mut g_e);
    let row_images_ms = row_images_started.elapsed().as_millis();

    let phase1_weights_started = std::time::Instant::now();
    let lambda = lagrange_weights(k_skip, fresh.z_skip, 0);
    let inner = build_eq_table(&fresh.x_inner_rest);
    let u_base = build_u_base_weights(k_log, k_skip, &lambda, &inner);
    let gamma_gate = gamma * gate;
    let incoming_base = build_eq_table_scaled(&incoming_row[..k_log], gamma_gate);
    let incoming_stack_point = incoming_row[k_log];
    let phase1_weights_ms = phase1_weights_started.elapsed().as_millis();
    let phase1_target = fresh.value + gamma_gate * incoming.value;
    let phase1_started = std::time::Instant::now();
    let (phase1_rounds, rho, phase1_claim, phase1_finals) = run_phase_stacked(
        phase1_target,
        u_base,
        fresh.alpha,
        g_v,
        incoming_base,
        incoming_stack_point,
        g_e,
        channel,
    );
    let phase1_ms = phase1_started.elapsed().as_millis();
    let g_v = phase1_finals[1];
    let g_e = phase1_finals[3];
    debug_assert_eq!(
        phase1_finals[0] * g_v + phase1_finals[2] * g_e,
        phase1_claim
    );
    channel.observe_f256(g_v);
    channel.observe_f256(g_e);
    let delta = channel.sample_f256();

    let column_image_started = std::time::Instant::now();
    let equality = FactoredEqTable::new(&rho);
    let h = rows.weighted_column_image_c1(&equality);
    let column_image_ms = column_image_started.elapsed().as_millis();
    let delta_gate = delta * gate;
    let mut phase2_weights = v_table;
    let phase2_mix_started = std::time::Instant::now();
    let first_round =
        mix_and_round_coefficients_one_product(&mut phase2_weights, &e_c, &h, delta_gate);
    let phase2_mix_ms = phase2_mix_started.elapsed().as_millis();
    let phase2_target = g_v + delta_gate * g_e;
    let phase2_started = std::time::Instant::now();
    let (phase2_rounds, sigma, phase2_claim, phase2_finals) =
        run_phase_one_product(phase2_target, phase2_weights, h, Some(first_round), channel);
    let phase2_ms = phase2_started.elapsed().as_millis();
    let final_matrix_eval = phase2_finals[1];
    debug_assert_eq!(phase2_finals[0] * final_matrix_eval, phase2_claim);
    channel.observe_f256(final_matrix_eval);

    if timing {
        eprintln!(
            "[matrix-fold-c1] storage={storage} k_log={k_log} gate={gate_enabled} tables={tables_ms}ms row-images={row_images_ms}ms phase1-weights={phase1_weights_ms}ms phase1={phase1_ms}ms column-image={column_image_ms}ms phase2-mix={phase2_mix_ms}ms phase2-round0=fused phase2={phase2_ms}ms total={}ms",
            total_started.elapsed().as_millis()
        );
    }

    let mut point = rho;
    point.extend(sigma);
    (
        C1MatrixFoldProof {
            phase1_rounds,
            g_v,
            g_e,
            phase2_rounds,
            final_matrix_eval,
        },
        C1MatrixAccClaim {
            point,
            value: final_matrix_eval,
        },
    )
}

pub fn prove_matrix_claim_fold_c1<Ch: Challenger>(
    r1cs: &FieldR1cs,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
    prove_from_rows(
        MatrixFoldRows::Resident(r1cs),
        fresh,
        incoming,
        gate,
        channel,
    )
}

pub fn prove_matrix_claim_fold_compact_c1<Ch: Challenger>(
    r1cs: &CompactFieldR1cs,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: bool,
    channel: &mut Ch,
) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
    prove_from_rows(
        MatrixFoldRows::Compact(r1cs),
        fresh,
        incoming,
        gate,
        channel,
    )
}

pub fn verify_matrix_claim_fold_c1<Ch: Challenger>(
    k_log: usize,
    k_skip: usize,
    fresh: &C1FreshLincheckClaim,
    incoming: &C1MatrixAccClaim,
    gate: F256,
    proof: &C1MatrixFoldProof,
    channel: &mut Ch,
) -> Result<C1MatrixAccClaim, MatrixFoldError> {
    if fresh.x_inner_rest.len() != k_log - k_skip
        || fresh.r_inner_rest.len() != k_log - k_skip
        || fresh.z_partial.len() != 1usize << k_skip
        || incoming.point.len() != 2 * k_log + 1
        || proof.phase1_rounds.len() != k_log + 1
        || proof.phase2_rounds.len() != k_log
    {
        return Err(MatrixFoldError::Shape);
    }

    absorb_fold_header(channel, fresh, incoming, gate);
    let gamma = channel.sample_f256();
    let gamma_gate = gamma * gate;
    let (incoming_row, incoming_column) = incoming.point.split_at(k_log + 1);

    let mut claim = fresh.value + gamma_gate * incoming.value;
    let mut rho = Vec::with_capacity(k_log + 1);
    for wire in &proof.phase1_rounds {
        channel.observe_f256_slice(wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        rho.push(challenge);
    }
    let fresh_weight = u_weight_eval(fresh, k_skip, &rho);
    let incoming_weight = eq_points(incoming_row, &rho);
    if fresh_weight * proof.g_v + gamma_gate * incoming_weight * proof.g_e != claim {
        return Err(MatrixFoldError::FinalMismatch);
    }
    channel.observe_f256(proof.g_v);
    channel.observe_f256(proof.g_e);
    let delta = channel.sample_f256();
    let delta_gate = delta * gate;

    let mut claim = proof.g_v + delta_gate * proof.g_e;
    let mut sigma = Vec::with_capacity(k_log);
    for wire in &proof.phase2_rounds {
        channel.observe_f256_slice(wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        sigma.push(challenge);
    }
    let fresh_weight = v_weight_eval(fresh, k_skip, &sigma);
    let incoming_weight = eq_points(incoming_column, &sigma);
    if (fresh_weight + delta_gate * incoming_weight) * proof.final_matrix_eval != claim {
        return Err(MatrixFoldError::FinalMismatch);
    }
    channel.observe_f256(proof.final_matrix_eval);

    let mut point = rho;
    point.extend(sigma);
    Ok(C1MatrixAccClaim {
        point,
        value: proof.final_matrix_eval,
    })
}

// ---- Merge of two accumulated claims on one matrix (ClientBatchStep, B) ----
//
// The v1.5 fold above turns one *fresh* lincheck claim plus one accumulated
// claim into one accumulated claim. An aggregation tree also has to join two
// accumulators that both already hold accumulated claims on the same matrix
// (two subtrees of client proofs of one registry entry). The merge is the
// same two-phase sumcheck with both sides in accumulated (eq) form:
//
//   phase 1 (rows):    g_l·eq(l_r, ·) + γ·g_r·eq(r_r, ·) against the row images
//                      of eq(l_c, ·) and eq(r_c, ·) under the matrix;
//   phase 2 (columns): the matrix row at ρ against g_l·eq(l_c, ·) + δ·g_r·eq(r_c, ·).
//
// Each side is gated by its liveness (0 or 1). Only a merge of two live
// claims runs the sumcheck; its outgoing claim is `M~(ρ‖σ)`. With one live
// side the outgoing claim IS that side's claim, carried as it is (no
// sumcheck, no matrix); with none it is the zero claim. Either way the
// transcript absorbs the zero proof, so the recorded schedule does not
// depend on liveness, and the zero proof is the only one-sided proof the
// verifier accepts.

fn zero_merge_proof(k_log: usize) -> C1MatrixFoldProof {
    C1MatrixFoldProof {
        phase1_rounds: vec![[F256::ZERO; 2]; k_log + 1],
        g_v: F256::ZERO,
        g_e: F256::ZERO,
        phase2_rounds: vec![[F256::ZERO; 2]; k_log],
        final_matrix_eval: F256::ZERO,
    }
}

fn absorb_merge_header<Ch: Challenger>(
    channel: &mut Ch,
    left: &C1MatrixAccClaim,
    left_gate: F256,
    right: &C1MatrixAccClaim,
    right_gate: F256,
) {
    channel.observe_label(b"client-batch-matrix-claim-merge-c1");
    channel.observe_f256_slice(&left.point);
    channel.observe_f256(left.value);
    channel.observe_f256(left_gate);
    channel.observe_f256_slice(&right.point);
    channel.observe_f256(right.value);
    channel.observe_f256(right_gate);
}

/// Merge two accumulated claims on `r1cs` (`left` then `right`, each live or
/// dead) into one accumulated claim, proving it on `channel` (see above: a
/// one-sided merge carries its live claim and proves nothing).
pub fn prove_matrix_claim_merge_c1<Ch: Challenger>(
    r1cs: &FieldR1cs,
    left: &C1MatrixAccClaim,
    left_live: bool,
    right: &C1MatrixAccClaim,
    right_live: bool,
    channel: &mut Ch,
) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
    let timing = std::env::var_os("NOIDH_MATRIX_FOLD_TIMING").is_some();
    let started = std::time::Instant::now();
    let k_log = r1cs.k_log;
    let width = 1usize << k_log;
    assert_eq!(left.point.len(), 2 * k_log + 1, "left merge claim width");
    assert_eq!(right.point.len(), 2 * k_log + 1, "right merge claim width");
    let left_gate = if left_live { F256::ONE } else { F256::ZERO };
    let right_gate = if right_live { F256::ONE } else { F256::ZERO };
    if !(left_live && right_live) {
        let proof = zero_merge_proof(k_log);
        let carried = verify_matrix_claim_merge_c1(
            k_log, left, left_gate, right, right_gate, &proof, channel,
        )
        .expect("the zero proof carries a one-sided merge");
        return (proof, carried);
    }
    absorb_merge_header(channel, left, left_gate, right, right_gate);
    let gamma = channel.sample_f256();
    let (left_row, left_column) = left.point.split_at(k_log + 1);
    let (right_row, right_column) = right.point.split_at(k_log + 1);

    // Laps, printed under NOIDH_MATRIX_FOLD_TIMING.
    let mut lap = std::time::Instant::now();
    let mut laps: Vec<(&str, u128)> = Vec::new();
    let mut mark = |label: &'static str, laps: &mut Vec<(&str, u128)>| {
        if timing {
            laps.push((label, lap.elapsed().as_millis()));
            lap = std::time::Instant::now();
        }
    };
    // The support: rows `2^row_bits` (of A, then of B), columns `2^column_bits`.
    let (support_rows, support_columns) = matrix_support(r1cs);
    let row_bits = support_bits(support_rows).min(k_log);
    let column_bits = support_bits(support_columns).min(k_log);
    let rows = 1usize << row_bits;
    let columns = 1usize << column_bits;
    mark("support", &mut laps);

    // Row images of both column weights (eq on the support's columns, the
    // zero high column bits folded in).
    let left_column_weights = build_eq_table_scaled(
        &left_column[..column_bits],
        eq_at_zero(&left_column[column_bits..]),
    );
    let right_column_weights = build_eq_table_scaled(
        &right_column[..column_bits],
        eq_at_zero(&right_column[column_bits..]),
    );
    let mut g_left = vec![F256::ZERO; 2 * rows];
    let mut g_right = vec![F256::ZERO; 2 * rows];
    mark("tables", &mut laps);
    fill_row_images_on_support(
        r1cs,
        &left_column_weights,
        &right_column_weights,
        rows,
        &mut g_left,
        &mut g_right,
    );
    mark("row-images", &mut laps);

    // Phase 1 over the k_log + 1 row variables (the stack bit last): the
    // support's row bits, the zero high row bits (tail), the stack bit.
    let right_scale = gamma * right_gate;
    let held = |row_point: &[F256]| {
        row_point[..row_bits]
            .iter()
            .chain(std::iter::once(&row_point[k_log]))
            .copied()
            .collect::<Vec<_>>()
    };
    let left_row_weights = build_eq_table_scaled(&held(left_row), left_gate);
    let right_row_weights = build_eq_table_scaled(&held(right_row), right_scale);
    mark("phase1-weights", &mut laps);
    let phase1_target = left_gate * left.value + right_scale * right.value;
    // A false input claim leaves the running claim inconsistent with the
    // final evaluations: the verifier's phase-1 terminal check refuses it.
    let (phase1_rounds, rho, _, finals) = run_phase_on_support(
        phase1_target,
        row_bits,
        left_row_weights,
        &left_row[row_bits..k_log],
        g_left,
        right_row_weights,
        &right_row[row_bits..k_log],
        Some(g_right),
        channel,
    );
    mark("phase1", &mut laps);
    let g_v = finals[1];
    let g_e = finals[3];
    channel.observe_f256(g_v);
    channel.observe_f256(g_e);
    let delta = channel.sample_f256();

    // Phase 2 over the k_log column variables of the matrix row at ρ: the
    // support's column bits, then the zero high column bits (tail).
    let equality = FactoredEqTable::new(&rho);
    let row_at_rho = column_image_on_support(r1cs, &equality, width, support_rows, columns);
    mark("column-image", &mut laps);
    let right_delta = delta * right_gate;
    let left_weights = build_eq_table_scaled(&left_column[..column_bits], left_gate);
    let right_weights = build_eq_table_scaled(&right_column[..column_bits], right_delta);
    let phase2_target = left_gate * g_v + right_delta * g_e;
    let (phase2_rounds, sigma, _, phase2_finals) = run_phase_on_support(
        phase2_target,
        column_bits,
        left_weights,
        &left_column[column_bits..],
        row_at_rho,
        right_weights,
        &right_column[column_bits..],
        None,
        channel,
    );
    mark("phase2", &mut laps);
    let final_matrix_eval = phase2_finals[1];
    channel.observe_f256(final_matrix_eval);
    if timing {
        let laps = laps
            .iter()
            .map(|(label, ms)| format!("{label}={ms}ms"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!(
            "[matrix-merge-c1] k_log={k_log} live=({left_live},{right_live}) support rows={support_rows} columns={support_columns} {laps} total={}ms",
            started.elapsed().as_millis()
        );
    }

    let mut point = rho;
    point.extend(sigma);
    (
        C1MatrixFoldProof {
            phase1_rounds,
            g_v,
            g_e,
            phase2_rounds,
            final_matrix_eval,
        },
        C1MatrixAccClaim {
            point,
            value: final_matrix_eval,
        },
    )
}

/// Verifier of [`prove_matrix_claim_merge_c1`]: the outgoing accumulated
/// claim (merged, or carried when at most one side is live), or the round
/// that failed. `left_gate`/`right_gate` are 0 or 1. The sumcheck runs gated
/// by both gates at once, so a one-sided merge checks the zero proof.
#[allow(clippy::too_many_arguments)]
pub fn verify_matrix_claim_merge_c1<Ch: Challenger>(
    k_log: usize,
    left: &C1MatrixAccClaim,
    left_gate: F256,
    right: &C1MatrixAccClaim,
    right_gate: F256,
    proof: &C1MatrixFoldProof,
    channel: &mut Ch,
) -> Result<C1MatrixAccClaim, MatrixFoldError> {
    if left.point.len() != 2 * k_log + 1
        || right.point.len() != 2 * k_log + 1
        || proof.phase1_rounds.len() != k_log + 1
        || proof.phase2_rounds.len() != k_log
    {
        return Err(MatrixFoldError::Shape);
    }
    absorb_merge_header(channel, left, left_gate, right, right_gate);
    let gamma = channel.sample_f256();
    let both = left_gate * right_gate;
    let right_scale = gamma * both;
    let (left_row, left_column) = left.point.split_at(k_log + 1);
    let (right_row, right_column) = right.point.split_at(k_log + 1);

    let mut claim = both * left.value + right_scale * right.value;
    let mut rho = Vec::with_capacity(k_log + 1);
    for wire in &proof.phase1_rounds {
        channel.observe_f256_slice(wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        rho.push(challenge);
    }
    if both * eq_points(left_row, &rho) * proof.g_v
        + right_scale * eq_points(right_row, &rho) * proof.g_e
        != claim
    {
        return Err(MatrixFoldError::FinalMismatch);
    }
    channel.observe_f256(proof.g_v);
    channel.observe_f256(proof.g_e);
    let delta = channel.sample_f256();
    let right_delta = delta * both;

    let mut claim = both * proof.g_v + right_delta * proof.g_e;
    let mut sigma = Vec::with_capacity(k_log);
    for wire in &proof.phase2_rounds {
        channel.observe_f256_slice(wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        sigma.push(challenge);
    }
    let weight =
        both * eq_points(left_column, &sigma) + right_delta * eq_points(right_column, &sigma);
    if weight * proof.final_matrix_eval != claim {
        return Err(MatrixFoldError::FinalMismatch);
    }
    channel.observe_f256(proof.final_matrix_eval);

    if both.is_zero() {
        // At most one live side: the zero proof only (its rounds are bound by
        // the checks above; its row images and final evaluation here), and
        // the live claim carried as it is (the zero claim with none).
        if !(proof.g_v.is_zero() && proof.g_e.is_zero() && proof.final_matrix_eval.is_zero()) {
            return Err(MatrixFoldError::FinalMismatch);
        }
        return Ok(C1MatrixAccClaim {
            point: left
                .point
                .iter()
                .zip(&right.point)
                .map(|(&left_coordinate, &right_coordinate)| {
                    left_gate * left_coordinate + right_gate * right_coordinate
                })
                .collect(),
            value: left_gate * left.value + right_gate * right.value,
        });
    }

    let mut point = rho;
    point.extend(sigma);
    Ok(C1MatrixAccClaim {
        point,
        value: proof.final_matrix_eval,
    })
}

struct FactoredC1EqTable {
    low: Vec<F256>,
    high: Vec<F256>,
    low_bits: usize,
    low_mask: usize,
}

impl FactoredC1EqTable {
    fn new(point: &[F256]) -> Self {
        let low_bits = point.len() / 2;
        Self {
            low: build_eq_table(&point[..low_bits]),
            high: build_eq_table(&point[low_bits..]),
            low_bits,
            low_mask: (1usize << low_bits) - 1,
        }
    }

    #[inline(always)]
    fn value(&self, index: usize) -> F256 {
        self.low[index & self.low_mask] * self.high[index >> self.low_bits]
    }
}

struct CompactC1FreshWeights<'a> {
    claim: &'a C1FreshLincheckClaim,
    mask: usize,
    lambda: Vec<F256>,
    row_rest: FactoredC1EqTable,
    column_rest: FactoredC1EqTable,
}

impl<'a> CompactC1FreshWeights<'a> {
    fn new(shape: FieldShape, claim: &'a C1FreshLincheckClaim) -> Self {
        Self {
            claim,
            mask: (1usize << shape.k_skip) - 1,
            lambda: lagrange_weights(shape.k_skip, claim.z_skip, 0),
            row_rest: FactoredC1EqTable::new(&claim.x_inner_rest),
            column_rest: FactoredC1EqTable::new(&claim.r_inner_rest),
        }
    }

    fn row_weight(&self, row: usize, k_skip: usize) -> F256 {
        self.lambda[row & self.mask] * self.row_rest.value(row >> k_skip)
    }

    fn column_weight(&self, column: usize, k_skip: usize) -> F256 {
        self.claim.z_partial[column & self.mask] * self.column_rest.value(column >> k_skip)
    }
}

struct CompactC1AccumulatedWeights {
    stack: F256,
    row: FactoredC1EqTable,
    column: FactoredC1EqTable,
}

impl CompactC1AccumulatedWeights {
    fn new(shape: FieldShape, claim: &C1MatrixAccClaim) -> Self {
        let (row, column) = claim.point.split_at(shape.k_log + 1);
        Self {
            stack: row[shape.k_log],
            row: FactoredC1EqTable::new(&row[..shape.k_log]),
            column: FactoredC1EqTable::new(column),
        }
    }
}

fn compact_matrix_claim_values_c1(
    r1cs: &CompactFieldR1cs,
    fresh: Option<&C1FreshLincheckClaim>,
    accumulated: Option<&C1MatrixAccClaim>,
) -> (Option<F256>, Option<F256>) {
    let shape = r1cs.shape();
    let fresh_weights = fresh.map(|claim| CompactC1FreshWeights::new(shape, claim));
    let accumulated_weights =
        accumulated.map(|claim| CompactC1AccumulatedWeights::new(shape, claim));
    if fresh_weights.is_none() && accumulated_weights.is_none() {
        return (None, None);
    }

    let (fresh_total, accumulated_total) = [FieldR1csArtifactMatrix::A, FieldR1csArtifactMatrix::B]
        .into_par_iter()
        .map(|side| {
            let (mut fresh_side, mut accumulated_side) = (0..r1cs.matrix_group_count(side))
                .into_par_iter()
                .map(|group| {
                    let mut fresh_sum = F256::ZERO;
                    let mut accumulated_sum = F256::ZERO;
                    let mut cached_row = usize::MAX;
                    let mut fresh_row = F256::ZERO;
                    let mut accumulated_row = F256::ZERO;
                    let visited = r1cs.for_each_matrix_group_entry(
                        side,
                        group,
                        |row, column, coefficient| {
                            if row != cached_row {
                                cached_row = row;
                                if let Some(weights) = &fresh_weights {
                                    fresh_row = weights.row_weight(row, shape.k_skip);
                                }
                                if let Some(weights) = &accumulated_weights {
                                    accumulated_row = weights.row.value(row);
                                }
                            }
                            let column = column as usize;
                            if let Some(weights) = &fresh_weights {
                                fresh_sum += (fresh_row
                                    * weights.column_weight(column, shape.k_skip))
                                .scale_base(coefficient);
                            }
                            if let Some(weights) = &accumulated_weights {
                                accumulated_sum += (accumulated_row * weights.column.value(column))
                                    .scale_base(coefficient);
                            }
                        },
                    );
                    assert!(visited, "enumerated compact matrix group exists");
                    (fresh_sum, accumulated_sum)
                })
                .reduce(
                    || (F256::ZERO, F256::ZERO),
                    |left, right| (left.0 + right.0, left.1 + right.1),
                );
            if let Some(weights) = &fresh_weights {
                fresh_side *= match side {
                    FieldR1csArtifactMatrix::A => weights.claim.alpha,
                    FieldR1csArtifactMatrix::B => F256::ONE,
                };
            }
            if let Some(weights) = &accumulated_weights {
                accumulated_side *= match side {
                    FieldR1csArtifactMatrix::A => F256::ONE + weights.stack,
                    FieldR1csArtifactMatrix::B => weights.stack,
                };
            }
            (fresh_side, accumulated_side)
        })
        .reduce(
            || (F256::ZERO, F256::ZERO),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );
    (
        fresh.map(|_| fresh_total),
        accumulated.map(|_| accumulated_total),
    )
}

impl CompactFieldR1cs {
    pub fn evaluate_matrix_claims_c1_authenticated(
        &self,
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
    ) -> Result<AuthenticatedC1MatrixClaimEvaluations, FieldR1csArtifactError> {
        validate_claim_shapes(self.shape(), fresh, accumulated)?;
        let (fresh_value, accumulated_value) =
            compact_matrix_claim_values_c1(self, fresh, accumulated);
        Ok(AuthenticatedC1MatrixClaimEvaluations::new(
            self.statement_digest(),
            fresh,
            accumulated,
            fresh_value,
            accumulated_value,
        ))
    }
}

impl C1MatrixClaimEvaluator for CompactFieldR1cs {
    fn field_shape(&self) -> FieldShape {
        self.shape()
    }

    fn evaluate_matrix_claims_c1(
        &mut self,
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
    ) -> Result<AuthenticatedC1MatrixClaimEvaluations, FieldR1csArtifactError> {
        self.evaluate_matrix_claims_c1_authenticated(fresh, accumulated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsLaneChallenger;
    use crate::field_r1cs::{synthetic_satisfiable, SparseFieldMatrix};

    fn value(seed: u64) -> F256 {
        F256::new(
            F128::new(seed, seed.rotate_left(7)),
            F128::new(!seed, seed ^ 0xC1),
        )
    }

    #[test]
    fn factored_stacked_phase_matches_dense_phase() {
        for k_log in 1..=8 {
            let width = 1usize << k_log;
            let alpha = value(0x1000 + k_log as u64);
            let stack_point = value(0x2000 + k_log as u64);
            let u_base = (0..width)
                .map(|index| value(0x3000 + index as u64))
                .collect::<Vec<_>>();
            let incoming_base = (0..width)
                .map(|index| value(0x4000 + index as u64))
                .collect::<Vec<_>>();
            let g_v = (0..2 * width)
                .map(|index| value(0x5000 + index as u64))
                .collect::<Vec<_>>();
            let g_e = (0..2 * width)
                .map(|index| value(0x6000 + index as u64))
                .collect::<Vec<_>>();

            let mut u_dense = Vec::with_capacity(2 * width);
            u_dense.extend(u_base.iter().map(|&weight| alpha * weight));
            u_dense.extend_from_slice(&u_base);
            let mut incoming_dense = Vec::with_capacity(2 * width);
            incoming_dense.extend(
                incoming_base
                    .iter()
                    .map(|&weight| (F256::ONE + stack_point) * weight),
            );
            incoming_dense.extend(incoming_base.iter().map(|&weight| stack_point * weight));
            let claim = u_dense
                .iter()
                .zip(&g_v)
                .chain(incoming_dense.iter().zip(&g_e))
                .fold(F256::ZERO, |sum, (&weight, &entry)| sum + weight * entry);

            let mut dense_channel = FsLaneChallenger::new_c1(b"matrix-fold-c1-stacked-test");
            let dense = run_phase(
                claim,
                u_dense,
                g_v.clone(),
                incoming_dense,
                g_e.clone(),
                &mut dense_channel,
            );
            let mut factored_channel = FsLaneChallenger::new_c1(b"matrix-fold-c1-stacked-test");
            let factored = run_phase_stacked(
                claim,
                u_base,
                alpha,
                g_v,
                incoming_base,
                stack_point,
                g_e,
                &mut factored_channel,
            );
            assert_eq!(factored, dense);
            assert_eq!(factored_channel.sample_f256(), dense_channel.sample_f256());
        }
    }

    #[test]
    fn c1_matrix_fold_roundtrip_and_tamper() {
        let k_log = 8;
        let (mut relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_ACC);
        let mut fresh = C1FreshLincheckClaim {
            alpha: value(1),
            z_skip: value(2),
            x_inner_rest: (0..k_log - relation.k_skip)
                .map(|index| value(10 + index as u64))
                .collect(),
            r_inner_rest: (0..k_log - relation.k_skip)
                .map(|index| value(20 + index as u64))
                .collect(),
            z_partial: (0..1usize << relation.k_skip)
                .map(|index| value(100 + index as u64))
                .collect(),
            value: value(3),
        };
        fresh.value = fresh_claim_value_c1(&relation, &fresh);
        let mut incoming = C1MatrixAccClaim {
            point: (0..2 * k_log + 1)
                .map(|index| value(200 + index as u64))
                .collect(),
            value: value(4),
        };
        incoming.value = stacked_matrix_mle_eval_c1(&relation, &incoming);

        let shape = FieldShape::of(&relation);
        let digest = relation.structural_statement_digest();
        let mut artifact = Vec::new();
        relation
            .write_artifact(&mut artifact)
            .expect("C1 fixture has a canonical artifact");
        let compact = CompactFieldR1cs::open_packed(artifact.into_boxed_slice(), shape, digest)
            .expect("C1 fixture authenticates as compact");
        let expected = relation
            .evaluate_matrix_claims_c1(Some(&fresh), Some(&incoming))
            .expect("resident C1 claims evaluate");
        let actual = compact
            .evaluate_matrix_claims_c1_authenticated(Some(&fresh), Some(&incoming))
            .expect("compact C1 claims evaluate");
        assert_eq!(actual, expected);
        assert!(actual.is_bound_to(Some(&fresh), Some(&incoming)));

        let mut prover = FsLaneChallenger::new_c1(b"matrix-fold-c1-test");
        let (proof, output) =
            prove_matrix_claim_fold_c1(&relation, &fresh, &incoming, true, &mut prover);
        let mut verifier = FsLaneChallenger::new_c1(b"matrix-fold-c1-test");
        let verified = verify_matrix_claim_fold_c1(
            k_log,
            relation.k_skip,
            &fresh,
            &incoming,
            F256::ONE,
            &proof,
            &mut verifier,
        )
        .expect("honest C1 matrix fold");
        assert_eq!(verified, output);
        assert_eq!(prover.sample_f256(), verifier.sample_f256());

        let mut tampered = proof;
        tampered.phase1_rounds[0][0] += F256::ONE;
        let mut verifier = FsLaneChallenger::new_c1(b"matrix-fold-c1-test");
        assert!(
            verify_matrix_claim_fold_c1(
                k_log,
                relation.k_skip,
                &fresh,
                &incoming,
                F256::ONE,
                &tampered,
                &mut verifier,
            )
            .is_err()
        );
    }

    // ---- merge of two accumulated claims (ClientBatchStep prototype) ----

    const MERGE_DOMAIN: &[u8] = b"matrix-merge-c1-test";

    /// A true accumulated claim on `relation` at a point derived from `seed`.
    fn true_claim(relation: &FieldR1cs, seed: u64) -> C1MatrixAccClaim {
        let mut claim = C1MatrixAccClaim {
            point: (0..2 * relation.k_log + 1)
                .map(|index| value(seed + 7 * index as u64))
                .collect(),
            value: F256::ZERO,
        };
        claim.value = stacked_matrix_mle_eval_c1(relation, &claim);
        claim
    }

    fn gate(live: bool) -> F256 {
        if live {
            F256::ONE
        } else {
            F256::ZERO
        }
    }

    fn merge_and_verify(
        relation: &FieldR1cs,
        left: &C1MatrixAccClaim,
        left_live: bool,
        right: &C1MatrixAccClaim,
        right_live: bool,
    ) -> (C1MatrixFoldProof, C1MatrixAccClaim, Result<C1MatrixAccClaim, MatrixFoldError>) {
        let mut prover = FsLaneChallenger::new_c1(MERGE_DOMAIN);
        let (proof, output) =
            prove_matrix_claim_merge_c1(relation, left, left_live, right, right_live, &mut prover);
        let mut verifier = FsLaneChallenger::new_c1(MERGE_DOMAIN);
        let verified = verify_matrix_claim_merge_c1(
            relation.k_log,
            left,
            gate(left_live),
            right,
            gate(right_live),
            &proof,
            &mut verifier,
        );
        if verified.is_ok() {
            assert_eq!(
                prover.sample_f256(),
                verifier.sample_f256(),
                "merge prover/verifier transcript lockstep"
            );
        }
        (proof, output, verified)
    }

    /// Two true claims merge into one true claim, which the verifier derives
    /// exactly as the prover did.
    #[test]
    fn c1_matrix_merge_of_two_true_claims_is_true() {
        let k_log = 8;
        let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_3E46);
        let left = true_claim(&relation, 0x100);
        let right = true_claim(&relation, 0x900);
        let (_, output, verified) = merge_and_verify(&relation, &left, true, &right, true);
        let verified = verified.expect("honest merge verifies");
        assert_eq!(verified, output);
        assert_eq!(output.point.len(), 2 * k_log + 1);
        assert_eq!(
            stacked_matrix_mle_eval_c1(&relation, &output),
            output.value,
            "the merged claim holds on the matrix"
        );
        assert_ne!(output.point, left.point);
        assert_ne!(output.point, right.point);
    }

    /// A false claim on either side never yields a verified merge: the honest
    /// prover's rounds fail the verifier, and a merged claim that verifies
    /// with a false side is false itself (refused at evaluation).
    #[test]
    fn c1_matrix_merge_of_a_false_claim_is_refused() {
        let k_log = 8;
        let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_3E47);
        let honest = true_claim(&relation, 0x200);
        let mut forged = true_claim(&relation, 0x300);
        forged.value += F256::ONE;
        for (left, right) in [(&forged, &honest), (&honest, &forged)] {
            // The honest prover evaluates the matrix itself, so its outgoing
            // claim is true; what must fail is the verifier, which binds that
            // claim to the (false) inputs.
            let (_, _, verified) = merge_and_verify(&relation, left, true, right, true);
            assert!(
                verified.is_err(),
                "a merge of a false claim verified"
            );
        }
    }

    /// Liveness gates: a dead side (any value) contributes nothing, so one
    /// live true claim merges into a true claim; two dead sides verify with
    /// the all-zero proof and claim nothing.
    #[test]
    fn c1_matrix_merge_respects_liveness() {
        let k_log = 8;
        let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_3E48);
        let live = true_claim(&relation, 0x400);
        let mut garbage = true_claim(&relation, 0x500);
        garbage.value += value(0x77);
        for (left, left_live, right, right_live) in
            [(&live, true, &garbage, false), (&garbage, false, &live, true)]
        {
            let (_, output, verified) =
                merge_and_verify(&relation, left, left_live, right, right_live);
            assert_eq!(verified.expect("one live side verifies"), output);
            assert_eq!(stacked_matrix_mle_eval_c1(&relation, &output), output.value);
        }
        // Declaring the garbage side live is refused.
        let (_, _, verified) = merge_and_verify(&relation, &live, true, &garbage, true);
        assert!(verified.is_err(), "a false live side verified");

        let (proof, _, verified) = merge_and_verify(&relation, &garbage, false, &garbage, false);
        verified.expect("two dead sides verify");
        assert!(proof
            .phase1_rounds
            .iter()
            .chain(&proof.phase2_rounds)
            .all(|round| round == &[F256::ZERO; 2]));
    }

    /// The two-sided merge prover as first written: every table at the full
    /// width (`2^(k_log+1)` rows, `2^k_log` columns). The reference the
    /// support-restricted prover must equal, round for round.
    fn dense_two_sided_merge_reference<Ch: Challenger>(
        r1cs: &FieldR1cs,
        left: &C1MatrixAccClaim,
        right: &C1MatrixAccClaim,
        channel: &mut Ch,
    ) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
        let rows = MatrixFoldRows::Resident(r1cs);
        let k_log = r1cs.k_log;
        let width = 1usize << k_log;
        absorb_merge_header(channel, left, F256::ONE, right, F256::ONE);
        let gamma = channel.sample_f256();
        let (left_row, left_column) = left.point.split_at(k_log + 1);
        let (right_row, right_column) = right.point.split_at(k_log + 1);
        let left_column_weights = build_eq_table(left_column);
        let right_column_weights = build_eq_table(right_column);
        let mut g_left = vec![F256::ZERO; 2 * width];
        let mut g_right = vec![F256::ZERO; 2 * width];
        rows.fill_row_images_c1(
            &left_column_weights,
            &right_column_weights,
            &mut g_left,
            &mut g_right,
        );
        let left_row_weights = build_eq_table_scaled(left_row, F256::ONE);
        let right_row_weights = build_eq_table_scaled(right_row, gamma);
        let (phase1_rounds, rho, _, finals) = run_phase(
            left.value + gamma * right.value,
            left_row_weights,
            g_left,
            right_row_weights,
            g_right,
            channel,
        );
        let (g_v, g_e) = (finals[1], finals[3]);
        channel.observe_f256(g_v);
        channel.observe_f256(g_e);
        let delta = channel.sample_f256();
        let row_at_rho = rows.weighted_column_image_c1(&FactoredEqTable::new(&rho));
        let weights = left_column_weights
            .iter()
            .zip(&right_column_weights)
            .map(|(&left_weight, &right_weight)| left_weight + delta * right_weight)
            .collect::<Vec<_>>();
        let (phase2_rounds, sigma, _, phase2_finals) =
            run_phase_one_product(g_v + delta * g_e, weights, row_at_rho, None, channel);
        let final_matrix_eval = phase2_finals[1];
        channel.observe_f256(final_matrix_eval);
        let mut point = rho;
        point.extend(sigma);
        (
            C1MatrixFoldProof {
                phase1_rounds,
                g_v,
                g_e,
                phase2_rounds,
                final_matrix_eval,
            },
            C1MatrixAccClaim {
                point,
                value: final_matrix_eval,
            },
        )
    }

    /// A k_log instance whose A rows below `rows_a`, B rows below `rows_b`
    /// carry three entries each, in columns below `columns` (the rest empty):
    /// the support a merge prover may restrict itself to.
    fn supported_instance(
        k_log: usize,
        rows_a: usize,
        rows_b: usize,
        columns: usize,
        seed: u64,
    ) -> FieldR1cs {
        let k = 1usize << k_log;
        let matrix = |rows: usize, salt: u64| {
            SparseFieldMatrix::from_rows(
                k,
                (0..k)
                    .map(|row| {
                        if row >= rows {
                            return Vec::new();
                        }
                        (0..3u64)
                            .map(|entry| {
                                let mix = seed ^ salt ^ (row as u64).wrapping_mul(0x9E37) ^ entry;
                                (
                                    ((mix.wrapping_mul(0x2545_F491) >> 7) % columns as u64) as u32,
                                    F128::new(mix | 1, mix.rotate_left(17)),
                                )
                            })
                            .collect()
                    })
                    .collect(),
            )
        };
        FieldR1cs {
            m: k_log,
            k_log,
            k_skip: 6.min(k_log),
            useful_rows: rows_a.max(rows_b),
            a_0: matrix(rows_a, 0xA),
            b_0: matrix(rows_b, 0xB),
            const_pin: Some(0),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }

    /// The two-sided merge prover equals the dense reference — rounds, row
    /// images, final evaluation, outgoing claim, transcript — on matrices
    /// whose support is a fraction of the width (rows and columns apart), the
    /// whole width, a single entry, no entry at all, and on points whose
    /// coordinates above the support are 1 (an `eq` factor of zero there).
    #[test]
    fn c1_matrix_merge_prover_equals_the_dense_reference() {
        let k_log = 8;
        let cases: [(&str, usize, usize, usize); 6] = [
            ("rows and columns apart", 37, 90, 50),
            ("A wider than B", 200, 3, 129),
            ("whole width", 256, 256, 256),
            ("one entry row", 1, 0, 1),
            ("no entry", 0, 0, 1),
            ("a last row", 256, 2, 7),
        ];
        for (index, (name, rows_a, rows_b, columns)) in cases.into_iter().enumerate() {
            let relation = supported_instance(k_log, rows_a, rows_b, columns, 0x5A + index as u64);
            for ones_above in [false, true] {
                let mut left = true_claim(&relation, 0xC00 + 16 * index as u64);
                let mut right = true_claim(&relation, 0xD00 + 16 * index as u64);
                if ones_above {
                    // Coordinates above any support: row bit 7 and column bit 7.
                    left.point[7] = F256::ONE;
                    right.point[k_log + 1 + 7] = F256::ONE;
                    left.value = stacked_matrix_mle_eval_c1(&relation, &left);
                    right.value = stacked_matrix_mle_eval_c1(&relation, &right);
                }
                let mut reference = FsLaneChallenger::new_c1(MERGE_DOMAIN);
                let expected = dense_two_sided_merge_reference(&relation, &left, &right, &mut reference);
                let mut prover = FsLaneChallenger::new_c1(MERGE_DOMAIN);
                let got =
                    prove_matrix_claim_merge_c1(&relation, &left, true, &right, true, &mut prover);
                assert_eq!(got, expected, "{name} (ones above: {ones_above})");
                assert_eq!(prover.sample_f256(), reference.sample_f256(), "{name}: transcript");
                assert_eq!(stacked_matrix_mle_eval_c1(&relation, &got.1), got.1.value, "{name}");
            }
        }
    }

    /// One live side: no sumcheck. The outgoing claim IS the live side's
    /// claim, carried exactly whatever the dead side holds; the proof is the
    /// zero proof and the verifier derives the same claim; a one-sided proof
    /// with a row image or a final evaluation set is refused (the zero proof
    /// is the only one). Two dead sides carry the zero claim.
    #[test]
    fn c1_matrix_merge_carries_a_one_sided_claim() {
        let k_log = 8;
        let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_3E4A);
        let live = true_claim(&relation, 0xA00);
        let mut garbage = true_claim(&relation, 0xB00);
        garbage.value += value(0x99);
        let zero = C1MatrixFoldProof {
            phase1_rounds: vec![[F256::ZERO; 2]; k_log + 1],
            g_v: F256::ZERO,
            g_e: F256::ZERO,
            phase2_rounds: vec![[F256::ZERO; 2]; k_log],
            final_matrix_eval: F256::ZERO,
        };
        type Alter = fn(&mut C1MatrixFoldProof);
        let alterations: [(&str, Alter); 3] = [
            ("left row image", |proof| proof.g_v += F256::ONE),
            ("right row image", |proof| proof.g_e += F256::ONE),
            ("final evaluation", |proof| proof.final_matrix_eval += F256::ONE),
        ];
        for (left, left_live, right, right_live) in
            [(&live, true, &garbage, false), (&garbage, false, &live, true)]
        {
            let (proof, output, verified) =
                merge_and_verify(&relation, left, left_live, right, right_live);
            assert_eq!(output, live, "the live claim is carried as it is");
            assert_eq!(verified.expect("a one-sided merge verifies"), live);
            assert_eq!(proof, zero, "a one-sided merge proves nothing");
            for (name, alter) in alterations {
                let mut altered = proof.clone();
                alter(&mut altered);
                let mut verifier = FsLaneChallenger::new_c1(MERGE_DOMAIN);
                assert!(
                    verify_matrix_claim_merge_c1(
                        k_log,
                        left,
                        gate(left_live),
                        right,
                        gate(right_live),
                        &altered,
                        &mut verifier,
                    )
                    .is_err(),
                    "a one-sided merge with a {name} verified"
                );
            }
        }
        let (proof, output, verified) =
            merge_and_verify(&relation, &garbage, false, &garbage, false);
        assert_eq!(output, C1MatrixAccClaim::zero(k_log), "two dead sides carry nothing");
        assert_eq!(verified.expect("two dead sides verify"), output);
        assert_eq!(proof, zero);
    }

    /// The transcript binds the order and the inputs: a merge proof of
    /// (left, right) is not a proof of (right, left), nor of other inputs,
    /// and a tampered round is refused.
    #[test]
    fn c1_matrix_merge_is_bound_to_its_inputs_and_order() {
        let k_log = 8;
        let (relation, _) = synthetic_satisfiable(k_log, k_log, 0xC1_3E49);
        let left = true_claim(&relation, 0x600);
        let right = true_claim(&relation, 0x700);
        let other = true_claim(&relation, 0x800);
        let (proof, _, verified) = merge_and_verify(&relation, &left, true, &right, true);
        verified.expect("honest merge");
        let replay = |l: &C1MatrixAccClaim, r: &C1MatrixAccClaim, proof: &C1MatrixFoldProof| {
            let mut verifier = FsLaneChallenger::new_c1(MERGE_DOMAIN);
            verify_matrix_claim_merge_c1(k_log, l, F256::ONE, r, F256::ONE, proof, &mut verifier)
        };
        assert!(replay(&right, &left, &proof).is_err(), "swapped order verified");
        assert!(replay(&left, &other, &proof).is_err(), "replayed on another right side");
        assert!(replay(&other, &right, &proof).is_err(), "replayed on another left side");
        let mut tampered = proof.clone();
        tampered.phase2_rounds[k_log - 1][1] += F256::ONE;
        assert!(replay(&left, &right, &tampered).is_err(), "tampered round verified");
        let mut tampered = proof;
        tampered.final_matrix_eval += F256::ONE;
        assert!(replay(&left, &right, &tampered).is_err(), "tampered final verified");
    }
}
