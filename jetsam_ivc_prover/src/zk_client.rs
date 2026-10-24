// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Prover of the hiding client form (form Z, v1.6). The transcript
//! follows `prove_field_c1_with_public_io_and_post_commit_context` with the
//! form-Z statement binding, the Libra-masked zerocheck, the masked lincheck
//! and the two mask claims batched into the opening.

use crate::field_prover::FieldPostCommitProverContext;
use jetsam_ivc_core::challenger::Challenger;
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::lincheck;
use jetsam_ivc_core::pcs::{self, C1QuirkyDirectClaim, Commitment, PcsParams};
use jetsam_ivc_core::proof::{C1FieldR1csProof, C1R1csClaim, C1ZClaim, C1ZkFieldR1csProof};
use jetsam_ivc_core::public_io::{
    PublicIoSpec, assert_witness_matches_io, bind_post_commit_class, bind_public_io_c1,
};
use jetsam_ivc_core::zerocheck;
use jetsam_ivc_core::zk_form::{
    ZkFormError, ZkFormLayout, ZkFormRelation, ZkFreshRandomness, draw_zk_form_randomness, wipe,
};

/// One attempt's hiding randomness from the operating system's generator.
/// Panics if the system generator fails: never fall back to a
/// weaker source.
pub fn draw_zk_form_randomness_from_system(layout: ZkFormLayout) -> ZkFreshRandomness {
    draw_zk_form_randomness(layout, &mut |bytes: &mut [u8]| {
        getrandom::getrandom(bytes).expect("the system random generator")
    })
}

/// Prove a form-Z client statement. The randomness is consumed: a second
/// proof needs a second draw.
#[allow(clippy::too_many_arguments)]
pub fn prove_field_c1_zk_with_public_io_and_post_commit_context<Ch, Aux, PostCommit>(
    relation: &ZkFormRelation,
    logical_witness: &[F128],
    randomness: ZkFreshRandomness,
    pcs_params: &PcsParams,
    spec: &PublicIoSpec,
    io: &[F128],
    post_commit_class_digest: &[u8; 32],
    challenger: &mut Ch,
    post_commit: PostCommit,
) -> Result<(C1ZkFieldR1csProof, Aux, Commitment, C1R1csClaim), ZkFormError>
where
    Ch: Challenger,
    PostCommit: FnOnce(&mut FieldPostCommitProverContext<'_, Ch>) -> Aux,
{
    let layout = *relation.layout();
    let physical = relation.physical();
    let m = layout.m;
    if pcs_params.m != m + pcs::LOG_PACKING || pcs_params.log_batch_size != 5 {
        return Err(ZkFormError::Shape("PCS parameters"));
    }
    let mut z = relation.embed(logical_witness, randomness)?;
    let timing = std::env::var_os("JETSAM_ZK_PROVE_TIMING").is_some();
    let mut phase = std::time::Instant::now();
    let mut lap = |label: &str| {
        if timing {
            eprintln!("[zk-client-prover] {label}: {:.1} ms", phase.elapsed().as_secs_f64() * 1e3);
        }
        phase = std::time::Instant::now();
    };

    let (commitment, prover_data) = pcs::commit(&z, pcs_params);
    lap("commit");
    jetsam_ivc_core::proof::bind_statement_zk_form_c1(
        challenger,
        &layout,
        &physical.statement_digest(),
        &commitment,
    );
    assert_witness_matches_io(&z, spec, io);
    let io_claims = bind_public_io_c1(challenger, spec, io, m);
    bind_post_commit_class(challenger, post_commit_class_digest);
    let (auxiliary, (auxiliary_claims, auxiliary_c1_claims)) = {
        let mut context = FieldPostCommitProverContext::new(&z, &commitment, m, challenger);
        let auxiliary = post_commit(&mut context);
        (auxiliary, context.finish_c1())
    };

    let (mut a, mut b) = relation.apply_ab(&z);
    lap("apply A/B");
    let zerocheck_mask = layout.zerocheck_mask(&z);
    let (zerocheck_proof, kappa, zerocheck) =
        zerocheck::field_c1::prove_zk(&a, &b, &z, m, &zerocheck_mask, challenger);
    wipe(&mut a);
    wipe(&mut b);
    drop((a, b));
    lap("zerocheck");
    let zerocheck_claim = &zerocheck.claim;
    let rounds = layout.rounds();
    let lincheck_point = lincheck::c1::C1QuirkyPoint {
        z_skip: zerocheck_claim.z,
        x_inner_rest: zerocheck_claim.mlv_challenges[..rounds].to_vec(),
        x_outer: Vec::new(),
    };
    let lincheck_mask = layout.lincheck_mask(&z);
    let u = layout.lincheck_mask_point();
    let circuit = relation.lincheck_circuit();
    let (lincheck_proof, lincheck) = lincheck::c1::prove_field_zk(
        &z,
        m,
        jetsam_ivc_core::zk_form::ZK_FORM_K_SKIP,
        &circuit,
        &lincheck_point,
        &lincheck_mask,
        &u,
        challenger,
    );
    lap("lincheck");

    let ab = C1ZClaim {
        point: lincheck::c1::C1QuirkyPoint {
            z_skip: lincheck.claim.r_inner_skip,
            x_inner_rest: lincheck.claim.r_inner_rest.clone(),
            x_outer: Vec::new(),
        },
        value: lincheck.claim.w,
    };
    let c = C1ZClaim {
        point: lincheck::c1::C1QuirkyPoint {
            z_skip: zerocheck_claim.z,
            x_inner_rest: zerocheck_claim.r_rest[..rounds].to_vec(),
            x_outer: Vec::new(),
        },
        value: zerocheck_claim.c_eval,
    };
    let mut claims = vec![
        C1QuirkyDirectClaim {
            z_skip: ab.point.z_skip,
            k_skip: jetsam_ivc_core::zk_form::ZK_FORM_K_SKIP,
            x_rest: ab.point.x_inner_rest.clone(),
            value: ab.value,
        },
        C1QuirkyDirectClaim {
            z_skip: c.point.z_skip,
            k_skip: jetsam_ivc_core::zk_form::ZK_FORM_K_SKIP,
            x_rest: c.point.x_inner_rest.clone(),
            value: c.value,
        },
    ];
    claims.extend(io_claims);
    claims.extend(auxiliary_claims.into_iter().map(|claim| C1QuirkyDirectClaim {
        z_skip: F256::from_base(claim.z_skip),
        k_skip: claim.k_skip,
        x_rest: claim.x_rest.into_iter().map(F256::from_base).collect(),
        value: F256::from_base(claim.value),
    }));
    claims.extend(auxiliary_c1_claims);
    let sparse = [
        layout.zerocheck_mask_claim(
            zerocheck_claim.z,
            &zerocheck_claim.mlv_challenges,
            zerocheck.mask_value,
        ),
        layout.lincheck_mask_claim(&lincheck.claim.r_inner_rest, lincheck.mask_value),
    ];
    debug_assert_eq!(sparse[0].evaluate(&z), sparse[0].value);
    debug_assert_eq!(sparse[1].evaluate(&z), sparse[1].value);
    let pcs_open = pcs::open_batch_quirky_direct_c1_with_sparse(
        &z,
        &prover_data,
        &commitment,
        &claims,
        &sparse,
        challenger,
    );
    lap("PCS open");
    wipe(&mut z);
    drop(prover_data);

    Ok((
        C1ZkFieldR1csProof {
            base: C1FieldR1csProof {
                zerocheck: zerocheck_proof,
                lincheck: lincheck_proof,
                pcs_open,
            },
            zerocheck_kappa: kappa,
            lincheck_mask_value: lincheck.mask_value,
        },
        auxiliary,
        commitment,
        C1R1csClaim { ab, c },
    ))
}
