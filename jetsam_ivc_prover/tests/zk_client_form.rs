// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The properties of the hiding client form ("form Z") — simulator
//! certificates (finite-field lemmas, exact ranks, no chi^2 except the one
//! labelled smoke test), soundness of the additions, randomness, neutrality.
//!
//! Code:
//! `jetsam_ivc_core::zk_form`, `zerocheck::field_c1::{prove_zk, verify_zk}`,
//! `lincheck::c1::{prove_field_zk, verify_deferred_zk}`, the sparse opening
//! in `pcs`, `verifier::verify_field_c1_zk*`, `jetsam_ivc_prover::zk_client`.
//! Moved here from `jetsam_ivc_core/tests` (the tests need the prover).
//!
//! Layout of form Z (physical cells, m = 22): cell 0 the constant, cells
//! 8..16 the public IO; every cell whose bits 3 and 4 are set (lanes 24..32)
//! is fresh companion, lanes 24 and 25 reserved for the change of variables
//! onto the row-batched message; the coefficients < 256 of lanes 0..24 (cells
//! below 8192, except 0 and 8..16) are rank coins; one high pattern of lanes
//! 26..32 holds R_B (16 type-B cells), the zerocheck's Libra mask (111 F256
//! coefficients, 222 cells) and the lincheck's mask (48, 96 cells); the
//! useful circuit lives on lanes 0..24 from 8192 on (pi, base
//! 24). Every free cell is a tautology row (type A `z_i * 1 = z_i`, type B
//! `1 * z_i = z_i`), never materialized: closed forms on both sides.

#![allow(dead_code)]


// ---- form-Z fixtures (real prover and verifier) -------------------------------

mod zform {
    use jetsam_ivc_core::challenger::FsLaneChallenger;
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_ivc_core::pcs::{Commitment, PcsParams};
    use jetsam_ivc_core::proof::{C1R1csClaim, C1ZkFieldR1csProof};
    use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
    use jetsam_ivc_core::verifier::VerifyError;
    use jetsam_ivc_core::zk_form::*;

    pub const DOMAIN: &[u8] = b"JTM/E8/ZK-FORM-TEST";
    pub const CLASS: [u8; 32] = [0xE8; 32];

    pub struct Rng(pub u64);
    impl Rng {
        pub fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub fn f128(&mut self) -> F128 {
            F128::new(self.u64(), self.u64())
        }
        pub fn below(&mut self, n: usize) -> usize {
            (self.u64() % n as u64) as usize
        }
    }

    /// A small catalogue relation in the client form's conventions: cell 0
    /// the constant (pinned), cells 1..8 zero pads, cells 8..16 public IO
    /// (tautology rows), then `gates` gates z_c = (k z_a + z_b) z_d.
    pub fn logical_relation(m: usize, gates: usize, seed: u64) -> (FieldR1cs, Vec<F128>) {
        logical_relation_with_inputs(m, gates, seed, None)
    }

    /// With `inputs`, cells 1..8 are private inputs (tautology rows) the
    /// gates read; the matrix depends on `seed` only, the IO too.
    pub fn logical_relation_with_inputs(
        m: usize,
        gates: usize,
        seed: u64,
        inputs: Option<[F128; 7]>,
    ) -> (FieldR1cs, Vec<F128>) {
        let n = 1usize << m;
        assert!(16 + gates <= n);
        let mut rng = Rng(seed);
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
        if let Some(values) = inputs {
            for (index, value) in values.iter().enumerate() {
                let cell = 1 + index;
                z[cell] = *value;
                a_rows[cell].push((cell as u32, F128::ONE));
                b_rows[cell].push((0, F128::ONE));
            }
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
        let relation = FieldR1cs {
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
        assert!(relation.satisfies(&z));
        (relation, z)
    }

    pub fn io_spec() -> PublicIoSpec {
        PublicIoSpec {
            io_slice: WitnessSlice { log2_len: 3, index: 1 },
            io_len: 8,
            claims: Vec::new(),
        }
    }

    pub fn pcs_params(m: usize) -> PcsParams {
        PcsParams {
            m: m + jetsam_ivc_core::pcs::LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 5,
            profile: Default::default(),
        }
    }

    /// Deterministic randomness for reproducible tests (production draws
    /// from the system generator).
    pub fn seeded_randomness(layout: ZkFormLayout, seed: u64) -> ZkFreshRandomness {
        let mut rng = Rng(seed);
        draw_zk_form_randomness(layout, &mut |bytes: &mut [u8]| {
            for chunk in bytes.chunks_mut(8) {
                let v = rng.u64().to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
        })
    }

    pub struct Proved {
        pub proof: C1ZkFieldR1csProof,
        pub commitment: Commitment,
        pub io: Vec<F128>,
        pub claim: C1R1csClaim,
    }

    pub fn prove(relation: &ZkFormRelation, witness: &[F128], randomness: ZkFreshRandomness) -> Proved {
        let m = relation.layout().m;
        let io = witness[8..16].to_vec();
        let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
        let (proof, (), commitment, claim) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                relation,
                witness,
                randomness,
                &pcs_params(m),
                &io_spec(),
                &io,
                &CLASS,
                &mut challenger,
                |_| (),
            )
            .expect("form-Z proof");
        Proved { proof, commitment, io, claim }
    }

    pub fn verify(
        relation: &ZkFormRelation,
        proof: &C1ZkFieldR1csProof,
        commitment: &Commitment,
        io: &[F128],
    ) -> Result<C1R1csClaim, VerifyError> {
        let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
        jetsam_ivc_core::verifier::verify_field_c1_zk(
            relation,
            commitment,
            proof,
            &io_spec(),
            io,
            &CLASS,
            &mut challenger,
        )
    }

    /// Replay the verifier transcript up to the masked zerocheck.
    pub fn replay_zerocheck(
        relation: &ZkFormRelation,
        proved: &Proved,
    ) -> jetsam_ivc_core::zerocheck::field_c1::C1ZkZerocheckClaim {
        let m = relation.layout().m;
        let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
        jetsam_ivc_core::proof::bind_statement_zk_form_c1(
            &mut challenger,
            relation.layout(),
            &relation.physical().statement_digest(),
            &proved.commitment,
        );
        jetsam_ivc_core::public_io::bind_public_io_c1(&mut challenger, &io_spec(), &proved.io, m);
        jetsam_ivc_core::public_io::bind_post_commit_class(&mut challenger, &CLASS);
        jetsam_ivc_core::zerocheck::field_c1::verify_zk(
            m,
            &proved.proof.base.zerocheck,
            proved.proof.zerocheck_kappa,
            &mut challenger,
        )
        .expect("honest zerocheck")
    }

    /// The deferred verifier: claims and the stripped fresh claim.
    pub fn deferred(
        relation: &ZkFormRelation,
        proved: &Proved,
    ) -> (
        C1R1csClaim,
        jetsam_ivc_core::matrix_claim::c1::C1FreshLincheckClaim,
    ) {
        let physical = relation.physical();
        let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
        jetsam_ivc_core::verifier::verify_field_c1_zk_deferred_matrix_with_post_commit_context(
            relation.layout(),
            &jetsam_ivc_core::proof::FieldShape::of(physical),
            &physical.statement_digest(),
            &proved.commitment,
            &proved.proof,
            &io_spec(),
            &proved.io,
            &CLASS,
            &(),
            &mut challenger,
            |_, _| Ok(()),
        )
        .expect("deferred verify")
    }

    /// A reduced form-Z instance: layout m, catalogue of `gates` gates.
    pub fn instance(m: usize, gates: usize, seed: u64) -> (ZkFormRelation, Vec<F128>) {
        let (logical, witness) = logical_relation(m - 1, gates, seed);
        let relation = ZkFormRelation::from_logical(ZkFormLayout::reduced(m), &logical).expect("rename");
        (relation, witness)
    }
}

// ---- exact certificate machinery (finite-field linear algebra, no chi^2) -----

mod cert {
    use jetsam_ivc_core::field::{F128, F256, PHI_8_TABLE};
    use jetsam_ivc_core::ntt::AdditiveNttF128;
    use std::sync::OnceLock;

    /// Explicit certificate profiles. `PRODUCTION` is the
    /// client form (m = 22, rate 1/4, 133 queries). `REDUCED` and `TINY` keep
    /// the structure (32 lanes, 6-bit skip, rate 1/4, lanes 24..32 companion,
    /// coins in the low coefficients of the B lanes) with fewer rest bits and
    /// queries so exact ranks are cheap. Their results do NOT transfer to the
    /// production constants by themselves: the reduction lemmas do, and the
    /// production-size checks below exercise their hypotheses.
    #[derive(Clone, Copy, Debug)]
    pub struct CertProfile {
        pub name: &'static str,
        /// Rest (x) bits of a cell index: cells = 2^(6 + x_bits).
        pub x_bits: usize,
        /// T1 query count.
        pub queries: usize,
        /// Coins of a B lane: x < coin_x_end (coefficients < 2 coin_x_end).
        pub coin_x_end: usize,
        /// Lemma W: coefficient bits of the low part (2^w_low_bits > queries).
        pub w_low_bits: usize,
    }

    impl CertProfile {
        pub const fn coeff_bits(&self) -> usize {
            self.x_bits + 1
        }
        /// log2 of the per-lane codeword length (rate 1/4).
        pub const fn k_code(&self) -> usize {
            self.x_bits + 3
        }
        pub const fn high_bits(&self) -> usize {
            self.coeff_bits() - self.w_low_bits
        }
        pub const fn coin_count(&self) -> usize {
            2 * self.coin_x_end
        }
        /// The excluded high pattern holding g and R_B (lemma W).
        pub const fn g_pattern(&self) -> usize {
            (1 << self.high_bits()) - 1
        }
    }

    pub const PRODUCTION: CertProfile = CertProfile {
        name: "production",
        x_bits: 16,
        queries: 133,
        coin_x_end: 128,
        w_low_bits: 8,
    };
    pub const REDUCED: CertProfile = CertProfile {
        name: "reduced",
        x_bits: 9,
        queries: 13,
        coin_x_end: 16,
        w_low_bits: 4,
    };
    pub const TINY: CertProfile = CertProfile {
        name: "tiny",
        x_bits: 3,
        queries: 3,
        coin_x_end: 8,
        w_low_bits: 2,
    };

    /// Lanes of the companion; 24 and 25 are the change of variables onto M.
    pub const C_LANES: std::ops::Range<usize> = 24..32;
    pub const C_PRIME_LANES: std::ops::Range<usize> = 26..32;

    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed)
        }
        pub fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub fn below(&mut self, n: usize) -> usize {
            (self.u64() % n as u64) as usize
        }
        pub fn f128(&mut self) -> F128 {
            F128::new(self.u64(), self.u64())
        }
        /// A C1 Fiat-Shamir challenge, through the transcript's own map (its
        /// high coordinate has trace one: never in F128).
        pub fn challenge(&mut self) -> F256 {
            let lo = self.f128();
            let raw_hi = self.f128();
            F256::from_raw_challenge_lanes(lo, raw_hi)
        }
        pub fn challenges(&mut self, n: usize) -> Vec<F256> {
            (0..n).map(|_| self.challenge()).collect()
        }
        pub fn distinct(&mut self, count: usize, domain: usize) -> Vec<usize> {
            let mut out: Vec<usize> = Vec::with_capacity(count);
            while out.len() < count {
                let v = self.below(domain);
                if !out.contains(&v) {
                    out.push(v);
                }
            }
            out
        }
    }

    /// `eq(point, x)` for every x, bit i of x against point[i] (the transcript
    /// convention of `build_eq_table`).
    pub fn eq_table(point: &[F256]) -> Vec<F256> {
        let mut out = vec![F256::ONE];
        for &c in point {
            let one_plus = F256::ONE + c;
            let len = out.len();
            let mut next = vec![F256::ZERO; 2 * len];
            for i in 0..len {
                next[i] = out[i] * one_plus;
                next[i + len] = out[i] * c;
            }
            out = next;
        }
        out
    }

    pub fn eq1(a: F256, b: F256) -> F256 {
        a * b + (F256::ONE + a) * (F256::ONE + b)
    }

    /// Lagrange weights of the 64-point skip set at `z` (same nodes as the
    /// zerocheck: `PHI_8_TABLE[0..64]`).
    pub fn skip_lagrange(z: F256) -> Vec<F256> {
        (0..64)
            .map(|i| {
                let mut num = F256::ONE;
                let mut den = F128::ONE;
                for j in 0..64 {
                    if i != j {
                        num *= z + F256::from_base(PHI_8_TABLE[j]);
                        den *= PHI_8_TABLE[i] + PHI_8_TABLE[j];
                    }
                }
                num.scale_base(den.inv())
            })
            .collect()
    }

    /// Incremental row-echelon basis over F128 (exact Gaussian elimination).
    #[derive(Clone)]
    pub struct Echelon {
        pub ncols: usize,
        rows: Vec<(usize, Vec<F128>)>,
    }

    impl Echelon {
        pub fn new(ncols: usize) -> Self {
            Self {
                ncols,
                rows: Vec::new(),
            }
        }
        fn reduce(&self, row: &mut [F128]) {
            for (pivot, basis) in &self.rows {
                let factor = row[*pivot];
                if !factor.is_zero() {
                    for (a, &b) in row.iter_mut().zip(basis) {
                        if !b.is_zero() {
                            *a += factor * b;
                        }
                    }
                }
            }
        }
        /// Adds `row`; returns whether it raised the rank.
        pub fn insert(&mut self, mut row: Vec<F128>) -> bool {
            assert_eq!(row.len(), self.ncols);
            self.reduce(&mut row);
            match row.iter().position(|v| !v.is_zero()) {
                Some(pivot) => {
                    let inv = row[pivot].inv();
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                    self.rows.push((pivot, row));
                    true
                }
                None => false,
            }
        }
        pub fn is_independent(&self, row: &[F128]) -> bool {
            let mut row = row.to_vec();
            self.reduce(&mut row);
            row.iter().any(|v| !v.is_zero())
        }
        pub fn rank(&self) -> usize {
            self.rows.len()
        }
    }

    /// An F256 functional on F128 cells is two F128 functionals (lo, hi):
    /// `(w c).lo = w.lo c`, `(w c).hi = w.hi c` for c in F128.
    pub fn lo_hi(row: &[F256]) -> [Vec<F128>; 2] {
        [
            row.iter().map(|v| v.lo).collect(),
            row.iter().map(|v| v.hi).collect(),
        ]
    }

    /// `w[j][pos] = X_{2^j}(x_pos) = Ŵ_j(x_pos)`: the production additive NTT
    /// (LCH novel basis) applied to the unit coefficient vector e_{2^j}.
    pub fn w_hat(k_code: usize, bits: usize) -> Vec<Vec<F128>> {
        let ntt = AdditiveNttF128::standard(k_code);
        (0..bits)
            .map(|j| {
                let mut v = vec![F128::ZERO; 1 << k_code];
                v[1 << j] = F128::ONE;
                ntt.forward_transform(&mut v);
                v
            })
            .collect()
    }

    pub fn w_hat_of(p: &CertProfile) -> &'static Vec<Vec<F128>> {
        static PROD: OnceLock<Vec<Vec<F128>>> = OnceLock::new();
        static RED: OnceLock<Vec<Vec<F128>>> = OnceLock::new();
        static TIN: OnceLock<Vec<Vec<F128>>> = OnceLock::new();
        let cell = match p.name {
            "production" => &PROD,
            "reduced" => &RED,
            _ => &TIN,
        };
        cell.get_or_init(|| w_hat(p.k_code(), p.coeff_bits()))
    }

    /// `X_k(x_pos)` in the novel basis: product of `Ŵ_j(x_pos)` over the bits
    /// of k (checked against the NTT of e_k in the T1 test).
    pub fn novel(w: &[Vec<F128>], k: usize, pos: usize) -> F128 {
        let mut acc = F128::ONE;
        let (mut rest, mut j) = (k, 0);
        while rest != 0 {
            if rest & 1 == 1 {
                acc *= w[j][pos];
            }
            rest >>= 1;
            j += 1;
        }
        acc
    }

    /// Lemma A: the Boolean specialisation of a rest point. Bit 0 = the F256
    /// generator X (lo 0, hi 1), bits 1..cb-1 = `u`, the top coin bit = 1,
    /// every higher bit = 0. On the coins, eq(point, .) then charges exactly
    /// the two x values 2^(cb-1) + 2u + {0, 1} with weights (1+X, X).
    pub fn special_point(p: &CertProfile, u: usize) -> Vec<F256> {
        let cb = p.coin_x_end.trailing_zeros() as usize;
        assert!(u < 1 << (cb - 2));
        let mut point = vec![F256::ZERO; p.x_bits];
        point[0] = F256::new(F128::ZERO, F128::ONE);
        for b in 1..cb - 1 {
            if (u >> (b - 1)) & 1 == 1 {
                point[b] = F256::ONE;
            }
        }
        point[cb - 1] = F256::ONE;
        point
    }

    /// The two `u` of lemma A: the lowest one whose coordinates clear the
    /// query count, and the next.
    pub fn special_us(p: &CertProfile) -> (usize, usize) {
        let base = p.coin_x_end; // 2 * 2^(cb-1)
        let need = (p.queries + 1).saturating_sub(base);
        let u = need.div_ceil(4);
        (u, u + 1)
    }

    /// B-only observations of one B lane on its coins (coefficients
    /// < 2 coin_x_end; cell 64x + s sits at coefficient 2x + s/32 of lane
    /// s mod 32): round1_c[l], round1_c[l+32] at r_rest and z_partial[l],
    /// z_partial[l+32] at r_inner_rest.
    pub fn b_only_rows(p: &CertProfile, r_rest: &[F256], r_inner: &[F256]) -> Vec<Vec<F256>> {
        let mut rows = Vec::new();
        for point in [r_rest, r_inner] {
            assert_eq!(point.len(), p.x_bits);
            let eq = eq_table(point);
            for parity in 0..2 {
                let mut row = vec![F256::ZERO; p.coin_count()];
                for x in 0..p.coin_x_end {
                    row[2 * x + parity] = eq[x];
                }
                rows.push(row);
            }
        }
        rows
    }

    /// Rank over F128 of [B-only rows (lo/hi); T1 rows at Q] on the coins of
    /// one lane, restricted to `cols`; also returns the expected full rank.
    pub fn coin_joint_rank(
        p: &CertProfile,
        rows: &[Vec<F256>],
        q_set: &[usize],
        cols: &[usize],
    ) -> usize {
        let w = w_hat_of(p);
        let mut ech = Echelon::new(cols.len());
        for row in rows {
            for half in lo_hi(row) {
                ech.insert(cols.iter().map(|&k| half[k]).collect());
            }
        }
        for &pos in q_set {
            ech.insert(cols.iter().map(|&k| novel(w, k, pos)).collect());
        }
        ech.rank()
    }

    /// Lemma W hypothesis: the high-bit factors (coefficient bits
    /// w_low_bits.. = x bits w_low_bits-1..) of every linear observation of
    /// the companion lanes, split lo/hi, plus the unit vector of the g/R_B
    /// pattern. Observations: round1_c (r_rest), z_partial and the batched
    /// opening's ab claim (r_inner), final_a and the lincheck's matrix point
    /// (rho), the lincheck rounds binding high x bits (at_one, at_infinity);
    /// the c claim of the opening shares r_rest.
    pub fn high_factor_rows(
        p: &CertProfile,
        r_rest: &[F256],
        r_inner: &[F256],
        rho: &[F256],
    ) -> Vec<Vec<F128>> {
        let hb = p.high_bits();
        let x0 = p.w_low_bits - 1;
        let mut rows256 = vec![
            eq_table(&r_rest[x0..]),
            eq_table(&r_inner[x0..]),
            eq_table(&rho[x0..]),
        ];
        for t in x0..p.x_bits {
            let below = eq_table(&rho[x0..t]);
            let above = eq_table(&r_inner[t + 1..]);
            let shift = t - x0;
            for (v0, v1) in [(F256::ZERO, rho[t]), (F256::ONE, F256::ONE)] {
                let mut row = vec![F256::ZERO; 1 << hb];
                for (ia, &a) in above.iter().enumerate() {
                    for (bit, v) in [(0usize, v0), (1usize, v1)] {
                        for (ib, &b) in below.iter().enumerate() {
                            row[ib | (bit << shift) | (ia << (shift + 1))] = b * v * a;
                        }
                    }
                }
                rows256.push(row);
            }
        }
        let mut rows: Vec<Vec<F128>> = rows256.iter().flat_map(|r| lo_hi(r)).collect();
        let mut g = vec![F128::ZERO; 1 << hb];
        g[p.g_pattern()] = F128::ONE;
        rows.push(g);
        rows
    }

    /// `h(x_pos)[pattern] = X_{pattern << w_low_bits}(x_pos)`: the high factor
    /// of the T1 functional, `X_{k_low + 2^w p}(x) = X_{k_low}(x) h(x)[p]`.
    pub fn t1_high_vector(p: &CertProfile, pos: usize) -> Vec<F128> {
        let w = w_hat_of(p);
        (0..1usize << p.high_bits())
            .map(|pattern| novel(w, pattern << p.w_low_bits, pos))
            .collect()
    }

    /// Structural upper bound on dim_F128 span(H), valid for EVERY challenge:
    /// round t spans R_t = eq(rho)_{high < t} (x) F^2 (x) eq(r_inner)_{> t};
    /// consecutive R_t meet in one line, so sum_t R_t has F256-dimension
    /// <= rounds + 1 and contains eq(rho_high) and eq(r_inner_high); plus
    /// eq(r_rest_high): F256-dimension <= rounds + 2, F128-dimension of the
    /// lo/hi parts <= 2 (rounds + 2); plus the g/R_B unit vector.
    pub fn high_span_bound(p: &CertProfile) -> usize {
        let rounds = p.x_bits - (p.w_low_bits - 1);
        2 * (rounds + 2) + 1
    }

    /// Lemma W certificate: rank of H and, for every high coset c (positions
    /// with the same pos >> w_low_bits), h_c outside span(H). The rank at the
    /// drawn point must reach the structural bound, so it IS the generic rank
    /// and each (rank + 1)-minor found for [H; h_c] is a nonzero polynomial.
    pub fn lemma_w_certificate(p: &CertProfile, rng: &mut Rng) -> (usize, usize, usize) {
        let r_rest = rng.challenges(p.x_bits);
        let r_inner = rng.challenges(p.x_bits);
        let rho = rng.challenges(p.x_bits);
        let rows = high_factor_rows(p, &r_rest, &r_inner, &rho);
        let n_rows = rows.len();
        let mut ech = Echelon::new(1 << p.high_bits());
        for row in rows {
            ech.insert(row);
        }
        assert_eq!(ech.rank(), high_span_bound(p), "{}: rank of H below its structural bound", p.name);
        let cosets = 1usize << (p.k_code() - p.w_low_bits);
        for c in 0..cosets {
            let h = t1_high_vector(p, c << p.w_low_bits);
            assert!(
                ech.is_independent(&h),
                "{}: h_c of coset {c} lies in span(H)",
                p.name
            );
        }
        (n_rows, ech.rank(), cosets)
    }

    /// Every linear observation that touches lanes 24..32, as F256 rows over
    /// (lane 24..32, coefficient); cell 64x + s is (s mod 32, 2x + s/32).
    /// round1_c[s] and z_partial[s] for the 16 companion skip indices (those
    /// of lanes 24, 25 are the 16 F128 observations),
    /// final_a, and the 2 x_bits lincheck rounds. Type-A tautology rows give
    /// the lincheck combination alpha L_s(z) eq(rho, x) on companion cells.
    pub struct CompanionModel {
        pub rows: Vec<Vec<F256>>,
        pub ncoeff: usize,
    }

    pub fn companion_model(p: &CertProfile, rng: &mut Rng) -> CompanionModel {
        let ncoeff = 1usize << p.coeff_bits();
        let nx = 1usize << p.x_bits;
        let r_rest = rng.challenges(p.x_bits);
        let r_inner = rng.challenges(p.x_bits);
        let rho = rng.challenges(p.x_bits);
        let z = rng.challenge();
        let alpha = rng.challenge();
        let lag = skip_lagrange(z);
        let lanes = C_LANES.len();
        let at = |s: usize, x: usize| (s % 32 - C_LANES.start) * ncoeff + 2 * x + s / 32;
        let c_skips: Vec<usize> = (0..64).filter(|s| C_LANES.contains(&(s % 32))).collect();
        let mut rows = Vec::new();
        for point in [&r_rest, &r_inner] {
            let eq = eq_table(point);
            for &s in &c_skips {
                let mut row = vec![F256::ZERO; lanes * ncoeff];
                for x in 0..nx {
                    row[at(s, x)] = eq[x];
                }
                rows.push(row);
            }
        }
        let eq_rho = eq_table(&rho);
        let mut final_a = vec![F256::ZERO; lanes * ncoeff];
        for &s in &c_skips {
            for x in 0..nx {
                final_a[at(s, x)] = lag[s] * eq_rho[x];
            }
        }
        rows.push(final_a);
        for t in (0..p.x_bits).rev() {
            let mut bound = alpha;
            for j in t + 1..p.x_bits {
                bound *= eq1(rho[j], r_inner[j]);
            }
            let below = eq_table(&rho[..t]);
            let above = eq_table(&r_inner[t + 1..]);
            for (v0, v1) in [(F256::ZERO, rho[t]), (F256::ONE, F256::ONE)] {
                let mut row = vec![F256::ZERO; lanes * ncoeff];
                for x in 0..nx {
                    let low = x & ((1 << t) - 1);
                    let bit = (x >> t) & 1;
                    let high = x >> (t + 1);
                    let v = if bit == 1 { v1 } else { v0 };
                    let f = bound * below[low] * v * above[high];
                    for &s in &c_skips {
                        row[at(s, x)] = lag[s] * f;
                    }
                }
                rows.push(row);
            }
        }
        // Change of variables (CERT-B): msg_24, msg_25 = pi(M - sum_l w_l msg_l).
        let rb = rng.challenges(5);
        let w = eq_table(&rb);
        let (w24, w25) = (w[24], w[25]);
        let det = w24.lo * w25.hi + w25.lo * w24.hi;
        assert!(!det.is_zero(), "row-batch weights of lanes 24, 25 F128-dependent");
        let det_inv = det.inv();
        let pi = |v: F256| -> (F128, F128) {
            let a = (v.lo * w25.hi + w25.lo * v.hi) * det_inv;
            let b = (w24.lo * v.hi + v.lo * w24.hi) * det_inv;
            (a, b)
        };
        for row in rows.iter_mut() {
            for l in C_PRIME_LANES {
                let (a, b) = pi(w[l]);
                for k in 0..ncoeff {
                    let v24 = row[(24 - C_LANES.start) * ncoeff + k];
                    let v25 = row[(25 - C_LANES.start) * ncoeff + k];
                    row[(l - C_LANES.start) * ncoeff + k] += v24.scale_base(a) + v25.scale_base(b);
                }
            }
        }
        CompanionModel { rows, ncoeff }
    }

    /// Columns of lemma W's C' block: lanes 26..32, every coefficient outside
    /// the g/R_B pattern.
    pub fn companion_columns(p: &CertProfile, ncoeff: usize) -> Vec<(usize, usize)> {
        let mut cols = Vec::new();
        for l in C_PRIME_LANES {
            for k in 0..ncoeff {
                if k >> p.w_low_bits != p.g_pattern() {
                    cols.push((l, k));
                }
            }
        }
        cols
    }

    pub fn companion_base(p: &CertProfile, model: &CompanionModel) -> Echelon {
        let cols = companion_columns(p, model.ncoeff);
        let mut ech = Echelon::new(cols.len());
        for row in &model.rows {
            for half in lo_hi(row) {
                ech.insert(
                    cols.iter()
                        .map(|&(l, k)| half[(l - C_LANES.start) * model.ncoeff + k])
                        .collect(),
                );
            }
        }
        ech
    }

    /// Adds the T1 rows of lanes 26..32 at `q_set` to `base`; returns the rank.
    pub fn with_companion_t1(
        p: &CertProfile,
        model: &CompanionModel,
        base: &Echelon,
        q_set: &[usize],
    ) -> usize {
        let w = w_hat_of(p);
        let cols = companion_columns(p, model.ncoeff);
        let mut ech = base.clone();
        for &pos in q_set {
            for lane in C_PRIME_LANES {
                ech.insert(
                    cols.iter()
                        .map(|&(l, k)| if l == lane { novel(w, k, pos) } else { F128::ZERO })
                        .collect(),
                );
            }
        }
        ech.rank()
    }

    /// Circuit stand-ins for the useful region of the full model.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CircuitKind {
        /// Every useful row has random A and B entries on random useful cells.
        Random,
        /// Every useful row is zero padding (A = B = 0).
        Padding,
        /// Random rows on the first half of the useful region, padding after.
        HalfPadding,
    }

    /// Roles of the full model (reduced profile).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Role {
        Public,
        Coin,
        Useful,
        /// Lanes 24, 25: substituted through M.
        ChangeOfVariable,
        Companion,
        TypeB,
        Libra,
    }

    pub struct FullModel {
        pub roles: Vec<Role>,
        /// Linear observation rows over every cell (F256), M-substituted.
        pub rows: Vec<(&'static str, Vec<F256>)>,
    }

    /// The full linear observation map of a form-Z client proof at profile
    /// `p`: round1_c[s] and z_partial[s] for all 64 s, final_a, final_b, and
    /// the lincheck rounds computed from the real bucket sums (circuit
    /// weights included), then the change of variables of lanes 24, 25.
    pub fn full_model(p: &CertProfile, kind: CircuitKind, libra_cells: usize, rng: &mut Rng) -> FullModel {
        let n = 1usize << (6 + p.x_bits);
        let nx = 1usize << p.x_bits;
        let mut roles = vec![Role::Useful; n];
        let g_lo = p.g_pattern() << p.w_low_bits;
        let mut type_b = 0usize;
        let mut libra = 0usize;
        for (i, role) in roles.iter_mut().enumerate() {
            let lane = i & 31;
            let x = i >> 6;
            let k = i >> 5;
            *role = if i == 0 || (8..16).contains(&i) {
                Role::Public
            } else if lane >= 24 {
                if lane < 26 {
                    Role::ChangeOfVariable
                } else if lane == 26 && k >= g_lo && type_b < 16 {
                    type_b += 1;
                    Role::TypeB
                } else if lane >= 27 && k >= g_lo && libra < libra_cells {
                    libra += 1;
                    Role::Libra
                } else {
                    Role::Companion
                }
            } else if x < p.coin_x_end {
                Role::Coin
            } else {
                Role::Useful
            };
        }
        let r_rest = rng.challenges(p.x_bits);
        let r_inner = rng.challenges(p.x_bits);
        let rho = rng.challenges(p.x_bits);
        let z = rng.challenge();
        let alpha = rng.challenge();
        let lag = skip_lagrange(z);
        let eq_rho = eq_table(&rho);
        let rowweight = |i: usize| lag[i & 63] * eq_rho[i >> 6];
        // Column weights: wa[col] = sum_rows rowweight A, wb[col] = sum_rows rowweight B.
        let mut wa = vec![F256::ZERO; n];
        let mut wb = vec![F256::ZERO; n];
        let useful: Vec<usize> = (0..n).filter(|&i| roles[i] == Role::Useful).collect();
        let half = useful.len() / 2;
        for i in 0..n {
            let rw = rowweight(i);
            match roles[i] {
                Role::Public => {
                    wa[i] += rw;
                    wb[0] += rw;
                }
                Role::Coin | Role::Companion | Role::ChangeOfVariable | Role::Libra => {
                    wa[i] += rw;
                    wb[0] += rw;
                }
                Role::TypeB => {
                    wa[0] += rw;
                    wb[i] += rw;
                }
                Role::Useful => {
                    let active = match kind {
                        CircuitKind::Random => true,
                        CircuitKind::Padding => false,
                        CircuitKind::HalfPadding => i < useful[half],
                    };
                    if active {
                        for _ in 0..2 {
                            let col = useful[rng.below(useful.len())];
                            wa[col] += rw.scale_base(rng.f128());
                        }
                        wa[0] += rw.scale_base(rng.f128());
                        let col = useful[rng.below(useful.len())];
                        wb[col] += rw.scale_base(rng.f128());
                        wb[0] += rw.scale_base(rng.f128());
                    }
                }
            }
        }
        let w: Vec<F256> = (0..n).map(|i| alpha * wa[i] + wb[i]).collect();
        let mut rows: Vec<(&'static str, Vec<F256>)> = Vec::new();
        for (name, point) in [("round1_c", &r_rest), ("z_partial", &r_inner)] {
            let eq = eq_table(point);
            for s in 0..64 {
                let mut row = vec![F256::ZERO; n];
                for x in 0..nx {
                    row[64 * x + s] = eq[x];
                }
                rows.push((name, row));
            }
        }
        rows.push(("final_a", wa.clone()));
        rows.push(("final_b", wb.clone()));
        // Lincheck rounds, top x bit first; buckets (x_<=t, s) summed over
        // x_>t with eq(r_inner_>t).
        for t in (0..p.x_bits).rev() {
            let above = eq_table(&r_inner[t + 1..]);
            let bucket_len = 64usize << (t + 1);
            let mut comb = vec![F256::ZERO; bucket_len];
            for i in 0..n {
                let high = i >> (6 + t + 1);
                comb[i & (bucket_len - 1)] += above[high] * w[i];
            }
            let mut at_one = vec![F256::ZERO; n];
            let mut at_inf = vec![F256::ZERO; n];
            for i in 0..n {
                let b = i & (bucket_len - 1);
                let high = i >> (6 + t + 1);
                let bit = (i >> (6 + t)) & 1;
                let b1 = b | (1 << (6 + t));
                let b0 = b & !(1 << (6 + t));
                if bit == 1 {
                    at_one[i] = comb[b1] * above[high];
                }
                at_inf[i] = (comb[b1] + comb[b0]) * above[high];
            }
            rows.push(("lincheck_one", at_one));
            rows.push(("lincheck_inf", at_inf));
        }
        // Change of variables: msg_24, msg_25 = pi(M - sum_{l != 24,25} w_l msg_l).
        let rb = rng.challenges(5);
        let wl = eq_table(&rb);
        let (w24, w25) = (wl[24], wl[25]);
        let det_inv = (w24.lo * w25.hi + w25.lo * w24.hi).inv();
        let pi = |v: F256| -> (F128, F128) {
            (
                (v.lo * w25.hi + w25.lo * v.hi) * det_inv,
                (w24.lo * v.hi + v.lo * w24.hi) * det_inv,
            )
        };
        let pis: Vec<(F128, F128)> = (0..32).map(|l| pi(wl[l])).collect();
        for (_, row) in rows.iter_mut() {
            for k in 0..n >> 5 {
                let v24 = row[32 * k + 24];
                let v25 = row[32 * k + 25];
                if v24.is_zero() && v25.is_zero() {
                    continue;
                }
                for l in 0..32 {
                    if l == 24 || l == 25 {
                        continue;
                    }
                    let (a, b) = pis[l];
                    row[32 * k + l] += v24.scale_base(a) + v25.scale_base(b);
                }
            }
        }
        FullModel { roles, rows }
    }

    /// Rank over F128 of the full model's rows on the random columns
    /// (coins, companion 26..32, type-B), plus T1 rows of `t1_lanes` at Q.
    pub fn full_rank(p: &CertProfile, model: &FullModel, q_set: &[usize], with_t1: bool) -> (usize, usize) {
        let cols: Vec<usize> = (0..model.roles.len())
            .filter(|&i| matches!(model.roles[i], Role::Coin | Role::Companion | Role::TypeB))
            .collect();
        let mut ech = Echelon::new(cols.len());
        let mut count = 0;
        for (_, row) in &model.rows {
            for half in lo_hi(row) {
                ech.insert(cols.iter().map(|&i| half[i]).collect());
                count += 1;
            }
        }
        if with_t1 {
            let w = w_hat_of(p);
            for &pos in q_set {
                for lane in (0..24).chain(26..32) {
                    if pos == 0 && (lane == 0 || (8..16).contains(&lane)) {
                        continue;
                    }
                    ech.insert(
                        cols.iter()
                            .map(|&i| if i & 31 == lane { novel(w, i >> 5, pos) } else { F128::ZERO })
                            .collect(),
                    );
                    count += 1;
                }
            }
        }
        (ech.rank(), count)
    }

    /// Query sets: random ones and adversarial shapes (0 included, the first
    /// positions, one coset of the low part, equal low bits across cosets).
    pub fn query_sets(p: &CertProfile, extra: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        let q = p.queries + extra;
        let domain = 1usize << p.k_code();
        let low = 1usize << p.w_low_bits;
        let mut sets = vec![(0..q).collect::<Vec<_>>()];
        let mut with_zero = rng.distinct(q - 1, domain - 1).iter().map(|v| v + 1).collect::<Vec<_>>();
        with_zero.push(0);
        sets.push(with_zero);
        let coset = rng.below(domain / low);
        sets.push(rng.distinct(q, low).iter().map(|v| (coset * low) | v).collect());
        let fixed_low = rng.below(low);
        sets.push(rng.distinct(q, domain / low).iter().map(|c| (c * low) | fixed_low).collect());
        for _ in 0..2 {
            sets.push(rng.distinct(q, domain));
        }
        sets
    }
}

// ---- layout ---------------------------------------------------------------

#[test]
fn form_z_layout_partitions_the_cells_and_has_the_designed_capacity() {
    // Every physical cell has exactly one role; Useful = 3_139_584;
    // Companion = cells with bits 3 and 4 set, minus the reserved R_B and
    // mask cells; Coin = 6_135 cells of lanes 0..24 below 8192 except 0 and
    // 8..16; LibraMask (222), LincheckMask (96) and TypeB
    // (16) in lanes 26..32, in ONE high pattern (lemma W), never in the
    // change-of-variable lanes 24, 25. pi (base 24) is injective, monotone
    // and onto the useful cells.
    use jetsam_ivc_core::zk_form::*;
    use std::collections::HashMap;
    let layout = ZkFormLayout::CLIENT;
    layout.validate().unwrap();
    let mut counts: HashMap<ZkCellRole, usize> = HashMap::new();
    for cell in 0..layout.cells() {
        let role = layout.role(cell);
        *counts.entry(role).or_default() += 1;
        let lane = cell & 31;
        let companion_bits = (cell >> 3) & 3 == 3;
        match role {
            ZkCellRole::Companion => assert!(companion_bits),
            ZkCellRole::LibraMask | ZkCellRole::LincheckMask | ZkCellRole::TypeB => {
                assert!(ZK_RESERVED_LANES.contains(&lane));
                assert_eq!(cell >> 13, 511, "the reserved high pattern");
            }
            ZkCellRole::Coin => assert!(lane < 24 && cell < 8192),
            ZkCellRole::Useful => assert!(lane < 24 && cell >= 8192),
            ZkCellRole::Constant => assert_eq!(cell, 0),
            ZkCellRole::Io => assert!((8..16).contains(&cell)),
        }
    }
    let expected = [
        (ZkCellRole::Constant, 1),
        (ZkCellRole::Io, 8),
        (ZkCellRole::Coin, 6_135),
        (ZkCellRole::TypeB, 16),
        (ZkCellRole::LibraMask, 222),
        (ZkCellRole::LincheckMask, 96),
        (ZkCellRole::Companion, 1_048_576 - 334),
        (ZkCellRole::Useful, 3_139_584),
    ];
    for (role, count) in expected {
        assert_eq!(counts.get(&role).copied().unwrap_or(0), count, "{role:?}");
    }
    assert_eq!(layout.useful_capacity(), 3_139_584);
    assert_eq!(layout.free_cells(), 6_135 + 1_048_576);
    let mut seen = vec![false; layout.cells()];
    let mut last = 15usize;
    let mut hits = 0usize;
    for logical in (1..8).chain(16..16 + layout.useful_capacity() - 7) {
        let cell = layout.physical_cell(logical).expect("inside capacity");
        assert_eq!(layout.role(cell), ZkCellRole::Useful);
        assert!(!seen[cell] && cell > last, "pi injective and monotone");
        seen[cell] = true;
        last = cell;
        hits += 1;
    }
    assert_eq!(hits, layout.useful_capacity(), "pi onto the useful cells");
    assert_eq!(layout.physical_cell(16 + layout.useful_capacity() - 7), None);
    assert_eq!(layout.physical_cell(0), Some(0));
    for io in 8..16 {
        assert_eq!(layout.physical_cell(io), Some(io));
    }
    println!("[layout] client form: {expected:?}");
}

#[test]
fn the_renaming_preserves_satisfiability_and_c_equals_identity() {
    // A satisfied relation renamed by pi (rows and columns) stays satisfied
    // with the tautology rows (C = I), the constant pin stays at column 0,
    // the IO slice [8, 16) is unchanged, and every catalogue entry (r, c, v)
    // is the physical entry (pi r, pi c, v): the support is kept.
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::zk_form::*;
    for m in [12usize, 13] {
        let gates = 1000;
        let (relation, witness) = zform::instance(m, gates, m as u64);
        let (logical, _) = zform::logical_relation(m - 1, gates, m as u64);
        let layout = *relation.layout();
        let physical = relation.physical();
        let z = relation
            .embed(&witness, zform::seeded_randomness(layout, 3))
            .unwrap();
        assert!(relation.satisfies(&z), "m = {m}");
        assert_eq!(z[0], F128::ONE);
        assert_eq!(&z[8..16], &witness[8..16]);
        assert_eq!(physical.const_pin, Some(0));
        check_zk_form_registration(&layout, physical).unwrap();
        for (lm, pm) in [(&logical.a_0, &physical.a_0), (&logical.b_0, &physical.b_0)] {
            assert_eq!(lm.col_indices.len(), pm.col_indices.len(), "support kept");
            for row in 0..lm.num_rows {
                let mut expect: Vec<(u32, F128)> = lm
                    .row(row)
                    .map(|(c, v)| (layout.physical_cell(c as usize).unwrap() as u32, v))
                    .collect();
                if expect.is_empty() {
                    continue;
                }
                expect.sort_by_key(|e| e.0);
                let got: Vec<(u32, F128)> = pm.row(layout.physical_cell(row).unwrap()).collect();
                assert_eq!(got, expect, "row {row}");
            }
        }
        let mut broken = z.clone();
        broken[layout.physical_cell(16 + gates / 2).unwrap()] += F128::ONE;
        assert!(!relation.satisfies(&broken));
    }
}

#[test]
fn the_closed_form_tautologies_equal_the_materialized_rows() {
    // At m = 12, 13, 14: both closed forms (skip-Lagrange basis of a fresh
    // lincheck claim, and eq basis of an accumulated claim) equal the
    // materialized type-A / type-B rows, both sides, including the 16 R_B
    // rows (A in column 0, B on the diagonal), the 9 excluded cells (0 and
    // 8..16) of the coin cube, and the column-0 side (B_i = e_0 on type-A
    // rows, A_i = e_0 on type-B rows). The evaluators are the production
    // matrix-claim ones.
    use jetsam_ivc_core::field::F256;
    use jetsam_ivc_core::matrix_claim::c1::{
        C1FreshLincheckClaim, C1MatrixAccClaim, fresh_claim_value_c1, stacked_matrix_mle_eval_c1,
    };
    use jetsam_ivc_core::zk_form::ZkFormLayout;
    let mut rng = cert::Rng::new(0xE8C1);
    for m in [12usize, 13, 14] {
        let layout = ZkFormLayout::reduced(m);
        let taut = layout.materialized_tautologies();
        for _ in 0..3 {
            let rounds = m - 6;
            let claim = C1FreshLincheckClaim {
                alpha: rng.challenge(),
                z_skip: rng.challenge(),
                x_inner_rest: rng.challenges(rounds),
                r_inner_rest: rng.challenges(rounds),
                z_partial: rng.challenges(64),
                value: F256::ZERO,
            };
            let closed = layout.tautology_fresh_value(
                claim.alpha,
                claim.z_skip,
                &claim.x_inner_rest,
                &claim.r_inner_rest,
                &claim.z_partial,
            );
            assert_eq!(fresh_claim_value_c1(&taut, &claim), closed, "fresh basis, m = {m}");
            let point = rng.challenges(2 * m + 1);
            let (a, b) = layout.tautology_mle_eq(&point[..m], &point[m + 1..]);
            let t = point[m];
            let materialized = stacked_matrix_mle_eval_c1(
                &taut,
                &C1MatrixAccClaim {
                    point: point.clone(),
                    value: F256::ZERO,
                },
            );
            assert_eq!(materialized, (F256::ONE + t) * a + t * b, "eq basis, m = {m}");
        }
    }
}

// ---- simulator certificates (finite-field lemmas, not ROM) ----------------

/// Every 3-subset of the TINY domain (exhaustive query sets).
fn all_triples(domain: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for a in 0..domain {
        for b in a + 1..domain {
            for c in b + 1..domain {
                out.push(vec![a, b, c]);
            }
        }
    }
    out
}

#[test]
fn zk_coins_cover_every_t1_opening_of_a_b_lane() {
    // For every B lane and the <= 133 distinct positions of a real proof,
    // coins -> (Enc(coins)(x_q))_q is onto; position 0 shows only the
    // constant, the IO, or a coin.
    use cert::*;
    use jetsam_ivc_core::field::F128;
    let mut rng = Rng::new(0xE8A1);

    // The T1 functional is the LCH novel basis of the production NTT:
    // X_k = prod_{j in k} Ŵ_j, Ŵ_j(x_pos) depends on pos >> j only, and
    // position 0 sees coefficient 0 only.
    for p in [&TINY, &REDUCED] {
        let w = w_hat_of(p);
        let ntt = jetsam_ivc_core::ntt::AdditiveNttF128::standard(p.k_code());
        for _ in 0..6 {
            let k = rng.below(1 << p.coeff_bits());
            let mut v = vec![F128::ZERO; 1 << p.k_code()];
            v[k] = F128::ONE;
            ntt.forward_transform(&mut v);
            for (pos, &value) in v.iter().enumerate() {
                assert_eq!(value, novel(w, k, pos), "{}: X_{k} at {pos}", p.name);
            }
        }
        for (j, column) in w.iter().enumerate() {
            for pos in 0..1usize << p.k_code() {
                assert_eq!(column[pos], column[pos & !((1usize << j) - 1)]);
            }
        }
        for k in 0..1usize << p.coeff_bits() {
            let expected = if k == 0 { F128::ONE } else { F128::ZERO };
            assert_eq!(novel(w, k, 0), expected);
        }
    }
    // The PCS layout: cell i is lane i mod 32, coefficient i >> 5.
    {
        let p = &REDUCED;
        let m_cells = 6 + p.x_bits;
        let cells: Vec<F128> = (0..1usize << m_cells).map(|_| rng.f128()).collect();
        let params = jetsam_ivc_core::pcs::PcsParams {
            m: m_cells + jetsam_ivc_core::pcs::LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 5,
            profile: Default::default(),
        };
        let (_, data) = jetsam_ivc_core::pcs::commit(&cells, &params);
        let w = w_hat_of(p);
        for _ in 0..8 {
            let lane = rng.below(32);
            let pos = rng.below(1 << p.k_code());
            let mut acc = F128::ZERO;
            for k in 0..1usize << p.coeff_bits() {
                acc += cells[32 * k + lane] * novel(w, k, pos);
            }
            assert_eq!(data.codeword[pos * 32 + lane], acc);
        }
    }
    // Rank of the T1 openings on the coins: every 3-subset at TINY, the
    // random and adversarial query sets at PRODUCTION; the lanes whose
    // coefficient 0 is public (lane 0: constant; lanes 8..16: IO) lose
    // position 0 only.
    for (p, sets) in [
        (&TINY, all_triples(1 << TINY.k_code())),
        (&PRODUCTION, query_sets(&PRODUCTION, 0, &mut rng)),
    ] {
        for public_zero in [false, true] {
            let coins: Vec<usize> = (public_zero as usize..p.coin_count()).collect();
            for q in &sets {
                let seen = q.iter().filter(|&&pos| !(public_zero && pos == 0)).count();
                assert_eq!(coin_joint_rank(p, &[], q, &coins), seen, "{} {q:?}", p.name);
            }
        }
        println!("[cert-A/T1] {}: {} query sets x 2 lane kinds, rank = |Q| (minus position 0 on public lanes)", p.name, sets.len());
    }
}

#[test]
fn zk_coins_cover_every_b_only_observation() {
    // With the companion fixed, the coins are onto the B-only observations
    // jointly, FOR EVERY query set Q: B-lane T1
    // openings, round1_c and z_partial at skip indices of B lanes. (The u_0
    // of row-batch rounds 3 and 4 is determined by round1_c, z_partial and
    // the public IO.) Proof: Boolean specialisation of r_rest and
    // r_inner_rest onto coordinates J above the query count, exact.
    use cert::*;
    use std::collections::BTreeSet;
    let mut rng = Rng::new(0xE8A2);
    for p in [&TINY, &REDUCED, &PRODUCTION] {
        let (u_rest, u_inner) = special_us(p);
        let special = b_only_rows(p, &special_point(p, u_rest), &special_point(p, u_inner));
        // Lemma A, step 1: at the specialisation the 8 F128 rows are the
        // coordinate functionals of a set J above the query count.
        let mut j_set = BTreeSet::new();
        for row in &special {
            for half in lo_hi(row) {
                for (k, v) in half.iter().enumerate() {
                    if !v.is_zero() {
                        j_set.insert(k);
                    }
                }
            }
        }
        let j_cols: Vec<usize> = j_set.iter().copied().collect();
        assert_eq!(j_cols.len(), 8, "{}", p.name);
        assert!(j_cols.iter().all(|&k| k > p.queries && k < p.coin_count()));
        assert_eq!(coin_joint_rank(p, &special, &[], &j_cols), 8);
        let random = b_only_rows(p, &rng.challenges(p.x_bits), &rng.challenges(p.x_bits));
        let sets = if p.name == "tiny" {
            all_triples(1 << p.k_code())
        } else {
            query_sets(p, 0, &mut rng)
        };
        for public_zero in [false, true] {
            let first = public_zero as usize;
            let coins: Vec<usize> = (first..p.coin_count()).collect();
            for q in &sets {
                let seen: Vec<usize> = q.iter().copied().filter(|&pos| !(public_zero && pos == 0)).collect();
                let expected = 8 + seen.len();
                // Step 2: the explicit minor, columns [first, first+|Q|) + J,
                // is nonzero at the specialisation (block triangular).
                let mut minor: Vec<usize> = (first..first + seen.len()).collect();
                minor.extend(&j_cols);
                assert_eq!(coin_joint_rank(p, &special, &seen, &minor), expected, "{} minor {q:?}", p.name);
                // And the rank holds at real (C1-distributed) challenges.
                assert_eq!(coin_joint_rank(p, &random, &seen, &coins), expected, "{} {q:?}", p.name);
            }
        }
        println!(
            "[cert-A] {}: J = {j_cols:?} > {} queries; {} query sets x 2 lane kinds: rank = 8 + |Q| at the specialisation minor and at random challenges",
            p.name,
            p.queries,
            sets.len()
        );
    }
    // Schwartz-Zippel: per query set, the minor is a polynomial in the F128
    // coordinates of r_rest and r_inner_rest; 8 rows of degree <= 16 per B
    // lane, 24 B lanes: degree <= 3072. The high coordinate of a C1
    // challenge ranges over 2^127 values: error <= 3072 / 2^127 < 2^-115
    // for each Q, and Q (drawn after every challenge, fresh oracle output)
    // is independent of them.
    println!("[cert-A] Schwartz-Zippel per query set: <= 24*8*16 / 2^127 < 2^-115");
}

#[test]
fn zk_companion_t1_openings_are_masked_for_every_query_set() {
    // The same for the companion lanes 26..32 (lemma W): if the
    // T1 vector h_c of no high coset lies in the span H of the high-bit
    // factors of the linear observations, then for EVERY query set Q of at
    // most 2^8 positions the T1 openings of lanes 26..32 add exactly 6|Q| to
    // the rank of the linear observations. Exact: H certificate at the
    // production size, joint rank at the reduced profile.
    use cert::*;
    let mut rng = Rng::new(0xE8A3);
    let certificate = lemma_w_certificate(&PRODUCTION, &mut rng);
    println!("[cert-W] production: H = {} F128 rows, rank {}, {} high cosets all outside span(H)", certificate.0, certificate.1, certificate.2);
    assert_eq!(certificate, (43, 23, 2048));
    let certificate = lemma_w_certificate(&REDUCED, &mut rng);
    println!("[cert-W] reduced: H = {} F128 rows, rank {}, {} high cosets all outside span(H)", certificate.0, certificate.1, certificate.2);
    assert_eq!(certificate, (31, 17, 256));
    // The conclusion, checked exactly on the reduced profile.
    let model = companion_model(&REDUCED, &mut rng);
    let base = companion_base(&REDUCED, &model);
    println!("[cert-W] reduced: {} F128 linear rows on lanes 26..32, rank {}", 2 * model.rows.len(), base.rank());
    // 102 F128 rows, rank 66 = 2 x 33: round1_c and z_partial of lanes
    // 26..32 (12 + 12 F256) plus the lincheck/final_a span, which on
    // tautology cells is a chain of rank rounds + 1 = 10 meeting z_partial
    // in one line. Deficiency 36 = 16 (lanes 24/25: functionals of M,
    // the F128 observations) + 2 (the lincheck chain closes on z_partial) + 18
    // (lincheck combinations the companion does NOT mask).
    // Lemma W does not need this rank: it only adds 6|Q| to it.
    assert_eq!(base.rank(), 66);
    for q in query_sets(&REDUCED, 0, &mut rng) {
        let rank = with_companion_t1(&REDUCED, &model, &base, &q);
        assert_eq!(rank, base.rank() + 6 * q.len(), "{q:?}");
    }
    println!("[cert-W] reduced: every query set adds exactly 6|Q|");
}

#[test]
fn zk_unopened_t1_leaves_keep_companion_entropy() {
    // For every opened set Q and every unopened position x, the six symbols
    // of lanes 26..32 at x stay uniform (768 bits) given every observation:
    // lemma W applied to Q + {x} (at most 134 <= 2^8 positions).
    use cert::*;
    let mut rng = Rng::new(0xE8A4);
    assert_eq!(lemma_w_certificate(&PRODUCTION, &mut rng), (43, 23, 2048));
    let model = companion_model(&REDUCED, &mut rng);
    let base = companion_base(&REDUCED, &model);
    let domain = 1usize << REDUCED.k_code();
    let low = 1usize << REDUCED.w_low_bits;
    for q in query_sets(&REDUCED, 0, &mut rng).into_iter().take(4) {
        let same_coset = (q[1] & !(low - 1)) | ((q[1] + 1) % low);
        let mut candidates = vec![same_coset, rng.below(domain)];
        if !q.contains(&0) {
            candidates.push(0);
        }
        for x in candidates {
            if q.contains(&x) {
                continue;
            }
            let mut opened = q.clone();
            opened.push(x);
            let rank = with_companion_t1(&REDUCED, &model, &base, &opened);
            assert_eq!(rank, base.rank() + 6 * opened.len(), "{q:?} + {x}");
        }
    }
    println!("[cert-G] the 6 companion symbols of every unopened leaf stay free: 6 x 128 = 768 bits");
}

#[test]
fn zk_companion_lanes_are_a_bijection_onto_the_row_batched_message() {
    // Two companion lanes with F128-independent row-batch weights map onto
    // the post-row-batch message M (rank 2^18 over F128); one lane alone
    // reaches only 2^17 (failure probability: w25/w24 in F128 iff r_0 in
    // F128, 2^-128; no abort, no redraw). Relations on M:
    // the 16 F128 observations round1_c[s], z_partial[s]
    // for s in {24, 25, 56, 57} equal psi_M(M) plus a combination of the
    // OBSERVED round1_c / z_partial of the other lanes: 16 F128 functionals
    // of M, rank 16. <M, b'> = v' adds nothing for the (ab, c) claims (in
    // their span) and 2 more with the IO / sparse cells of b': rank <= 18,
    // all computable from the transcript. M is the last block:
    // the simulator samples M on that affine subspace and computes the
    // whole BaseFold transcript (T2, epochs, tail, rounds) from it.
    use cert::*;
    use jetsam_ivc_core::field::{F128, F256};
    let mut rng = Rng::new(0xE8B1);
    for _ in 0..256 {
        let w = eq_table(&rng.challenges(5));
        let det = w[24].lo * w[25].hi + w[25].lo * w[24].hi;
        assert!(!det.is_zero(), "(C24, C25) -> M is a bijection per coefficient");
        // One lane: c -> w24 c has F128 rank 1 per coefficient (2^17 in all).
        assert!(!w[24].hi.is_zero());
    }
    // The 18 functionals at production size (columns: M_lo[k], M_hi[k]).
    let p = &PRODUCTION;
    let ncoeff = 1usize << p.coeff_bits();
    let rb = rng.challenges(5);
    let w = eq_table(&rb);
    let (w24, w25) = (w[24], w[25]);
    let det_inv = (w24.lo * w25.hi + w25.lo * w24.hi).inv();
    // pi_24(v) = a24.lo v.lo + a24.hi v.hi (as F128 coefficients), same for 25.
    let pi24 = (w25.hi * det_inv, w25.lo * det_inv);
    let pi25 = (w24.hi * det_inv, w24.lo * det_inv);
    let r_rest = rng.challenges(p.x_bits);
    let r_inner = rng.challenges(p.x_bits);
    let mut rows: Vec<Vec<F128>> = Vec::new();
    for point in [&r_rest, &r_inner] {
        let eq = eq_table(point);
        for (coefficients, parity) in [(pi24, 0usize), (pi24, 1), (pi25, 0), (pi25, 1)] {
            for half in 0..2 {
                let mut row = vec![F128::ZERO; 2 * ncoeff];
                for (x, &e) in eq.iter().enumerate() {
                    let k = 2 * x + parity;
                    let e = if half == 0 { e.lo } else { e.hi };
                    row[2 * k] = e * coefficients.0;
                    row[2 * k + 1] = e * coefficients.1;
                }
                rows.push(row);
            }
        }
    }
    // b' = sum_l w_l b(32 k + l), b = gamma_ab L(z_ab) eq(r_inner) + gamma_c L(z) eq(r_rest).
    let (gamma_ab, gamma_c) = (rng.challenge(), rng.challenge());
    let lag_ab = skip_lagrange(rng.challenge());
    let lag_c = skip_lagrange(rng.challenge());
    let eq_ab = eq_table(&r_inner);
    let eq_c = eq_table(&r_rest);
    let mut b_folded = vec![F256::ZERO; ncoeff];
    let _ = ();
    for (k, slot) in b_folded.iter_mut().enumerate() {
        let x = k >> 1;
        for l in 0..32 {
            let s = l + 32 * (k & 1);
            *slot += w[l] * (gamma_ab * lag_ab[s] * eq_ab[x] + gamma_c * lag_c[s] * eq_c[x]);
        }
    }
    // The (ab, c) part of <M, b'> is an F256 combination of the psi sums
    // (sum_x eq M[2x + p] = w24 psi_24 + w25 psi_25): in their span. The IO
    // and sparse claims add a few cells of b' outside that structure.
    let check_in_span = b_folded.clone();
    let reserved_k = ncoeff - 3;
    let x_gen = jetsam_ivc_core::zk_form::F256_X;
    let functional = |b_folded: &[F256], half: usize| {
        let mut row = vec![F128::ZERO; 2 * ncoeff];
        for (k, &b) in b_folded.iter().enumerate() {
            let (on_lo, on_hi) = (b, x_gen * b);
            row[2 * k] = if half == 0 { on_lo.lo } else { on_lo.hi };
            row[2 * k + 1] = if half == 0 { on_hi.lo } else { on_hi.hi };
        }
        row
    };
    {
        let mut psi = Echelon::new(2 * ncoeff);
        for row in &rows {
            psi.insert(row.clone());
        }
        assert_eq!(psi.rank(), 16, "the 16 observations of lanes 24, 25");
        for half in 0..2 {
            assert!(!psi.is_independent(&functional(&check_in_span, half)), "<M, b'> of (ab, c) in the span");
        }
    }
    b_folded[reserved_k] += rng.challenge();
    for half in 0..2 {
        rows.push(functional(&b_folded, half));
    }
    let mut ech = Echelon::new(2 * ncoeff);
    for row in rows {
        ech.insert(row);
    }
    println!("[cert-B] production: 18 F128 functionals of M (2^18 F128 coordinates), rank {}", ech.rank());
    assert_eq!(ech.rank(), 18);
}

#[test]
fn a_high_half_companion_leaks_on_the_vanishing_subspace() {
    // The rejected layout: a companion in the high message half of each lane
    // (coefficient bit 16 set) enters the codeword through Ŵ_16, which
    // vanishes on V_16: at 1/8 of the positions every T1 opening shows the
    // B part with no companion term (kept for capacity
    // reasons).
    let w = cert::w_hat_of(&cert::PRODUCTION);
    assert!((0..1usize << 16).all(|pos| w[16][pos].is_zero()));
    assert!((1usize << 16..1usize << 19).all(|pos| !w[16][pos].is_zero()));
    println!("[cert-B neg] Ŵ_16 = 0 on 2^16 of the 2^19 positions");
}

#[test]
fn zk_joint_rank_at_test_scale() {
    // With the lincheck masked, the linear observations
    // (round1_c and z_partial at the 64 skip indices, final_a, final_b) and
    // the T1 openings of the B and companion lanes, on every random cell
    // (coins, lanes 26..32, R_B; lanes 24/25 through M, the masks in their
    // own Libra blocks) have rank = rows - 16: exactly the lanes-24/25
    // identities, for a random circuit, a half-padding and a padding
    // circuit. Without the mask, the padding circuit loses one more
    // dimension: that is what the lincheck mask removes.
    use cert::*;
    let profile = CertProfile {
        name: "reduced",
        x_bits: 9,
        queries: 13,
        coin_x_end: 16,
        w_low_bits: 4,
    };
    let mut rng = Rng::new(0xE8C3);
    let q = query_sets(&profile, 0, &mut rng).pop().unwrap();
    for (kind, with_t1) in [
        (CircuitKind::Random, true),
        (CircuitKind::HalfPadding, false),
        (CircuitKind::Padding, false),
    ] {
        let mut model = full_model(&profile, kind, 32, &mut rng);
        let (rank_raw, rows_raw) = full_rank(&profile, &model, &[], false);
        model.rows.retain(|(name, _)| !name.starts_with("lincheck"));
        let (rank, rows) = full_rank(&profile, &model, &q, with_t1);
        println!(
            "[cert-J] {kind:?}: masked block {rows} rows rank {rank} (deficiency {}); unmasked lincheck: deficiency {}",
            rows - rank,
            rows_raw - rank_raw
        );
        assert_eq!(rows - rank, 16, "{kind:?}");
        if kind == CircuitKind::Padding {
            assert_eq!(rows_raw - rank_raw, 19);
        }
    }
}

#[test]
fn zk_joint_rank_production_reduction() {
    // m = 22: the joint rank reduces, block by block, to
    // Q-independent production-size facts. (1) lemma A (pieces, every Q):
    // `zk_coins_cover_every_b_only_observation`; (2) lemma W (lanes 26..32,
    // every Q): `zk_companion_t1_openings_are_masked_for_every_query_set`;
    // (3) here: the 25 F256 linear observations on lanes 26..32 (round1_c,
    // z_partial at their 12 skip indices, final_a) have F128 rank 50 already
    // on a column subset; (4) the Libra blocks (CERT-D', CERT-D''), (5) R_B
    // (CERT-E), (6) M (CERT-B) have their own tests. The lambda and lambda'
    // limits make the joint determinant the product of the block minors.
    use cert::*;
    use jetsam_ivc_core::field::{F128, F256};
    let p = &PRODUCTION;
    let mut rng = Rng::new(0xE8C4);
    let eq_rest = eq_table(&rng.challenges(p.x_bits));
    let eq_inner = eq_table(&rng.challenges(p.x_bits));
    let eq_rho = eq_table(&rng.challenges(p.x_bits));
    let lag = skip_lagrange(rng.challenge());
    let x_end = 1usize << 11;
    let col = |lane: usize, k: usize| (lane - 26) * 2 * x_end + k;
    let ncols = 6 * 2 * x_end;
    let skips: Vec<usize> = (0..64).filter(|s| (26..32).contains(&(s % 32))).collect();
    let mut rows: Vec<Vec<F256>> = Vec::new();
    for eq in [&eq_rest, &eq_inner] {
        for &s in &skips {
            let mut row = vec![F256::ZERO; ncols];
            for x in 0..x_end {
                row[col(s % 32, 2 * x + s / 32)] = eq[x];
            }
            rows.push(row);
        }
    }
    let mut final_a = vec![F256::ZERO; ncols];
    for &s in &skips {
        for x in 0..x_end {
            final_a[col(s % 32, 2 * x + s / 32)] = lag[s] * eq_rho[x];
        }
    }
    rows.push(final_a);
    let mut ech = Echelon::new(ncols);
    for row in &rows {
        for half in lo_hi(row) {
            ech.insert(half);
        }
    }
    println!("[cert-J prod] lanes 26..32: 25 F256 observations, F128 rank {} on {ncols} columns", ech.rank());
    assert_eq!(ech.rank(), 50);
    let _ = F128::ZERO;
    println!(
        "[cert-J prod] error budget per proof: lemma A 2^-115 (per Q), lemma W 2^-108, CERT-B 2^-128, \
         Libra blocks and joint minor: degree <= rows x max entry degree (~2^15) / 2^127 ~ 2^-112"
    );
}

/// Rows of a linear map given on F256 coefficients, as F128 rows on the
/// (lo, hi) cells of each coefficient: `w c = w c_lo + (w X) c_hi`.
fn f128_rows_of(observations: &[Vec<jetsam_ivc_core::field::F256>]) -> Vec<Vec<jetsam_ivc_core::field::F128>> {
    let x = jetsam_ivc_core::zk_form::F256_X;
    let mut out = Vec::new();
    for obs in observations {
        for half in 0..2 {
            let mut row = Vec::with_capacity(2 * obs.len());
            for &w in obs {
                let (a, b) = (w, w * x);
                row.push(if half == 0 { a.lo } else { a.hi });
                row.push(if half == 0 { b.lo } else { b.hi });
            }
            out.push(row);
        }
    }
    out
}

#[test]
fn zk_libra_mask_observation_rank() {
    // CERT-D' at m = 22 (lambda limit): the zerocheck mask's observations
    // are kappa, the 64 skip-round values Z_S(l_j) h(l_j) + kappa, and the
    // 16 x 2 round values; over the 111 F256 coefficients (222 cells) they
    // have rank 96 F256 (the 64th skip value is determined: deg h <= 62),
    // and g_final = Z_S(z) h(z) + sum_i g_i(rho_i) is the telescope
    // relation (in the span).
    use cert::*;
    use jetsam_ivc_core::field::{F256, PHI_8_TABLE};
    use jetsam_ivc_core::zk_form::{ZK_SKIP_MASK_COEFFICIENTS, ZkFormLayout, skip_vanishing};
    let layout = ZkFormLayout::CLIENT;
    let rounds = layout.rounds();
    let n = layout.zerocheck_mask_coefficients();
    let mut rng = Rng::new(0xE8D1);
    let r = rng.challenges(rounds);
    let rho = rng.challenges(rounds);
    let z = rng.challenge();
    let g = |i: usize, d: usize| ZK_SKIP_MASK_COEFFICIENTS + 3 * i + d;
    let kappa_row = || {
        let mut row = vec![F256::ZERO; n];
        for i in 0..rounds {
            row[g(i, 0)] += F256::ONE;
            row[g(i, 1)] += r[i];
            row[g(i, 2)] += r[i];
        }
        row
    };
    let zs_h_row = |x: F256| {
        let mut row = vec![F256::ZERO; n];
        let zs = skip_vanishing(x);
        let mut power = F256::ONE;
        for k in 0..ZK_SKIP_MASK_COEFFICIENTS {
            row[k] = zs * power;
            power *= x;
        }
        row
    };
    let add = |a: &mut Vec<F256>, b: &[F256]| a.iter_mut().zip(b).for_each(|(a, &b)| *a += b);
    let mut observations = vec![kappa_row()];
    for j in 0..64 {
        let mut row = zs_h_row(F256::from_base(PHI_8_TABLE[64 + j]));
        add(&mut row, &kappa_row());
        observations.push(row);
    }
    for i in 0..rounds {
        let mut one = zs_h_row(z);
        for j in 0..i {
            one[g(j, 0)] += F256::ONE;
            one[g(j, 1)] += rho[j];
            one[g(j, 2)] += rho[j] * rho[j];
        }
        for j in i + 1..rounds {
            one[g(j, 0)] += F256::ONE;
            one[g(j, 1)] += r[j];
            one[g(j, 2)] += r[j];
        }
        one[g(i, 0)] += F256::ONE;
        one[g(i, 1)] += F256::ONE;
        one[g(i, 2)] += F256::ONE;
        let mut infinity = vec![F256::ZERO; n];
        infinity[g(i, 2)] = F256::ONE;
        observations.push(one);
        observations.push(infinity);
    }
    let mut ech = Echelon::new(2 * n);
    for row in f128_rows_of(&observations) {
        ech.insert(row);
    }
    assert_eq!(observations.len(), 1 + 64 + 2 * rounds);
    assert_eq!(ech.rank(), 2 * 96, "96 F256 independent observations");
    let mut g_final = zs_h_row(z);
    for i in 0..rounds {
        g_final[g(i, 0)] += F256::ONE;
        g_final[g(i, 1)] += rho[i];
        g_final[g(i, 2)] += rho[i] * rho[i];
    }
    for row in f128_rows_of(&[g_final]) {
        assert!(!ech.is_independent(&row), "g_final is the telescope relation");
    }
    println!("[cert-D'] zerocheck mask: 97 F256 observations, rank 96 on 111 coefficients; g_final determined");
}

/// The lincheck mask's observations (16 x 2 round terms, then M(r)) for
/// coefficients `g` (3 per round), by the prover's recurrence.
fn lincheck_mask_observations(
    g: &[[jetsam_ivc_core::field::F256; 3]],
    u: &[jetsam_ivc_core::field::F256],
    rho: &[jetsam_ivc_core::field::F256],
) -> Vec<jetsam_ivc_core::field::F256> {
    use jetsam_ivc_core::field::F256;
    let r = g.len();
    let mut prefix = vec![F256::ZERO; r + 1];
    for t in 0..r {
        prefix[t + 1] = prefix[t] + g[t][1] + g[t][2];
    }
    let (mut e, mut d) = (F256::ONE, F256::ZERO);
    let mut out = Vec::new();
    for t in (0..r).rev() {
        out.push(e * (g[t][0] + g[t][1] + g[t][2]) + u[t] * (d + e * prefix[t]));
        out.push(e * g[t][2]);
        let eq_t = F256::ONE + u[t] + rho[t];
        d = d * eq_t + (g[t][0] + g[t][1] * rho[t] + g[t][2] * rho[t] * rho[t]) * e;
        e *= eq_t;
    }
    out.push(d);
    out
}

#[test]
fn zk_lincheck_mask_observation_rank() {
    // CERT-D'': the lincheck mask
    // M(x) = sum_t g_t(x_t) prod_{k != t} eq(u_k, x_k), deg g_t = 2, u fixed
    // (`zk_form::ZkLincheckMask`), with g_{0,1} fixed by sum_x M = 0.
    // (1) The prover's round terms are the definition: brute force over the
    //     cube at 6 rounds, at X = 0, 1, w (degree-2 interpolation).
    // (2) At m = 22, its observations (16 x 2 round terms, M(r_inner)) have
    //     rank 32 F256 on the 47 free coefficients: onto the 32 message
    //     coordinates, one relation (the chain from the zero sum to M(r)).
    use cert::*;
    use jetsam_ivc_core::field::F256;
    use jetsam_ivc_core::zk_form::{ZkFormLayout, ZkLincheckMask};
    let mut rng = Rng::new(0xE8D2);
    // (1) definition.
    let small = ZkFormLayout::reduced(12);
    let r = small.rounds();
    let u = small.lincheck_mask_point();
    let rho = rng.challenges(r);
    let mut g: Vec<[F256; 3]> = (0..r).map(|_| [rng.challenge(), rng.challenge(), rng.challenge()]).collect();
    let fix = g.iter().fold(F256::ZERO, |s, c| s + c[1] + c[2]) + g[0][1];
    g[0][1] = fix;
    let mask = ZkLincheckMask { rounds: g.clone() };
    assert!(mask.cube_sum().is_zero());
    let formula = lincheck_mask_observations(&g, &u, &rho);
    let w = rng.challenge();
    let mut expected = Vec::new();
    for t in (0..r).rev() {
        let at = |x_t: F256| {
            let mut sum = F256::ZERO;
            for low in 0..1usize << t {
                let mut point = rho.clone();
                point[t] = x_t;
                for (k, slot) in point.iter_mut().enumerate().take(t) {
                    *slot = if (low >> k) & 1 == 1 { F256::ONE } else { F256::ZERO };
                }
                sum += mask.evaluate(&point, &u);
            }
            sum
        };
        let (m0, m1, mw) = (at(F256::ZERO), at(F256::ONE), at(w));
        let lead = (mw + m0 + (m1 + m0) * w) * (w * w + w).inv();
        expected.push(m1);
        expected.push(lead);
    }
    expected.push(mask.evaluate(&rho, &u));
    assert_eq!(formula, expected, "round terms = definition");
    // (2) rank at production.
    let layout = ZkFormLayout::CLIENT;
    let r = layout.rounds();
    let u = layout.lincheck_mask_point();
    let rho = rng.challenges(r);
    // Free coefficients: every (t, d) but (0, 1); g_{0,1} = sum of the others' (d = 1, 2).
    let free: Vec<(usize, usize)> = (0..r)
        .flat_map(|t| (0..3).map(move |d| (t, d)))
        .filter(|&(t, d)| (t, d) != (0, 1))
        .collect();
    let columns: Vec<Vec<F256>> = free
        .iter()
        .map(|&(t, d)| {
            let mut g = vec![[F256::ZERO; 3]; r];
            g[t][d] = F256::ONE;
            if d != 0 {
                g[0][1] += F256::ONE;
            }
            lincheck_mask_observations(&g, &u, &rho)
        })
        .collect();
    let observations: Vec<Vec<F256>> = (0..2 * r + 1)
        .map(|o| columns.iter().map(|col| col[o]).collect())
        .collect();
    let mut ech = Echelon::new(2 * free.len());
    for row in f128_rows_of(&observations) {
        ech.insert(row);
    }
    println!(
        "[cert-D''] lincheck mask: {} F256 observations on {} free coefficients, F128 rank {}",
        observations.len(),
        free.len(),
        ech.rank()
    );
    assert_eq!(ech.rank(), 2 * 2 * r, "32 F256: onto the message coordinates");
}

#[test]
fn zk_skip_round_mask_vanishes_on_the_skip_set() {
    // Z_S * h is zero on S whatever h, so P = 0 on S iff P' = lambda * kappa
    // on S; Z_S is nonzero on Lambda (the 64 sent points).
    use jetsam_ivc_core::field::{F256, PHI_8_TABLE};
    use jetsam_ivc_core::zk_form::skip_vanishing;
    for s in 0..64 {
        assert!(skip_vanishing(F256::from_base(PHI_8_TABLE[s])).is_zero());
        assert!(!skip_vanishing(F256::from_base(PHI_8_TABLE[64 + s])).is_zero());
    }
    // A real proof whose skip round is shifted by lambda kappa verifies,
    // with kappa nonzero (the shift is live).
    let (relation, witness) = zform::instance(12, 600, 21);
    let proved = zform::prove(&relation, &witness, zform::seeded_randomness(*relation.layout(), 22));
    assert!(!proved.proof.zerocheck_kappa.is_zero());
    zform::verify(&relation, &proved.proof, &proved.commitment, &proved.io).expect("honest");
}

#[test]
fn zk_final_b_is_masked_by_the_type_b_cells() {
    // CERT-E at m = 22: conditioned on R_B every other observation is
    // affine in the remaining randomness (R_B enters the zerocheck only
    // through cross terms, which vanish in the lambda limit); final_b's
    // weight on the 16 R_B cells, L_s(z) eq(rho, x), has F128 rank 2.
    use cert::*;
    use jetsam_ivc_core::zk_form::{ZK_TYPE_B_CELLS, ZkFormLayout};
    let layout = ZkFormLayout::CLIENT;
    let mut rng = Rng::new(0xE8E1);
    for _ in 0..8 {
        let lag = skip_lagrange(rng.challenge());
        let rho = rng.challenges(layout.rounds());
        let weights: Vec<_> = (0..ZK_TYPE_B_CELLS)
            .map(|j| {
                let cell = layout.type_b_cell(j);
                let x = cell >> 6;
                let eq = rho.iter().enumerate().fold(jetsam_ivc_core::field::F256::ONE, |p, (i, &c)| {
                    if (x >> i) & 1 == 1 { p * c } else { p * (jetsam_ivc_core::field::F256::ONE + c) }
                });
                lag[cell & 63] * eq
            })
            .collect();
        let mut ech = Echelon::new(ZK_TYPE_B_CELLS);
        for half in lo_hi(&weights) {
            ech.insert(half);
        }
        assert_eq!(ech.rank(), 2);
    }
}

/// The BaseFold row-batch messages (rounds over the 5 lane bits) by their
/// definition: fold a (F128 witness) and b (the wide tensor) pairwise.
fn row_batch_messages(
    a: &[jetsam_ivc_core::field::F128],
    b: &[jetsam_ivc_core::field::F256],
    challenges: &[jetsam_ivc_core::field::F256],
) -> Vec<(jetsam_ivc_core::field::F256, jetsam_ivc_core::field::F256)> {
    use jetsam_ivc_core::field::F256;
    let mut a: Vec<F256> = a.iter().map(|&v| F256::from_base(v)).collect();
    let mut b = b.to_vec();
    let mut out = Vec::new();
    for &c in challenges {
        let half = a.len() / 2;
        let (mut u0, mut u2) = (F256::ZERO, F256::ZERO);
        for i in 0..half {
            u0 += a[2 * i] * b[2 * i];
            u2 += (a[2 * i] + a[2 * i + 1]) * (b[2 * i] + b[2 * i + 1]);
        }
        out.push((u0, u2));
        let fold = |v: &Vec<F256>| (0..half).map(|i| v[2 * i] + c * (v[2 * i] + v[2 * i + 1])).collect::<Vec<_>>();
        a = fold(&a);
        b = fold(&b);
    }
    out
}

#[test]
fn zk_simulator_determined_coordinates_are_exact() {
    // CERT-F (exact identities the simulator uses to compute the determined
    // coordinates instead of sampling them), on a real form-Z proof and on
    // the production tensor algebra:
    // (1) final_c = round1_c interpolated at z (verifier-checked);
    // (2) the 64th skip value: round1_ab' + round1_c + lambda kappa vanishes
    //     on S and has degree <= 126, so its 64 values on Lambda over Z_S lie
    //     on a degree-62 polynomial (zero X^63 coefficient);
    // (3) g_final = G(z, rho), recomputed from the committed mask cells;
    // (4) G'(r_inner) recomputed from the committed lincheck mask;
    // (5) the 5 row-batch rounds are a function of round1_c, z_partial and the
    //     public IO: unchanged by a witness change invisible
    //     to every per-skip partial sum, changed by a visible one;
    // (6) T1 of lanes 24, 25 is determined by the T2 value and the other
    //     lanes (CERT-B bijection; [D], not replayed here).
    use cert::*;
    use jetsam_ivc_core::field::{F128, F256, PHI_8_TABLE};
    use jetsam_ivc_core::zk_form::skip_vanishing;
    let (relation, witness) = zform::instance(13, 1500, 31);
    let layout = *relation.layout();
    let proved = zform::prove(&relation, &witness, zform::seeded_randomness(layout, 32));
    let zc = zform::replay_zerocheck(&relation, &proved);
    let proof = &proved.proof.base.zerocheck;
    // (2)
    let shift = zc.lambda * proved.proof.zerocheck_kappa;
    let nodes: Vec<F256> = (0..64).map(|j| F256::from_base(PHI_8_TABLE[64 + j])).collect();
    let mut top = F256::ZERO;
    for j in 0..64 {
        let value = (proof.round1_ab[j] + proof.round1_c[j] + shift) * skip_vanishing(nodes[j]).inv();
        let mut denominator = F256::ONE;
        for i in 0..64 {
            if i != j {
                denominator *= nodes[j] + nodes[i];
            }
        }
        top += value * denominator.inv();
    }
    assert!(top.is_zero(), "the 64th skip value is determined");
    // (3), (4)
    let z = relation.embed(&witness, zform::seeded_randomness(layout, 32)).unwrap();
    let zm = layout.zerocheck_mask(&z);
    let h_at_z = zm.skip.iter().rev().fold(F256::ZERO, |acc, &c| acc * zc.claim.z + c);
    let mut g = skip_vanishing(zc.claim.z) * h_at_z;
    for (gi, &r) in zm.rounds.iter().zip(&zc.claim.mlv_challenges) {
        g += gi[0] + gi[1] * r + gi[2] * r * r;
    }
    assert_eq!(g, zc.mask_value);
    let (claim, _) = zform::deferred(&relation, &proved);
    assert_eq!(
        layout
            .lincheck_mask(&z)
            .evaluate(&claim.ab.point.x_inner_rest, &layout.lincheck_mask_point()),
        proved.proof.lincheck_mask_value
    );
    // (5) at m = 12, random witness, two quirky claims and the IO claim.
    let mut rng = Rng::new(0xE8F5);
    let m = 12usize;
    let n = 1usize << m;
    let rounds = m - 6;
    let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
    let points = [rng.challenges(rounds), rng.challenges(rounds)];
    let lags = [skip_lagrange(rng.challenge()), skip_lagrange(rng.challenge())];
    let gammas = [rng.challenge(), rng.challenge(), rng.challenge()];
    let io_eq = eq_table(&rng.challenges(3));
    let eqs = [eq_table(&points[0]), eq_table(&points[1])];
    let mut b = vec![F256::ZERO; n];
    for (cell, slot) in b.iter_mut().enumerate() {
        let (s, x) = (cell & 63, cell >> 6);
        for c in 0..2 {
            *slot += gammas[c] * lags[c][s] * eqs[c][x];
        }
        if (8..16).contains(&cell) {
            *slot += gammas[2] * io_eq[cell - 8];
        }
    }
    let rb = rng.challenges(5);
    let base = row_batch_messages(&a, &b, &rb);
    // A change invisible to every per-skip partial: 5 cells of one skip
    // column s = 37 solving the 4 F128 equations sum_j eq_c(x_j) d_j = 0.
    let xs = [3usize, 9, 17, 40, 55];
    let mut ech = Echelon::new(5);
    let rows: Vec<Vec<F128>> = (0..2)
        .flat_map(|c| lo_hi(&xs.iter().map(|&x| eqs[c][x]).collect::<Vec<_>>()))
        .collect();
    for row in &rows {
        ech.insert(row.clone());
    }
    assert_eq!(ech.rank(), 4);
    // Kernel vector by brute elimination: solve with d_4 = 1.
    let kernel = {
        let mut m4: Vec<Vec<F128>> = rows.iter().map(|r| r.clone()).collect();
        // Gaussian elimination on the 4 x 5 system, d_4 = 1.
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..4 {
            if let Some(p) = (r..4).find(|&i| !m4[i][c].is_zero()) {
                m4.swap(r, p);
                let inv = m4[r][c].inv();
                for v in m4[r].iter_mut() {
                    *v *= inv;
                }
                for i in 0..4 {
                    if i != r && !m4[i][c].is_zero() {
                        let f = m4[i][c];
                        let pivot_row = m4[r].clone();
                        for (v, &pv) in m4[i].iter_mut().zip(&pivot_row) {
                            *v += f * pv;
                        }
                    }
                }
                pivots.push(c);
                r += 1;
            }
        }
        assert_eq!(pivots, vec![0, 1, 2, 3]);
        let mut d = [F128::ZERO; 5];
        d[4] = F128::ONE;
        for (i, &c) in pivots.iter().enumerate() {
            d[c] = m4[i][4];
        }
        d
    };
    let mut invisible = a.clone();
    for (j, &x) in xs.iter().enumerate() {
        invisible[64 * x + 37] += kernel[j];
    }
    assert_eq!(row_batch_messages(&invisible, &b, &rb), base, "row batch is determined");
    let mut visible = a.clone();
    visible[64 * 3 + 37] += F128::ONE;
    assert_ne!(row_batch_messages(&visible, &b, &rb), base);
    println!("[cert-F] final_c, 64th skip value, g_final, G'(r), row batch: exact");
}

#[test]
fn zk_simulated_transcript_smoke_test() {
    // SMOKE TEST ONLY, NOT A CERTIFICATE: two witnesses
    // of one statement (same matrix, same IO, different private inputs), 32
    // proofs each at m = 12; F2 projections of the masked messages and of
    // the opened T1 symbols; chi^2 of the two histograms far below the
    // 6-sigma threshold.
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::zk_form::{ZkFormLayout, ZkFormRelation};
    let m = 12usize;
    let inputs = |seed: u64| {
        let mut rng = zform::Rng(seed);
        let mut v = [F128::ZERO; 7];
        for slot in v.iter_mut() {
            *slot = rng.f128();
        }
        v
    };
    let (logical, w1) = zform::logical_relation_with_inputs(m - 1, 700, 41, Some(inputs(1)));
    let (logical2, w2) = zform::logical_relation_with_inputs(m - 1, 700, 41, Some(inputs(2)));
    assert_eq!(logical.statement_digest(), logical2.statement_digest());
    assert_eq!(&w1[8..16], &w2[8..16]);
    assert_ne!(w1, w2);
    let relation = ZkFormRelation::from_logical(ZkFormLayout::reduced(m), &logical).unwrap();
    let bits = |proved: &zform::Proved| -> Vec<bool> {
        let p = &proved.proof.base;
        let mut out = Vec::new();
        out.extend(p.zerocheck.round1_ab.iter().map(|v| v.lo.lo & 1 == 1));
        out.extend(p.zerocheck.multilinear_rounds.iter().map(|v| v.0.lo.lo & 1 == 1));
        out.extend(p.lincheck.rounds.iter().map(|v| v.0.hi.lo & 1 == 1));
        out.extend(p.lincheck.z_partial.iter().map(|v| v.lo.lo & 1 == 1));
        out.push(p.zerocheck.final_a_eval.lo.lo & 1 == 1);
        out.push(p.zerocheck.final_b_eval.lo.lo & 1 == 1);
        for query in p.pcs_open.queries.iter().take(2) {
            out.extend(query.initial_leaf.iter().map(|v| v.lo & 1 == 1));
        }
        out
    };
    let runs = 32usize;
    let mut counts = [Vec::new(), Vec::new()];
    for (side, witness) in [&w1, &w2].into_iter().enumerate() {
        for run in 0..runs {
            let proved = zform::prove(
                &relation,
                witness,
                zform::seeded_randomness(*relation.layout(), 1000 * side as u64 + run as u64),
            );
            let b = bits(&proved);
            if counts[side].is_empty() {
                counts[side] = vec![0usize; b.len()];
            }
            for (c, bit) in counts[side].iter_mut().zip(b) {
                *c += bit as usize;
            }
        }
    }
    let mut chi2 = 0.0f64;
    let mut df = 0usize;
    for (&c1, &c2) in counts[0].iter().zip(&counts[1]) {
        if c1 + c2 == 0 || c1 + c2 == 2 * runs {
            continue;
        }
        let (c1, c2) = (c1 as f64, c2 as f64);
        chi2 += (c1 - c2).powi(2) / (c1 + c2);
        df += 1;
    }
    let threshold = df as f64 + 6.0 * (2.0 * df as f64).sqrt();
    println!("[cert-F smoke] chi2 = {chi2:.1} over {df} projections (threshold {threshold:.1})");
    assert!(df > 100);
    assert!(chi2 < threshold);
}

/// A challenger that forwards to the production one and records the order of
/// transcript events (CERT-H).
struct Recording {
    inner: jetsam_ivc_core::challenger::FsLaneChallenger,
    log: Vec<String>,
}

impl jetsam_ivc_core::challenger::Challenger for Recording {
    fn observe_label(&mut self, label: &[u8]) {
        self.log.push(format!("L:{}", String::from_utf8_lossy(label)));
        self.inner.observe_label(label);
    }
    fn observe_f128(&mut self, value: jetsam_ivc_core::field::F128) {
        self.log.push("o".into());
        self.inner.observe_f128(value);
    }
    fn observe_f128_slice(&mut self, values: &[jetsam_ivc_core::field::F128]) {
        self.log.push(format!("o{}", values.len()));
        self.inner.observe_f128_slice(values);
    }
    fn observe_f256(&mut self, value: jetsam_ivc_core::field::F256) {
        self.log.push("O".into());
        self.inner.observe_f256(value);
    }
    fn observe_f256_slice(&mut self, values: &[jetsam_ivc_core::field::F256]) {
        self.log.push(format!("O{}", values.len()));
        self.inner.observe_f256_slice(values);
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.log.push(if bytes.len() == 24 { format!("B24:{bytes:?}") } else { "B".into() });
        self.inner.observe_bytes(bytes);
    }
    fn sample_f128(&mut self) -> jetsam_ivc_core::field::F128 {
        self.log.push("s".into());
        self.inner.sample_f128()
    }
    fn sample_f256(&mut self) -> jetsam_ivc_core::field::F256 {
        self.log.push("S".into());
        self.inner.sample_f256()
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.log.push("pow".into());
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.log.push("pow".into());
        self.inner.verify_pow(nonce, bits)
    }
}

#[test]
fn zk_free_coordinates_are_causal() {
    // CERT-H: the constraint on message i depends only on challenges drawn
    // before i. On the prover's real transcript: the commitment (T1 root,
    // holding both masks) and kappa are absorbed before lambda; lambda is
    // drawn before the skip round; lambda' (lincheck mask) is drawn after
    // alpha, beta and before the first lincheck round; G'(r) is absorbed
    // after z_partial and before r_inner_skip; the sparse values before the
    // batching scalars. (A simulator can therefore sample in order.)
    use jetsam_ivc_core::field::F128;
    let (relation, witness) = zform::instance(12, 500, 51);
    let m = relation.layout().m;
    let mut challenger = Recording {
        inner: jetsam_ivc_core::challenger::FsLaneChallenger::new_c1(zform::DOMAIN),
        log: Vec::new(),
    };
    let io: Vec<F128> = witness[8..16].to_vec();
    jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
        &relation,
        &witness,
        zform::seeded_randomness(*relation.layout(), 52),
        &zform::pcs_params(m),
        &zform::io_spec(),
        &io,
        &zform::CLASS,
        &mut challenger,
        |_| (),
    )
    .unwrap();
    let log = &challenger.log;
    let at = |needle: &str| {
        log.iter()
            .position(|e| e == needle)
            .unwrap_or_else(|| panic!("{needle} missing"))
    };
    let first_sample = log.iter().position(|e| e == "S").unwrap();
    assert!(at("L:history-field-r1cs-c1") < first_sample, "statement and T1 root first");
    let form_label = at("L:history-zk-client-form-v1");
    let layout = relation.layout();
    let mut expected = Vec::new();
    for value in [layout.m, layout.coin_coefficients, layout.reserved_coefficients] {
        expected.extend_from_slice(&(value as u64).to_le_bytes());
    }
    assert_eq!(log[form_label + 1], format!("B24:{expected:?}"), "the layout is bound");
    assert!(form_label < at("L:history-field-r1cs-c1"));
    let zk = at("L:history-field-zerocheck-c1-zk");
    assert_eq!(&log[zk + 1..zk + 3], &["O".to_string(), "S".to_string()], "kappa then lambda");
    assert_eq!(log[zk + 3], "O64", "the masked skip round after lambda");
    let lin = at("L:history-lincheck-c1");
    let lin_zk = at("L:history-lincheck-c1-zk");
    assert_eq!(&log[lin + 1..lin_zk], &["S".to_string(), "S".to_string()], "alpha, beta");
    assert_eq!(log[lin_zk + 1], "S", "lambda' before the rounds");
    assert_eq!(&log[lin_zk + 2..lin_zk + 4], &["O".to_string(), "O".to_string()]);
    let z_partial = log[lin_zk..].iter().position(|e| e == "O64").unwrap() + lin_zk;
    assert_eq!(&log[z_partial + 1..z_partial + 3], &["O".to_string(), "S".to_string()], "G'(r) then r_inner_skip");
    let sparse = at("L:history-pcs-sparse-c1");
    assert_eq!(&log[sparse + 1..sparse + 5], &["O", "O", "O", "O"].map(String::from));
    assert_eq!(log[sparse + 5], "S", "sparse values before the batching scalars");
    println!("[cert-H] {} transcript events in causal order", log.len());
}

// ---- soundness of the additions -------------------------------------------

// Refusal variants: each refusal is pinned to the check
// that must catch it, not to `is_err()`. `κ`, the skip round and the IO are
// all observed before the skip challenge `z`, so altering any of them moves
// `z` and the first local check, `final_c_eval` against the round-1 C
// message, fails. The masks' final values are not checked locally: they are
// derived (zerocheck) or sent (lincheck) and bound by the sparse claims of
// the batched opening, so their alteration fails in the PCS.
fn c_eval_mismatch(result: &Result<jetsam_ivc_core::proof::C1R1csClaim, jetsam_ivc_core::verifier::VerifyError>) -> bool {
    matches!(
        result,
        Err(jetsam_ivc_core::verifier::VerifyError::Zerocheck(
            jetsam_ivc_core::zerocheck::VerifyError::CEvalMismatch
        ))
    )
}

fn pcs_refusal(result: &Result<jetsam_ivc_core::proof::C1R1csClaim, jetsam_ivc_core::verifier::VerifyError>) -> bool {
    matches!(result, Err(jetsam_ivc_core::verifier::VerifyError::PcsAb(_)))
}

fn reduced_proof(seed: u64) -> (jetsam_ivc_core::zk_form::ZkFormRelation, Vec<jetsam_ivc_core::field::F128>, zform::Proved) {
    let (relation, witness) = zform::instance(12, 600, seed);
    let proved = zform::prove(&relation, &witness, zform::seeded_randomness(*relation.layout(), seed + 1));
    (relation, witness, proved)
}

#[test]
fn a_form_z_proof_of_a_false_statement_is_refused() {
    use jetsam_ivc_core::field::F128;
    let (relation, mut witness) = zform::instance(12, 600, 61);
    witness[16 + 300] += F128::ONE;
    let proved = zform::prove(&relation, &witness, zform::seeded_randomness(*relation.layout(), 62));
    // The honest messages of a non-zero constraint polynomial make the derived
    // zerocheck mask value differ from the committed mask: the PCS refuses.
    let result = zform::verify(&relation, &proved.proof, &proved.commitment, &proved.io);
    assert!(pcs_refusal(&result), "{result:?}");
    // And a valid proof under another statement (other IO) is refused.
    let (relation, _, proved) = reduced_proof(63);
    let mut io = proved.io.clone();
    io[2] += F128::ONE;
    let result = zform::verify(&relation, &proved.proof, &proved.commitment, &io);
    assert!(c_eval_mismatch(&result), "{result:?}");
}

#[test]
fn a_wrong_kappa_is_refused() {
    use jetsam_ivc_core::field::F256;
    let (relation, _, proved) = reduced_proof(71);
    for delta in [F256::ONE, jetsam_ivc_core::zk_form::F256_X] {
        let mut proof = proved.proof.clone();
        proof.zerocheck_kappa += delta;
        let result = zform::verify(&relation, &proof, &proved.commitment, &proved.io);
        assert!(c_eval_mismatch(&result), "{result:?}");
    }
    let mut proof = proved.proof.clone();
    proof.zerocheck_kappa = F256::ZERO;
    let result = zform::verify(&relation, &proof, &proved.commitment, &proved.io);
    assert!(c_eval_mismatch(&result), "{result:?}");
}

#[test]
fn a_skip_round_assumed_zero_on_s_is_refused() {
    // A prover that treats P' as zero on S (skip round sent without the
    // lambda kappa shift) is refused, as is any single altered skip value.
    let (relation, _, proved) = reduced_proof(81);
    let zc = zform::replay_zerocheck(&relation, &proved);
    let shift = zc.lambda * proved.proof.zerocheck_kappa;
    let mut proof = proved.proof.clone();
    for value in proof.base.zerocheck.round1_ab.iter_mut() {
        *value += shift;
    }
    let result = zform::verify(&relation, &proof, &proved.commitment, &proved.io);
    assert!(c_eval_mismatch(&result), "{result:?}");
    let mut proof = proved.proof.clone();
    proof.base.zerocheck.round1_ab[17] += jetsam_ivc_core::field::F256::ONE;
    let result = zform::verify(&relation, &proof, &proved.commitment, &proved.io);
    assert!(c_eval_mismatch(&result), "{result:?}");
}

#[test]
fn a_sparse_claim_with_wrong_weights_is_refused() {
    // The masks' final values are bound by the sparse claims: a wrong
    // G'(r) is refused; at the PCS level, an opening checked against one
    // altered weight, cell or value is refused.
    use cert::Rng;
    use jetsam_ivc_core::challenger::FsLaneChallenger;
    use jetsam_ivc_core::field::{F128, F256};
    use jetsam_ivc_core::pcs;
    use jetsam_ivc_core::zk_form::ZkSparseClaim;
    let (relation, _, proved) = reduced_proof(91);
    let mut proof = proved.proof.clone();
    proof.lincheck_mask_value += F256::ONE;
    let result = zform::verify(&relation, &proof, &proved.commitment, &proved.io);
    assert!(pcs_refusal(&result), "{result:?}");

    let m = 12usize;
    let mut rng = Rng::new(0xE891);
    let witness: Vec<F128> = (0..1usize << m).map(|_| rng.f128()).collect();
    let params = zform::pcs_params(m);
    let (commitment, data) = pcs::commit(&witness, &params);
    let x_rest = rng.challenges(m - 6);
    let z_skip = rng.challenge();
    let lag = cert::skip_lagrange(z_skip);
    let eq = cert::eq_table(&x_rest);
    let value = (0..1usize << m).fold(F256::ZERO, |s, i| s + (lag[i & 63] * eq[i >> 6]).scale_base(witness[i]));
    let claim = pcs::C1QuirkyDirectClaim { z_skip, k_skip: 6, x_rest: x_rest.clone(), value };
    let cells = vec![5usize, 700, 4000];
    let weights = rng.challenges(3);
    let mut sparse = ZkSparseClaim { cells, weights, value: F256::ZERO };
    sparse.value = sparse.evaluate(&witness);
    let mut prover = FsLaneChallenger::new_c1(b"e8-sparse");
    let opening = pcs::open_batch_quirky_direct_c1_with_sparse(
        &witness,
        &data,
        &commitment,
        std::slice::from_ref(&claim),
        std::slice::from_ref(&sparse),
        &mut prover,
    );
    let check = |sparse: &ZkSparseClaim| {
        let mut verifier = FsLaneChallenger::new_c1(b"e8-sparse");
        let refs = [pcs::C1QuirkyDirectClaimRef { z_skip, k_skip: 6, x_rest: &x_rest, value }];
        pcs::verify_opening_batch_quirky_direct_c1_with_sparse(
            &commitment,
            &refs,
            std::slice::from_ref(sparse),
            &opening,
            &mut verifier,
        )
    };
    check(&sparse).expect("honest sparse opening");
    // Weights and cells are the verifier's own (not observed): a change moves
    // only the transparent final_b. The value is observed before the batching
    // scalars: a change moves the BaseFold target and transcript.
    let mut wrong = sparse.clone();
    wrong.weights[1] += F256::ONE;
    assert_eq!(check(&wrong), Err(pcs::VerifyError::FinalBMismatch), "altered weight");
    let mut wrong = sparse.clone();
    wrong.cells[2] = 4001;
    assert_eq!(check(&wrong), Err(pcs::VerifyError::FinalBMismatch), "altered cell");
    let mut wrong = sparse.clone();
    wrong.value += F256::ONE;
    let result = check(&wrong);
    assert!(matches!(result, Err(pcs::VerifyError::BaseFold(_))), "altered value: {result:?}");
    let mut wrong = sparse.clone();
    wrong.cells[0] += 1usize << m;
    assert_eq!(check(&wrong), Err(pcs::VerifyError::FinalBMismatch), "cell outside the committed vector");
}

#[test]
fn a_form_z_matrix_with_an_entry_on_a_free_row_or_column_is_refused() {
    // A form-Z matrix carries nothing on the free rows or columns (coins,
    // companion, Libra cells, R_B): the closed-form tautologies would add to
    // a materialized entry, and a circuit reading the companion would break
    // CERT-E and Libra. The native check refuses it at registration; a
    // catalogue relation beyond the useful capacity is refused by pi.
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_ivc_core::zk_form::*;
    let (relation, _) = zform::instance(12, 400, 101);
    let layout = *relation.layout();
    let physical = relation.physical();
    check_zk_form_registration(&layout, physical).unwrap();
    let with_extra = |row: usize, column: usize| -> FieldR1cs {
        let n = layout.cells();
        let mut rows: Vec<Vec<(u32, F128)>> = (0..n).map(|r| physical.a_0.row(r).collect()).collect();
        rows[row].push((column as u32, F128::ONE));
        rows[row].sort_by_key(|e| e.0);
        FieldR1cs {
            m: physical.m,
            k_log: physical.k_log,
            k_skip: physical.k_skip,
            useful_rows: physical.useful_rows,
            a_0: SparseFieldMatrix::from_rows(n, rows),
            b_0: physical.b_0.clone(),
            const_pin: physical.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    };
    let coin = 64 * 3 + 5;
    let companion = 64 * 10 + 27;
    let useful = layout.physical_cell(16 + 10).unwrap();
    assert_eq!(layout.role(coin), ZkCellRole::Coin);
    assert_eq!(layout.role(companion), ZkCellRole::Companion);
    assert!(matches!(
        check_zk_form_registration(&layout, &with_extra(coin, useful)),
        Err(ZkFormError::FreeCellEntry { .. })
    ));
    assert!(matches!(
        check_zk_form_registration(&layout, &with_extra(useful, companion)),
        Err(ZkFormError::FreeCellEntry { .. })
    ));
    assert!(matches!(
        check_zk_form_registration(&layout, &with_extra(useful, layout.type_b_cell(3))),
        Err(ZkFormError::FreeCellEntry { .. })
    ));
    let capacity = layout.useful_capacity();
    let (too_big, _) = zform::logical_relation(12, capacity, 102);
    assert!(matches!(
        ZkFormRelation::from_logical(layout, &too_big),
        Err(ZkFormError::Capacity { .. })
    ));
}

// ---- structured provers (support lever) ------------------------------------

#[test]
fn the_structured_fold_merge_and_receipt_provers_equal_the_full_width_ones() {
    // No structured prover is kept for form Z: the verifier strips the
    // tautology rows in closed form, so the deferred matrix claim concerns
    // the renamed catalogue matrix only, whose support is the catalogue's.
    // Equality with the full width (catalogue + materialized tautologies):
    // on a real proof's fresh claim (receipt) and on an accumulated claim
    // (fold, merge), full = support part + closed form.
    use jetsam_ivc_core::field::{F128, F256};
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_ivc_core::matrix_claim::c1::{
        C1MatrixAccClaim, fresh_claim_value_c1, stacked_matrix_mle_eval_c1,
    };
    let (relation, witness) = zform::instance(13, 1500, 111);
    let layout = *relation.layout();
    let proved = zform::prove(&relation, &witness, zform::seeded_randomness(layout, 112));
    let (_, fresh) = zform::deferred(&relation, &proved);
    let physical = relation.physical();
    assert_eq!(fresh_claim_value_c1(physical, &fresh), fresh.value, "receipt on the support");
    let taut = layout.materialized_tautologies();
    let n = layout.cells();
    let union = |p: &SparseFieldMatrix, t: &SparseFieldMatrix| {
        let rows: Vec<Vec<(u32, F128)>> = (0..n)
            .map(|r| {
                let mut row: Vec<(u32, F128)> = p.row(r).chain(t.row(r)).collect();
                row.sort_by_key(|e| e.0);
                row
            })
            .collect();
        SparseFieldMatrix::from_rows(n, rows)
    };
    let full = FieldR1cs {
        m: physical.m,
        k_log: physical.k_log,
        k_skip: physical.k_skip,
        useful_rows: n,
        a_0: union(&physical.a_0, &taut.a_0),
        b_0: union(&physical.b_0, &taut.b_0),
        const_pin: physical.const_pin,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    };
    let closed = layout.tautology_fresh_value(
        fresh.alpha,
        fresh.z_skip,
        &fresh.x_inner_rest,
        &fresh.r_inner_rest,
        &fresh.z_partial,
    );
    assert_eq!(fresh_claim_value_c1(&full, &fresh), fresh.value + closed);
    let mut rng = cert::Rng::new(0xE8A9);
    let m = layout.m;
    let point = rng.challenges(2 * m + 1);
    let acc = C1MatrixAccClaim { point: point.clone(), value: F256::ZERO };
    let (a, b) = layout.tautology_mle_eq(&point[..m], &point[m + 1..]);
    let t = point[m];
    assert_eq!(
        stacked_matrix_mle_eval_c1(&full, &acc),
        stacked_matrix_mle_eval_c1(physical, &acc) + (F256::ONE + t) * a + t * b
    );
    println!(
        "[structured] support: {} entries (catalogue) vs {} full width",
        physical.a_0.col_indices.len() + physical.b_0.col_indices.len(),
        full.a_0.col_indices.len() + full.b_0.col_indices.len()
    );
}

/// A v1.5 client proof (catalogue relation as is, no form Z) on a fixed
/// instance and transcript: FNV-1a 64 of its bincode bytes and the root.
fn v15_client_proof_fingerprint() -> (u64, usize, [u8; 32]) {
    use jetsam_ivc_core::challenger::FsLaneChallenger;
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_ivc_core::pcs::PcsParams;
    use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
    let m = 12usize;
    let n = 1usize << m;
    let mut state = 0x15u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut z = vec![F128::ZERO; n];
    let mut a_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
    let mut b_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
    z[0] = F128::ONE;
    a_rows[0].push((0, F128::ONE));
    b_rows[0].push((0, F128::ONE));
    for io in 8..16 {
        z[io] = F128::new(next(), next());
        a_rows[io].push((io as u32, F128::ONE));
        b_rows[io].push((0, F128::ONE));
    }
    for c in 16..716 {
        let (x, w) = ((next() % c as u64) as usize, (next() % c as u64) as usize);
        let k = F128::new(next(), next());
        a_rows[c].push((x as u32, k));
        b_rows[c].push((w as u32, F128::ONE));
        z[c] = k * z[x] * z[w];
    }
    let relation = FieldR1cs {
        m,
        k_log: m,
        k_skip: 6,
        useful_rows: 716,
        a_0: SparseFieldMatrix::from_rows(n, a_rows),
        b_0: SparseFieldMatrix::from_rows(n, b_rows),
        const_pin: Some(0),
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    };
    assert!(relation.satisfies(&z));
    let spec = PublicIoSpec {
        io_slice: WitnessSlice { log2_len: 3, index: 1 },
        io_len: 8,
        claims: Vec::new(),
    };
    let params = PcsParams {
        m: m + jetsam_ivc_core::pcs::LOG_PACKING,
        log_inv_rate: 2,
        log_batch_size: 5,
        profile: Default::default(),
    };
    let io = z[8..16].to_vec();
    let mut challenger = FsLaneChallenger::new_c1(b"JTM/E8/V15-NEUTRALITY");
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &relation,
            &z,
            &params,
            &spec,
            &io,
            &[0x15; 32],
            &mut challenger,
            |_| (),
        );
    let mut verifier = FsLaneChallenger::new_c1(b"JTM/E8/V15-NEUTRALITY");
    let (_, fresh) = jetsam_ivc_core::verifier::verify_field_c1_deferred_matrix_with_post_commit_context(
        &jetsam_ivc_core::proof::FieldShape::of(&relation),
        &relation.statement_digest(),
        &commitment,
        &proof,
        &spec,
        &io,
        &[0x15; 32],
        &(),
        &mut verifier,
        |_, _| Ok(()),
    )
    .expect("v1.5 proof verifies");
    assert_eq!(jetsam_ivc_core::matrix_claim::c1::fresh_claim_value_c1(&relation, &fresh), fresh.value);
    let bytes = bincode::serialize(&proof).expect("serialize");
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in &bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    (hash, bytes.len(), commitment.root)
}

#[test]
fn the_v1_5_zerocheck_and_opening_are_unchanged() {
    // The form-Z variants sit beside the v1.5 paths: a v1.5 client proof of
    // a fixed instance and transcript is byte for byte the one the code
    // produced before the form-Z variants (fingerprint measured on that
    // code), and it still verifies and discharges.
    let (hash, len, root) = v15_client_proof_fingerprint();
    assert_eq!(len, 212_336);
    assert_eq!(hash, 0x91bd_4139_f4cd_1ff5);
    assert_eq!(
        root.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "4f1a960fe454c8d1c99f8415df35dd2a8e77b988026eaed560b32216dd0ca145"
    );
}

// ---- randomness -------------------------------------------------------------

#[test]
fn two_proofs_of_one_witness_differ() {
    // Fresh randomness per attempt: two proofs of one witness differ (roots,
    // masked messages), both verify; two system draws differ too.
    let (relation, witness) = zform::instance(12, 500, 121);
    let layout = *relation.layout();
    let first = zform::prove(&relation, &witness, zform::seeded_randomness(layout, 1));
    let second = zform::prove(&relation, &witness, zform::seeded_randomness(layout, 2));
    assert_ne!(first.commitment.root, second.commitment.root);
    assert_ne!(first.proof.base.lincheck.z_partial, second.proof.base.lincheck.z_partial);
    assert_ne!(first.proof.zerocheck_kappa, second.proof.zerocheck_kappa);
    for proved in [&first, &second] {
        zform::verify(&relation, &proved.proof, &proved.commitment, &proved.io).expect("verifies");
    }
    let system = |_: ()| {
        zform::prove(
            &relation,
            &witness,
            jetsam_ivc_prover::zk_client::draw_zk_form_randomness_from_system(layout),
        )
    };
    let (a, b) = (system(()), system(()));
    assert_ne!(a.commitment.root, b.commitment.root);
    zform::verify(&relation, &a.proof, &a.commitment, &a.io).expect("system draw verifies");
}

#[test]
fn the_hiding_randomness_is_consumed_and_wiped() {
    // ZkFreshRandomness is not Clone, is moved into the prover, and its
    // buffer is zeroized on drop; the generator fills the whole buffer in one
    // call; randomness of another form is refused.
    use jetsam_ivc_core::field::F128;
    use jetsam_ivc_core::zk_form::*;
    use std::marker::PhantomData;
    struct Probe<T>(PhantomData<T>);
    trait Cloneable {
        fn is_clone(&self) -> bool {
            true
        }
    }
    impl<T: Clone> Cloneable for Probe<T> {}
    trait NotCloneable {
        fn is_clone(&self) -> bool {
            false
        }
    }
    impl<T> NotCloneable for &Probe<T> {}
    assert!(!(&Probe::<ZkFreshRandomness>(PhantomData)).is_clone());
    assert!((&Probe::<Vec<u8>>(PhantomData)).is_clone());
    assert!(std::mem::needs_drop::<ZkFreshRandomness>());

    let mut values = vec![F128::new(5, 6); 33];
    wipe(&mut values);
    assert!(values.iter().all(|v| v.is_zero()));

    let layout = ZkFormLayout::reduced(12);
    let mut calls = Vec::new();
    let randomness = draw_zk_form_randomness(layout, &mut |bytes: &mut [u8]| {
        calls.push(bytes.len());
        bytes.fill(0xA5);
    });
    assert_eq!(calls, vec![16 * layout.free_cells()]);
    assert_eq!(randomness.len(), layout.free_cells());
    let (relation, witness) = zform::instance(13, 300, 131);
    assert!(matches!(
        relation.embed(&witness, randomness),
        Err(ZkFormError::Randomness)
    ));
}

#[test]
#[ignore = "exploration (release): rank of the full linear observation map"]
fn exploration_full_linear_rank() {
    use cert::*;
    let mut rng = Rng::new(0xE8B0);
    for kind in [CircuitKind::Random, CircuitKind::HalfPadding, CircuitKind::Padding] {
        let model = full_model(&REDUCED, kind, 32, &mut rng);
        let (rank, rows) = full_rank(&REDUCED, &model, &[], false);
        println!("[explore] {kind:?}: {rows} F128 linear rows, rank {rank}, deficiency {}", rows - rank);
        let q = query_sets(&REDUCED, 0, &mut rng).pop().unwrap();
        let (rank_t1, rows_t1) = full_rank(&REDUCED, &model, &q, true);
        println!("[explore] {kind:?} + T1: {rows_t1} rows, rank {rank_t1}, deficiency {}", rows_t1 - rank_t1);
        // Option "Libra on the lincheck": its rounds leave the linear block.
        let mut masked = model;
        masked.rows.retain(|(name, _)| !name.starts_with("lincheck"));
        let (rank_m, rows_m) = full_rank(&REDUCED, &masked, &q, true);
        println!("[explore] {kind:?} without lincheck rows + T1: {rows_m} rows, rank {rank_m}, deficiency {}", rows_m - rank_m);
    }
}
