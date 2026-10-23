// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The hiding client form ("form Z", v1.6): the client proof form
//! that hides the client's witness.
//!
//! A form-Z relation is a catalogue relation renamed by `pi` into the useful
//! zone of an `m`-variable square shape, plus implicit tautology rows on every
//! free cell. The free cells carry fresh randomness drawn by the client:
//!
//! * cell 0 is the constant, cells 8..16 the public IO (both untouched);
//! * **coins**: the cells of lanes 0..24 below `32 * coin_coefficients`
//!   (except 0 and 8..16), type A;
//! * **companion**: every cell of lanes 24..32 (bits 3 and 4 of the cell index
//!   set), type A; the top `reserved_coefficients` coefficients of lanes
//!   26..32 hold, in this order, the 16 **type-B** cells `R_B` (they mask
//!   `final_b`), the Libra mask of the zerocheck (`63 + 3 (m - 6)` F256
//!   coefficients, two cells each) and the mask of the lincheck
//!   (`3 (m - 6)` coefficients); the rest of that block is companion;
//! * **useful**: lanes 0..24 from `32 * coin_coefficients` on: logical cell
//!   number `n` (constant and IO excluded) sits at
//!   `32 * coin_coefficients + 32 * (n / 24) + n % 24`.
//!
//! Tautology rows: type A `z_i * 1 = z_i` (`A_i = e_i`, `B_i = e_0`), type B
//! `1 * z_i = z_i` (`A_i = e_0`, `B_i = e_i`). They are never materialized:
//! the prover adds them to `A z`, `B z` and to the lincheck combination; the
//! verifier subtracts their closed form from the deferred matrix claim, so the
//! matrix side only ever sees the (sparse) catalogue part.
//!
//! Nothing here is consensus code until a V1_6 pack carries it; the v1.5
//! client form and its paths are untouched.

use crate::field::{F128, F256, PHI_8_TABLE};
use crate::field_r1cs::{FieldR1cs, SparseFieldMatrix};
use crate::lincheck::LincheckCircuit;
use crate::zerocheck::field_c1::{build_eq_table, lagrange_weights};
use rayon::prelude::*;

/// Skip dimension of the form (the protocol constant).
pub const ZK_FORM_K_SKIP: usize = 6;
/// Lanes of the PCS batch.
pub const ZK_FORM_LANES: usize = 32;
/// Lanes 24..32: the companion (cells with bits 3 and 4 set).
pub const ZK_COMPANION_LANES: std::ops::Range<usize> = 24..32;
/// Lanes 24 and 25 carry the change of variables onto the row-batched
/// message M (CERT-B); nothing else is placed in them.
pub const ZK_CHANGE_OF_VARIABLE_LANES: [usize; 2] = [24, 25];
/// Lanes whose top block holds R_B and the two Libra masks.
pub const ZK_RESERVED_LANES: std::ops::Range<usize> = 26..32;
/// Type-B cells masking `final_b` (CERT-E).
pub const ZK_TYPE_B_CELLS: usize = 16;
/// Degree bound of `h` in the skip-round mask `Z_S * h` is 62.
pub const ZK_SKIP_MASK_COEFFICIENTS: usize = 63;
/// The useful lanes.
pub const ZK_USEFUL_LANES: usize = 24;

/// Role of a physical cell of form Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZkCellRole {
    Constant,
    Io,
    Coin,
    Companion,
    /// Libra mask of the zerocheck.
    LibraMask,
    /// Mask of the lincheck.
    LincheckMask,
    TypeB,
    Useful,
}

impl ZkCellRole {
    /// Free cells carry randomness and an implicit tautology row.
    pub const fn is_free(self) -> bool {
        matches!(
            self,
            Self::Coin | Self::Companion | Self::LibraMask | Self::LincheckMask | Self::TypeB
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZkFormError {
    Layout(&'static str),
    /// The catalogue relation does not have the client form's shape.
    Shape(&'static str),
    /// A used logical cell falls beyond the useful capacity.
    Capacity { logical_cell: usize, capacity: usize },
    /// A form-Z matrix carries an entry on a free row or column
    /// (registration rule).
    FreeCellEntry { row: usize, column: usize },
    WitnessLength { expected: usize, got: usize },
    Randomness,
}

/// The geometry of form Z at `m` variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZkFormLayout {
    pub m: usize,
    /// Coins: the coefficients `< coin_coefficients` of lanes 0..24.
    pub coin_coefficients: usize,
    /// Top coefficients of lanes 26..32 reserved for R_B and the masks.
    pub reserved_coefficients: usize,
}

impl ZkFormLayout {
    /// The client form: m = 22, coins below cell 8192, a 256-coefficient
    /// reserved block (top coefficients of lanes 26..32, one high pattern).
    pub const CLIENT: Self = Self {
        m: 22,
        coin_coefficients: 256,
        reserved_coefficients: 256,
    };

    /// A test-scale form (m >= 12): 16 coin coefficients, 64 reserved.
    pub const fn reduced(m: usize) -> Self {
        Self {
            m,
            coin_coefficients: 16,
            reserved_coefficients: 64,
        }
    }

    pub fn validate(&self) -> Result<(), ZkFormError> {
        if self.m < 12 || self.m > 26 {
            return Err(ZkFormError::Layout("m out of range"));
        }
        if !self.coin_coefficients.is_power_of_two() || self.coin_coefficients < 2 {
            return Err(ZkFormError::Layout("coin coefficients"));
        }
        if self.coin_coefficients + self.reserved_coefficients > self.coefficients() {
            return Err(ZkFormError::Layout("coins and reserved block overlap"));
        }
        if self.reserved_cells_needed() > ZK_RESERVED_LANES.len() * self.reserved_coefficients {
            return Err(ZkFormError::Layout("reserved block too small"));
        }
        Ok(())
    }

    pub const fn cells(&self) -> usize {
        1usize << self.m
    }
    /// Coefficients per lane.
    pub const fn coefficients(&self) -> usize {
        1usize << (self.m - 5)
    }
    /// Multilinear rounds of the zerocheck and of the lincheck (square form).
    pub const fn rounds(&self) -> usize {
        self.m - ZK_FORM_K_SKIP
    }
    pub const fn zerocheck_mask_coefficients(&self) -> usize {
        ZK_SKIP_MASK_COEFFICIENTS + 3 * self.rounds()
    }
    /// Three coefficients (`deg g_t = 2`) per lincheck round.
    pub const fn lincheck_mask_coefficients(&self) -> usize {
        3 * self.rounds()
    }
    pub const fn reserved_cells_needed(&self) -> usize {
        ZK_TYPE_B_CELLS + 2 * self.zerocheck_mask_coefficients() + 2 * self.lincheck_mask_coefficients()
    }
    /// First useful cell.
    pub const fn useful_start(&self) -> usize {
        32 * self.coin_coefficients
    }
    /// Coins sit at rest index x < coin_x_end.
    pub const fn coin_x_end(&self) -> usize {
        self.coin_coefficients / 2
    }
    /// Useful cells: 24 lanes times the coefficients above the coins.
    pub const fn useful_capacity(&self) -> usize {
        ZK_USEFUL_LANES * (self.coefficients() - self.coin_coefficients)
    }
    pub const fn coin_cells(&self) -> usize {
        ZK_USEFUL_LANES * self.coin_coefficients - 9
    }
    const fn reserved_base(&self) -> usize {
        self.coefficients() - self.reserved_coefficients
    }

    /// Physical cell of reserved slot `index` (lanes 26..32 interleaved).
    pub const fn reserved_cell(&self, index: usize) -> usize {
        let lanes = 6;
        let coefficient = self.reserved_base() + index / lanes;
        32 * coefficient + ZK_RESERVED_LANES.start + index % lanes
    }
    pub const fn type_b_cell(&self, j: usize) -> usize {
        self.reserved_cell(j)
    }
    /// Cell of half `half` (0 = lo, 1 = hi) of zerocheck mask coefficient `j`.
    pub const fn zerocheck_mask_cell(&self, j: usize, half: usize) -> usize {
        self.reserved_cell(ZK_TYPE_B_CELLS + 2 * j + half)
    }
    pub const fn lincheck_mask_cell(&self, j: usize, half: usize) -> usize {
        self.reserved_cell(ZK_TYPE_B_CELLS + 2 * self.zerocheck_mask_coefficients() + 2 * j + half)
    }

    fn reserved_index(&self, cell: usize) -> Option<usize> {
        let lane = cell & 31;
        let coefficient = cell >> 5;
        if ZK_RESERVED_LANES.contains(&lane) && coefficient >= self.reserved_base() {
            Some((coefficient - self.reserved_base()) * 6 + (lane - ZK_RESERVED_LANES.start))
        } else {
            None
        }
    }

    pub fn role(&self, cell: usize) -> ZkCellRole {
        debug_assert!(cell < self.cells());
        if cell == 0 {
            return ZkCellRole::Constant;
        }
        if (8..16).contains(&cell) {
            return ZkCellRole::Io;
        }
        let lane = cell & 31;
        if lane >= ZK_COMPANION_LANES.start {
            if let Some(index) = self.reserved_index(cell) {
                let zc = ZK_TYPE_B_CELLS + 2 * self.zerocheck_mask_coefficients();
                return if index < ZK_TYPE_B_CELLS {
                    ZkCellRole::TypeB
                } else if index < zc {
                    ZkCellRole::LibraMask
                } else if index < zc + 2 * self.lincheck_mask_coefficients() {
                    ZkCellRole::LincheckMask
                } else {
                    ZkCellRole::Companion
                };
            }
            return ZkCellRole::Companion;
        }
        if cell < self.useful_start() {
            ZkCellRole::Coin
        } else {
            ZkCellRole::Useful
        }
    }

    /// `pi`: logical cell of a catalogue relation to its physical cell.
    /// Identity on the constant and the IO slice [8, 16); every other logical
    /// cell `c` is numbered `n = c - 1` (c < 8) or `n = c - 9` (c >= 16) and
    /// goes to `useful_start + 32 (n / 24) + n % 24`. `None` past capacity.
    pub fn physical_cell(&self, logical: usize) -> Option<usize> {
        if logical == 0 || (8..16).contains(&logical) {
            return Some(logical);
        }
        let n = if logical < 8 { logical - 1 } else { logical - 9 };
        if n >= self.useful_capacity() {
            return None;
        }
        Some(self.useful_start() + 32 * (n / ZK_USEFUL_LANES) + n % ZK_USEFUL_LANES)
    }

    /// Number of free cells (one F128 of randomness each).
    pub fn free_cells(&self) -> usize {
        (0..self.cells())
            .filter(|&cell| self.role(cell).is_free())
            .count()
    }

    /// The fixed point `u` weighting the lincheck mask: never in F128, never
    /// 0 or 1 (the high coordinate has trace one). Public constant.
    pub fn lincheck_mask_point(&self) -> Vec<F256> {
        (0..self.rounds())
            .map(|t| {
                F256::from_raw_challenge_lanes(
                    F128::new(0x4A45_5453_414D_5A00 ^ t as u64, 0x6C69_6E63_6865_636B),
                    F128::new(0x6D61_736B_0000_0001 + t as u64, 0x7531_3600_0000_0000),
                )
            })
            .collect()
    }

    /// Closed form, fresh basis (the deferred lincheck claim): for every skip
    /// index s, `T[s] = sum_x eq(r_inner, x) W_taut(64 x + s)` where `W_taut`
    /// is the lincheck combination of the tautology rows at the row point
    /// `(z_skip, rho)`: type A puts `alpha L_s(z) eq(rho, x)` on its own column
    /// and `L_s(z) eq(rho, x)` on column 0, type B the converse. The tautology
    /// part of a fresh claim is `sum_s T[s] z_partial[s]`.
    pub fn tautology_fresh_columns(
        &self,
        alpha: F256,
        z_skip: F256,
        rho: &[F256],
        r_inner: &[F256],
    ) -> Vec<F256> {
        assert_eq!(rho.len(), self.rounds());
        assert_eq!(r_inner.len(), self.rounds());
        let skip = lagrange_weights(ZK_FORM_K_SKIP, z_skip, 0);
        let c = self.coin_x_end().trailing_zeros() as usize;
        let one = F256::ONE;
        let eq1 = |a: F256, b: F256| one + a + b;
        let mut p_all = one;
        let mut p_coin = one;
        let mut e0 = one;
        let mut rho_coin = one;
        let mut rho_zero = one;
        let mut r_zero = one;
        for j in 0..self.rounds() {
            p_all *= eq1(rho[j], r_inner[j]);
            let both_zero = (one + rho[j]) * (one + r_inner[j]);
            p_coin *= if j < c { eq1(rho[j], r_inner[j]) } else { both_zero };
            e0 *= both_zero;
            if j >= c {
                rho_coin *= one + rho[j];
            }
            rho_zero *= one + rho[j];
            r_zero *= one + r_inner[j];
        }
        let public = |s: usize| s == 0 || (8..16).contains(&s);
        let mut out = vec![F256::ZERO; 64];
        let mut sum_a = F256::ZERO;
        let mut sum_b = F256::ZERO;
        for s in 0..64 {
            let lane = s & 31;
            if lane < ZK_COMPANION_LANES.start {
                let mut base = p_coin;
                let mut rows = rho_coin;
                if public(s) {
                    base += e0;
                    rows += rho_zero;
                }
                out[s] = skip[s] * alpha * base;
                sum_a += skip[s] * rows;
            } else {
                out[s] = skip[s] * alpha * p_all;
                sum_a += skip[s];
            }
        }
        let eq_rho = |x: usize| eq_at(rho, x);
        let eq_r = |x: usize| eq_at(r_inner, x);
        for j in 0..ZK_TYPE_B_CELLS {
            let cell = self.type_b_cell(j);
            let (s, x) = (cell & 63, cell >> 6);
            let rw = skip[s] * eq_rho(x);
            out[s] += (alpha + one) * rw * eq_r(x);
            sum_a += rw;
            sum_b += rw;
        }
        out[0] += r_zero * (sum_a + alpha * sum_b);
        out
    }

    /// The tautology part of a fresh lincheck claim.
    pub fn tautology_fresh_value(
        &self,
        alpha: F256,
        z_skip: F256,
        rho: &[F256],
        r_inner: &[F256],
        z_partial: &[F256],
    ) -> F256 {
        assert_eq!(z_partial.len(), 64);
        self.tautology_fresh_columns(alpha, z_skip, rho, r_inner)
            .iter()
            .zip(z_partial)
            .fold(F256::ZERO, |sum, (&t, &z)| sum + t * z)
    }

    /// Closed form, eq basis: the MLEs of the tautology A and B matrices at
    /// a row point `y` and a column point `w` (both `m` coordinates, cell bit
    /// i against coordinate i). Equal to the materialized rows (tested).
    pub fn tautology_mle_eq(&self, y: &[F256], w: &[F256]) -> (F256, F256) {
        assert_eq!(y.len(), self.m);
        assert_eq!(w.len(), self.m);
        let one = F256::ONE;
        let eq_w0 = w.iter().fold(one, |p, &v| p * (one + v));
        // sum over the type-A set of prod_b phi_b(i_b), for a per-bit phi.
        let type_a_sum = |phi: &dyn Fn(usize, usize) -> F256| -> F256 {
            let sigma = |b: usize| phi(b, 0) + phi(b, 1);
            let mut companion = phi(3, 1) * phi(4, 1);
            for b in (0..self.m).filter(|&b| b != 3 && b != 4) {
                companion *= sigma(b);
            }
            let coin_bits = self.useful_start().trailing_zeros() as usize;
            let mut coins = sigma(0) * sigma(1) * sigma(2);
            coins *= sigma(3) * sigma(4) + phi(3, 1) * phi(4, 1);
            for b in 5..coin_bits {
                coins *= sigma(b);
            }
            for b in coin_bits..self.m {
                coins *= phi(b, 0);
            }
            let at = |cell: usize| (0..self.m).fold(one, |p, b| p * phi(b, (cell >> b) & 1));
            for cell in std::iter::once(0).chain(8..16) {
                coins += at(cell);
            }
            let mut total = companion + coins;
            for j in 0..ZK_TYPE_B_CELLS {
                total += at(self.type_b_cell(j));
            }
            total
        };
        let eqb = |p: F256, bit: usize| if bit == 1 { p } else { one + p };
        let diag = type_a_sum(&|b, bit| eqb(y[b], bit) * eqb(w[b], bit));
        let rows = type_a_sum(&|b, bit| eqb(y[b], bit));
        let mut b_diag = F256::ZERO;
        let mut b_rows = F256::ZERO;
        for j in 0..ZK_TYPE_B_CELLS {
            let cell = self.type_b_cell(j);
            let ey = (0..self.m).fold(one, |p, b| p * eqb(y[b], (cell >> b) & 1));
            let ew = (0..self.m).fold(one, |p, b| p * eqb(w[b], (cell >> b) & 1));
            b_diag += ey * ew;
            b_rows += ey;
        }
        (diag + b_rows * eq_w0, rows * eq_w0 + b_diag)
    }

    /// Materialized tautology rows (tests and the registration digest only).
    pub fn materialized_tautologies(&self) -> FieldR1cs {
        let n = self.cells();
        let mut a_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
        let mut b_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
        for cell in 0..n {
            match self.role(cell) {
                ZkCellRole::TypeB => {
                    a_rows[cell].push((0, F128::ONE));
                    b_rows[cell].push((cell as u32, F128::ONE));
                }
                role if role.is_free() => {
                    a_rows[cell].push((cell as u32, F128::ONE));
                    b_rows[cell].push((0, F128::ONE));
                }
                _ => {}
            }
        }
        FieldR1cs {
            m: self.m,
            k_log: self.m,
            k_skip: ZK_FORM_K_SKIP,
            useful_rows: n,
            a_0: SparseFieldMatrix::from_rows(n, a_rows),
            b_0: SparseFieldMatrix::from_rows(n, b_rows),
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }
}

/// `eq(point, x)` for an integer x (bit i against point[i]).
fn eq_at(point: &[F256], x: usize) -> F256 {
    point.iter().enumerate().fold(F256::ONE, |p, (i, &c)| {
        if (x >> i) & 1 == 1 { p * c } else { p * (F256::ONE + c) }
    })
}

/// `Z_S(x)`: the vanishing polynomial of the 64-point skip set.
pub fn skip_vanishing(x: F256) -> F256 {
    PHI_8_TABLE[..1 << ZK_FORM_K_SKIP]
        .iter()
        .fold(F256::ONE, |p, &node| p * (x + F256::from_base(node)))
}

/// The F256 generator X of F256 over F128 (lo 0, hi 1): an F256 coefficient
/// `c = c.lo + c.hi X` stored in two F128 cells is weighted `(w, w X)`.
pub const F256_X: F256 = F256::new(F128::ZERO, F128::ONE);

/// Overwrite with zeros through volatile writes.
pub fn wipe(values: &mut [F128]) {
    use zeroize::Zeroize;
    // SAFETY: F128 is repr(C, align(16)) with two u64 fields and no padding.
    let words =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u64, 2 * values.len()) };
    words.zeroize();
}

/// Fresh hiding randomness of ONE proving attempt:
/// one F128 per free cell, drawn from the system generator. Not `Clone`,
/// moved into the prover, wiped on drop. A retry needs a new draw.
pub struct ZkFreshRandomness {
    layout: ZkFormLayout,
    cells: Vec<F128>,
}

impl Drop for ZkFreshRandomness {
    fn drop(&mut self) {
        wipe(&mut self.cells);
    }
}

impl ZkFreshRandomness {
    pub fn layout(&self) -> ZkFormLayout {
        self.layout
    }
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    /// The buffer address, for the wipe test only.
    #[doc(hidden)]
    pub fn buffer_address(&self) -> usize {
        self.cells.as_ptr() as usize
    }
}

/// Draw one attempt's randomness. `fill` is the system generator (the client
/// passes `getrandom`); it is called once on the whole buffer.
pub fn draw_zk_form_randomness(
    layout: ZkFormLayout,
    fill: &mut dyn FnMut(&mut [u8]),
) -> ZkFreshRandomness {
    let count = layout.free_cells();
    let mut cells = vec![F128::ZERO; count];
    {
        // SAFETY: F128 is plain old data (two u64), every bit pattern valid.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(cells.as_mut_ptr() as *mut u8, 16 * count) };
        fill(bytes);
    }
    ZkFreshRandomness { layout, cells }
}

/// The zerocheck's Libra mask `g(s, x) = Z_S(s) h(s) + sum_i g_i(x_i)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkZerocheckMask {
    /// `h_0 .. h_62`.
    pub skip: Vec<F256>,
    /// `(g_{i,0}, g_{i,1}, g_{i,2})` for the rest bit i.
    pub rounds: Vec<[F256; 3]>,
}

/// The lincheck's mask, a Libra mask for a PLAIN sum:
/// `M(x) = sum_t g_t(x_t) prod_{k != t} eq(u_k, x_k)`, `deg g_t = 2`, `u` a
/// fixed public point. Summing over a free bit k multiplies by
/// `sum_{x_k} eq(u_k, x_k) = 1` (no characteristic-2 collapse), and the
/// round of bit t carries `E_t g_t(X) + eq(u_t, X) (D_t + E_t B_t)`: a
/// generic quadratic, two fresh coordinates per round (CERT-D''). The prover
/// fixes `g_{0,1}` so that `sum_x M(x) = sum_t (g_{t,1} + g_{t,2}) = 0`.
/// (An `eq(u, x) G'(x)` mask keeps the factor `eq(u_t, X)` on every round:
/// one fresh coordinate per round only.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkLincheckMask {
    /// `(g_{t,0}, g_{t,1}, g_{t,2})` for rest bit t.
    pub rounds: Vec<[F256; 3]>,
}

impl ZkLincheckMask {
    /// Weights of `M(point)` on the coefficients `g_{t,d}` (index 3t + d).
    pub fn weights_at(point: &[F256], u: &[F256]) -> Vec<F256> {
        let r = point.len();
        let eqs: Vec<F256> = point.iter().zip(u).map(|(&p, &u)| F256::ONE + p + u).collect();
        let mut out = Vec::with_capacity(3 * r);
        for t in 0..r {
            let others = (0..r).filter(|&k| k != t).fold(F256::ONE, |acc, k| acc * eqs[k]);
            out.extend([others, others * point[t], others * point[t] * point[t]]);
        }
        out
    }

    /// `M(point)`.
    pub fn evaluate(&self, point: &[F256], u: &[F256]) -> F256 {
        let flat = self.rounds.iter().flatten();
        Self::weights_at(point, u)
            .iter()
            .zip(flat)
            .fold(F256::ZERO, |sum, (&w, &c)| sum + w * c)
    }

    /// `sum_x M(x) = sum_t (g_{t,1} + g_{t,2})`.
    pub fn cube_sum(&self) -> F256 {
        self.rounds.iter().fold(F256::ZERO, |sum, g| sum + g[1] + g[2])
    }
}

/// A linear claim on a few committed cells (the masks' final values),
/// batched into the PCS opening beside the point claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkSparseClaim {
    pub cells: Vec<usize>,
    pub weights: Vec<F256>,
    pub value: F256,
}

impl ZkSparseClaim {
    pub fn evaluate(&self, z: &[F128]) -> F256 {
        self.cells
            .iter()
            .zip(&self.weights)
            .fold(F256::ZERO, |sum, (&cell, &w)| sum + w.scale_base(z[cell]))
    }
}

impl ZkFormLayout {
    fn read_f256(z: &[F128], cell_of: impl Fn(usize) -> usize, j: usize) -> F256 {
        F256::new(z[cell_of(2 * j)], z[cell_of(2 * j + 1)])
    }

    pub fn zerocheck_mask(&self, z: &[F128]) -> ZkZerocheckMask {
        let cell = |k: usize| self.zerocheck_mask_cell(k / 2, k % 2);
        let coefficient = |j: usize| Self::read_f256(z, cell, j);
        ZkZerocheckMask {
            skip: (0..ZK_SKIP_MASK_COEFFICIENTS).map(coefficient).collect(),
            rounds: (0..self.rounds())
                .map(|i| {
                    let base = ZK_SKIP_MASK_COEFFICIENTS + 3 * i;
                    [coefficient(base), coefficient(base + 1), coefficient(base + 2)]
                })
                .collect(),
        }
    }

    pub fn lincheck_mask(&self, z: &[F128]) -> ZkLincheckMask {
        let cell = |k: usize| self.lincheck_mask_cell(k / 2, k % 2);
        let coefficient = |j: usize| Self::read_f256(z, cell, j);
        ZkLincheckMask {
            rounds: (0..self.rounds())
                .map(|t| [coefficient(3 * t), coefficient(3 * t + 1), coefficient(3 * t + 2)])
                .collect(),
        }
    }

    /// Weights of `G(z, rho) = Z_S(z) h(z) + sum_i g_i(rho_i)` on the
    /// zerocheck mask cells (the value is supplied by the caller).
    pub fn zerocheck_mask_claim(&self, z_skip: F256, rho: &[F256], value: F256) -> ZkSparseClaim {
        assert_eq!(rho.len(), self.rounds());
        let mut coefficient_weights = Vec::with_capacity(self.zerocheck_mask_coefficients());
        let zs = skip_vanishing(z_skip);
        let mut power = F256::ONE;
        for _ in 0..ZK_SKIP_MASK_COEFFICIENTS {
            coefficient_weights.push(zs * power);
            power *= z_skip;
        }
        for &r in rho {
            coefficient_weights.extend([F256::ONE, r, r * r]);
        }
        self.sparse(coefficient_weights, |j, half| self.zerocheck_mask_cell(j, half), value)
    }

    /// Weights of `M(r_inner)` on the lincheck mask cells.
    pub fn lincheck_mask_claim(&self, r_inner: &[F256], value: F256) -> ZkSparseClaim {
        assert_eq!(r_inner.len(), self.rounds());
        let coefficient_weights = ZkLincheckMask::weights_at(r_inner, &self.lincheck_mask_point());
        self.sparse(coefficient_weights, |j, half| self.lincheck_mask_cell(j, half), value)
    }

    fn sparse(
        &self,
        coefficient_weights: Vec<F256>,
        cell: impl Fn(usize, usize) -> usize,
        value: F256,
    ) -> ZkSparseClaim {
        let mut cells = Vec::with_capacity(2 * coefficient_weights.len());
        let mut weights = Vec::with_capacity(2 * coefficient_weights.len());
        for (j, w) in coefficient_weights.into_iter().enumerate() {
            cells.push(cell(j, 0));
            weights.push(w);
            cells.push(cell(j, 1));
            weights.push(w * F256_X);
        }
        ZkSparseClaim {
            cells,
            weights,
            value,
        }
    }
}

/// A catalogue relation in form Z: the renamed catalogue matrices (no
/// tautology materialized) and the layout.
pub struct ZkFormRelation {
    layout: ZkFormLayout,
    physical: FieldR1cs,
}

/// The registration rule: a form-Z (physical) matrix has no entry on a free
/// row or a free column.
pub fn check_zk_form_registration(
    layout: &ZkFormLayout,
    physical: &FieldR1cs,
) -> Result<(), ZkFormError> {
    if physical.m != layout.m || physical.k_log != layout.m || physical.k_skip != ZK_FORM_K_SKIP {
        return Err(ZkFormError::Shape("physical shape"));
    }
    for matrix in [&physical.a_0, &physical.b_0] {
        for row in 0..matrix.num_rows {
            let free_row = layout.role(row).is_free();
            for (column, _) in matrix.row(row) {
                if free_row || layout.role(column as usize).is_free() {
                    return Err(ZkFormError::FreeCellEntry {
                        row,
                        column: column as usize,
                    });
                }
            }
        }
    }
    Ok(())
}

impl ZkFormRelation {
    /// Rename a catalogue relation (square, skip 6, constant at 0, IO in
    /// [8, 16)) into form Z.
    pub fn from_logical(layout: ZkFormLayout, logical: &FieldR1cs) -> Result<Self, ZkFormError> {
        layout.validate()?;
        if logical.k_skip != ZK_FORM_K_SKIP || logical.k_log != logical.m {
            return Err(ZkFormError::Shape("catalogue relation not square with skip 6"));
        }
        if logical.const_pin.is_some_and(|column| column != 0) {
            return Err(ZkFormError::Shape("constant pin not at cell 0"));
        }
        let n_phys = layout.cells();
        let map = |cell: usize| -> Result<usize, ZkFormError> {
            layout.physical_cell(cell).ok_or(ZkFormError::Capacity {
                logical_cell: cell,
                capacity: layout.useful_capacity(),
            })
        };
        let rename = |matrix: &SparseFieldMatrix| -> Result<SparseFieldMatrix, ZkFormError> {
            let mut rows: Vec<(usize, usize)> = Vec::new();
            for row in 0..matrix.num_rows {
                if !matrix.row_range(row).is_empty() {
                    rows.push((map(row)?, row));
                }
            }
            rows.sort_unstable();
            let mut row_offsets = vec![0usize; n_phys + 1];
            let mut col_indices = Vec::with_capacity(matrix.col_indices.len());
            let mut value_indices = Vec::with_capacity(matrix.value_indices.len());
            let mut next_phys_row = 0usize;
            for (phys_row, row) in rows {
                for slot in &mut row_offsets[next_phys_row..=phys_row] {
                    *slot = col_indices.len();
                }
                let range = matrix.row_range(row);
                let mut entries: Vec<(u32, u32)> = Vec::with_capacity(range.len());
                for index in range {
                    let column = map(matrix.col_indices[index] as usize)? as u32;
                    entries.push((column, matrix.value_indices[index]));
                }
                entries.sort_by_key(|&(column, _)| column);
                for (column, value) in entries {
                    col_indices.push(column);
                    value_indices.push(value);
                }
                next_phys_row = phys_row + 1;
            }
            for slot in &mut row_offsets[next_phys_row..=n_phys] {
                *slot = col_indices.len();
            }
            Ok(SparseFieldMatrix {
                num_rows: n_phys,
                num_cols: n_phys,
                col_indices,
                value_indices,
                value_table: matrix.value_table.clone(),
                row_offsets,
            })
        };
        let physical = FieldR1cs {
            m: layout.m,
            k_log: layout.m,
            k_skip: ZK_FORM_K_SKIP,
            useful_rows: n_phys,
            a_0: rename(&logical.a_0)?,
            b_0: rename(&logical.b_0)?,
            const_pin: logical.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        check_zk_form_registration(&layout, &physical)?;
        Ok(Self { layout, physical })
    }

    pub fn layout(&self) -> &ZkFormLayout {
        &self.layout
    }
    pub fn physical(&self) -> &FieldR1cs {
        &self.physical
    }

    /// The physical witness: the logical witness renamed, every free cell
    /// filled from `randomness` (consumed, then wiped), the lincheck mask
    /// constant set so that `sum_x M(x) = 0`.
    pub fn embed(
        &self,
        logical_witness: &[F128],
        randomness: ZkFreshRandomness,
    ) -> Result<Vec<F128>, ZkFormError> {
        let layout = &self.layout;
        if randomness.layout != *layout || randomness.cells.len() != layout.free_cells() {
            return Err(ZkFormError::Randomness);
        }
        let mut z = vec![F128::ZERO; layout.cells()];
        for (cell, &value) in logical_witness.iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            let phys = layout.physical_cell(cell).ok_or(ZkFormError::Capacity {
                logical_cell: cell,
                capacity: layout.useful_capacity(),
            })?;
            z[phys] = value;
        }
        let mut next = 0usize;
        for (cell, slot) in z.iter_mut().enumerate() {
            if layout.role(cell).is_free() {
                *slot = randomness.cells[next];
                next += 1;
            }
        }
        drop(randomness);
        let mask = layout.lincheck_mask(&z);
        // sum_x M(x) = 0: g_{0,1} absorbs the cube sum.
        let fixed = mask.cube_sum() + mask.rounds[0][1];
        z[layout.lincheck_mask_cell(1, 0)] = fixed.lo;
        z[layout.lincheck_mask_cell(1, 1)] = fixed.hi;
        Ok(z)
    }

    /// `(A z, B z)` of form Z: the catalogue part plus the tautology rows.
    pub fn apply_ab(&self, z: &[F128]) -> (Vec<F128>, Vec<F128>) {
        let mut a = self.physical.apply_a(z);
        let mut b = self.physical.apply_b(z);
        let layout = &self.layout;
        let constant = z[0];
        a.par_iter_mut()
            .zip(b.par_iter_mut())
            .enumerate()
            .for_each(|(cell, (a, b))| match layout.role(cell) {
                ZkCellRole::TypeB => {
                    *a += constant;
                    *b += z[cell];
                }
                role if role.is_free() => {
                    *a += z[cell];
                    *b += constant;
                }
                _ => {}
            });
        (a, b)
    }

    pub fn satisfies(&self, z: &[F128]) -> bool {
        let (a, b) = self.apply_ab(z);
        a.iter().zip(&b).zip(z).all(|((&a, &b), &z)| a * b == z)
    }

    pub fn lincheck_circuit(&self) -> ZkLincheckCircuit<'_> {
        ZkLincheckCircuit { relation: self }
    }
}

/// The lincheck combination of form Z: catalogue fold plus the tautology rows.
pub struct ZkLincheckCircuit<'a> {
    relation: &'a ZkFormRelation,
}

impl ZkLincheckCircuit<'_> {
    fn add_tautologies<T>(&self, comb: &mut [T], alpha: T, weight: impl Fn(usize) -> T + Sync)
    where
        T: Copy
            + Default
            + Send
            + Sync
            + core::ops::Add<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::AddAssign,
    {
        let layout = &self.relation.layout;
        let (sum_a, sum_b) = comb
            .par_iter_mut()
            .enumerate()
            .map(|(cell, slot)| match layout.role(cell) {
                ZkCellRole::TypeB => {
                    let w = weight(cell);
                    *slot += w;
                    (T::default(), w)
                }
                role if role.is_free() => {
                    let w = weight(cell);
                    *slot += alpha * w;
                    (w, T::default())
                }
                _ => (T::default(), T::default()),
            })
            .reduce(
                || (T::default(), T::default()),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        comb[0] += sum_a + alpha * sum_b;
    }
}

impl LincheckCircuit for ZkLincheckCircuit<'_> {
    fn n_cols(&self) -> usize {
        self.relation.layout.cells()
    }

    fn const_pin_col(&self) -> Option<usize> {
        self.relation.physical.const_pin
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        let mut comb = self.relation.physical.fold_alpha_batched(alpha, eq_inner);
        self.add_tautologies(&mut comb, alpha, |cell| eq_inner[cell]);
        comb
    }

    fn fold_alpha_batched_quirky_c1(
        &self,
        alpha: F256,
        z_skip: F256,
        x_inner_rest: &[F256],
        k_skip: usize,
    ) -> Vec<F256> {
        let mut comb =
            self.relation
                .physical
                .fold_alpha_batched_quirky_c1(alpha, z_skip, x_inner_rest, k_skip);
        let skip = lagrange_weights(k_skip, z_skip, 0);
        let rest = build_eq_table(x_inner_rest);
        let mask = (1usize << k_skip) - 1;
        self.add_tautologies(&mut comb, alpha, |cell| skip[cell & mask] * rest[cell >> k_skip]);
        comb
    }
}
