// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Public IO of a ClientBatchStep proof (prototype).
//!
//! | lanes | width | role |
//! |---|---:|---|
//! | `batch_digest` | 2 | `D_batch`, the batch matrix digest (carried, checked natively) |
//! | `post_commit` | 2 | the batch post-commit class digest (carried, checked natively) |
//! | `leaves` | 2·C | the registry leaves the whole tree verified clients against |
//! | `out_entries` | C·E | the outgoing accumulator lane of every registry entry |
//! | `in_entries` | C·E | incoming lanes (junction only; dead below the top) |
//! | `out_self` | S | the accumulated lincheck claims on `D_batch` |
//! | `in_self` | S | incoming `D_batch` lane (junction only) |
//! | `root` | 2 | ordered hash tree of the `(D, io_commitment)` covered |
//! | `count` | 1 | `g^n`, n = number of client proofs covered |
//! | `mask` | C | registry entries used by this tree |
//!
//! `E = 2·(2·k + 1) + 3` (point, value, live) for a client matrix of side
//! `2^k`, `S` likewise for the batch matrix.

use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;
use jetsam_ivc_core::merkle::Hash;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};

use crate::acceptance::trace::self_verify::flat_digest_lanes;
use crate::acceptance::trace::LinExpr;

/// One accumulator lane: `2·(2k+1)` point lanes, two value lanes, `live`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AccLane {
    pub(crate) point: usize,
    pub(crate) value: usize,
    pub(crate) live: usize,
    pub(crate) k_log: usize,
}

impl AccLane {
    pub(crate) const fn width(k_log: usize) -> usize {
        2 * (2 * k_log + 1) + 3
    }

    const fn at(offset: usize, k_log: usize) -> Self {
        let value = offset + 2 * (2 * k_log + 1);
        Self {
            point: offset,
            value,
            live: value + 2,
            k_log,
        }
    }

    pub(crate) fn install(&self, claim: Option<&C1MatrixAccClaim>, io: &mut [F128]) {
        io[self.point..=self.live].fill(F128::ZERO);
        let Some(claim) = claim else {
            return;
        };
        assert_eq!(claim.point.len(), 2 * self.k_log + 1, "lane width");
        for (pair, coordinate) in io[self.point..self.value]
            .chunks_exact_mut(2)
            .zip(&claim.point)
        {
            pair[0] = coordinate.lo;
            pair[1] = coordinate.hi;
        }
        io[self.value] = claim.value.lo;
        io[self.value + 1] = claim.value.hi;
        io[self.live] = F128::ONE;
    }

    /// The claim a canonical lane carries: `None` for a dead lane (all
    /// zero), `Err` for a non-canonical one.
    pub(crate) fn parse(&self, io: &[F128]) -> Result<Option<C1MatrixAccClaim>, ()> {
        let live = io[self.live];
        if live == F128::ZERO {
            return if io[self.point..self.live].iter().all(|cell| *cell == F128::ZERO) {
                Ok(None)
            } else {
                Err(())
            };
        }
        if live != F128::ONE {
            return Err(());
        }
        Ok(Some(C1MatrixAccClaim {
            point: io[self.point..self.value]
                .chunks_exact(2)
                .map(|pair| F256::new(pair[0], pair[1]))
                .collect(),
            value: F256::new(io[self.value], io[self.value + 1]),
        }))
    }

    /// The lane's cells in `io`, in order: point, value, live.
    pub(crate) fn cells<'a>(&self, io: &'a [LinExpr]) -> &'a [LinExpr] {
        &io[self.point..=self.live]
    }
}

/// Lane offsets of the batch public IO.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClientBatchIoLayout {
    pub(crate) k_log: usize,
    pub(crate) k_log_batch: usize,
    pub(crate) capacity: usize,
    pub(crate) batch_digest: usize,
    pub(crate) post_commit: usize,
    pub(crate) leaves: usize,
    pub(crate) out_entries: usize,
    pub(crate) in_entries: usize,
    pub(crate) out_self: usize,
    pub(crate) in_self: usize,
    pub(crate) root: usize,
    pub(crate) count: usize,
    pub(crate) mask: usize,
    pub(crate) len: usize,
}

impl ClientBatchIoLayout {
    pub(crate) fn new(k_log: usize, k_log_batch: usize, capacity: usize) -> Self {
        let entry = AccLane::width(k_log);
        let batch_digest = 0;
        let post_commit = batch_digest + 2;
        let leaves = post_commit + 2;
        let out_entries = leaves + 2 * capacity;
        let in_entries = out_entries + capacity * entry;
        let out_self = in_entries + capacity * entry;
        let in_self = out_self + AccLane::width(k_log_batch);
        let root = in_self + AccLane::width(k_log_batch);
        let count = root + 2;
        let mask = count + 1;
        let len = mask + capacity;
        Self {
            k_log,
            k_log_batch,
            capacity,
            batch_digest,
            post_commit,
            leaves,
            out_entries,
            in_entries,
            out_self,
            in_self,
            root,
            count,
            mask,
            len,
        }
    }

    pub(crate) fn out_entry(&self, index: usize) -> AccLane {
        assert!(index < self.capacity);
        AccLane::at(self.out_entries + index * AccLane::width(self.k_log), self.k_log)
    }

    pub(crate) fn in_entry(&self, index: usize) -> AccLane {
        assert!(index < self.capacity);
        AccLane::at(self.in_entries + index * AccLane::width(self.k_log), self.k_log)
    }

    pub(crate) fn out_self_lane(&self) -> AccLane {
        AccLane::at(self.out_self, self.k_log_batch)
    }

    pub(crate) fn in_self_lane(&self) -> AccLane {
        AccLane::at(self.in_self, self.k_log_batch)
    }

    pub(crate) fn leaf(&self, index: usize) -> usize {
        assert!(index < self.capacity);
        self.leaves + 2 * index
    }

    /// The public-IO spec: the IO block at slice index one.
    pub(crate) fn spec(&self) -> PublicIoSpec {
        let log2_len = self.len.next_power_of_two().trailing_zeros() as usize;
        PublicIoSpec {
            io_slice: WitnessSlice { log2_len, index: 1 },
            io_len: self.len,
            claims: Vec::new(),
        }
    }
}

/// The value every count lane is a power of: `count = g^n`.
pub(crate) fn count_generator() -> F128 {
    F128::new(2, 0)
}

/// `g^n` (native).
pub(crate) fn count_value(n: usize) -> F128 {
    let mut value = F128::ONE;
    for _ in 0..n {
        value *= count_generator();
    }
    value
}

/// `n` such that `count = g^n`, searched up to `bound`.
pub(crate) fn decode_count(count: F128, bound: usize) -> Option<usize> {
    let mut value = F128::ONE;
    for n in 0..=bound {
        if value == count {
            return Some(n);
        }
        value *= count_generator();
    }
    None
}

/// The native values of a batch IO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClientBatchPublic {
    pub(crate) batch_digest: Hash,
    pub(crate) post_commit: Hash,
    pub(crate) leaves: Vec<Hash>,
    pub(crate) out_entries: Vec<Option<C1MatrixAccClaim>>,
    pub(crate) in_entries: Vec<Option<C1MatrixAccClaim>>,
    pub(crate) out_self: Option<C1MatrixAccClaim>,
    pub(crate) in_self: Option<C1MatrixAccClaim>,
    pub(crate) root: [F128; 2],
    pub(crate) count: F128,
    pub(crate) mask: Vec<bool>,
}

impl ClientBatchPublic {
    pub(crate) fn install(&self, layout: &ClientBatchIoLayout) -> Vec<F128> {
        let mut io = vec![F128::ZERO; layout.len];
        io[layout.batch_digest..layout.batch_digest + 2]
            .copy_from_slice(&flat_digest_lanes(&self.batch_digest));
        io[layout.post_commit..layout.post_commit + 2]
            .copy_from_slice(&flat_digest_lanes(&self.post_commit));
        for (index, leaf) in self.leaves.iter().enumerate() {
            let at = layout.leaf(index);
            io[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
        }
        for index in 0..layout.capacity {
            layout
                .out_entry(index)
                .install(self.out_entries[index].as_ref(), &mut io);
            layout
                .in_entry(index)
                .install(self.in_entries[index].as_ref(), &mut io);
            io[layout.mask + index] = if self.mask[index] {
                F128::ONE
            } else {
                F128::ZERO
            };
        }
        layout.out_self_lane().install(self.out_self.as_ref(), &mut io);
        layout.in_self_lane().install(self.in_self.as_ref(), &mut io);
        io[layout.root..layout.root + 2].copy_from_slice(&self.root);
        io[layout.count] = self.count;
        io
    }

    pub(crate) fn parse(layout: &ClientBatchIoLayout, io: &[F128]) -> Result<Self, ()> {
        if io.len() != layout.len {
            return Err(());
        }
        let digest = |at: usize| -> Hash {
            let mut bytes = [0u8; 32];
            for (chunk, lane) in bytes.chunks_exact_mut(16).zip(&io[at..at + 2]) {
                chunk[..8].copy_from_slice(&lane.lo.to_le_bytes());
                chunk[8..].copy_from_slice(&lane.hi.to_le_bytes());
            }
            bytes
        };
        let mut mask = Vec::with_capacity(layout.capacity);
        for index in 0..layout.capacity {
            mask.push(match io[layout.mask + index] {
                value if value == F128::ONE => true,
                value if value == F128::ZERO => false,
                _ => return Err(()),
            });
        }
        let parsed = Self {
            batch_digest: digest(layout.batch_digest),
            post_commit: digest(layout.post_commit),
            leaves: (0..layout.capacity).map(|i| digest(layout.leaf(i))).collect(),
            out_entries: (0..layout.capacity)
                .map(|i| layout.out_entry(i).parse(io))
                .collect::<Result<_, _>>()?,
            in_entries: (0..layout.capacity)
                .map(|i| layout.in_entry(i).parse(io))
                .collect::<Result<_, _>>()?,
            out_self: layout.out_self_lane().parse(io)?,
            in_self: layout.in_self_lane().parse(io)?,
            root: [io[layout.root], io[layout.root + 1]],
            count: io[layout.count],
            mask,
        };
        // Canonical digest encoding: the round trip must be exact.
        if flat_digest_lanes(&parsed.batch_digest)[..] != io[layout.batch_digest..layout.batch_digest + 2]
        {
            return Err(());
        }
        Ok(parsed)
    }
}
