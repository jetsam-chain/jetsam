// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Poseidon2b proof-of-work validation.
//!
//! The mining digest is a fixed-field Poseidon2b sponge over the semantic
//! header fields, under the dedicated `POWHDR__` capacity IV.
//!
//! The absorbed field schedule is fixed at 16 `Block128` elements:
//!
//! ```text
//! prev_block_hash       2 fields
//! state_root            2 fields
//! tx_root               2 fields
//! timestamp             1 field   zero-extended LE u64
//! height                1 field   zero-extended LE u64
//! miner_address         2 fields
//! nonce                 1 field   LE u128
//! difficulty_target     2 fields
//! log_slots             1 field   zero-extended LE u32
//! active_slot_count     1 field   zero-extended LE u64
//! alloc_counter         1 field   zero-extended LE u64
//! ```
//!
//! With Poseidon2b `rate = 2`, this is exactly eight rate blocks and uses
//! `finalize_no_pad()`. Domain separation comes from `POWHDR__`; the semantic
//! chain-link id uses the distinct `BLOCKHDR` domain.

use crate::block_header::BlockHeader;
use crate::consensus::{difficulty::le256_lt, ConsensusError};
use jetsam_core::{Block128, TowerField};
use jetsam_poseidon2b::batch::FixedFieldNonceBatch;
use jetsam_poseidon2b::native::compression::Poseidon2bSponge;
use jetsam_poseidon2b::native::domain::{capacity_iv, TAG_POWHDR};

/// Consensus semantic block id.
pub type BlockHash = [u8; 32];

/// Number of `Block128` field elements absorbed by `poseidon_pow_digest`.
pub const POW_HEADER_FIELD_COUNT: usize = 16;

/// Index of the nonce field in the fixed PoW field schedule.
///
/// JETSAM CHANGE: 0, was 10 upstream.
///
/// A Poseidon2b sponge absorbs two fields per permutation, so with the nonce at
/// lane 10 of 16 the first five permutations cover only template-fixed fields
/// and can be precomputed once as a midstate: a highly optimised miner replays
/// three permutations per attempt instead of eight, a 2.67x advantage over a
/// straightforward implementation. A sponge cannot be run backwards, so putting
/// the nonce in the FIRST absorbed pair removes that shortcut entirely — every
/// attempt runs the full eight permutations, for everyone.
///
/// This costs us nothing: the in-tree hasher never exploited the midstate. It
/// removes an edge that a third party could have taken, and narrows the gap a
/// future ASIC could open.
pub const POW_NONCE_FIELD_INDEX: usize = 0;

pub type PowHeaderFields = [Block128; POW_HEADER_FIELD_COUNT];

/// Prepared production PoW hasher. One value should be retained by each mining
/// worker so the fixed flat-basis header and scratch allocation are reused
/// across nonce batches.
pub struct PowNonceBatchHasher {
    inner: FixedFieldNonceBatch,
}

impl PowNonceBatchHasher {
    pub fn new(fields: &PowHeaderFields) -> Self {
        Self {
            inner: FixedFieldNonceBatch::new(TAG_POWHDR, fields, POW_NONCE_FIELD_INDEX),
        }
    }

    #[inline]
    pub fn hash_into(&mut self, start_nonce: u128, out: &mut [[u8; 32]]) {
        self.inner.hash_into(start_nonce, out);
    }
}

/// Compute the semantic block id (used as `prev_block_hash` in the next block).
#[inline]
pub fn block_id(h: &BlockHeader) -> BlockHash {
    crate::block_header::block_id(h)
}

/// Fill the fixed Poseidon2b PoW field schedule for a header.
/// JETSAM CHANGE: the nonce is emitted FIRST, so it lands in the first absorbed
/// pair and no midstate can be precomputed across attempts. `block_header::
/// hash_block_header` MUST absorb in exactly this order — nothing links the two
/// at compile time, and a divergence would not fail to build, it would make
/// every HistoryStep proof unsatisfiable. `header_order_matches_block_id` in
/// this module is the guard.
pub fn pow_header_fields_into(h: &BlockHeader, out: &mut PowHeaderFields) {
    let mut i = 0usize;
    debug_assert_eq!(i, POW_NONCE_FIELD_INDEX);
    out[i] = Block128::from(h.nonce);
    i += 1;
    put_digest(out, &mut i, &h.prev_block_hash);
    put_digest(out, &mut i, &h.state_root);
    put_digest(out, &mut i, &h.tx_root);
    out[i] = Block128::from(h.timestamp as u128);
    i += 1;
    out[i] = Block128::from(h.height as u128);
    i += 1;
    put_digest(out, &mut i, h.miner_address.as_bytes());
    put_digest(out, &mut i, &h.difficulty_target);
    out[i] = Block128::from(h.log_slots as u128);
    i += 1;
    out[i] = Block128::from(h.active_slot_count as u128);
    i += 1;
    out[i] = Block128::from(h.alloc_counter as u128);
    i += 1;
    debug_assert_eq!(i, POW_HEADER_FIELD_COUNT);
}

/// Return the fixed Poseidon2b PoW field schedule for a header.
pub fn pow_header_fields(h: &BlockHeader) -> PowHeaderFields {
    let mut fields = [Block128::ZERO; POW_HEADER_FIELD_COUNT];
    pow_header_fields_into(h, &mut fields);
    fields
}

/// Compute `H_POSEIDON_POW(header)`.
#[inline]
pub fn poseidon_pow_digest(header: &BlockHeader) -> BlockHash {
    poseidon_pow_digest_from_fields(&pow_header_fields(header))
}

/// Compute `H_POSEIDON_POW(fields)` from an already materialized field schedule.
pub fn poseidon_pow_digest_from_fields(fields: &PowHeaderFields) -> BlockHash {
    let mut sponge = Poseidon2bSponge::with_iv(capacity_iv(TAG_POWHDR));
    for chunk in fields.chunks_exact(2) {
        sponge.absorb_pair(chunk[0], chunk[1]);
    }
    sponge.finalize_no_pad()
}

/// Compute `H_POSEIDON_POW` for consecutive nonce values.
///
/// `out[i]` is the digest for `fields` with nonce `start_nonce + i`.
/// The packed path is consensus-equivalent to [`poseidon_pow_digest_from_fields`]
/// and only changes how many independent permutations are evaluated together.
pub fn poseidon_pow_digest_nonce_batch(
    fields: &PowHeaderFields,
    start_nonce: u128,
    out: &mut [[u8; 32]],
) {
    PowNonceBatchHasher::new(fields).hash_into(start_nonce, out);
}

/// Validate that the block satisfies the declared PoW target.
///
/// Computes `pow_digest = H_POSEIDON_POW(header)` and checks
/// `pow_digest < header.difficulty_target` as little-endian 256-bit integers.
/// Equality is rejected.
pub fn validate_pow(header: &BlockHeader) -> Result<BlockHash, ConsensusError> {
    let digest = pow_digest(header);
    if le256_lt(&digest, &header.difficulty_target) {
        Ok(digest)
    } else {
        Err(ConsensusError::InvalidPoW)
    }
}

/// The mining digest a block at this height must satisfy.
///
/// Below [`params::V1_4_ACTIVATION_HEIGHT`] — which ships as `None`, so everywhere
/// today — this is exactly [`poseidon_pow_digest`], byte for byte. From that height
/// on, the sponge digest becomes the **seed** of the cache-resident walk and the
/// walk's output is what must beat the target.
///
/// Seeding the walk from the sponge rather than from the raw header is what keeps
/// the fork free: the field schedule, the domain tag and the nonce position are
/// untouched, so `pow_header_fields` and every proof that depends on it keep
/// working, and the nonce still never enters the circuit.
///
/// **Cost, measured 2026-09-23.** Best of ten runs of 150 headers on each machine,
/// one binary, production dispatch:
///
/// | | sponge only | sponge + walk | factor |
/// |---|---:|---:|---:|
/// | EPYC 7742, Zen 2 | 21.6 µs | 1.523 ms | **71x** |
/// | Ryzen 5950X, Zen 3 | 18.6 µs | 1.241 ms | 67x |
/// | Ryzen 7950X3D, Zen 4 | 15.1 µs | 1.069 ms | 71x |
///
/// A second harness, on an *exclusive* EPYC 7742 core and reporting the **mean** of
/// five runs rather than the best of ten, gives **21.4 µs and 1.854 ms — 87x** on
/// the same production function. Both are real; best-of-ten on a busy machine picks
/// the luckiest window, a mean does not. **Size anything operational on the slower
/// end**: a resync estimate that is optimistic is worse than one that is not.
///
/// The factor barely moves across three microarchitectures. Its machine-independent
/// form, from `perf stat` — instruction counts, unlike cycles, are trustworthy on a
/// loaded machine — is **133 149 instructions per header before, 10 991 148 after:
/// 82.5x**. Wall-clock comes out lower than that because the walk retires more per
/// cycle than the sponge does (IPC 2.10 against 1.80).
///
/// Resyncing 500 000 headers goes from 11 s to between **12.7 and 15.5 minutes
/// single-threaded** on Zen 2, the spread being the two harnesses above. Header PoW depends on nothing outside its own header, so the mitigation is
/// threads and not consensus: on the same machine 32 of them do it in **33 s, at
/// 78 % scaling efficiency**. Past one thread per *physical* core, two pads share
/// one 512 KiB L2 and the efficiency falls — bulk verification should be sized on
/// cores, not on hyperthreads.
///
/// Against a node running *today's* release the factor is smaller: that release has
/// no PCLMULQDQ kernel, so its sponge costs several times the 21.6 µs above. The
/// release that carries this fork carries the kernel too. Both numbers are true and
/// they answer different questions; quote the one that matches what is compared.
#[inline]
pub fn pow_digest(header: &BlockHeader) -> BlockHash {
    pow_digest_with(header, crate::consensus::params::V1_4_ACTIVATION_HEIGHT)
}

/// Testable twin of [`pow_digest`] with the activation height injected.
///
/// The production constant is `None`, so without this the walk branch of the
/// consensus digest cannot be executed by any test in the tree — the first header
/// ever to take it would be a real one, on a real network, at a height chosen
/// months earlier. Same dormant-twin device as `params::v1_4_active_with` and
/// `header::asert_anchor_height_with`.
pub(crate) fn pow_digest_with(header: &BlockHeader, activation: Option<u64>) -> BlockHash {
    let seed = poseidon_pow_digest(header);
    if crate::consensus::params::v1_4_active_with(header.height, activation) {
        // The scratchpad is held per thread, not allocated per header. A fresh
        // `Vec` would be 512 KiB of `calloc` and 128 page faults on top of a hash
        // that already costs 1.5 ms, on the one path — initial sync, snapshot
        // staging — that runs it thousands of times in a row. The fill overwrites
        // every cell before it is read, so a reused pad gives a bit-identical
        // digest (`a_reused_scratchpad_leaks_no_state` is the guard).
        thread_local! {
            static SCRATCH: std::cell::RefCell<crate::consensus::pow_walk::Scratch> =
                std::cell::RefCell::new(crate::consensus::pow_walk::Scratch::new());
        }
        SCRATCH.with(|scratch| {
            crate::consensus::pow_walk::towerwalk_digest_with(&mut scratch.borrow_mut(), &seed)
        })
    } else {
        seed
    }
}

/// Search for a valid PoW nonce in `[start, start + range)`.
///
/// Above [`params::V1_4_ACTIVATION_HEIGHT`] each candidate seed is walked before it
/// is compared, so the batched sponge stays exactly as profitable as it was — which
/// is to say, not at all: the walk dominates by four orders of magnitude and the
/// batch only amortises the seed. The scratchpad is allocated once for the whole
/// search, never per nonce.
pub fn search_pow(header_template: &BlockHeader, start_nonce: u128, range: u128) -> Option<u128> {
    search_pow_with(
        header_template,
        start_nonce,
        range,
        crate::consensus::params::V1_4_ACTIVATION_HEIGHT,
    )
}

/// Testable twin of [`search_pow`], for the same reason as [`pow_digest_with`]:
/// a miner and a node that disagreed about which digest a height demands would
/// mine forever and be refused forever, and nothing below the activation height
/// can tell them apart.
pub(crate) fn search_pow_with(
    header_template: &BlockHeader,
    start_nonce: u128,
    range: u128,
    activation: Option<u64>,
) -> Option<u128> {
    let fields = pow_header_fields(header_template);
    let mut hasher = PowNonceBatchHasher::new(&fields);
    let target = &header_template.difficulty_target;
    let walk = crate::consensus::params::v1_4_active_with(header_template.height, activation);
    let mut scratch = walk.then(crate::consensus::pow_walk::Scratch::new);
    let mut digests = [[0u8; 32]; 64];
    let mut done = 0u128;
    while done < range {
        let batch_len = ((range - done).min(digests.len() as u128)) as usize;
        let nonce_base = start_nonce.saturating_add(done);
        hasher.hash_into(nonce_base, &mut digests[..batch_len]);
        for (i, digest) in digests[..batch_len].iter().enumerate() {
            let candidate = match scratch.as_mut() {
                Some(sc) => crate::consensus::pow_walk::towerwalk_digest_with(sc, digest),
                None => *digest,
            };
            if le256_lt(&candidate, target) {
                return Some(nonce_base.saturating_add(i as u128));
            }
        }
        done += batch_len as u128;
    }
    None
}

#[inline]
fn put_digest(out: &mut PowHeaderFields, index: &mut usize, digest: &[u8; 32]) {
    out[*index] = Block128::from(u128::from_le_bytes(digest[..16].try_into().unwrap()));
    *index += 1;
    out[*index] = Block128::from(u128::from_le_bytes(digest[16..].try_into().unwrap()));
    *index += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_header::BlockHeader;
    use jetsam_core::packed::PACKED_LANES;
    use jetsam_poseidon2b::native::domain::{capacity_iv, TAG_BLOCKHDR};
    use jetsam_poseidon2b::primitives::Address;

    const TEST_TARGET: [u8; 32] = [0xFF; 32];

    fn dummy_header() -> BlockHeader {
        BlockHeader {
            prev_block_hash: [0u8; 32],
            state_root: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 1_700_000_000,
            height: 1,
            miner_address: Address([3u8; 32]),
            nonce: 0,
            difficulty_target: TEST_TARGET,
            log_slots: 24,
            active_slot_count: 0,
            alloc_counter: 0,
        }
    }

    #[test]
    fn pow_field_schedule_is_fixed_and_canonical() {
        let mut h = dummy_header();
        h.prev_block_hash = [0x10; 32];
        h.state_root = [0x20; 32];
        h.tx_root = [0x30; 32];
        h.timestamp = 0x0102_0304_0506_0708;
        h.height = 0x1112_1314_1516_1718;
        h.miner_address = Address([0x40; 32]);
        h.nonce = 0x2122_2324_2526_2728_3132_3334_3536_3738;
        h.difficulty_target = [0x50; 32];
        h.log_slots = 0x6162_6364;
        h.active_slot_count = 0x7172_7374_7576_7778;
        h.alloc_counter = 0x8182_8384_8586_8788;

        let fields = pow_header_fields(&h);
        assert_eq!(fields.len(), POW_HEADER_FIELD_COUNT);
        // JETSAM CHANGE: the nonce leads the schedule so no midstate is possible.
        assert_eq!(POW_NONCE_FIELD_INDEX, 0);
        assert_eq!(fields[0], Block128::from(h.nonce));
        assert_eq!(fields[1], digest_half(&h.prev_block_hash, 0));
        assert_eq!(fields[2], digest_half(&h.prev_block_hash, 1));
        assert_eq!(fields[3], digest_half(&h.state_root, 0));
        assert_eq!(fields[4], digest_half(&h.state_root, 1));
        assert_eq!(fields[5], digest_half(&h.tx_root, 0));
        assert_eq!(fields[6], digest_half(&h.tx_root, 1));
        assert_eq!(fields[7], Block128::from(h.timestamp as u128));
        assert_eq!(fields[8], Block128::from(h.height as u128));
        assert_eq!(fields[9], digest_half(h.miner_address.as_bytes(), 0));
        assert_eq!(fields[10], digest_half(h.miner_address.as_bytes(), 1));
        assert_eq!(fields[11], digest_half(&h.difficulty_target, 0));
        assert_eq!(fields[12], digest_half(&h.difficulty_target, 1));
        assert_eq!(fields[13], Block128::from(h.log_slots as u128));
        assert_eq!(fields[14], Block128::from(h.active_slot_count as u128));
        assert_eq!(fields[15], Block128::from(h.alloc_counter as u128));
    }

    /// JETSAM — the guard against a SILENT divergence.
    ///
    /// `pow_header_fields_into` and `block_header::hash_block_header` must
    /// absorb the same fields in the same order: the recursive parent-seal
    /// replays the block id in-circuit from the PoW field vector. Nothing links
    /// the two functions at compile time, so if one moves without the other the
    /// build stays green and every HistoryStep proof becomes unsatisfiable.
    /// This test recomputes the block id from the PoW schedule and requires the
    /// two to agree.
    #[test]
    fn header_order_matches_block_id() {
        use jetsam_poseidon2b::native::compression::Poseidon2bSponge;

        let mut h = dummy_header();
        h.prev_block_hash = [0xA1; 32];
        h.state_root = [0xB2; 32];
        h.tx_root = [0xC3; 32];
        h.timestamp = 1_700_000_000;
        h.height = 987_654;
        h.miner_address = Address([0xD4; 32]);
        h.nonce = 0x1234_5678_9abc_def0_1122_3344_5566_7788;
        h.difficulty_target = [0xE5; 32];
        h.log_slots = 24;
        h.active_slot_count = 4_242;
        h.alloc_counter = 9_999;

        let mut sponge = Poseidon2bSponge::with_iv(capacity_iv(TAG_BLOCKHDR));
        for field in pow_header_fields(&h) {
            sponge.absorb(field);
        }
        assert_eq!(
            sponge.finalize(),
            crate::block_header::hash_block_header(&h),
            "the PoW field order and hash_block_header have diverged"
        );
    }

    /// The nonce must sit in the FIRST absorbed pair, or a midstate becomes
    /// possible again. A Poseidon2b sponge absorbs two fields per permutation.
    #[test]
    fn nonce_is_inside_the_first_absorbed_pair() {
        assert!(
            POW_NONCE_FIELD_INDEX < 2,
            "nonce at lane {POW_NONCE_FIELD_INDEX} leaves a precomputable prefix"
        );
    }

    #[test]
    fn pow_digest_uses_distinct_domain_from_block_id() {
        let h = dummy_header();
        assert_ne!(capacity_iv(TAG_POWHDR), capacity_iv(TAG_BLOCKHDR));
        assert_ne!(poseidon_pow_digest(&h), block_id(&h));
    }

    #[test]
    fn nonce_change_changes_pow_digest() {
        let mut h = dummy_header();
        let digest1 = poseidon_pow_digest(&h);
        h.nonce = 42;
        let digest2 = poseidon_pow_digest(&h);
        assert_ne!(digest1, digest2);
    }

    #[test]
    fn pow_nonce_batch_matches_scalar_for_partial_and_full_chunks() {
        let h = dummy_header();
        let fields = pow_header_fields(&h);
        let start = 123_456u128;
        let n = PACKED_LANES * 3 + 1;
        let mut batch = vec![[0u8; 32]; n];

        poseidon_pow_digest_nonce_batch(&fields, start, &mut batch);

        for (i, digest) in batch.iter().enumerate() {
            let mut scalar_fields = fields;
            scalar_fields[POW_NONCE_FIELD_INDEX] = Block128::from(start + i as u128);
            assert_eq!(*digest, poseidon_pow_digest_from_fields(&scalar_fields));
        }
    }

    #[test]
    fn search_pow_returns_nonce_that_validates() {
        let mut h = dummy_header();
        let nonce = search_pow(&h, 0, 16).expect("max target should accept quickly");
        h.nonce = nonce;
        assert!(validate_pow(&h).is_ok());
    }

    /// The fork's money path, on the one branch no production constant can reach.
    ///
    /// `V1_4_ACTIVATION_HEIGHT` is `None`, so the walk branch of `pow_digest` and
    /// the walk branch of `search_pow` are dead code in every build that ships. The
    /// first header to take either of them would otherwise be a real one, at a
    /// height chosen months earlier, on a network that cannot be rolled back. This
    /// arms both through their dormant twins and checks the only property that
    /// matters: **what the miner searched for is what the node demands.**
    ///
    /// The target is deliberately not the maximum one — at `[0xFF; 32]` the first
    /// nonce tried wins on either algorithm and the test would pass without the
    /// walk ever mattering. One byte in sixteen means the search has to reject
    /// walked candidates before it accepts one.
    #[test]
    fn the_armed_walk_is_what_the_miner_searches_and_what_the_node_demands() {
        let armed = Some(7u64);
        let mut h = dummy_header();
        h.height = 7;
        h.difficulty_target = {
            let mut t = [0xFFu8; 32];
            t[31] = 0x10;
            t
        };

        let nonce = search_pow_with(&h, 0, 4_096, armed)
            .expect("a target this easy is found in a few dozen walks");
        h.nonce = nonce;

        let walked = pow_digest_with(&h, armed);
        let sponge = poseidon_pow_digest(&h);

        assert!(
            le256_lt(&walked, &h.difficulty_target),
            "the miner returned a nonce the node would reject"
        );
        assert_ne!(
            walked, sponge,
            "the armed digest is the sponge: the walk branch did not run"
        );
        assert_eq!(
            walked,
            crate::consensus::pow_walk::towerwalk_digest(&sponge),
            "the node's armed digest is not the published walk of the sponge"
        );

        // One height below activation the same header keeps the old digest, and the
        // old search still satisfies it. The fork is a switch, not a replacement.
        assert_eq!(
            pow_digest_with(&h, Some(8)),
            sponge,
            "height 7 took the walk under an activation height of 8"
        );
        let mut old = dummy_header();
        old.height = 7;
        old.difficulty_target = h.difficulty_target;
        let old_nonce =
            search_pow_with(&old, 0, 4_096, Some(8)).expect("the sponge search still terminates");
        old.nonce = old_nonce;
        assert!(
            le256_lt(&pow_digest_with(&old, Some(8)), &old.difficulty_target),
            "the pre-fork path no longer agrees with itself"
        );
    }

    #[test]
    fn validate_pow_rejects_zero_target() {
        let mut h = dummy_header();
        h.difficulty_target = [0u8; 32];
        assert_eq!(validate_pow(&h), Err(ConsensusError::InvalidPoW));
    }

    #[test]
    fn validate_pow_accepts_easy_target() {
        let mut h = dummy_header();
        h.nonce = 0;
        assert!(validate_pow(&h).is_ok());
    }

    #[test]
    fn le256_lt_correctness() {
        let zero = [0u8; 32];
        let mut one = [0u8; 32];
        one[0] = 1;
        let mut big = [0u8; 32];
        big[31] = 1;

        assert!(le256_lt(&zero, &one));
        assert!(le256_lt(&one, &big));
        assert!(!le256_lt(&big, &zero));
        assert!(!le256_lt(&one, &one));
    }

    #[inline]
    fn digest_half(digest: &[u8; 32], half: usize) -> Block128 {
        let offset = half * 16;
        Block128::from(u128::from_le_bytes(
            digest[offset..offset + 16].try_into().unwrap(),
        ))
    }
}
