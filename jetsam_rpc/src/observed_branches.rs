// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `getObservedBranches {since_unix}` (plan T3.10, design T1.3 §10.2): the
//! branches this node refused, recorded and weighed by the node's branch
//! observer (`jetsam_node::branch_observer`).
//!
//! The node holds the observer; the RPC reaches it through the
//! [`ObservedBranchesSource`] handed to the server when it starts. The
//! response types live here, with the method that serves them.
//!
//! The observer bounds a response at 16 branches of 2 048 headers. In hex
//! that is about 14 MB, more than the 10 MiB the transport sends in one
//! response: it would refuse the response whole, exactly when the most is to
//! be seen. The method also bounds the headers of one response, all branches
//! together ([`MAX_RESPONSE_TOTAL_HEADERS`]); the heaviest branches, listed
//! first, keep theirs.

use jetsam_p2p::object_protocol::ChainPoint;
use jsonrpsee::core::RpcResult;
use jsonrpsee::types::ErrorObject;
use serde::{Deserialize, Serialize};

/// Headers in one response at most, all branches together: eight branches
/// of the observer's 2 048, about 7 MB of hex, under the transport's 10 MiB.
pub const MAX_RESPONSE_TOTAL_HEADERS: usize = 16_384;

/// What this node made of a recorded branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchVerdict {
    /// This node reorganized onto it.
    Accepted,
    /// It forks below the highest pin this node carries (rule (o)).
    PinMismatch,
    /// One of its headers fails a native header rule.
    HeaderRule,
    /// Its headers pass, but it forks deeper than finality allows.
    DeeperThanFinality,
    /// Its headers pass and it forks inside finality: fork choice decides.
    Pending,
}

/// A block, hex id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedPoint {
    pub height: u64,
    pub id: String,
}

/// One branch of `getObservedBranches`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedBranchEntry {
    pub fork_height: u64,
    pub fork_base_id: String,
    pub tip_height: u64,
    pub tip_id: String,
    /// Every header from `fork + 1` passes the native header rules.
    pub headers_valid: bool,
    pub verdict: BranchVerdict,
    /// The first refused header and its rule.
    pub rule: Option<String>,
    pub below_last_pin: bool,
    pub reorg_depth: Option<u64>,
    /// Recomputed work through the last valid header, 32 bytes little-endian.
    pub work_hex: String,
    pub first_seen_unix: u64,
    pub last_seen_unix: u64,
    pub canonical_height_at_last_seen: u64,
    pub canonical_id_at_last_seen: String,
    /// Raw 212-byte headers from `fork + 1`, hex.
    pub headers: Vec<String>,
    pub headers_truncated: bool,
}

/// The `getObservedBranches` response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedBranchesResponse {
    pub now_unix: u64,
    /// Start of the record, restarts included.
    pub recording_since_unix: u64,
    pub canonical: ObservedPoint,
    /// The node's own tip work: to be used only when the list is truncated.
    pub node_tip_work_hex: String,
    pub last_pin_height: Option<u64>,
    /// Branches were left out.
    pub truncated: bool,
    pub branches: Vec<ObservedBranchEntry>,
}

/// What the node's branch observer answers.
pub trait ObservedBranchesSource: Send + Sync {
    /// The branches seen since `since_unix`, the heaviest first, as the
    /// observer bounds them, against this node's canonical tip and its work.
    /// It may wait for the observer: it is called on a blocking worker.
    fn observed_branches(
        &self,
        since_unix: u64,
        canonical_tip: ChainPoint,
        node_tip_work: [u8; 32],
        now_unix: u64,
    ) -> ObservedBranchesResponse;
}

pub type SharedObservedBranches = std::sync::Arc<dyn ObservedBranchesSource>;

/// `response` with at most [`MAX_RESPONSE_TOTAL_HEADERS`] headers: the
/// branches keep theirs in order (the heaviest first) while the bound
/// allows; a branch past it keeps what is left and says so
/// (`headers_truncated`). Every branch stays listed.
pub fn bounded_for_transport(mut response: ObservedBranchesResponse) -> ObservedBranchesResponse {
    let mut left = MAX_RESPONSE_TOTAL_HEADERS;
    for branch in &mut response.branches {
        if branch.headers.len() > left {
            branch.headers.truncate(left);
            branch.headers_truncated = true;
        }
        left -= branch.headers.len();
    }
    response
}

/// `getObservedBranches {since_unix}`: `source` asked on a blocking worker,
/// the response bounded for the transport.
pub async fn get_observed_branches(
    source: SharedObservedBranches,
    since_unix: u64,
    canonical_tip: ChainPoint,
    node_tip_work: [u8; 32],
) -> RpcResult<ObservedBranchesResponse> {
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let response = tokio::task::spawn_blocking(move || {
        source.observed_branches(since_unix, canonical_tip, node_tip_work, now_unix)
    })
    .await
    .map_err(|error| {
        ErrorObject::owned(
            -32000,
            format!("branch observer failed: {error}"),
            None::<()>,
        )
    })?;
    Ok(bounded_for_transport(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transport's limit on one response (jsonrpsee's default).
    const TRANSPORT_LIMIT: usize = 10 * 1024 * 1024;

    fn entry(rank: u8, headers: usize, truncated: bool) -> ObservedBranchEntry {
        ObservedBranchEntry {
            fork_height: 38_399,
            fork_base_id: hex::encode([rank; 32]),
            tip_height: 38_399 + headers as u64,
            tip_id: hex::encode([rank.wrapping_add(1); 32]),
            headers_valid: true,
            verdict: BranchVerdict::PinMismatch,
            rule: None,
            below_last_pin: true,
            reorg_depth: None,
            work_hex: hex::encode([0xFF - rank; 32]),
            first_seen_unix: 1_800_000_000,
            last_seen_unix: 1_800_000_600,
            canonical_height_at_last_seen: 39_000,
            canonical_id_at_last_seen: hex::encode([0xC0; 32]),
            // A raw header is 212 bytes: 424 hex digits.
            headers: vec![hex::encode([rank; 212]); headers],
            headers_truncated: truncated,
        }
    }

    fn response(branches: Vec<ObservedBranchEntry>) -> ObservedBranchesResponse {
        ObservedBranchesResponse {
            now_unix: 1_800_000_700,
            recording_since_unix: 1_799_990_000,
            canonical: ObservedPoint {
                height: 39_000,
                id: hex::encode([0xC0; 32]),
            },
            node_tip_work_hex: hex::encode([0x11; 32]),
            last_pin_height: Some(38_400),
            truncated: false,
            branches,
        }
    }

    fn total_headers(response: &ObservedBranchesResponse) -> usize {
        response
            .branches
            .iter()
            .map(|branch| branch.headers.len())
            .sum()
    }

    /// The observer's own bounds, 16 branches of 2 048 headers, make a
    /// response the transport refuses whole. Bounded, it is sent: every
    /// branch is listed, the heaviest (first) whole, the others marked.
    #[test]
    fn observed_branches_response_fits_the_transport_limit() {
        let full = response((0..16).map(|rank| entry(rank, 2_048, false)).collect());
        assert!(
            serde_json::to_vec(&full).unwrap().len() > TRANSPORT_LIMIT,
            "unbounded, the transport refuses it"
        );
        let sent = bounded_for_transport(full.clone());
        let bytes = serde_json::to_vec(&sent).unwrap().len();
        assert!(bytes < TRANSPORT_LIMIT - 64 * 1024, "{bytes} bytes");
        assert_eq!(total_headers(&sent), MAX_RESPONSE_TOTAL_HEADERS);
        assert_eq!(sent.branches.len(), 16, "every branch is still listed");
        for (rank, (sent, full)) in sent.branches.iter().zip(&full.branches).enumerate() {
            let kept = if rank < 8 { 2_048 } else { 0 };
            assert_eq!(sent.headers[..], full.headers[..kept], "branch {rank}");
            assert_eq!(sent.headers_truncated, rank >= 8, "branch {rank}");
            assert_eq!(
                (sent.tip_height, &sent.work_hex),
                (full.tip_height, &full.work_hex),
                "the branch itself is unchanged"
            );
        }
        // A branch reaching the bound keeps what is left.
        let partial = bounded_for_transport(response(vec![
            entry(0, 2_048, false),
            entry(1, MAX_RESPONSE_TOTAL_HEADERS - 2_048 - 100, false),
            entry(2, 300, false),
            entry(3, 5, false),
        ]));
        let kept = partial
            .branches
            .iter()
            .map(|branch| (branch.headers.len(), branch.headers_truncated))
            .collect::<Vec<_>>();
        assert_eq!(
            kept,
            [
                (2_048, false),
                (MAX_RESPONSE_TOTAL_HEADERS - 2_048 - 100, false),
                (100, true),
                (0, true),
            ]
        );
        // Within the bound nothing changes, an observer's own truncation
        // included.
        let small = response(vec![entry(0, 3, false), entry(1, 2_048, true)]);
        assert_eq!(bounded_for_transport(small.clone()), small);
    }

    struct Recorded {
        response: ObservedBranchesResponse,
        asked: std::sync::Mutex<Vec<(u64, ChainPoint, [u8; 32], u64)>>,
    }

    impl ObservedBranchesSource for Recorded {
        fn observed_branches(
            &self,
            since_unix: u64,
            canonical_tip: ChainPoint,
            node_tip_work: [u8; 32],
            now_unix: u64,
        ) -> ObservedBranchesResponse {
            self.asked
                .lock()
                .unwrap()
                .push((since_unix, canonical_tip, node_tip_work, now_unix));
            self.response.clone()
        }
    }

    /// The method asks the node's observer with the caller's `since_unix`,
    /// the canonical tip and its work, at the current time, and sends the
    /// response bounded for the transport.
    #[tokio::test]
    async fn get_observed_branches_asks_the_observer_and_bounds_the_response() {
        let source = std::sync::Arc::new(Recorded {
            response: response((0..10).map(|rank| entry(rank, 2_048, false)).collect()),
            asked: std::sync::Mutex::new(Vec::new()),
        });
        let tip = ChainPoint::new(39_000, [0xC0; 32]);
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let answered = get_observed_branches(source.clone(), 1_799_999_000, tip, [0x11; 32])
            .await
            .unwrap();
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let asked = source.asked.lock().unwrap().clone();
        assert_eq!(asked.len(), 1);
        let (since, asked_tip, work, now) = asked[0];
        assert_eq!((since, asked_tip, work), (1_799_999_000, tip, [0x11; 32]));
        assert!((before..=after).contains(&now), "asked at the current time");
        assert_eq!(answered, bounded_for_transport(source.response.clone()));
        assert_eq!(total_headers(&answered), MAX_RESPONSE_TOTAL_HEADERS);
    }
}
