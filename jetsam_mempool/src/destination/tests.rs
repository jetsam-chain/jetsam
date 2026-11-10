// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The destination intent's bytes and the pool that holds intents (T2.5),
//! over the real L1 vectors of the destination module.

use super::*;
use jetsam_chain::consensus::destination::ssz::LightClientUpdate;
use jetsam_chain::consensus::vault::{
    decode_proof_nodes, encode_bridge_account_proof, encode_proof_nodes, VAULT_ADDRESS,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

fn testdata(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../jetsam_chain/src/consensus/destination/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture")
}

/// The real committee update 1894 of the L1 (Fulu payload: tag 1, then SSZ).
pub(crate) fn committee_update_payload() -> Vec<u8> {
    let served = testdata("up_lodestar-mainnet.bin");
    let mut payload = vec![1u8];
    payload.extend_from_slice(&served[12..]);
    payload
}

/// A real finality update of the L1.
pub(crate) fn finality_payload() -> Vec<u8> {
    let mut payload = vec![1u8];
    payload.extend(testdata("fu_lodestar-mainnet.bin"));
    payload
}

/// A2-M1: the synthetic **signed** L1 of `testdata/l1_synthetic.txt` (test
/// keys only, written by a temporary test of `jetsam_chain` from its
/// destination test helpers): an anchor committee (its anchor update), the
/// update it signs for the next period, a finality update it signs whose
/// finalized root holds the bridge account, its genesis (A5-8) and
/// `withdrawals[n]` of one burn, the same finality update signed by a
/// stranger committee.
#[derive(Clone)]
pub(crate) struct SyntheticL1 {
    pub anchor: jetsam_chain::consensus::destination::state::L1Anchor,
    pub bridge: jetsam_chain::consensus::destination::l1_read::L1BridgeParams,
    pub release: VaultRelease,
    pub anchor_update: Vec<u8>,
    pub next_update: Vec<u8>,
    pub finality: Vec<u8>,
    pub finality_stranger: Vec<u8>,
    pub account_proof: Vec<u8>,
}

impl SyntheticL1 {
    /// The fixture, read once per test binary.
    pub(crate) fn load() -> Self {
        static FIXTURE: std::sync::OnceLock<SyntheticL1> = std::sync::OnceLock::new();
        FIXTURE.get_or_init(Self::read).clone()
    }

    fn read() -> Self {
        let text = std::fs::read_to_string(format!(
            "{}/src/destination/testdata/l1_synthetic.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture");
        let field = |key: &str| -> String {
            text.lines()
                .find_map(|line| line.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("fixture field {key}"))
                .to_string()
        };
        let bytes = |key: &str| -> Vec<u8> {
            let hex = field(key);
            (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
                .collect()
        };
        let number = |key: &str| -> u64 { field(key).parse().expect("number") };
        let storage = bytes("storage_proof");
        let storage_proof = decode_proof_nodes(&mut storage.as_slice()).expect("storage proof");
        Self {
            anchor: jetsam_chain::consensus::destination::state::L1Anchor {
                period: number("anchor_period"),
                committee_root: bytes("anchor_root").try_into().unwrap(),
            },
            bridge: jetsam_chain::consensus::destination::l1_read::L1BridgeParams {
                address: bytes("bridge_address").try_into().unwrap(),
                code_hash: bytes("bridge_code_hash").try_into().unwrap(),
                withdrawals_slot: number("bridge_withdrawals_slot"),
                genesis_id_slot: number("bridge_genesis_id_slot"),
                genesis_id: bytes("bridge_genesis_id").try_into().unwrap(),
            },
            release: VaultRelease {
                n: number("release_n"),
                owner: Address(bytes("release_owner").try_into().unwrap()),
                amount: number("release_amount"),
                storage_proof,
            },
            anchor_update: bytes("anchor_update"),
            next_update: bytes("next_update"),
            finality: bytes("finality"),
            finality_stranger: bytes("finality_stranger"),
            account_proof: bytes("account_proof"),
        }
    }

    /// The destination rules of this L1, the vault gate (and the v1.7 clock)
    /// `vault`'s.
    pub(crate) fn rules(
        &self,
        vault: jetsam_chain::consensus::vault::VaultRules,
    ) -> jetsam_chain::consensus::destination::state::DestinationRules {
        jetsam_chain::consensus::destination::state::DestinationRules {
            activation_height: vault.activation_height,
            network: jetsam_chain::consensus::destination::forks::L1_MAINNET,
            anchor: Some(self.anchor),
            bridge: Some(self.bridge),
            vault,
            base_return: jetsam_chain::consensus::destination::state::BaseReturnRules::CONSENSUS,
        }
    }

    /// The anchor committee, as a branch knows it after its anchor update.
    pub(crate) fn anchor_committee(&self) -> jetsam_chain::consensus::destination::state::CommitteeRef {
        jetsam_chain::consensus::destination::state::CommitteeRef {
            period: self.anchor.period,
            root: self.anchor.committee_root,
        }
    }

    /// The keys of the anchor committee, as a node holds them (D57),
    /// decompressed once per test binary (≈ 0.1 s each in a debug build).
    pub(crate) fn anchor_keys(&self) -> jetsam_chain::consensus::destination::state::CommitteeKeyCache {
        static KEYS: std::sync::OnceLock<jetsam_chain::consensus::destination::state::CommitteeKeyCache> =
            std::sync::OnceLock::new();
        KEYS.get_or_init(|| {
            let mut keys =
                jetsam_chain::consensus::destination::state::CommitteeKeyCache::default();
            assert_eq!(
                keys.insert_update_payload(&self.anchor_update).unwrap(),
                self.anchor.committee_root
            );
            keys
        })
        .clone()
    }
}

/// One page from `owner` into `outputs` (≤ 2), one input covering them and
/// the fee.
pub(crate) fn payment(owner: Address, outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 5,
        amount: outputs.iter().map(|(_, a)| a).sum::<u64>() + fee,
        creation_id: 7_777,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (i, (o, a)) in outputs.iter().enumerate() {
        page_outputs[i] = TxOutput {
            slot_index: 6 + i as u32,
            amount: *a,
            owner: *o,
        };
        bitmap |= output_bitmap_bit(i);
    }
    let page = TxPage::new(TxBody {
        epoch_anchor: [3u8; 32],
        fee,
        input_owner: owner,
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
    .unwrap();
    PagedSpendIntent::new(vec![page], vec![0xA5; 64]).unwrap()
}

pub(crate) fn committee_intent(payload: Vec<u8>) -> DestinationIntent {
    let marker = committee_update_marker(&payload);
    DestinationIntent::CommitteeUpdate(CommitteeUpdateIntent {
        payment: payment(
            Address([0x51; 32]),
            &[(Address([0x52; 32]), 10), (marker, 0)],
            50_000,
        ),
        payload,
    })
}

pub(crate) fn release_intent(n: u64, owner: Address, amount: u64) -> DestinationIntent {
    let release = VaultRelease {
        n,
        owner,
        amount,
        storage_proof: vec![vec![0xee; 40]],
    };
    let proof = encode_bridge_account_proof(&[vec![0xdd; 64]], &[vec![0xdc; 64]]);
    let rf = jetsam_chain::consensus::vault::RELEASE_FEE_MICRO;
    DestinationIntent::Release(ReleaseIntent {
        payment: payment(
            VAULT_ADDRESS,
            &[(owner, amount - rf), (release.marker(), 0)],
            rf,
        ),
        release,
        finality: finality_payload(),
        account_proof: proof,
    })
}

/// An update of another period (the attested slot moved), re-encoded.
fn update_of_period(period_shift: u64) -> Vec<u8> {
    let mut update = LightClientUpdate::decode_payload(&committee_update_payload()).unwrap();
    update.attested_header.beacon.slot += period_shift * 8_192;
    update.encode_payload()
}

#[test]
fn destination_intent_round_trips_and_refuses_damage() {
    for intent in [
        committee_intent(committee_update_payload()),
        release_intent(7, Address([0xa1; 32]), 5_000_000),
    ] {
        let bytes = intent.encode().unwrap();
        assert!(bytes.len() <= MAX_DESTINATION_INTENT_BYTES);
        assert_eq!(DestinationIntent::decode(&bytes).unwrap(), intent);
        // Every strict prefix is refused, and so is one more byte.
        for cut in [0, 8, 9, 13, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                DestinationIntent::decode(&bytes[..cut]).is_err(),
                "cut {cut}"
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(
            DestinationIntent::decode(&longer),
            Err(DestinationIntentError::TrailingBytes)
        );
        let mut magic = bytes.clone();
        magic[0] ^= 1;
        assert_eq!(
            DestinationIntent::decode(&magic),
            Err(DestinationIntentError::BadMagic)
        );
        let mut kind = bytes.clone();
        kind[8] = 9;
        assert_eq!(
            DestinationIntent::decode(&kind),
            Err(DestinationIntentError::UnknownKind(9))
        );
    }
    assert!(matches!(
        DestinationIntent::decode(&vec![0u8; MAX_DESTINATION_INTENT_BYTES + 1]),
        Err(DestinationIntentError::TooLarge { .. })
    ));
}

/// The transaction of an intent carries exactly its object's marker: no
/// marker, another object's marker, or one marker too many is refused.
#[test]
fn destination_intent_payment_carries_exactly_its_marker() {
    let payload = committee_update_payload();
    let other = update_of_period(1);
    let marker = committee_update_marker(&payload);
    let with = |outputs: &[(Address, u64)]| {
        DestinationIntent::CommitteeUpdate(CommitteeUpdateIntent {
            payload: payload.clone(),
            payment: payment(Address([0x51; 32]), outputs, 50_000),
        })
        .encode()
        .unwrap()
    };
    assert!(DestinationIntent::decode(&with(&[(Address([1; 32]), 10), (marker, 0)])).is_ok());
    for outputs in [
        vec![(Address([1; 32]), 10)],
        vec![(Address([1; 32]), 10), (committee_update_marker(&other), 0)],
        vec![(marker, 0), (marker, 0)],
        vec![
            (marker, 0),
            (
                VaultRelease {
                    n: 1,
                    owner: Address([2; 32]),
                    amount: 1,
                    storage_proof: vec![vec![1]],
                }
                .marker(),
                0,
            ),
        ],
    ] {
        assert!(
            matches!(
                DestinationIntent::decode(&with(&outputs)),
                Err(DestinationIntentError::Payment(_))
            ),
            "{outputs:?}"
        );
    }
    // A release's group carries its own marker, not another n's.
    let DestinationIntent::Release(mut release) = release_intent(7, Address([0xa1; 32]), 5_000_000)
    else {
        unreachable!()
    };
    release.release.n = 8;
    assert!(matches!(
        DestinationIntent::decode(&DestinationIntent::Release(release).encode().unwrap()),
        Err(DestinationIntentError::Payment(_))
    ));
}

/// Object payloads that do not decode as their object are refused.
#[test]
fn destination_intent_objects_must_decode() {
    let DestinationIntent::Release(release) = release_intent(7, Address([0xa1; 32]), 5_000_000)
    else {
        unreachable!()
    };
    let mut bad_finality = release.clone();
    bad_finality.finality = vec![1, 2, 3];
    assert_eq!(
        DestinationIntent::decode(&DestinationIntent::Release(bad_finality).encode().unwrap()),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::FinalityUpdate
        ))
    );
    let mut bad_proof = release.clone();
    bad_proof.account_proof.push(0);
    assert_eq!(
        DestinationIntent::decode(&DestinationIntent::Release(bad_proof).encode().unwrap()),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::BridgeAccountProof
        ))
    );
    // A5-8: the account proof alone, without the genesis proof after it.
    let mut account_only = release.clone();
    account_only.account_proof.clear();
    encode_proof_nodes(&[vec![0xdd; 64]], &mut account_only.account_proof);
    assert_eq!(
        DestinationIntent::decode(&DestinationIntent::Release(account_only).encode().unwrap()),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::BridgeAccountProof
        ))
    );
    let bad_update = DestinationIntent::CommitteeUpdate(CommitteeUpdateIntent {
        payload: vec![1, 0, 0],
        payment: payment(
            Address([0x51; 32]),
            &[(committee_update_marker(&[1, 0, 0]), 0)],
            50_000,
        ),
    });
    assert_eq!(
        DestinationIntent::decode(&bad_update.encode().unwrap()),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::CommitteeUpdate
        ))
    );
}

/// One committee update per period, one release per index, within bounds; a
/// newer intent replaces the held one; pruning forgets paid, stale and
/// already-known.
#[test]
fn destination_pool_holds_one_per_period_and_index_within_bounds() {
    let mut pool = DestinationPool::default();
    for n in 1..=MAX_HELD_RELEASES as u64 {
        pool.hold(release_intent(n, Address([0xa1; 32]), 5_000_000))
            .unwrap();
    }
    assert_eq!(
        pool.hold(release_intent(999, Address([0xa1; 32]), 5_000_000)),
        Err(HoldError::Full)
    );
    // Replacing a held index is not growth.
    pool.hold(release_intent(3, Address([0xb0; 32]), 6_000_000))
        .unwrap();
    assert_eq!(pool.releases().count(), MAX_HELD_RELEASES);
    assert_eq!(
        pool.releases().nth(2).unwrap().release.owner,
        Address([0xb0; 32])
    );
    let ns: Vec<u64> = pool.releases().map(|r| r.release.n).collect();
    assert!(ns.windows(2).all(|w| w[0] < w[1]), "increasing n");

    let base = committee_intent(committee_update_payload());
    let DestinationIntent::CommitteeUpdate(first) = &base else {
        unreachable!()
    };
    let period = first.next_period().unwrap();
    pool.hold(base.clone()).unwrap();
    for shift in 1..MAX_HELD_COMMITTEE_UPDATES as u64 {
        pool.hold(committee_intent(update_of_period(shift)))
            .unwrap();
    }
    assert_eq!(
        pool.hold(committee_intent(update_of_period(99))),
        Err(HoldError::Full)
    );
    pool.hold(base).unwrap();
    assert_eq!(pool.committee_updates().count(), MAX_HELD_COMMITTEE_UPDATES);

    let forgotten = pool.prune(|n| n == 3, |r| r.release.n == 5, Some(period + 1));
    assert_eq!(
        forgotten,
        2 + 2,
        "two releases, the updates for P and P + 1"
    );
    assert!(pool
        .releases()
        .all(|r| r.release.n != 3 && r.release.n != 5));
    assert!(pool
        .committee_updates()
        .all(|u| u.next_period().unwrap() > period + 1));
    pool.clear();
    assert!(pool.is_empty());
}

/// A2-M1: a full release pool never refuses a lower `n` than what it holds
/// (the template serves releases by increasing `n`): the largest held index
/// gives way to it; a higher `n` than all is refused; a held index is
/// replaced, not grown.
#[test]
fn full_release_pool_makes_room_for_a_lower_n() {
    let mut pool = DestinationPool::default();
    let first = 1_000_000u64;
    let last = first + MAX_HELD_RELEASES as u64 - 1;
    for n in first..=last {
        pool.hold(release_intent(n, Address([0xa1; 32]), 5_000_000))
            .unwrap();
    }
    let held = |pool: &DestinationPool| -> Vec<u64> { pool.releases().map(|r| r.release.n).collect() };
    assert_eq!(
        pool.hold(release_intent(last + 1, Address([0xa1; 32]), 5_000_000)),
        Err(HoldError::Full),
        "nothing lower to give way"
    );
    pool.hold(release_intent(7, Address([0xa1; 32]), 5_000_000))
        .expect("a lower n makes room");
    assert_eq!(pool.releases().count(), MAX_HELD_RELEASES);
    assert_eq!(held(&pool).first(), Some(&7));
    assert!(!held(&pool).contains(&last), "the largest gave way");
    assert!(held(&pool).contains(&(last - 1)));
    // A held index is replaced in place: nothing gives way.
    pool.hold(release_intent(last - 1, Address([0xb0; 32]), 6_000_000))
        .unwrap();
    assert_eq!(pool.releases().count(), MAX_HELD_RELEASES);
    assert!(held(&pool).contains(&(last - 2)));
    // Between the held ones: the largest gives way again.
    pool.hold(release_intent(first - 1, Address([0xa1; 32]), 5_000_000))
        .unwrap();
    assert!(!held(&pool).contains(&(last - 1)));
    assert_eq!(held(&pool)[..2], [7, first - 1]);
}
