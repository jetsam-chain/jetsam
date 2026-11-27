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
    pub(crate) fn anchor_committee(
        &self,
    ) -> jetsam_chain::consensus::destination::state::CommitteeRef {
        jetsam_chain::consensus::destination::state::CommitteeRef {
            period: self.anchor.period,
            root: self.anchor.committee_root,
        }
    }

    /// The keys of the anchor committee, as a node holds them (D57),
    /// decompressed once per test binary (≈ 0.1 s each in a debug build).
    pub(crate) fn anchor_keys(
        &self,
    ) -> jetsam_chain::consensus::destination::state::CommitteeKeyCache {
        static KEYS: std::sync::OnceLock<
            jetsam_chain::consensus::destination::state::CommitteeKeyCache,
        > = std::sync::OnceLock::new();
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

/// RET3: the synthetic **signed** L1 of `testdata/base_synthetic.txt` (test
/// keys only, written by a temporary test of `jetsam_chain` from its
/// return-over-Base test helpers): a finality update signed by the anchor
/// committee of [`SyntheticL1`] whose finalized root holds Base's five
/// contracts (pinned as `BASE_SEPOLIA`) and a game Base holds valid at its
/// L1 time, claiming a Base state root under which the bridge (the same
/// contract and genesis) holds `withdrawals[n]` of one burn.
#[derive(Clone)]
pub(crate) struct SyntheticBase {
    pub release: VaultRelease,
    pub finality: Vec<u8>,
    pub game_proof: Vec<u8>,
    pub output_preimage: Vec<u8>,
    pub account_proof: Vec<u8>,
}

impl SyntheticBase {
    pub(crate) fn load() -> Self {
        let text = std::fs::read_to_string(format!(
            "{}/src/destination/testdata/base_synthetic.txt",
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
        let storage = bytes("storage_proof");
        Self {
            release: VaultRelease {
                n: field("release_n").parse().unwrap(),
                owner: Address(bytes("release_owner").try_into().unwrap()),
                amount: field("release_amount").parse().unwrap(),
                storage_proof: decode_proof_nodes(&mut storage.as_slice()).expect("storage"),
            },
            finality: bytes("finality"),
            game_proof: bytes("game_proof"),
            output_preimage: bytes("output_preimage"),
            account_proof: bytes("account_proof"),
        }
    }

    pub(crate) fn proofs(&self) -> BaseReleaseProofs {
        BaseReleaseProofs {
            game_proof: self.game_proof.clone(),
            output_preimage: self.output_preimage.clone(),
        }
    }

    /// The destination rules of [`SyntheticL1`] with the return over Base
    /// armed at `height` on Base Sepolia's pins. The fixture signs for the L1
    /// main network, the L1 the return reads from the height on; below it the
    /// light client reads Sepolia here, so that only the rules of the next
    /// block (`DestinationRules::effective_at`) verify the fixture.
    pub(crate) fn rules(
        l1: &SyntheticL1,
        vault: jetsam_chain::consensus::vault::VaultRules,
        height: u64,
    ) -> jetsam_chain::consensus::destination::state::DestinationRules {
        let mut rules = l1.rules(vault);
        rules.network = jetsam_chain::consensus::destination::forks::L1_SEPOLIA;
        rules.base_return = jetsam_chain::consensus::destination::state::BaseReturnRules {
            activation_height: Some(height),
            network: jetsam_chain::consensus::destination::forks::L1_MAINNET,
            base: jetsam_chain::consensus::destination::base_read::BASE_SEPOLIA,
            // FIXRC: the fix not armed (the test clock is above the test
            // chain's fix height); its own tests arm it.
            advance: jetsam_chain::consensus::destination::advance::AdvanceRules {
                fix_height: None,
                ..jetsam_chain::consensus::destination::advance::AdvanceRules::CONSENSUS
            },
            fast: jetsam_chain::consensus::destination::fast::FastLaneRules::CONSENSUS,
            anchor: None,
            bridge: None,
            bridge_rotations: &[],
            fixext_height: None,
        };
        rules
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
        base: None,
        fast: None,
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

/// A release over Base (kind 3): its game proof and output root preimage
/// between the finality update and the account proof. It decodes only under
/// rules that arm the return over Base: while the height is `None` (the
/// public network, the rehearsal) its kind is unknown, exactly as before; the
/// test chain's build is armed. Damage and malformed objects are refused.
#[test]
fn a_release_over_base_round_trips_only_once_the_height_is_armed() {
    let base = SyntheticBase::load();
    let DestinationIntent::Release(mut release) = release_intent(3, base.release.owner, 5_000_000)
    else {
        unreachable!()
    };
    release.base = Some(base.proofs());
    let intent = DestinationIntent::Release(release.clone());
    let bytes = intent.encode().unwrap();
    assert_eq!(bytes[8], 3, "kind 3");
    assert!(bytes.len() <= MAX_DESTINATION_INTENT_BYTES);
    assert_eq!(
        DestinationIntent::decode_with(&bytes, true),
        Ok(intent.clone())
    );
    // This build: dormant (the kind of a build without the return over
    // Base) on the public network and the rehearsal, armed on the test chain.
    let armed_build = jetsam_chain::consensus::params::V1_7_BASE_RETURN_HEIGHT.is_some();
    assert_eq!(
        DestinationIntent::decode(&bytes),
        if armed_build {
            Ok(intent.clone())
        } else {
            Err(DestinationIntentError::UnknownKind(3))
        }
    );
    assert_eq!(
        DestinationIntent::decode_with(&bytes, false),
        Err(DestinationIntentError::UnknownKind(3))
    );
    // A release of before is still read either way.
    let plain = release_intent(3, base.release.owner, 5_000_000)
        .encode()
        .unwrap();
    assert!(DestinationIntent::decode_with(&plain, true).is_ok());
    for cut in [9, 13, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            DestinationIntent::decode_with(&bytes[..cut], true).is_err(),
            "cut {cut}"
        );
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(
        DestinationIntent::decode_with(&longer, true),
        Err(DestinationIntentError::TrailingBytes)
    );
    let mut bad_game = release.clone();
    bad_game.base.as_mut().unwrap().game_proof.push(0);
    assert_eq!(
        DestinationIntent::decode_with(
            &DestinationIntent::Release(bad_game).encode().unwrap(),
            true
        ),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::BaseGameProof
        ))
    );
    let mut bad_preimage = release;
    bad_preimage.base.as_mut().unwrap().output_preimage.push(0);
    assert_eq!(
        DestinationIntent::decode_with(
            &DestinationIntent::Release(bad_preimage).encode().unwrap(),
            true
        ),
        Err(DestinationIntentError::MalformedObject(
            DestinationObjectKind::OutputRootPreimage
        ))
    );
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

    let forgotten = pool.prune(|r| r.release.n == 3, |r| r.release.n == 5, Some(period + 1));
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
    let held =
        |pool: &DestinationPool| -> Vec<u64> { pool.releases().map(|r| r.release.n).collect() };
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

// ---------------------------------------------------------------------------
// The fast lane (D127, D130): fast release, contradiction, revocation
// ---------------------------------------------------------------------------

mod fast_lane {
    use super::*;
    use blst::min_pk::SecretKey;
    use jetsam_chain::consensus::destination::fast::{
        ContradictionEntry, FastLaneRules, FastLaneState, FastStatement, FAST_RELEASE_DST,
        FAST_REVOKE_DST,
    };
    use jetsam_chain::consensus::destination::mpt::keccak256;
    use jetsam_chain::consensus::guardian::network_id;

    fn key(label: &str) -> (SecretKey, [u8; 48]) {
        let sk = SecretKey::key_gen(&keccak256(label.as_bytes()), &[]).expect("a test key");
        let pk = sk.sk_to_pk().compress();
        (sk, pk)
    }

    const L2: u64 = 56_993_063;

    fn fast_release() -> ReleaseIntent {
        let base = SyntheticBase::load();
        let (relay, _) = key("jetsam-v17-FAST-TEST-ONLY/relay");
        let statement = FastStatement {
            network: network_id(),
            l1_chain_id: 1,
            base_chain_id: 8_453,
            bridge: [0x1b; 20],
            game: [0x5e; 20],
            l2_block: L2,
            root_claim: [0x7b; 32],
        };
        let attestation = FastAttestation {
            statement,
            signature: relay.sign(&statement.message(), FAST_RELEASE_DST, &[]).compress(),
        };
        let DestinationIntent::Release(mut release) =
            release_intent(base.release.n, base.release.owner, base.release.amount)
        else {
            unreachable!()
        };
        let rf = jetsam_chain::consensus::vault::RELEASE_FEE_MICRO;
        let marker = fast_release_marker(base.release.n, &base.release.owner, base.release.amount, L2);
        release.payment = payment(
            VAULT_ADDRESS,
            &[(base.release.owner, base.release.amount - rf), (marker, 0)],
            rf,
        );
        release.base = Some(base.proofs());
        release.fast = Some(attestation.encode_payload());
        release
    }

    fn entry() -> ContradictionEntry {
        ContradictionEntry {
            n: 3,
            owner: Address([0xa1; 32]),
            amount: 5_000_000,
            l2_block: L2,
            height: 77,
            key: Some(key("jetsam-v17-FAST-TEST-ONLY/relay").1),
        }
    }

    fn contradiction() -> FastContradictionIntent {
        let base = SyntheticBase::load();
        FastContradictionIntent {
            payload: FastContradiction {
                n: 3,
                storage_proof: vec![vec![0xee; 40]],
            }
            .encode_payload(),
            finality: base.finality.clone(),
            base: base.proofs(),
            account_proof: base.account_proof.clone(),
            payment: payment(Address([0x51; 32]), &[(entry().marker(), 0), (Address([0x52; 32]), 10)], 50_000),
        }
    }

    fn revocation_by(signer: &str, revoked: [u8; 48]) -> FastRevocationIntent {
        let (cold, _) = key(signer);
        let unsigned = FastRevocation {
            network: network_id(),
            key: revoked,
            signature: [0; 96],
        };
        let payload = FastRevocation {
            signature: cold.sign(&unsigned.message(), FAST_REVOKE_DST, &[]).compress(),
            ..unsigned
        }
        .encode_payload();
        FastRevocationIntent {
            payload,
            payment: payment(
                Address([0x51; 32]),
                &[(revocation_marker(&revoked), 0), (Address([0x52; 32]), 10)],
                50_000,
            ),
        }
    }

    fn revocation() -> FastRevocationIntent {
        revocation_by("jetsam-v17-FAST-TEST-ONLY/cold", key("jetsam-v17-FAST-TEST-ONLY/relay").1)
    }

    #[test]
    fn the_fast_lanes_intents_round_trip_only_once_the_height_is_armed() {
        for (intent, kind) in [
            (DestinationIntent::Release(fast_release()), 4u8),
            (DestinationIntent::FastContradiction(contradiction()), 5),
            (DestinationIntent::FastRevocation(revocation()), 6),
        ] {
            let bytes = intent.encode().unwrap();
            assert_eq!(bytes[8], kind);
            assert!(bytes.len() <= MAX_DESTINATION_INTENT_BYTES);
            assert_eq!(DestinationIntent::decode_with(&bytes, true), Ok(intent.clone()));
            // Dormant (the public network, the rehearsal): a kind of a build
            // without them; the test chain's build is armed.
            assert_eq!(
                DestinationIntent::decode_with(&bytes, false),
                Err(DestinationIntentError::UnknownKind(kind))
            );
            assert_eq!(
                DestinationIntent::decode(&bytes),
                if jetsam_chain::consensus::params::V1_7_BASE_RETURN_HEIGHT.is_some() {
                    Ok(intent.clone())
                } else {
                    Err(DestinationIntentError::UnknownKind(kind))
                }
            );
            let mut longer = bytes.clone();
            longer.push(0);
            assert!(DestinationIntent::decode_with(&longer, true).is_err());
        }
        let mut bad = fast_release();
        bad.fast.as_mut().unwrap().push(0);
        assert_eq!(
            DestinationIntent::decode_with(&DestinationIntent::Release(bad).encode().unwrap(), true),
            Err(DestinationIntentError::MalformedObject(
                DestinationObjectKind::FastGameAttestation
            ))
        );
    }

    /// Each transaction carries exactly its object's marker: a fast release
    /// the fast lane's (never the slow one), a revocation the marker of the
    /// key it revokes, a contradiction one `JTMFCT01`.
    #[test]
    fn a_fast_intents_transaction_carries_exactly_its_marker() {
        let rf = jetsam_chain::consensus::vault::RELEASE_FEE_MICRO;
        let mut slow_marker = fast_release();
        slow_marker.payment = payment(
            VAULT_ADDRESS,
            &[(slow_marker.release.owner, slow_marker.release.amount - rf), (slow_marker.release.marker(), 0)],
            rf,
        );
        let mut other_key = revocation();
        other_key.payment = payment(
            Address([0x51; 32]),
            &[(revocation_marker(&[0x11; 48]), 0), (Address([0x52; 32]), 10)],
            50_000,
        );
        let mut none = contradiction();
        none.payment = payment(Address([0x51; 32]), &[(Address([0x52; 32]), 10)], 50_000);
        let mut two = revocation();
        two.payment = payment(
            Address([0x51; 32]),
            &[(revocation_marker(&key("jetsam-v17-FAST-TEST-ONLY/relay").1), 0), (entry().marker(), 0)],
            50_000,
        );
        for intent in [
            DestinationIntent::Release(slow_marker),
            DestinationIntent::FastRevocation(other_key),
            DestinationIntent::FastContradiction(none),
            DestinationIntent::FastRevocation(two),
        ] {
            assert!(matches!(
                DestinationIntent::decode_with(&intent.encode().unwrap(), true),
                Err(DestinationIntentError::Payment(_))
            ));
        }
    }

    /// A revocation is judged as the block judges it: the cold key of the
    /// build, the relay's key open on the branch.
    #[test]
    fn a_revocation_is_judged_against_the_cold_key_and_the_branch() {
        let l1 = SyntheticL1::load();
        let mut rules = SyntheticBase::rules(&l1, jetsam_chain::consensus::vault::VaultRules::current(), 10);
        rules.base_return.fast = FastLaneRules {
            release_key: Some(key("jetsam-v17-FAST-TEST-ONLY/relay").1),
            revoke_key: Some(key("jetsam-v17-FAST-TEST-ONLY/cold").1),
            ..FastLaneRules::CONSENSUS
        };
        let judge = |intent: &DestinationIntent, lane: &FastLaneState| {
            verify_destination_intent(intent, &[], &rules, lane, &CommitteeKeyCache::default())
        };
        let open = FastLaneState::default();
        assert_eq!(judge(&DestinationIntent::FastRevocation(revocation()), &open), Ok(()));
        let mut closed = FastLaneState::default();
        closed.revocations.push(key("jetsam-v17-FAST-TEST-ONLY/relay").1);
        assert_eq!(
            judge(&DestinationIntent::FastRevocation(revocation()), &closed),
            Err(DestinationBlockError::FastKeyClosed)
        );
        let by_relay = revocation_by(
            "jetsam-v17-FAST-TEST-ONLY/relay",
            key("jetsam-v17-FAST-TEST-ONLY/relay").1,
        );
        assert_eq!(
            judge(&DestinationIntent::FastRevocation(by_relay), &open),
            Err(DestinationBlockError::FastBadSignature)
        );
        // A contradiction of an `n` the branch holds no fast release of is
        // refused before anything is read under Base.
        assert!(judge(&DestinationIntent::FastContradiction(contradiction()), &open).is_err());
    }

    #[test]
    fn the_pool_holds_the_fast_lanes_intents_and_forgets_the_useless() {
        let mut pool = DestinationPool::default();
        pool.hold(DestinationIntent::FastContradiction(contradiction())).unwrap();
        pool.hold(DestinationIntent::FastRevocation(revocation())).unwrap();
        pool.hold(DestinationIntent::Release(fast_release())).unwrap();
        assert_eq!(pool.len(), 3);
        assert_eq!(pool.contradictions().count(), 1);
        assert!(pool.revocation().is_some());
        // The branch holds the record of 3, open, and the key open: kept.
        let mut lane = FastLaneState::default();
        lane.records.insert(
            3,
            jetsam_chain::consensus::destination::fast::FastRecord {
                owner: Address([0xa1; 32]),
                amount: 5_000_000,
                l2_block: L2,
                height: 77,
                contradicted: false,
                consumed: false,
            },
        );
        assert_eq!(pool.prune_fast(&lane), 0);
        // Contradicted, and the key closed: both forgotten.
        lane.records.get_mut(&3).unwrap().contradicted = true;
        lane.revocations.push(key("jetsam-v17-FAST-TEST-ONLY/relay").1);
        assert_eq!(pool.prune_fast(&lane), 2);
        assert_eq!(pool.len(), 1);
        pool.hold(DestinationIntent::FastContradiction(contradiction())).unwrap();
        pool.hold(DestinationIntent::FastRevocation(revocation())).unwrap();
        assert_eq!(pool.forget_fast(&[3], true), 2);
        pool.clear();
        assert!(pool.is_empty());
    }
}
