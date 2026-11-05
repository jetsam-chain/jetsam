// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Intents of both bridge families for the transport tests (T7.1): exactly
//! one intent each by the mempool's canonical decoding. Nothing here is
//! valid against a chain (no state, no proof): the reactor does not judge
//! that, the node's mempool does.

#![allow(dead_code)]

use jetsam_chain::consensus::guardian::StakeLock;
use jetsam_chain::consensus::vault::{encode_bridge_account_proof, VaultRelease};
use jetsam_mempool::destination::{DestinationIntent, ReleaseIntent};
use jetsam_mempool::guardian::GuardianIntent;
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

/// One page from `owner` into `outputs` (≤ 2) and the fee.
pub fn payment(owner: Address, outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
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

/// A stake lock of `amount` by a fixed owner, its marker in the payment.
pub fn guardian_intent_bytes(amount: u64) -> Vec<u8> {
    let owner = Address([0x61; 32]);
    let object = StakeLock { owner, amount }.object();
    let marker = object.marker();
    GuardianIntent {
        object,
        payment: payment(owner, &[(marker, 0)], 50_000),
    }
    .encode()
    .unwrap()
}

/// A release of `n` with a real finality update, its marker in the group.
pub fn destination_intent_bytes(n: u64) -> Vec<u8> {
    let release = VaultRelease {
        n,
        owner: Address([0xa1; 32]),
        amount: 5_000_000,
        storage_proof: vec![vec![0xee; 40]],
    };
    let mut finality = vec![1u8];
    finality.extend_from_slice(include_bytes!(
        "../../../jetsam_chain/src/consensus/destination/testdata/fu_lodestar-mainnet.bin"
    ));
    // The account proof, then the genesis proof (A5-8): well-formed bytes.
    let account_proof = encode_bridge_account_proof(&[vec![0xdd; 64]], &[vec![0xdc; 64]]);
    let marker = release.marker();
    DestinationIntent::Release(ReleaseIntent {
        payment: payment(
            jetsam_chain::consensus::vault::VAULT_ADDRESS,
            &[(release.owner, 4_900_000), (marker, 0)],
            100_000,
        ),
        release,
        finality,
        account_proof,
    })
    .encode()
    .unwrap()
}

/// An attestation whose body is all `fill` (well-formed bytes: a statement
/// and a signature that verify nowhere), its marker in the payment.
pub fn attestation_intent_bytes(fill: u8) -> Vec<u8> {
    use jetsam_chain::consensus::guardian::GuardianObjectKind;
    use jetsam_chain::consensus::guardian::{GuardianObject, ATTESTATION_BODY_BYTES};
    let object = GuardianObject::new(
        GuardianObjectKind::Attestation,
        vec![fill; ATTESTATION_BODY_BYTES],
    );
    let marker = object.marker();
    GuardianIntent {
        object,
        payment: payment(Address([0x62; 32]), &[(marker, 0)], 50_000),
    }
    .encode()
    .unwrap()
}
