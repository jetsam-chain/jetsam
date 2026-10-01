// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use crate::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use crate::consensus::params::GENESIS_TARGET;
use crate::consensus::pow::block_id;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const ACTIVATION: u64 = 100;
const HEIGHT: u64 = 200;
const LICENSE: u64 = 1_000 * MICRO_PER_JTM;
const PAYER: Address = Address([0x51; 32]);
const TREASURY: Address = Address([0x7e; 32]);

fn rules(destination: LicenseDestination) -> ClientObjectRules {
    ClientObjectRules {
        activation_height: Some(ACTIVATION),
        license_micro: LICENSE,
        destination,
        submission_fee_micro: MICRO_PER_JTM,
        activation_delay: 480,
        dividend_blocks: 480,
        registry_capacity: CLIENT_REGISTRY_CAPACITY,
        max_matrix_file_bytes: CLIENT_MATRIX_MAX_FILE_BYTES,
    }
}

fn burn_rules() -> ClientObjectRules {
    rules(LicenseDestination::Burn)
}

fn header(height: u64) -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn coinbase(parent: &BlockHeader, owner: Address) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 15_000_000,
        amount: 1,
        owner,
    };
    Transaction::new(TxBody {
        epoch_anchor: block_id(parent),
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: output_bitmap_bit(0),
        is_coinbase: true,
    })
}

/// One logical transaction: one input from `owner` on the first page, the
/// outputs packed two per page, `fee` on the first page.
fn logical(
    slot_base: u32,
    owner: Address,
    outputs: &[(Address, u64)],
    fee: u64,
) -> Vec<Transaction> {
    let pages = outputs.len().div_ceil(TX_OUTPUTS).max(1);
    let input_amount = outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee;
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            if page == 0 {
                inputs[0] = TxInput {
                    slot_index: slot_base,
                    amount: input_amount,
                    creation_id: 1,
                };
                bitmap |= 1 | PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (owner, amount)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[slot] = TxOutput {
                    slot_index: slot_base + 1 + (page * TX_OUTPUTS + slot) as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(slot);
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: [7u8; 32],
                fee: if page == 0 { fee } else { 0 },
                input_owner: owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

fn required_fee(n_inputs: u64, n_outputs: u64) -> u64 {
    let parent = header(HEIGHT - 1);
    fee_breakdown(
        n_inputs,
        n_outputs,
        parent.active_slot_count,
        parent.log_slots,
    )
    .required_total
}

fn block(groups: Vec<Vec<Transaction>>) -> Block {
    block_at(HEIGHT, groups)
}

fn block_at(height: u64, groups: Vec<Vec<Transaction>>) -> Block {
    let parent = header(height - 1);
    let mut transactions = vec![coinbase(&parent, Address([9u8; 32]))];
    for group in groups {
        transactions.extend(group);
    }
    Block {
        header: header(height),
        transactions,
    }
}

fn digest(byte: u8) -> Digest {
    [byte; 32]
}

fn registration(byte: u8) -> ClientRegistration {
    ClientRegistration {
        matrix_digest: digest(byte),
        matrix_file_root: digest(byte.wrapping_add(0x80)),
        matrix_file_len: 3_800_000,
    }
}

fn submission(byte: u8, io: u8) -> ClientSubmission {
    ClientSubmission {
        matrix_digest: digest(byte),
        io_commitment: digest(io),
    }
}

/// A registration paid with exactly the burn license.
fn paid_registration(slot_base: u32, object: ClientRegistration) -> Vec<Transaction> {
    let marker = ClientObject::Registration(object).marker();
    logical(
        slot_base,
        PAYER,
        &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE), (marker, 0)],
        required_fee(1, 2),
    )
}

fn paid_submission(slot_base: u32, object: ClientSubmission, extra_fee: u64) -> Vec<Transaction> {
    let marker = ClientObject::Submission(object).marker();
    logical(
        slot_base,
        PAYER,
        &[(marker, 0)],
        required_fee(1, 1) + extra_fee,
    )
}

fn entry(index: u8, byte: u8, registered_at: u64, license: LicenseSplit) -> ClientRegistryEntry {
    let object = registration(byte);
    ClientRegistryEntry {
        index,
        matrix_digest: object.matrix_digest,
        matrix_file_root: object.matrix_file_root,
        matrix_file_len: object.matrix_file_len,
        registered_at,
        active_from: registered_at + 480,
        license,
    }
}

fn registry_with(count: u8, registered_at: u64) -> ClientRegistryState {
    let mut registry = ClientRegistryState::new();
    let effect = ClientObjectsEffect {
        registrations: (0..count)
            .map(|index| {
                entry(
                    index,
                    0x10 + index,
                    registered_at,
                    LicenseSplit {
                        burn: LICENSE,
                        ..LicenseSplit::default()
                    },
                )
            })
            .collect(),
        ..ClientObjectsEffect::default()
    };
    registry.apply(&effect);
    registry
}

fn validate(
    block: &Block,
    objects: &[ClientObject],
    registry: &ClientRegistryState,
    rules: &ClientObjectRules,
) -> Result<ClientObjectsEffect, ClientObjectError> {
    validate_block_client_objects(
        block,
        objects,
        &header(block.header.height - 1),
        registry,
        rules,
    )
}

// ---------------------------------------------------------------------------
// Parameters, addresses, D5
// ---------------------------------------------------------------------------

#[test]
fn locked_addresses_are_their_public_derivations() {
    assert_eq!(
        CLIENT_LICENSE_BURN_ADDRESS,
        Address(poseidon2b_hash_bytes(
            CLIENT_LICENSE_BURN_ADDRESS_DOMAIN,
            b""
        ))
    );
    assert_eq!(
        CLIENT_LICENSE_POOL_ADDRESS,
        Address(poseidon2b_hash_bytes(
            CLIENT_LICENSE_POOL_ADDRESS_DOMAIN,
            b""
        ))
    );
    for address in [CLIENT_LICENSE_BURN_ADDRESS, CLIENT_LICENSE_POOL_ADDRESS] {
        assert_ne!(address, Address([0u8; 32]));
        assert_ne!(address, NETWORK_FUND_ADDRESS);
        assert_ne!(address, LAB_FUND_ADDRESS);
        assert_eq!(client_object_marker_kind(&address), None);
        assert!(is_client_locked_address(&address));
    }
    assert_ne!(CLIENT_LICENSE_BURN_ADDRESS, CLIENT_LICENSE_POOL_ADDRESS);
    assert!(!is_client_locked_address(&PAYER));
    assert!(!is_client_locked_address(&NETWORK_FUND_ADDRESS));
}

#[test]
fn provisional_parameters_are_the_declared_ones() {
    // D5 is open: changing any of these is a deliberate edit of this pin.
    assert_eq!(CLIENT_LICENSE_MICRO, 1_000_000_000);
    assert_eq!(CLIENT_SUBMISSION_FEE_MICRO, 1_000_000);
    assert_eq!(CLIENT_LICENSE_DESTINATION, LicenseDestination::Burn);
    assert_eq!(CLIENT_ACTIVATION_DELAY_BLOCKS, 480);
    assert_eq!(CLIENT_LICENSE_DIVIDEND_BLOCKS, 480);
    assert_eq!(CLIENT_REGISTRY_CAPACITY, 16);
    assert!(CLIENT_LICENSE_DESTINATION.is_admissible());
    assert_eq!(
        ClientObjectRules::CONSENSUS.activation_height,
        V1_5_ACTIVATION_HEIGHT
    );
}

#[test]
fn every_destination_splits_the_whole_license() {
    let odd = 1_000_000_001;
    assert_eq!(
        LicenseDestination::Burn.split(odd),
        LicenseSplit {
            burn: odd,
            miners: 0,
            treasury: 0
        }
    );
    assert_eq!(
        LicenseDestination::Miners.split(odd),
        LicenseSplit {
            burn: 0,
            miners: odd,
            treasury: 0
        }
    );
    assert_eq!(
        LicenseDestination::Treasury(TREASURY).split(odd),
        LicenseSplit {
            burn: 0,
            miners: 0,
            treasury: odd
        }
    );
    assert_eq!(
        LicenseDestination::BurnAndMiners { burn_bps: 5_000 }.split(LICENSE),
        LicenseSplit {
            burn: LICENSE / 2,
            miners: LICENSE / 2,
            treasury: 0
        }
    );
    // Rounding favours the burn.
    let mix = LicenseDestination::BurnAndMiners { burn_bps: 3_333 }.split(odd);
    assert_eq!(mix.miners, odd * 6_667 / 10_000);
    assert_eq!(mix.total(), odd);
    for bps in [0u16, 1, 5_000, 9_999, 10_000] {
        assert_eq!(
            LicenseDestination::BurnAndMiners { burn_bps: bps }
                .split(odd)
                .total(),
            odd
        );
    }
}

#[test]
fn destination_admissibility() {
    assert!(LicenseDestination::Burn.is_admissible());
    assert!(LicenseDestination::Miners.is_admissible());
    assert!(LicenseDestination::BurnAndMiners { burn_bps: 10_000 }.is_admissible());
    assert!(!LicenseDestination::BurnAndMiners { burn_bps: 10_001 }.is_admissible());
    assert!(LicenseDestination::Treasury(TREASURY).is_admissible());
    for locked in [
        Address([0u8; 32]),
        CLIENT_LICENSE_BURN_ADDRESS,
        CLIENT_LICENSE_POOL_ADDRESS,
        ClientObject::Registration(registration(1)).marker(),
    ] {
        assert!(!LicenseDestination::Treasury(locked).is_admissible());
    }
}

// ---------------------------------------------------------------------------
// Objects, markers, wire
// ---------------------------------------------------------------------------

#[test]
fn objects_round_trip_and_reject_malformed_bytes() {
    for object in [
        ClientObject::Registration(registration(3)),
        ClientObject::Submission(submission(3, 4)),
    ] {
        let mut bytes = Vec::new();
        object.encode(&mut bytes);
        assert_eq!(bytes.len(), object.wire_len());
        let mut src = bytes.as_slice();
        assert_eq!(ClientObject::decode(&mut src), Ok(object));
        assert!(src.is_empty());
        for cut in 0..bytes.len() {
            assert!(
                ClientObject::decode(&mut &bytes[..cut]).is_err(),
                "cut {cut}"
            );
        }
        let mut bad_tag = bytes.clone();
        bad_tag[0] = 9;
        assert!(ClientObject::decode(&mut bad_tag.as_slice()).is_err());
    }
}

#[test]
fn the_section_is_absent_when_empty_and_canonical_when_present() {
    let mut empty = Vec::new();
    encode_client_objects_section(&[], &mut empty);
    assert!(empty.is_empty());
    assert_eq!(client_objects_section_wire_len(&[]), 0);
    assert_eq!(
        decode_client_objects_section(&mut [].as_slice()),
        Ok(Vec::new())
    );

    let objects = vec![
        ClientObject::Registration(registration(1)),
        ClientObject::Submission(submission(2, 3)),
    ];
    let mut bytes = Vec::new();
    encode_client_objects_section(&objects, &mut bytes);
    assert_eq!(bytes.len(), client_objects_section_wire_len(&objects));
    assert_eq!(bytes[0], CLIENT_OBJECTS_SECTION_MARKER);
    let mut src = bytes.as_slice();
    assert_eq!(decode_client_objects_section(&mut src), Ok(objects.clone()));
    assert!(src.is_empty());

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(decode_client_objects_section(&mut trailing.as_slice()).is_err());
    let mut wrong_marker = bytes.clone();
    wrong_marker[0] = 0xC2;
    assert!(decode_client_objects_section(&mut wrong_marker.as_slice()).is_err());
    let zero_count = vec![CLIENT_OBJECTS_SECTION_MARKER, 0];
    assert!(decode_client_objects_section(&mut zero_count.as_slice()).is_err());
    let too_many = vec![
        CLIENT_OBJECTS_SECTION_MARKER,
        (MAX_BLOCK_CLIENT_OBJECTS + 1) as u8,
    ];
    assert!(decode_client_objects_section(&mut too_many.as_slice()).is_err());
}

#[test]
fn markers_commit_to_every_field_and_carry_their_tag() {
    let base = registration(5);
    let marker = ClientObject::Registration(base).marker();
    assert_eq!(&marker.0[..8], &CLIENT_REGISTRATION_MARKER_TAG);
    assert_eq!(
        client_object_marker_kind(&marker),
        Some(ClientObjectKind::Registration)
    );
    assert!(is_client_locked_address(&marker));
    let mut variants = vec![base, base, base];
    variants[0].matrix_digest[31] ^= 1;
    variants[1].matrix_file_root[0] ^= 1;
    variants[2].matrix_file_len += 1;
    for variant in variants {
        assert_ne!(ClientObject::Registration(variant).marker(), marker);
    }

    let sub = submission(5, 6);
    let sub_marker = ClientObject::Submission(sub).marker();
    assert_eq!(&sub_marker.0[..8], &CLIENT_SUBMISSION_MARKER_TAG);
    assert_eq!(
        client_object_marker_kind(&sub_marker),
        Some(ClientObjectKind::Submission)
    );
    assert_ne!(
        ClientObject::Submission(submission(5, 7)).marker(),
        sub_marker
    );
    assert_ne!(
        ClientObject::Submission(submission(4, 6)).marker(),
        sub_marker
    );
    // Same 64 payload bytes in the two kinds never give one marker.
    assert_eq!(client_object_marker_kind(&PAYER), None);
}

#[test]
fn matrix_file_root_binds_every_byte_and_the_length() {
    let file = vec![0xA5u8; CLIENT_MATRIX_CHUNK_BYTES + 17];
    let root = matrix_file_root(&file);
    let mut flipped = file.clone();
    flipped[CLIENT_MATRIX_CHUNK_BYTES + 3] ^= 1;
    assert_ne!(matrix_file_root(&flipped), root);
    let mut first = file.clone();
    first[0] ^= 1;
    assert_ne!(matrix_file_root(&first), root);
    assert_ne!(matrix_file_root(&file[..file.len() - 1]), root);
    let mut longer = file.clone();
    longer.push(0);
    assert_ne!(matrix_file_root(&longer), root);
}

// ---------------------------------------------------------------------------
// Registry state
// ---------------------------------------------------------------------------

#[test]
fn registry_root_is_the_zero_padded_tree() {
    let node =
        |a: &Digest, b: &Digest| compress_flat_feed_forward_with_tag(REGISTRY_NODE_TAG, a, b);
    let mut level = vec![[0u8; 32]; CLIENT_REGISTRY_CAPACITY];
    level[0] = digest(1);
    level[1] = digest(2);
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| node(&pair[0], &pair[1]))
            .collect();
    }
    assert_eq!(client_registry_root(&[digest(1), digest(2)]), level[0]);
    assert_ne!(client_registry_root(&[digest(2), digest(1)]), level[0]);
    assert_ne!(client_registry_root(&[]), level[0]);
}

#[test]
fn registry_is_append_only_and_follows_a_reorg() {
    let mut registry = ClientRegistryState::new();
    let empty_root = registry.root();
    registry.apply(&ClientObjectsEffect {
        registrations: vec![entry(0, 1, 150, LicenseSplit::default())],
        ..ClientObjectsEffect::default()
    });
    registry.apply(&ClientObjectsEffect {
        registrations: vec![entry(1, 2, 160, LicenseSplit::default())],
        ..ClientObjectsEffect::default()
    });
    assert_eq!(registry.digests(), vec![digest(1), digest(2)]);
    assert_eq!(
        registry.root(),
        client_registry_root(&[digest(1), digest(2)])
    );
    assert_ne!(registry.root(), empty_root);

    let mut reorged = registry.clone();
    reorged.truncate_above(160);
    assert_eq!(reorged, registry);
    reorged.truncate_above(159);
    assert_eq!(reorged.digests(), vec![digest(1)]);
    reorged.truncate_above(149);
    assert!(reorged.is_empty());
    assert_eq!(reorged.root(), empty_root);
}

#[test]
fn registry_state_round_trips() {
    let registry = registry_with(3, 150);
    let bytes = registry.encode();
    assert_eq!(ClientRegistryState::decode(&bytes), Ok(registry.clone()));
    assert_eq!(
        ClientRegistryState::decode(&ClientRegistryState::new().encode()),
        Ok(ClientRegistryState::new())
    );
    for cut in 0..bytes.len() {
        assert!(
            ClientRegistryState::decode(&bytes[..cut]).is_err(),
            "cut {cut}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(ClientRegistryState::decode(&trailing).is_err());
}

#[test]
fn an_entry_is_carriable_only_once_active() {
    let registry = registry_with(1, 150);
    let d = digest(0x10);
    assert_eq!(
        registry.check_carriable(&d, 629),
        Err(ClientObjectError::ClientNotYetActive {
            matrix_digest: d,
            active_from: 630
        })
    );
    assert_eq!(
        registry.check_carriable(&d, 630).map(|entry| entry.index),
        Ok(0)
    );
    assert_eq!(
        registry.check_carriable(&digest(0x77), 630),
        Err(ClientObjectError::ClientNotRegistered {
            matrix_digest: digest(0x77)
        })
    );
    assert_eq!(registry.carriable_at(629).count(), 0);
    assert_eq!(registry.carriable_at(630).count(), 1);
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

#[test]
fn nothing_is_interpreted_below_the_v1_5_height() {
    let height = ACTIVATION - 1;
    let object = registration(1);
    let marked = logical(
        100,
        CLIENT_LICENSE_BURN_ADDRESS,
        &[(ClientObject::Registration(object).marker(), 5)],
        10,
    );
    let block = block_at(height, vec![marked]);
    // Tagged outputs with value, spends from the burn address: ordinary data.
    assert_eq!(
        validate(&block, &[], &ClientRegistryState::new(), &burn_rules()),
        Ok(ClientObjectsEffect::default())
    );
    assert_eq!(
        validate(
            &block,
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::ObjectsBeforeActivation)
    );
    // The dormant consensus rules never interpret anything.
    let dormant = ClientObjectRules {
        activation_height: None,
        ..burn_rules()
    };
    let high = block_at(1_000_000, vec![paid_registration(100, object)]);
    assert_eq!(
        validate(&high, &[], &ClientRegistryState::new(), &dormant),
        Ok(ClientObjectsEffect::default())
    );
    assert_eq!(
        validate(
            &high,
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &dormant
        ),
        Err(ClientObjectError::ObjectsBeforeActivation)
    );
}

// ---------------------------------------------------------------------------
// Registration (M3.5)
// ---------------------------------------------------------------------------

#[test]
fn a_paid_registration_is_appended() {
    let object = registration(1);
    let block = block(vec![paid_registration(100, object)]);
    let mut registry = ClientRegistryState::new();
    let effect = validate(
        &block,
        &[ClientObject::Registration(object)],
        &registry,
        &burn_rules(),
    )
    .unwrap();
    assert_eq!(effect.submission, None);
    assert_eq!(effect.license_dividend, 0);
    assert_eq!(
        effect.registrations,
        vec![ClientRegistryEntry {
            index: 0,
            matrix_digest: object.matrix_digest,
            matrix_file_root: object.matrix_file_root,
            matrix_file_len: object.matrix_file_len,
            registered_at: HEIGHT,
            active_from: HEIGHT + 480,
            license: LicenseSplit {
                burn: LICENSE,
                miners: 0,
                treasury: 0
            },
        }]
    );
    registry.apply(&effect);
    assert_eq!(registry.digests(), vec![object.matrix_digest]);

    // The next one takes the next index.
    let second = registration(2);
    let block = block_at(HEIGHT + 1, vec![paid_registration(100, second)]);
    let effect = validate(
        &block,
        &[ClientObject::Registration(second)],
        &registry,
        &burn_rules(),
    )
    .unwrap();
    assert_eq!(effect.registrations[0].index, 1);
}

#[test]
fn several_registrations_in_one_block_take_consecutive_indices() {
    let (a, b) = (registration(1), registration(2));
    let block = block(vec![paid_registration(100, a), paid_registration(200, b)]);
    let registry = registry_with(3, 150);
    let effect = validate(
        &block,
        &[ClientObject::Registration(a), ClientObject::Registration(b)],
        &registry,
        &burn_rules(),
    )
    .unwrap();
    assert_eq!(
        effect
            .registrations
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        vec![3, 4]
    );
}

#[test]
fn a_registration_without_license_is_refused() {
    let object = registration(1);
    let marker = ClientObject::Registration(object).marker();
    let unpaid = logical(100, PAYER, &[(marker, 0)], required_fee(1, 1));
    assert_eq!(
        validate(
            &block(vec![unpaid]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::LicenseMissing {
            destination: CLIENT_LICENSE_BURN_ADDRESS,
            required: LICENSE,
        })
    );
    // A license sent to the wrong locked address is no license.
    let misdirected = logical(
        100,
        PAYER,
        &[(CLIENT_LICENSE_POOL_ADDRESS, LICENSE), (marker, 0)],
        required_fee(1, 2),
    );
    assert!(matches!(
        validate(
            &block(vec![misdirected]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::LicenseMissing { .. })
    ));
}

#[test]
fn an_insufficient_license_is_refused() {
    let object = registration(1);
    let marker = ClientObject::Registration(object).marker();
    let short = logical(
        100,
        PAYER,
        &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE - 1), (marker, 0)],
        required_fee(1, 2),
    );
    assert_eq!(
        validate(
            &block(vec![short]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::LicenseUnderpaid {
            destination: CLIENT_LICENSE_BURN_ADDRESS,
            required: LICENSE,
            paid: LICENSE - 1,
        })
    );
    // Paying more is allowed; the excess is burned with the rest.
    let generous = logical(
        100,
        PAYER,
        &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE + 1), (marker, 0)],
        required_fee(1, 2),
    );
    assert!(validate(
        &block(vec![generous]),
        &[ClientObject::Registration(object)],
        &ClientRegistryState::new(),
        &burn_rules()
    )
    .is_ok());
}

#[test]
fn a_full_registry_refuses_registrations() {
    let object = registration(0xEE);
    let full = registry_with(16, 150);
    assert_eq!(
        validate(
            &block(vec![paid_registration(100, object)]),
            &[ClientObject::Registration(object)],
            &full,
            &burn_rules()
        ),
        Err(ClientObjectError::RegistryFull { capacity: 16 })
    );
    // Fifteen registered, two in one block: the second does not fit.
    let (a, b) = (registration(0xE0), registration(0xE1));
    assert_eq!(
        validate(
            &block(vec![paid_registration(100, a), paid_registration(200, b)]),
            &[ClientObject::Registration(a), ClientObject::Registration(b)],
            &registry_with(15, 150),
            &burn_rules(),
        ),
        Err(ClientObjectError::RegistryFull { capacity: 16 })
    );
}

#[test]
fn a_digest_is_registered_once() {
    let registry = registry_with(2, 150);
    let again = ClientRegistration {
        matrix_file_root: digest(0x42),
        ..registration(0x11)
    };
    assert_eq!(
        validate(
            &block(vec![paid_registration(100, again)]),
            &[ClientObject::Registration(again)],
            &registry,
            &burn_rules()
        ),
        Err(ClientObjectError::DuplicateRegistration {
            matrix_digest: digest(0x11)
        })
    );
    // Twice in the same block, two different payments and file roots.
    let first = registration(0x33);
    let second = ClientRegistration {
        matrix_file_len: 1,
        ..first
    };
    assert_eq!(
        validate(
            &block(vec![
                paid_registration(100, first),
                paid_registration(200, second)
            ]),
            &[
                ClientObject::Registration(first),
                ClientObject::Registration(second)
            ],
            &ClientRegistryState::new(),
            &burn_rules(),
        ),
        Err(ClientObjectError::DuplicateRegistration {
            matrix_digest: digest(0x33)
        })
    );
}

#[test]
fn malformed_registrations_are_refused() {
    let null_d = ClientRegistration {
        matrix_digest: [0u8; 32],
        ..registration(1)
    };
    let null_root = ClientRegistration {
        matrix_file_root: [0u8; 32],
        ..registration(1)
    };
    let empty_file = ClientRegistration {
        matrix_file_len: 0,
        ..registration(1)
    };
    let huge_file = ClientRegistration {
        matrix_file_len: CLIENT_MATRIX_MAX_FILE_BYTES + 1,
        ..registration(1)
    };
    let cases = [
        (null_d, ClientObjectError::NullMatrixDigest),
        (null_root, ClientObjectError::NullMatrixFileRoot),
        (
            empty_file,
            ClientObjectError::MatrixFileLength {
                len: 0,
                max: CLIENT_MATRIX_MAX_FILE_BYTES,
            },
        ),
        (
            huge_file,
            ClientObjectError::MatrixFileLength {
                len: CLIENT_MATRIX_MAX_FILE_BYTES + 1,
                max: CLIENT_MATRIX_MAX_FILE_BYTES,
            },
        ),
    ];
    for (object, expected) in cases {
        assert_eq!(
            validate(
                &block(vec![paid_registration(100, object)]),
                &[ClientObject::Registration(object)],
                &ClientRegistryState::new(),
                &burn_rules()
            ),
            Err(expected)
        );
    }
}

#[test]
fn the_object_list_is_exactly_the_markers_openings() {
    let (a, b) = (registration(1), registration(2));
    let two = block(vec![paid_registration(100, a), paid_registration(200, b)]);
    let registry = ClientRegistryState::new();
    let ok = [ClientObject::Registration(a), ClientObject::Registration(b)];
    assert!(validate(&two, &ok, &registry, &burn_rules()).is_ok());

    let altered = ClientRegistration {
        matrix_file_len: a.matrix_file_len + 1,
        ..a
    };
    let mismatches: Vec<Vec<ClientObject>> = vec![
        vec![],                                                          // objects dropped
        vec![ClientObject::Registration(a)],                             // one dropped
        vec![ok[1], ok[0]],                                              // reordered
        vec![ClientObject::Registration(altered), ok[1]],                // altered
        vec![ok[0], ok[1], ClientObject::Registration(registration(3))], // added
        vec![ClientObject::Submission(submission(1, 2)), ok[1]],         // wrong kind
    ];
    for objects in mismatches {
        assert_eq!(
            validate(&two, &objects, &registry, &burn_rules()),
            Err(ClientObjectError::ObjectsDoNotMatchMarkers),
            "{objects:?}"
        );
    }
}

#[test]
fn one_payment_names_one_object() {
    // "Paiement réutilisé": a transaction cannot pay two objects, and one
    // object cannot be paid twice (its marker opens one object, in order).
    let (a, b) = (registration(1), registration(2));
    let double = logical(
        100,
        PAYER,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, 2 * LICENSE),
            (ClientObject::Registration(a).marker(), 0),
            (ClientObject::Registration(b).marker(), 0),
        ],
        required_fee(1, 3),
    );
    assert_eq!(
        validate(
            &block(vec![double]),
            &[ClientObject::Registration(a), ClientObject::Registration(b)],
            &ClientRegistryState::new(),
            &burn_rules(),
        ),
        Err(ClientObjectError::MultipleMarkersInTransaction { group: 0 })
    );
    let paid_twice = block(vec![paid_registration(100, a), paid_registration(200, a)]);
    assert_eq!(
        validate(
            &paid_twice,
            &[ClientObject::Registration(a)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::ObjectsDoNotMatchMarkers)
    );
}

#[test]
fn markers_carry_no_value_and_stay_out_of_system_records() {
    let object = registration(1);
    let marker = ClientObject::Registration(object).marker();
    let valued = logical(
        100,
        PAYER,
        &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE), (marker, 1)],
        required_fee(1, 2),
    );
    assert_eq!(
        validate(
            &block(vec![valued]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::MarkerCarriesValue { group: 0 })
    );
    let mut in_coinbase = block(vec![]);
    in_coinbase.transactions[0] = coinbase(&header(HEIGHT - 1), marker);
    assert_eq!(
        validate(
            &in_coinbase,
            &[],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Err(ClientObjectError::MarkerInSystemRecord)
    );
}

#[test]
fn locked_addresses_cannot_spend() {
    let marker = ClientObject::Submission(submission(1, 2)).marker();
    for owner in [
        CLIENT_LICENSE_BURN_ADDRESS,
        CLIENT_LICENSE_POOL_ADDRESS,
        marker,
    ] {
        let spend = logical(100, owner, &[(PAYER, 10)], required_fee(1, 1));
        assert_eq!(
            validate(
                &block(vec![spend]),
                &[],
                &ClientRegistryState::new(),
                &burn_rules()
            ),
            Err(ClientObjectError::SpendFromLockedAddress { owner })
        );
    }
    // Paying them is fine: burning without registering is allowed.
    let burn = logical(
        100,
        PAYER,
        &[(CLIENT_LICENSE_BURN_ADDRESS, 10)],
        required_fee(1, 1),
    );
    assert_eq!(
        validate(
            &block(vec![burn]),
            &[],
            &ClientRegistryState::new(),
            &burn_rules()
        ),
        Ok(ClientObjectsEffect::default())
    );
}

// ---------------------------------------------------------------------------
// D5: the four destinations
// ---------------------------------------------------------------------------

fn registration_paying(
    slot_base: u32,
    object: ClientRegistration,
    payments: &[(Address, u64)],
) -> Vec<Transaction> {
    let mut outputs = payments.to_vec();
    outputs.push((ClientObject::Registration(object).marker(), 0));
    let n_outputs = outputs.len() as u64;
    logical(slot_base, PAYER, &outputs, required_fee(1, n_outputs))
}

#[test]
fn destination_miners_pays_the_pool_and_owes_a_dividend() {
    let rules = rules(LicenseDestination::Miners);
    let object = registration(1);
    let paid = registration_paying(100, object, &[(CLIENT_LICENSE_POOL_ADDRESS, LICENSE)]);
    let effect = validate(
        &block(vec![paid]),
        &[ClientObject::Registration(object)],
        &ClientRegistryState::new(),
        &rules,
    )
    .unwrap();
    assert_eq!(
        effect.registrations[0].license,
        LicenseSplit {
            burn: 0,
            miners: LICENSE,
            treasury: 0
        }
    );

    let burned = registration_paying(100, object, &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE)]);
    assert_eq!(
        validate(
            &block(vec![burned]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &rules
        ),
        Err(ClientObjectError::LicenseMissing {
            destination: CLIENT_LICENSE_POOL_ADDRESS,
            required: LICENSE
        })
    );

    // The dividend: spread over the next 480 blocks, remainder first.
    let mut registry = ClientRegistryState::new();
    registry.apply(&effect);
    let each = LICENSE / 480;
    let first = each + LICENSE % 480;
    assert_eq!(registry.license_dividend_at(HEIGHT, &rules), 0);
    assert_eq!(registry.license_dividend_at(HEIGHT + 1, &rules), first);
    assert_eq!(registry.license_dividend_at(HEIGHT + 2, &rules), each);
    assert_eq!(registry.license_dividend_at(HEIGHT + 480, &rules), each);
    assert_eq!(registry.license_dividend_at(HEIGHT + 481, &rules), 0);
    let total: u64 = (HEIGHT..=HEIGHT + 481)
        .map(|height| registry.license_dividend_at(height, &rules))
        .sum();
    assert_eq!(total, LICENSE);

    // The block after the registration is owed its share in its effect.
    let next = block_at(HEIGHT + 1, vec![]);
    assert_eq!(
        validate(&next, &[], &registry, &rules)
            .unwrap()
            .license_dividend,
        first
    );
}

#[test]
fn destination_mix_needs_both_payments() {
    let rules = rules(LicenseDestination::BurnAndMiners { burn_bps: 5_000 });
    let object = registration(1);
    let half = LICENSE / 2;
    let paid = registration_paying(
        100,
        object,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, half),
            (CLIENT_LICENSE_POOL_ADDRESS, half),
        ],
    );
    let effect = validate(
        &block(vec![paid]),
        &[ClientObject::Registration(object)],
        &ClientRegistryState::new(),
        &rules,
    )
    .unwrap();
    assert_eq!(
        effect.registrations[0].license,
        LicenseSplit {
            burn: half,
            miners: half,
            treasury: 0
        }
    );
    let mut registry = ClientRegistryState::new();
    registry.apply(&effect);
    assert_eq!(registry.license_dividend_at(HEIGHT + 2, &rules), half / 480);

    let burn_only = registration_paying(100, object, &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE)]);
    assert_eq!(
        validate(
            &block(vec![burn_only]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &rules
        ),
        Err(ClientObjectError::LicenseMissing {
            destination: CLIENT_LICENSE_POOL_ADDRESS,
            required: half
        })
    );
}

#[test]
fn destination_treasury_pays_the_declared_address() {
    let rules = rules(LicenseDestination::Treasury(TREASURY));
    let object = registration(1);
    let paid = registration_paying(100, object, &[(TREASURY, LICENSE)]);
    let effect = validate(
        &block(vec![paid]),
        &[ClientObject::Registration(object)],
        &ClientRegistryState::new(),
        &rules,
    )
    .unwrap();
    assert_eq!(
        effect.registrations[0].license,
        LicenseSplit {
            burn: 0,
            miners: 0,
            treasury: LICENSE
        }
    );
    let mut registry = ClientRegistryState::new();
    registry.apply(&effect);
    assert_eq!(registry.license_dividend_at(HEIGHT + 1, &rules), 0);

    let burned = registration_paying(100, object, &[(CLIENT_LICENSE_BURN_ADDRESS, LICENSE)]);
    assert_eq!(
        validate(
            &block(vec![burned]),
            &[ClientObject::Registration(object)],
            &ClientRegistryState::new(),
            &rules
        ),
        Err(ClientObjectError::LicenseMissing {
            destination: TREASURY,
            required: LICENSE
        })
    );
    // The treasury is an ordinary, spendable address.
    let spend = logical(300, TREASURY, &[(PAYER, 10)], required_fee(1, 1));
    assert!(validate(&block(vec![spend]), &[], &registry, &rules).is_ok());
}

#[test]
fn destination_burn_owes_no_dividend() {
    let rules = burn_rules();
    let object = registration(1);
    let effect = validate(
        &block(vec![paid_registration(100, object)]),
        &[ClientObject::Registration(object)],
        &ClientRegistryState::new(),
        &rules,
    )
    .unwrap();
    let mut registry = ClientRegistryState::new();
    registry.apply(&effect);
    for height in HEIGHT..HEIGHT + 500 {
        assert_eq!(registry.license_dividend_at(height, &rules), 0);
    }
}

// ---------------------------------------------------------------------------
// Submission (M3.6)
// ---------------------------------------------------------------------------

#[test]
fn a_paid_submission_for_an_active_client_is_accepted() {
    let registry = registry_with(2, 100); // active from 580
    let object = submission(0x11, 0x99);
    let height = 600;
    let block = block_at(height, vec![paid_submission(100, object, MICRO_PER_JTM)]);
    let effect = validate(
        &block,
        &[ClientObject::Submission(object)],
        &registry,
        &burn_rules(),
    )
    .unwrap();
    assert_eq!(effect.submission, Some(object));
    assert!(effect.registrations.is_empty());
}

#[test]
fn a_submission_for_an_unregistered_or_inactive_client_is_refused() {
    let registry = registry_with(1, 100);
    let unknown = submission(0x77, 1);
    assert_eq!(
        validate(
            &block_at(600, vec![paid_submission(100, unknown, MICRO_PER_JTM)]),
            &[ClientObject::Submission(unknown)],
            &registry,
            &burn_rules()
        ),
        Err(ClientObjectError::ClientNotRegistered {
            matrix_digest: digest(0x77)
        })
    );
    let early = submission(0x10, 1);
    assert_eq!(
        validate(
            &block_at(579, vec![paid_submission(100, early, MICRO_PER_JTM)]),
            &[ClientObject::Submission(early)],
            &registry,
            &burn_rules()
        ),
        Err(ClientObjectError::ClientNotYetActive {
            matrix_digest: digest(0x10),
            active_from: 580
        })
    );
    // Registered in this very block: not in the parent's registry.
    let fresh = registration(0x20);
    let same_block = block_at(
        600,
        vec![
            paid_registration(100, fresh),
            paid_submission(200, submission(0x20, 1), MICRO_PER_JTM),
        ],
    );
    assert_eq!(
        validate(
            &same_block,
            &[
                ClientObject::Registration(fresh),
                ClientObject::Submission(submission(0x20, 1))
            ],
            &registry,
            &burn_rules(),
        ),
        Err(ClientObjectError::ClientNotRegistered {
            matrix_digest: digest(0x20)
        })
    );
}

#[test]
fn a_block_pays_at_most_one_submission() {
    let registry = registry_with(2, 100);
    let (a, b) = (submission(0x10, 1), submission(0x11, 2));
    assert_eq!(
        validate(
            &block_at(
                600,
                vec![
                    paid_submission(100, a, MICRO_PER_JTM),
                    paid_submission(200, b, MICRO_PER_JTM)
                ]
            ),
            &[ClientObject::Submission(a), ClientObject::Submission(b)],
            &registry,
            &burn_rules(),
        ),
        Err(ClientObjectError::TooManySubmissions)
    );
}

#[test]
fn a_submission_fee_below_the_minimum_is_refused() {
    let registry = registry_with(1, 100);
    let object = submission(0x10, 1);
    let required = required_fee(1, 1) + MICRO_PER_JTM;
    assert_eq!(
        validate(
            &block_at(600, vec![paid_submission(100, object, MICRO_PER_JTM - 1)]),
            &[ClientObject::Submission(object)],
            &registry,
            &burn_rules()
        ),
        Err(ClientObjectError::SubmissionFeeTooLow {
            required,
            paid: required - 1
        })
    );
}

#[test]
fn a_paid_submission_must_be_the_carried_client() {
    let registry = registry_with(2, 100);
    let object = submission(0x10, 0x99);
    let effect = ClientObjectsEffect {
        submission: Some(object),
        ..ClientObjectsEffect::default()
    };
    let carried = CarriedClient {
        matrix_digest: digest(0x10),
        io_commitment: digest(0x99),
    };
    assert_eq!(
        check_carried_client(&effect, Some(carried), &registry, 600),
        Ok(())
    );
    assert_eq!(
        check_carried_client(&effect, None, &registry, 600),
        Err(ClientObjectError::SubmissionNotCarried)
    );
    for other in [
        CarriedClient {
            io_commitment: digest(0x98),
            ..carried
        },
        CarriedClient {
            matrix_digest: digest(0x11),
            ..carried
        },
    ] {
        assert_eq!(
            check_carried_client(&effect, Some(other), &registry, 600),
            Err(ClientObjectError::SubmissionNotCarried)
        );
    }

    // Carried without payment: the miner's gift, if the client is carriable.
    let none = ClientObjectsEffect::default();
    assert_eq!(check_carried_client(&none, None, &registry, 600), Ok(()));
    assert_eq!(
        check_carried_client(&none, Some(carried), &registry, 600),
        Ok(())
    );
    assert_eq!(
        check_carried_client(&none, Some(carried), &registry, 579),
        Err(ClientObjectError::ClientNotYetActive {
            matrix_digest: digest(0x10),
            active_from: 580
        })
    );
    let stranger = CarriedClient {
        matrix_digest: digest(0x55),
        ..carried
    };
    assert_eq!(
        check_carried_client(&none, Some(stranger), &registry, 600),
        Err(ClientObjectError::ClientNotRegistered {
            matrix_digest: digest(0x55)
        })
    );
}
