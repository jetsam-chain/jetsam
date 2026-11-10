// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! F4 of the final audit: a deposit as the wallet builds it — its Base/EVM
//! recipient, its outputs, and the build's rules that admit or refuse it.

use super::*;
use jetsam_chain::consensus::fees::fee_breakdown;
use jetsam_chain::consensus::params::MICRO_PER_JTM;
use jetsam_chain::consensus::vault::{
    check_plain_vault_transaction, deposit_marker, VaultError, VaultRules, MAX_DEPOSIT_MICRO,
    MIN_DEPOSIT_MICRO, VAULT_ADDRESS,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};
use jetsam_vault_rule::GroupClass;

const JTM: u64 = MICRO_PER_JTM;
const PAYER: Address = Address([0xa1; 32]);
const RECIPIENT: [u8; 20] = [0x5a; 20];

/// Mixed-case checksummed Base/EVM addresses (the published test vectors of
/// the checksum, EIP-55), with two all-capitals forms that checksum to
/// themselves.
const CHECKSUMMED: [&str; 6] = [
    "0x52908400098527886E0F7030069857D2E4169EE7",
    "0x8617E340B3D01FA5F11F306F4090FD50E238070D",
    "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
    "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
    "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
    "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
];

/// The build's rules with the v1.7 height `j` armed (the public profiles
/// carry none yet): F = J + the opening delay.
fn armed(j: u64) -> VaultRules {
    let mut rules = VaultRules::current();
    rules.activation_height = Some(j);
    rules.deposits_from_height =
        jetsam_chain::consensus::vault::bridge_deposits_from_height(Some(j));
    rules
}

/// One page from `owner` spending one input, paying `outputs` (at most two).
fn page(owner: Address, outputs: &[(Address, u64)], fee: u64) -> Vec<TxPage> {
    assert!(outputs.len() <= TX_OUTPUTS);
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 900,
        amount: outputs.iter().map(|(_, a)| a).sum::<u64>() + fee,
        creation_id: 77,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1u16 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (slot, (owner, amount)) in outputs.iter().enumerate() {
        page_outputs[slot] = TxOutput {
            slot_index: 70_001 + slot as u32,
            amount: *amount,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(slot);
    }
    vec![TxPage::new(TxBody {
        epoch_anchor: [3u8; 32],
        fee,
        input_owner: owner,
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
    .expect("page")]
}

// ---------------------------------------------------------------------------
// The recipient
// ---------------------------------------------------------------------------

/// An address in lower case or in capitals carries no checksum and is taken
/// as written; a mixed-case one must carry the right checksum. Either way the
/// address shown back to the user is the checksummed form.
#[test]
fn a_recipient_is_read_and_shown_back_with_its_checksum() {
    for text in CHECKSUMMED {
        let bytes = parse_recipient(text).expect(text);
        assert_eq!(checksummed(&bytes), text, "{text}");
        let lower = format!("0x{}", text[2..].to_ascii_lowercase());
        assert_eq!(parse_recipient(&lower), Ok(bytes), "{lower}");
        let upper = format!("0x{}", text[2..].to_ascii_uppercase());
        assert_eq!(parse_recipient(&upper), Ok(bytes), "{upper}");
        assert_eq!(
            parse_recipient(&format!("  {text}\n")),
            Ok(bytes),
            "trimmed"
        );
        // `0X` is the same prefix.
        assert_eq!(parse_recipient(&format!("0X{}", &text[2..])), Ok(bytes));
    }
    assert_eq!(
        hex::encode(parse_recipient(CHECKSUMMED[2]).unwrap()),
        "5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"
    );
}

/// One letter typed in the wrong case is a typo the checksum catches: the
/// deposit is refused and the right form is given.
#[test]
fn a_mixed_case_recipient_with_a_wrong_checksum_is_refused() {
    let typo = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD";
    assert_eq!(
        parse_recipient(typo),
        Err(RecipientError::Checksum {
            expected: CHECKSUMMED[2].to_string()
        })
    );
    let shown = parse_recipient(typo).unwrap_err().to_string();
    assert!(shown.contains(CHECKSUMMED[2]), "{shown}");
}

/// Anything but `0x` and forty hexadecimal digits is not a Base/EVM address:
/// a Jetsam address, a txid, a short or long string, a non-hex digit.
#[test]
fn a_recipient_is_0x_and_forty_hex_digits() {
    assert_eq!(
        parse_recipient("5aaeb6053f3e94c9b9a09f33669435e7ef1beaed"),
        Err(RecipientError::MissingPrefix)
    );
    assert_eq!(
        parse_recipient("tj1qf8ch9cqqsq5aww0wthql8302x82k04c6j70epjfz4w2ddz9q3ysvjt40d"),
        Err(RecipientError::MissingPrefix)
    );
    assert_eq!(
        parse_recipient("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beae"),
        Err(RecipientError::Length(39))
    );
    assert_eq!(
        parse_recipient("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed00"),
        Err(RecipientError::Length(42))
    );
    assert_eq!(parse_recipient("0x"), Err(RecipientError::Length(0)));
    assert_eq!(
        parse_recipient(&format!("0x{}", "ab".repeat(32))),
        Err(RecipientError::Length(64)),
        "a 32-byte hex string (a txid, a Jetsam owner) is not a recipient"
    );
    assert_eq!(
        parse_recipient("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaeg"),
        Err(RecipientError::NotHex)
    );
    assert_eq!(
        parse_recipient("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beae\u{e9}"),
        Err(RecipientError::NotHex)
    );
}

// ---------------------------------------------------------------------------
// The outputs and the rule
// ---------------------------------------------------------------------------

/// A deposit pays the vault, then its marker naming the recipient, in that
/// order: what the rehearsal tool built (`jetsam-rehearsal deposit`), now
/// the one definition both use.
#[test]
fn a_deposit_pays_the_vault_then_its_marker() {
    assert_eq!(
        deposit_outputs(12 * JTM, &RECIPIENT),
        vec![(VAULT_ADDRESS, 12 * JTM), (deposit_marker(&RECIPIENT), 0)]
    );
}

/// The outputs, with the payer's change after them, are a deposit under V4
/// itself (the shared rule's classifier, which the node's block gate and the
/// proof program run), and the mempool's policy admits them as one page above
/// F and refuses them at F.
#[test]
fn deposit_outputs_are_a_deposit_under_v4_above_f_only() {
    let rules = armed(9_600);
    let f = rules.deposits_from_height.expect("F");
    let mut outputs = deposit_outputs(25 * JTM, &RECIPIENT);
    outputs.push((PAYER, 7_000));
    let owners: Vec<([u8; 32], u64)> = outputs.iter().map(|(o, a)| (o.0, *a)).collect();
    assert_eq!(
        jetsam_vault_rule::classify_group(0, &PAYER.0, &owners, f + 1, &rules.deposit_rule()),
        Ok(GroupClass::Deposit {
            recipient: RECIPIENT,
            amount: 25 * JTM
        })
    );
    let pages = page(PAYER, &deposit_outputs(25 * JTM, &RECIPIENT), 50_000);
    assert_eq!(check_plain_vault_transaction(&pages, f + 1, &rules), Ok(()));
    assert!(matches!(
        check_plain_vault_transaction(&pages, f, &rules),
        Err(VaultError::DepositBeforeBridgeOpening { .. })
    ));
}

/// While the build's v1.7 height is none, every deposit is refused, at any
/// height: the vault is then an ordinary address whose spend secret is
/// published (anyone could take what is sent there). This is the case of
/// the public network today; the test chain is armed at 8 640 (D75).
#[test]
fn every_deposit_is_refused_while_the_v1_7_height_is_none() {
    let mut unarmed = VaultRules::current();
    unarmed.activation_height = None;
    unarmed.deposits_from_height = None;
    for height in [0, 96, 129, 46_081, 1 << 40] {
        assert_eq!(
            check_deposit(10 * JTM, &RECIPIENT, height, &unarmed),
            Err(DepositRefusal::NotArmed),
            "height {height}"
        );
    }
    let current = VaultRules::current();
    if current.activation_height.is_none() {
        assert_eq!(
            check_deposit(10 * JTM, &RECIPIENT, 1 << 40, &current),
            Err(DepositRefusal::NotArmed)
        );
    }
    // Armed today: the private rehearsal build, and the test chain at 8 160
    // (D78), where the armed rules decide (`a_deposit_waits_for_the_block_after_f`).
    // The public network is not armed.
    if !jetsam_vault_rule::REHEARSAL_CLOCK {
        let expected = if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
            Some(8_160)
        } else {
            None
        };
        assert_eq!(
            jetsam_chain::consensus::params::V1_7_ACTIVATION_HEIGHT,
            expected,
            "a public profile's v1.7 height moved: the refusal above no longer covers it as it did"
        );
        // ... and its bridge opening: F = 8 704 on the test chain (J + 544, D104), none on the public network.
        let expected_f = if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
            Some(8_704)
        } else {
            None
        };
        assert_eq!(
            current.deposits_from_height, expected_f,
            "a public profile's bridge opening moved: the refusal above no longer covers it as it did"
        );
    }
}

/// Armed, a deposit is refused at and below F (V4-d), and also below J,
/// where the mempool would take a payment to the vault as an ordinary one;
/// it is admitted from F + 1 on.
#[test]
fn a_deposit_waits_for_the_block_after_f() {
    let rules = armed(9_600);
    let f = rules.deposits_from_height.expect("F");
    for next in [0, 9_599, 9_600, f - 1, f] {
        assert_eq!(
            check_deposit(10 * JTM, &RECIPIENT, next, &rules),
            Err(DepositRefusal::NotOpenYet {
                next_height: next,
                opening: f
            }),
            "next block {next}"
        );
    }
    assert_eq!(check_deposit(10 * JTM, &RECIPIENT, f + 1, &rules), Ok(()));
    assert_eq!(
        check_deposit(10 * JTM, &RECIPIENT, f + 10_000, &rules),
        Ok(())
    );
}

/// V4-c: from the minimum to the maximum of one deposit, both included.
#[test]
fn a_deposit_is_between_the_minimum_and_the_maximum() {
    let rules = armed(9_600);
    let next = rules.deposits_from_height.unwrap() + 1;
    assert_eq!(MIN_DEPOSIT_MICRO, JTM);
    // D123: the bond, 50 000 JTM (the test chain keeps 1 000 below its caps height)
    assert_eq!(MAX_DEPOSIT_MICRO, 50_000 * JTM);
    let rules = VaultRules {
        legacy: None,
        ..rules
    };
    assert_eq!(
        check_deposit(MIN_DEPOSIT_MICRO, &RECIPIENT, next, &rules),
        Ok(())
    );
    assert_eq!(
        check_deposit(MAX_DEPOSIT_MICRO, &RECIPIENT, next, &rules),
        Ok(())
    );
    assert_eq!(
        check_deposit(MIN_DEPOSIT_MICRO - 1, &RECIPIENT, next, &rules),
        Err(DepositRefusal::BelowMinimum {
            amount: MIN_DEPOSIT_MICRO - 1,
            minimum: MIN_DEPOSIT_MICRO
        })
    );
    assert_eq!(
        check_deposit(0, &RECIPIENT, next, &rules),
        Err(DepositRefusal::BelowMinimum {
            amount: 0,
            minimum: MIN_DEPOSIT_MICRO
        })
    );
    assert_eq!(
        check_deposit(MAX_DEPOSIT_MICRO + 1, &RECIPIENT, next, &rules),
        Err(DepositRefusal::AboveMaximum {
            amount: MAX_DEPOSIT_MICRO + 1,
            maximum: MAX_DEPOSIT_MICRO
        })
    );
}

/// V4-b: the zero address receives nothing.
#[test]
fn the_zero_recipient_is_refused() {
    let rules = armed(9_600);
    let next = rules.deposits_from_height.unwrap() + 1;
    assert_eq!(
        check_deposit(10 * JTM, &[0u8; 20], next, &rules),
        Err(DepositRefusal::ZeroRecipient)
    );
    assert!(parse_recipient("0x0000000000000000000000000000000000000000").is_ok());
}

/// The automatic fee of a deposit pays the consensus minimum of any deposit
/// group of one input page (1 to 8 inputs, with or without change), and the
/// relay floor when that is higher.
#[test]
fn the_automatic_fee_pays_any_deposit_of_one_input_page() {
    for (active, log_slots) in [(1_055u64, 24u32), (40_000, 24), (3_000_000, 26)] {
        let fee = deposit_fee(active, log_slots, 0);
        for inputs in 1..=TX_INPUTS as u64 {
            for outputs in [2u64, 3] {
                let minimum = fee_breakdown(inputs, outputs, active, log_slots).required_total;
                assert!(
                    fee >= minimum,
                    "{inputs} in, {outputs} out: {fee} < {minimum}"
                );
            }
        }
        assert_eq!(
            deposit_fee(active, log_slots, fee + 1),
            fee + 1,
            "the relay floor"
        );
    }
}

/// Public text: these messages say Base/EVM, never the network's name nor
/// the two-letter name of the proof technique.
#[test]
fn the_messages_say_base_evm_only() {
    let texts = [
        RecipientError::MissingPrefix.to_string(),
        RecipientError::Length(3).to_string(),
        RecipientError::NotHex.to_string(),
        RecipientError::Checksum {
            expected: CHECKSUMMED[2].into(),
        }
        .to_string(),
        DepositRefusal::NotArmed.to_string(),
        DepositRefusal::NotOpenYet {
            next_height: 5,
            opening: 9,
        }
        .to_string(),
        DepositRefusal::BelowMinimum {
            amount: 1,
            minimum: JTM,
        }
        .to_string(),
        DepositRefusal::AboveMaximum {
            amount: 2_000 * JTM,
            maximum: 1_000 * JTM,
        }
        .to_string(),
        DepositRefusal::ZeroRecipient.to_string(),
    ];
    let forbidden = [["Ether", "eum"].concat(), ["Z", "K"].concat()];
    for text in &texts {
        for word in &forbidden {
            assert!(
                !text
                    .to_ascii_lowercase()
                    .contains(&word.to_ascii_lowercase()),
                "{text}"
            );
        }
    }
    assert!(texts[0].contains("Base/EVM"));
    assert!(texts[4].contains("not armed"));
}

/// D123: the largest deposit a refusal names is the next block's: on a chain raising its caps at `c` (the test
/// chain), 1 000 JTM below `c`, 50 000 JTM from `c` on.
#[test]
fn d123_a_deposit_is_checked_against_the_maximum_of_the_next_block() {
    use jetsam_chain::consensus::vault::{LegacyVaultCaps, TESTNET_LEGACY_VAULT_CAPS};
    let f = armed(9_600).deposits_from_height.unwrap();
    let c = f + 100;
    let rules = VaultRules {
        legacy: Some(LegacyVaultCaps {
            raised_at: Some(c),
            ..TESTNET_LEGACY_VAULT_CAPS
        }),
        ..armed(9_600)
    };
    assert_eq!(check_deposit(1_000 * JTM, &RECIPIENT, c - 1, &rules), Ok(()));
    assert_eq!(
        check_deposit(1_000 * JTM + 1, &RECIPIENT, c - 1, &rules),
        Err(DepositRefusal::AboveMaximum {
            amount: 1_000 * JTM + 1,
            maximum: 1_000 * JTM
        })
    );
    assert_eq!(check_deposit(50_000 * JTM, &RECIPIENT, c, &rules), Ok(()));
    assert_eq!(
        check_deposit(50_000 * JTM + 1, &RECIPIENT, c, &rules),
        Err(DepositRefusal::AboveMaximum {
            amount: 50_000 * JTM + 1,
            maximum: 50_000 * JTM
        })
    );
}
