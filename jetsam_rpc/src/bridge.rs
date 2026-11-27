// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7: a deposit to the bridge vault as a wallet builds it (F4 of the final
//! audit: no user path to deposit existed outside the private rehearsal tool).
//!
//! A deposit is one ordinary group of the depositor's active address: it pays
//! the vault, then a zero-value marker `JTMBRDG1 ‖ recipient ‖ 0000` naming
//! the Base/EVM address the bridge mints to, then the change
//! ([`deposit_outputs`]). What makes it a deposit is V4, the shared rule
//! (`jetsam_vault_rule`) the node's block gate and the proof program both run:
//! [`check_deposit`] asks that rule, it never restates it. V8 (the room left
//! under the vault's ceiling) needs the branch state: the mempool applies it
//! at admission (`check_vault_deposit_room`), and so does every block.
//!
//! Before the build's v1.7 height the vault is an ordinary address whose spend
//! secret is published: anything sent there could be taken by anyone, and the
//! mempool takes such a payment as an ordinary one. A deposit is therefore
//! refused here until the block after the bridge opening `F`
//! ([`DepositRefusal::NotArmed`], [`DepositRefusal::NotOpenYet`]), on the
//! node's side (`walletPlanBridgeDeposit`, `walletBridgeDeposit`) as on the
//! CLI's.

use jetsam_chain::consensus::destination::mpt::keccak256;
use jetsam_chain::consensus::fees::fee_breakdown;
use jetsam_chain::consensus::identity::TICKER;
use jetsam_chain::consensus::params::MICRO_PER_JTM;
use jetsam_chain::consensus::vault::{deposit_marker, VaultRules, VAULT_ADDRESS};
use jetsam_poseidon2b::primitives::Address;
use jetsam_vault_rule::RuleError;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// Live outputs of a deposit group with its change: the vault, the marker,
/// the change back to the payer.
pub const DEPOSIT_OUTPUTS_WITH_CHANGE: u64 = 3;

/// Why a text is not a Base/EVM address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecipientError {
    /// No `0x` in front.
    MissingPrefix,
    /// Not forty digits after `0x` (how many there are).
    Length(usize),
    /// A character that is not a hexadecimal digit.
    NotHex,
    /// Upper and lower case mixed, against the address's checksum: the form
    /// these digits are written in.
    Checksum { expected: String },
}

impl std::fmt::Display for RecipientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingPrefix => write!(
                f,
                "a Base/EVM address starts with 0x, followed by 40 hexadecimal digits"
            ),
            Self::Length(digits) => write!(
                f,
                "a Base/EVM address has 40 hexadecimal digits after 0x, this one has {digits}"
            ),
            Self::NotHex => write!(
                f,
                "a Base/EVM address has only hexadecimal digits (0-9, a-f) after 0x"
            ),
            Self::Checksum { expected } => write!(
                f,
                "the capital letters of this Base/EVM address do not match its checksum, a sign of \
                 a typo: with these digits it is written {expected}. Check the address where you \
                 copied it"
            ),
        }
    }
}

impl std::error::Error for RecipientError {}

/// The 20 bytes of a Base/EVM address written `0x` and forty hexadecimal
/// digits (surrounding blanks ignored). Digits all in lower case or all in
/// capitals carry no checksum and are taken as written; mixed case must
/// match the address's checksum (EIP-55), which catches most typos.
pub fn parse_recipient(text: &str) -> Result<[u8; 20], RecipientError> {
    let text = text.trim();
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .ok_or(RecipientError::MissingPrefix)?;
    if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(RecipientError::NotHex);
    }
    if digits.len() != 40 {
        return Err(RecipientError::Length(digits.len()));
    }
    let mut recipient = [0u8; 20];
    hex::decode_to_slice(digits, &mut recipient).map_err(|_| RecipientError::NotHex)?;
    let lower = digits.bytes().any(|byte| byte.is_ascii_lowercase());
    let upper = digits.bytes().any(|byte| byte.is_ascii_uppercase());
    if lower && upper {
        let expected = checksummed(&recipient);
        if expected[2..] != *digits {
            return Err(RecipientError::Checksum { expected });
        }
    }
    Ok(recipient)
}

/// `0x` and the address's digits with its checksum (EIP-55): a letter is a
/// capital where the keccak-256 of the lower-case digits has a nibble of 8 or
/// more.
pub fn checksummed(recipient: &[u8; 20]) -> String {
    let lower = hex::encode(recipient);
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (index, digit) in lower.chars().enumerate() {
        let nibble = if index % 2 == 0 {
            hash[index / 2] >> 4
        } else {
            hash[index / 2] & 0x0f
        };
        out.push(if nibble >= 8 {
            digit.to_ascii_uppercase()
        } else {
            digit
        });
    }
    out
}

/// The outputs of a deposit of `amount` for the Base/EVM `recipient`, before
/// the change: the vault, then the marker naming the recipient (V4). The one
/// definition the wallet's deposits and the rehearsal tool use.
pub fn deposit_outputs(amount: u64, recipient: &[u8; 20]) -> Vec<(Address, u64)> {
    vec![(VAULT_ADDRESS, amount), (deposit_marker(recipient), 0)]
}

/// Why a deposit is refused before anything is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepositRefusal {
    /// The build carries no v1.7 height (none of the public profiles does
    /// yet): no deposit is valid, the vault is an ordinary address.
    NotArmed,
    /// V4-d: a deposit is valid only in a block above the bridge opening `F`.
    NotOpenYet { next_height: u64, opening: u64 },
    /// V4-c.
    BelowMinimum { amount: u64, minimum: u64 },
    /// V4-c: the per-deposit cap of the bridge contract.
    AboveMaximum { amount: u64, maximum: u64 },
    /// V4-b.
    ZeroRecipient,
    /// Any other refusal of the rule (none for the outputs of
    /// [`deposit_outputs`]).
    Rule(RuleError),
}

/// `amount` µ in coins, with the build's ticker.
fn coins(amount: u64) -> String {
    format!(
        "{}.{:06} {TICKER}",
        amount / MICRO_PER_JTM,
        amount % MICRO_PER_JTM
    )
}

impl std::fmt::Display for DepositRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotArmed => write!(
                f,
                "the bridge is not armed in this build (it carries no v1.7 activation height): \
                 the vault is then an ordinary address whose spend secret is published, and \
                 anyone could take what is sent there"
            ),
            Self::NotOpenYet {
                next_height,
                opening,
            } => write!(
                f,
                "deposits are valid from block {} on (the block after the bridge opening {opening}); \
                 the next block is {next_height}",
                opening.saturating_add(1)
            ),
            Self::BelowMinimum { amount, minimum } => write!(
                f,
                "a deposit is at least the minimum of {}; this one is {}",
                coins(*minimum),
                coins(*amount)
            ),
            Self::AboveMaximum { amount, maximum } => write!(
                f,
                "a deposit is at most the maximum of {} (the per-deposit cap of the bridge); this \
                 one is {}: make several deposits",
                coins(*maximum),
                coins(*amount)
            ),
            Self::ZeroRecipient => write!(f, "the zero address cannot receive a deposit"),
            Self::Rule(error) => write!(f, "the deposit rule refuses it: {error:?}"),
        }
    }
}

impl std::error::Error for DepositRefusal {}

/// Whether a deposit of `amount` for `recipient`, offered for the block at
/// `next_height`, is one under `rules`: V4 itself on [`deposit_outputs`]
/// (shape, recipient, amount, `F`), refused outright while `rules` carry no
/// v1.7 height. V8 is the mempool's (it needs the vault's balance).
pub fn check_deposit(
    amount: u64,
    recipient: &[u8; 20],
    next_height: u64,
    rules: &VaultRules,
) -> Result<(), DepositRefusal> {
    let (Some(_), Some(opening)) = (rules.activation_height, rules.deposits_from_height) else {
        return Err(DepositRefusal::NotArmed);
    };
    let outputs: Vec<([u8; 32], u64)> = deposit_outputs(amount, recipient)
        .into_iter()
        .map(|(owner, amount)| (owner.0, amount))
        .collect();
    match jetsam_vault_rule::deposit_shape(0, &outputs, next_height, &rules.deposit_rule()) {
        Ok(Some(_)) => Ok(()),
        Err(RuleError::DepositAmount { amount, .. }) if amount < rules.min_deposit => {
            Err(DepositRefusal::BelowMinimum {
                amount,
                minimum: rules.min_deposit,
            })
        }
        // D123: the largest deposit of the next block.
        Err(RuleError::DepositAmount { amount, .. }) => Err(DepositRefusal::AboveMaximum {
            amount,
            maximum: rules.at(next_height).max_deposit,
        }),
        // One vault output and one marker whose reserved bytes are zero: the
        // recipient is the only part of the shape that can be wrong (V4-b).
        Err(RuleError::DepositShape { .. }) => Err(DepositRefusal::ZeroRecipient),
        Err(RuleError::DepositBeforeBridgeOpening { height, .. }) => {
            Err(DepositRefusal::NotOpenYet {
                next_height: height,
                opening,
            })
        }
        Err(error) => Err(DepositRefusal::Rule(error)),
        Ok(None) => Err(DepositRefusal::Rule(RuleError::DepositShape { group: 0 })),
    }
}

/// The fee of a deposit when none is given: the consensus minimum of a
/// deposit group of one input page (1 to 8 inputs) with its change, and at
/// least the relay floor of that shape, the most of them.
pub fn deposit_fee(
    active_slot_count: u64,
    log_slots: u32,
    relay_floor: jetsam_mempool::RelayFloor,
) -> u64 {
    (1..=jetsam_tx::TX_INPUTS as u64)
        .map(|inputs| {
            fee_breakdown(
                inputs,
                DEPOSIT_OUTPUTS_WITH_CHANGE,
                active_slot_count,
                log_slots,
            )
            .required_total
            .max(relay_floor.required(inputs, DEPOSIT_OUTPUTS_WITH_CHANGE))
        })
        .max()
        .unwrap_or(0)
}

/// `walletPlanBridgeDeposit`: what `walletBridgeDeposit` builds now, with
/// the same checks. Nothing is built, reserved or sent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeDepositPlan {
    /// The Base/EVM recipient, with its checksum.
    pub recipient: String,
    pub amount_micro_jtm: u64,
    pub fee_micro_jtm: u64,
    /// Amount and fee.
    pub total_micro_jtm: u64,
    /// The active address, which pays.
    pub from: String,
    /// Its spendable balance.
    pub spendable_micro_jtm: u64,
    /// The bridge vault (bech32m).
    pub vault: String,
    /// The marker's owner (hex): `JTMBRDG1 ‖ recipient ‖ 00000000`.
    pub marker: String,
    /// The block the deposit is offered for.
    pub next_height: u64,
    /// `F`: deposits only in blocks above it.
    pub deposits_from_height: u64,
}

/// `walletBridgeDeposit`: the deposit the mempool admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeDepositResult {
    pub txid: String,
    /// The Base/EVM recipient, with its checksum.
    pub recipient: String,
    pub amount_micro_jtm: u64,
    pub fee_micro_jtm: u64,
    pub input_count: usize,
    pub output_count: usize,
    /// The block it was offered for.
    pub next_height: u64,
}
