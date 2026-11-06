// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.
//! jetsam-cli — Jetsam full node CLI client.
//!
//! Connects to a running `jetsam` daemon via JSON-RPC (no local keys, no crypto).
//! All operations happen inside the daemon; the CLI is a thin terminal UI.
//!
//! Quick start:
//!   jetsam-cli status            — node health at a glance
//!   jetsam-cli balance           — wallet balance
//!   jetsam-cli send <addr> 10.5  — send 10.5 JTM
//!   jetsam-cli help              — full command list

#![allow(clippy::format_in_format_args, clippy::print_literal)]

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use jetsam_chain::consensus::identity::TICKER;
use jetsam_chain::consensus::params::RETAINED_BLOCK_SERVING_DEPTH;
use jetsam_chain::consensus::vault::VaultRules;

/// Where the daemon of *this* build listens by default.
///
/// A testnet daemon listens on 9711, not 9701. A client that assumes the
/// public network's port refuses to connect to the very daemon it shipped
/// with, and says "connection refused" rather than "wrong network" — which
/// sends the reader looking for a firewall.
///
/// `clap` needs a literal here, so the port cannot be interpolated from
/// [`identity::DEFAULT_RPC_PORT`]. The two are pinned to each other by
/// `the_default_endpoint_matches_this_build` below: change one without the
/// other and the test fails.
///
/// The profile is read from `jetsam_chain` (`IS_TEST_CHAIN`), not from this
/// crate's own `testnet` feature, which is off when the profile is selected
/// with `--features jetsam_chain/testnet`.
const DEFAULT_RPC_URL: &str = if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
    "http://127.0.0.1:9711"
} else {
    "http://127.0.0.1:9701"
};
use serde_json::Value;
use std::io::{self, IsTerminal, Write};

// ---------------------------------------------------------------------------
// ANSI terminal colours (no external crate needed)
// ---------------------------------------------------------------------------

const RST: &str = "\x1b[0m"; // reset
const BOLD: &str = "\x1b[1m"; // bold
const DIM: &str = "\x1b[2m"; // dim
const RED: &str = "\x1b[31m"; // error
const GRN: &str = "\x1b[32m"; // success / positive
const YLW: &str = "\x1b[33m"; // warning
const CYN: &str = "\x1b[36m"; // label / key
const WHT: &str = "\x1b[97m"; // bright white value

/// Check whether stdout is a real terminal (disable colours when piped).
fn is_tty() -> bool {
    io::stdout().is_terminal()
        && std::env::var("TERM").is_ok_and(|t| t != "dumb")
        && std::env::var("NO_COLOR").is_err()
        && std::env::var("CI").is_err()
}

/// Return coloured string only when outputting to a terminal.
macro_rules! c {
    ($colour:expr, $text:expr) => {{
        if is_tty() {
            format!("{}{}{}", $colour, $text, RST)
        } else {
            $text.to_string()
        }
    }};
}

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

const MICRO_PER_ELD: u64 = 1_000_000;

/// Parse a human amount like "10.5" or "0.000001" as JTM → μJTM.
fn parse_jetsam_amount(s: &str) -> anyhow::Result<u64> {
    let (whole, fractional) = match s.split_once('.') {
        Some((whole, fractional)) if !fractional.contains('.') => (whole, fractional),
        Some(_) => bail!("invalid amount {s:?}: more than one decimal point"),
        None => (s, ""),
    };
    if whole.is_empty() && fractional.is_empty() {
        bail!("amount cannot be empty");
    }
    if whole.starts_with('-') || fractional.starts_with('-') {
        bail!("amount cannot be negative");
    }
    if fractional.len() > 6 {
        bail!("amount has more than 6 decimal places");
    }
    if (!whole.is_empty() && !whole.bytes().all(|byte| byte.is_ascii_digit()))
        || !fractional.bytes().all(|byte| byte.is_ascii_digit())
    {
        bail!("invalid amount {s:?}: expected a decimal number like 10.5 or 0.000001");
    }
    let whole = if whole.is_empty() {
        0
    } else {
        whole
            .parse::<u64>()
            .with_context(|| format!("amount {s:?} is out of range"))?
    };
    let fractional = if fractional.is_empty() {
        0
    } else {
        let digits = fractional
            .parse::<u64>()
            .with_context(|| format!("invalid fractional amount {s:?}"))?;
        digits
            .checked_mul(10u64.pow(6 - fractional.len() as u32))
            .ok_or_else(|| anyhow::anyhow!("amount {s:?} is out of range"))?
    };
    whole
        .checked_mul(MICRO_PER_ELD)
        .and_then(|value| value.checked_add(fractional))
        .ok_or_else(|| anyhow::anyhow!("amount {s:?} is out of range"))
}

fn jetsam_str(micro_jtm: u64) -> String {
    format!(
        "{}.{:06}",
        micro_jtm / MICRO_PER_ELD,
        micro_jtm % MICRO_PER_ELD
    )
}

fn fmt_hash(h: &str) -> &str {
    h.trim_start_matches("0x")
}

// ---------------------------------------------------------------------------
// CLI structure
// ---------------------------------------------------------------------------

/// The top-level long help, in this chain's ticker.
fn cli_long_about() -> String {
    format!(
        "\
Jetsam thin client. Connects to a running jetsam daemon via JSON-RPC.

QUICK START:
  jetsam-cli status              Node info (height, hash, slots)
  jetsam-cli balance             Wallet balance
  jetsam-cli send <addr> 10.5   Send 10.5 {TICKER} to address
  jetsam-cli history             Transaction history
  jetsam-cli mempool             Pending transactions
  jetsam-cli help                All commands

AMOUNT FORMAT:
  Amounts are in {TICKER} (e.g. 10.5, 0.000001).
  1 {TICKER} = 1,000,000 \u{03bc}{TICKER} \u{2014} the CLI converts automatically.

DAEMON:
  The daemon must be running: jetsam --mode miner --data-dir ~/.jetsam/data"
    )
}

#[derive(Parser)]
#[command(
    name = "jetsam-cli",
    about = "Jetsam thin client — control a running jetsam daemon",
    version = env!("CARGO_PKG_VERSION"),
    long_about = cli_long_about(),
)]
struct Cli {
    /// JSON-RPC endpoint of the running jetsam daemon.
    #[arg(
        long,
        short = 'r',
        default_value = DEFAULT_RPC_URL,
        env = "JETSAM_RPC",
        value_name = "URL",
        global = true
    )]
    rpc: String,

    /// Output raw JSON (for scripting / piping to jq).
    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    // ---- Chain ----------------------------------------------------------------
    /// Node status: height, best hash, difficulty, active UTXOs.
    Status,

    /// Block hash at a given height.
    #[command(name = "block-hash", alias = "bh")]
    BlockHash {
        /// Block height.
        height: u64,
    },

    /// Decoded block header at a given height (all fields as structured data).
    #[command(name = "block-header", alias = "bhead")]
    BlockHeader {
        /// Block height.
        height: u64,
    },

    /// Full raw block at a given height (retained recent blocks only).
    #[command(alias = "blk")]
    Block {
        /// Block height to query.
        height: u64,
    },

    /// Raw 212-byte block header hex (for developers).
    Header {
        /// Block height to query.
        height: u64,
    },

    /// Current finalized HistoryStep terminal.
    HistoryStepTerminal,

    /// UTXO slot by index: value and owner.
    Slot {
        /// Slot index (0-based).
        index: u32,
    },

    /// All UTXOs owned by an address (bech32m).
    #[command(name = "utxos-of")]
    UtxosOf {
        /// Owner address (bech32m j1…).
        #[arg(value_name = "ADDRESS")]
        address: String,
    },

    /// Confirmed transaction info by hash.
    Tx {
        /// Transaction body hash (64-char hex).
        #[arg(value_name = "TX_HASH")]
        txhash: String,
    },

    /// UTXO state dimensions: capacity, fill %, size on disk, expansion headroom.
    State,

    /// Mining info: difficulty, block reward, HistoryStep height.
    Mining,

    /// Number of connected peers.
    Peers,

    /// Adoption of a release: the versions this node's peers announce.
    ///
    /// Counts the peers whose network profile was verified (on this chain),
    /// connected now and seen within --window seconds, by announced version,
    /// and the share running --target (by default, what the asked node
    /// itself announces). Answers of several nodes (--node, --answer) are
    /// merged by peer identity, given to a loopback or bearer RPC only. A
    /// version is what a peer claims; a release older than the announcement
    /// shows `rust-libp2p/<version>`. Hashrate is not measured here.
    Adoption {
        /// Release (`1.6.0`) or exact announcement
        /// (`jetsam/1.6.0 (v1.6=4800)`) to count.
        #[arg(long, value_name = "RELEASE")]
        target: Option<String>,
        /// A departed peer still counts if seen within this many seconds.
        #[arg(long, value_name = "SECONDS", default_value_t = 3_600)]
        window: u64,
        /// Another node's RPC endpoint to ask as well (repeatable).
        #[arg(long = "node", value_name = "URL")]
        nodes: Vec<String>,
        /// A saved `getPeerVersions` answer (bare or JSON-RPC envelope) to
        /// merge (repeatable).
        #[arg(long = "answer", value_name = "FILE")]
        answers: Vec<std::path::PathBuf>,
        /// Do not ask the --rpc node; use --node and --answer only.
        #[arg(long)]
        offline: bool,
    },

    /// Estimate minimum relay fee for live inputs/outputs.
    #[command(name = "estimate-fee")]
    EstimateFee {
        /// Number of outputs in the transaction (default: 2).
        #[arg(default_value_t = 2)]
        n_outputs: u32,
        /// Number of live inputs in the transaction (default: 1).
        #[arg(long, default_value_t = 1, value_name = "N")]
        inputs: u32,
    },

    /// Validate an address and show its canonical bech32m form.
    Validate {
        /// Address to validate (bech32m).
        #[arg(value_name = "ADDRESS")]
        address: String,
    },

    /// Current epoch anchor hash (needed by wallets to build transactions).
    #[command(alias = "anchor")]
    Epoch,

    /// Pending transactions in the mempool.
    Mempool,

    /// Single pending transaction by hash.
    #[command(name = "mempool-tx")]
    MempoolTx {
        /// Transaction body hash (64-char hex).
        #[arg(value_name = "TX_HASH")]
        txhash: String,
    },

    // ---- Wallet ---------------------------------------------------------------
    /// Show and manage wallet addresses.
    #[command(alias = "addr")]
    Address {
        /// Generate the next inactive address.
        #[arg(long)]
        new: bool,
        /// List locally generated addresses (the active one is marked).
        #[arg(long)]
        list: bool,
        /// Show address at a specific key index.
        #[arg(long, value_name = "INDEX")]
        index: Option<u32>,
        /// Switch the ACTIVE address to this key index. Sends spend from the
        /// active address only (one owner per transaction); change returns
        /// to it.
        #[arg(long = "use", value_name = "INDEX")]
        use_index: Option<u32>,
    },

    #[command(
        alias = "bal",
        about = format!("Confirmed wallet balance ({TICKER} and \u{03bc}{TICKER}).")
    )]
    Balance,

    /// List all confirmed UTXOs with slot index, value, and height.
    #[command(alias = "ls")]
    Utxos,

    #[command(
        about = format!("Send {TICKER} to a recipient address."),
        long_about = format!(
            "Send {TICKER} to a recipient address.\n\n\
             Amount is in {TICKER} (e.g. 10.5 \u{2192} 10,500,000 \u{03bc}{TICKER}).\n\
             Fee is auto-computed if not specified (recommended).\n\n\
             Examples:\n  \
             jetsam-cli send f784...b61e 10.5\n  \
             jetsam-cli send f784...b61e 10.5 --fee 0.01"
        )
    )]
    Send {
        /// Recipient address (canonical bech32m j1...).
        #[arg(value_name = "ADDRESS")]
        to: String,
        #[arg(
            value_name = "AMOUNT",
            help = format!(
                "Amount in {TICKER}  (1 {TICKER} = 1 000 000 \u{03bc}{TICKER}). \
                 Examples: \"50\" = 50 {TICKER}, \"0.5\" = 500 000 \u{03bc}{TICKER}, \
                 \"0.000001\" = 1 \u{03bc}{TICKER} (minimum). \
                 Tip: for programmatic use the RPC walletSend accepts raw \u{03bc}{TICKER} directly."
            )
        )]
        amount: String,
        #[arg(
            long,
            value_name = "FEE",
            help = format!("Transaction fee in {TICKER}. Omit for automatic fee calculation.")
        )]
        fee: Option<String>,
        /// Show the ordinary-transaction plan without submitting.
        #[arg(long)]
        dry_run: bool,
    },

    /// Transaction history: received and sent transactions.
    #[command(alias = "hist", alias = "txs")]
    History {
        /// Filter by a specific wallet address (bech32m).
        #[arg(long, value_name = "ADDRESS")]
        address: Option<String>,
        /// Show only the last N entries.
        #[arg(long, value_name = "N")]
        last: Option<usize>,
    },

    /// Reload the active address from the verified owner index.
    Scan,

    /// Export a Merkle payment receipt for a confirmed transaction.
    /// Redirect output to a file: jetsam-cli receipt <hash> > receipt.hex
    Receipt {
        /// Transaction hash (64-char hex).
        #[arg(value_name = "TX_HASH")]
        txhash: String,
    },

    /// Verify a Merkle payment receipt against the canonical chain.
    #[command(alias = "check")]
    Verify {
        /// Receipt bytes as hex string (from 'receipt' command).
        #[arg(value_name = "RECEIPT_HEX")]
        receipt: String,
    },

    // ---- Delegation (v1.7) ----------------------------------------------------
    /// Delegated stake beside the guardians (v1.7): lock, unlock, claim, info,
    /// payout. Dormant until the delegation height J_D and the first funding
    /// of the reward pool by both development funds in one block.
    #[command(subcommand)]
    Stake(StakeCommand),

    // ---- Bridge (v1.7) --------------------------------------------------------
    /// The bridge to Base/EVM (v1.7): deposit to the bridge vault for a
    /// Base/EVM address, and read the bridge's state. Deposits are refused
    /// while this build carries no v1.7 activation height.
    #[command(subcommand)]
    Bridge(BridgeCommand),

    // ---- Node control ---------------------------------------------------------
    /// Gracefully stop the Jetsam daemon.
    Stop,

    // ---- Mining (external miner API) ------------------------------------------
    /// Get a block template for an external Poseidon2b PoW miner.
    #[command(name = "block-template", alias = "template")]
    BlockTemplate {
        /// Coinbase address for this template (bech32m j1...). Defaults to wallet address.
        #[arg(long, value_name = "ADDRESS", default_value = "")]
        miner_addr: String,
    },

    /// Submit a nonce for a node-owned, single-use mining template.
    #[command(name = "submit-block", alias = "submit")]
    SubmitBlock {
        /// Opaque template identifier returned by `block-template`.
        #[arg(value_name = "TEMPLATE_ID")]
        template_id: String,
        /// Exactly 16 little-endian nonce bytes as 32 lowercase hex characters.
        #[arg(value_name = "NONCE_HEX")]
        nonce_hex: String,
    },
}

/// `jetsam-cli stake …` (v1.7, T3.8). No graphical interface (D13).
#[derive(Subcommand)]
enum StakeCommand {
    /// Lock an amount of the active address into the stake vault; it earns
    /// from the next reward period on.
    Lock {
        /// Amount, e.g. 10.5 (at least the minimum lock).
        #[arg(value_name = "AMOUNT")]
        amount: String,
        /// Explicit fee of the paying transaction.
        #[arg(long, value_name = "AMOUNT")]
        fee: Option<String>,
    },

    /// Ask for staked coins back. Each entry is payable by anyone after the
    /// unbonding delay; an amount larger than one payout can pay is split
    /// into several entries. The fee is paid from another small coin of the
    /// active address: keep one.
    Unlock {
        /// Amount, e.g. 10.5.
        #[arg(value_name = "AMOUNT")]
        amount: String,
        /// Explicit fee of each paying transaction.
        #[arg(long, value_name = "AMOUNT")]
        fee: Option<String>,
    },

    /// Claim reward of the active address, payable from the next block. The
    /// fee is paid from another small coin of the active address: keep one.
    Claim {
        /// Amount, e.g. 2.
        #[arg(value_name = "AMOUNT")]
        amount: String,
        /// Claim the guardians' vested share (its reward owner only).
        #[arg(long)]
        guardian: bool,
        /// Explicit fee of the paying transaction.
        #[arg(long, value_name = "AMOUNT")]
        fee: Option<String>,
    },

    /// Stake, entries and reward of an address (default: the active one).
    Info {
        /// Owner address (bech32m).
        #[arg(value_name = "ADDRESS")]
        address: Option<String>,
    },

    /// Pay a due entry out of its pool; anyone may, no key is needed.
    Payout {
        /// Entry id (see `stake info`).
        #[arg(value_name = "ENTRY_ID")]
        entry_id: u64,
    },
}

/// `jetsam-cli bridge …` (v1.7, F4 of the final audit): deposit to the bridge
/// vault for a Base/EVM address, and read the bridge's state. No graphical
/// interface yet.
#[derive(Subcommand)]
enum BridgeCommand {
    #[command(
        about = format!(
            "Deposit {TICKER} of the active address to the bridge vault, minted on Base/EVM."
        ),
        long_about = format!(
            "Deposit {TICKER} of the active address to the bridge vault for a Base/EVM address.\n\n\
             The daemon's wallet builds and signs one transaction: the amount to the vault, a \
             marker naming the Base/EVM address, the change back. The bridge mints the same \
             amount to that address on Base/EVM: by the slow lane, at the next checkpoint (about \
             every 1 h 50) then about 24 h; or sooner, less a fee, if an advancer advances it. \
             Deposits left out beyond 22 a day wait; none is lost.\n\n\
             A deposit cannot be cancelled. The plan is shown first and nothing is sent until \
             you type yes (or pass --yes). Refused while this build carries no v1.7 activation \
             height (never on mainnet before that), until the block after the bridge opening, \
             and outside the limits of one deposit (see 'jetsam-cli bridge status').\n\n\
             Examples:\n  \
             jetsam-cli bridge deposit 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed 25 --dry-run\n  \
             jetsam-cli bridge deposit 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed 25"
        )
    )]
    Deposit {
        /// The Base/EVM address that receives the minted coins: 0x and 40
        /// hexadecimal digits (mixed case must match its checksum). Use an
        /// address you control.
        #[arg(value_name = "BASE_EVM_ADDRESS")]
        recipient: String,
        #[arg(
            value_name = "AMOUNT",
            help = format!(
                "Amount in {TICKER}, e.g. 25 or 12.5, within the limits of one deposit \
                 ('jetsam-cli bridge status')."
            )
        )]
        amount: String,
        /// Explicit fee of the deposit transaction. Omit for the automatic one.
        #[arg(long, value_name = "AMOUNT")]
        fee: Option<String>,
        /// Show the plan only: nothing is built or sent.
        #[arg(long)]
        dry_run: bool,
        /// Send without asking (scripts). Required when there is no terminal
        /// to type yes in.
        #[arg(long)]
        yes: bool,
    },

    /// The bridge to Base/EVM as the node sees it (read only): armed or not,
    /// the block from which deposits are valid, the vault's balance and room,
    /// the limits of one deposit.
    Status,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Suppress broken-pipe panics: happen when stdout is piped to `head`, etc.
    // Without this, `jetsam-cli utxos | head -5` prints a Rust panic traceback.
    // The payload can be either &str or String depending on the panic site.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg_str = info.payload().downcast_ref::<&str>().copied().unwrap_or("");
        let msg_string = info
            .payload()
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .unwrap_or("");
        if msg_str.contains("Broken pipe") || msg_string.contains("Broken pipe") {
            std::process::exit(0);
        }
        default_hook(info);
    }));

    let cli = Cli::parse();

    if let Err(e) = run(cli).await {
        // Use {:#} to get the full error chain (context: cause: ...)
        // so print_error can detect "Node is not responding" in any layer.
        print_error(&format!("{e:#}"));
        std::process::exit(1);
    }

    // Flush stdout before exit (avoids broken-pipe on some platforms).
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("jetsam-cli/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let ctx = Ctx {
        client: &client,
        rpc: &cli.rpc,
        json: cli.json,
    };

    match &cli.cmd {
        Command::Status => cmd_status(&ctx).await,
        Command::BlockHash { height } => cmd_block_hash(&ctx, *height).await,
        Command::BlockHeader { height } => cmd_block_header(&ctx, *height).await,
        Command::Block { height } => cmd_block(&ctx, *height).await,
        Command::Header { height } => cmd_header(&ctx, *height).await,
        Command::HistoryStepTerminal => cmd_history_step_terminal(&ctx).await,
        Command::Slot { index } => cmd_slot(&ctx, *index).await,
        Command::UtxosOf { address } => cmd_utxos_of(&ctx, address).await,
        Command::Tx { txhash } => cmd_tx(&ctx, txhash).await,
        Command::State => cmd_state(&ctx).await,
        Command::Mining => cmd_mining(&ctx).await,
        Command::Peers => cmd_peers(&ctx).await,
        Command::Adoption {
            target,
            window,
            nodes,
            answers,
            offline,
        } => cmd_adoption(&ctx, target.as_deref(), *window, nodes, answers, *offline).await,
        Command::EstimateFee { n_outputs, inputs } => {
            cmd_estimate_fee(&ctx, *inputs, *n_outputs).await
        }
        Command::Validate { address } => cmd_validate(&ctx, address).await,
        Command::Epoch => cmd_epoch(&ctx).await,
        Command::Mempool => cmd_mempool(&ctx).await,
        Command::MempoolTx { txhash } => cmd_mempool_tx(&ctx, txhash).await,
        Command::Address {
            new,
            list,
            index,
            use_index,
        } => cmd_address(&ctx, *new, *list, *index, *use_index).await,
        Command::Balance => cmd_balance(&ctx).await,
        Command::Utxos => cmd_utxos(&ctx).await,
        Command::Send {
            to,
            amount,
            fee,
            dry_run,
        } => cmd_send(&ctx, to, amount, fee.as_deref(), *dry_run).await,
        Command::History { address, last } => cmd_history(&ctx, address.as_deref(), *last).await,
        Command::Scan => cmd_scan(&ctx).await,
        Command::Receipt { txhash } => cmd_receipt(&ctx, txhash).await,
        Command::Verify { receipt } => cmd_verify(&ctx, receipt).await,
        Command::Stake(command) => cmd_stake(&ctx, command).await,
        Command::Bridge(command) => cmd_bridge(&ctx, command).await,
        Command::Stop => cmd_stop(&ctx).await,
        Command::BlockTemplate { miner_addr } => cmd_block_template(&ctx, miner_addr).await,
        Command::SubmitBlock {
            template_id,
            nonce_hex,
        } => cmd_submit_block(&ctx, template_id, nonce_hex).await,
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct Ctx<'a> {
    client: &'a reqwest::Client,
    rpc: &'a str,
    json: bool,
}

impl<'a> Ctx<'a> {
    fn h<'h>(&self, hash: &'h str) -> &'h str {
        fmt_hash(hash)
    }
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

fn print_error(msg: &str) {
    // Check for common connection errors
    let human = if msg.contains("Connection refused")
        || msg.contains("connection refused")
        || msg.contains("ConnectError")
        || msg.contains("Node is not responding")
    {
        format!(
            "Node is not responding.\n\
             Is the jetsam daemon running?  Try: jetsam --mode miner\n\
             Default RPC: {DEFAULT_RPC_URL}  (override with --rpc)"
        )
    } else if msg.contains("Insufficient") || msg.contains("insufficient") {
        // Extract amounts from error if possible
        msg.replace("InsufficientFunds", "Insufficient funds")
            .replace("_", " ")
    } else if msg.contains("Bad address")
        || msg.contains("bad address")
        || msg.contains("invalid address")
        || msg.contains("WrongHrp")
    {
        "Invalid address.\nUse a bech32m address (j1…).\nExample: jetsam-cli send j1q9gnyj0zwhqj9tm5sf… 10.5".to_string()
    } else {
        msg.to_string()
    };

    eprintln!("{} {}", c!(RED, "Error:"), human);
}

fn section(title: &str) {
    if is_tty() {
        println!("{}{}{}", BOLD, title, RST);
    } else {
        println!("{title}");
    }
}

fn kv(key: &str, val: &str) {
    if is_tty() {
        println!("  {}{:<18}{} {}{}{}", CYN, key, RST, WHT, val, RST);
    } else {
        println!("  {key:<18} {val}");
    }
}

fn kv2(key: &str, main: &str, sub: &str) {
    if is_tty() {
        println!(
            "  {}{:<18}{} {}{}{} {}{}{}",
            CYN, key, RST, WHT, main, RST, DIM, sub, RST
        );
    } else {
        println!("  {key:<18} {main}  {sub}");
    }
}

fn ok_msg(msg: &str) {
    println!("{} {}", c!(GRN, "✓"), msg);
}

fn warn_msg(msg: &str) {
    println!("{} {}", c!(YLW, "⚠"), msg);
}

fn separator(width: usize) {
    println!("  {}", c!(DIM, &"─".repeat(width)));
}

// ---------------------------------------------------------------------------
// Chain commands
// ---------------------------------------------------------------------------

async fn cmd_status(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let info = rpc(ctx, "getChainInfo", &[])
        .await
        .context("getChainInfo")?;

    if ctx.json {
        return print_json(&info);
    }

    let height = info["height"].as_u64().unwrap_or(0);
    let best_hash = info["best_hash"].as_str().unwrap_or("?");
    let diff = info["difficulty_target"].as_str().unwrap_or("?");
    let slots = info["active_slot_count"].as_u64().unwrap_or(0);
    let log_slots = info["log_slots"].as_u64().unwrap_or(0);
    let capacity = 1u64 << log_slots.min(63);
    let fill_pct = slots.saturating_mul(100).checked_div(capacity).unwrap_or(0);

    let diff_bits = hex::decode(diff)
        .ok()
        .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())
        .map(|target| jetsam_chain::consensus::target_leading_zero_bits(&target));

    section("Jetsam node status");
    kv("Height", &height.to_string());
    kv("Best hash", ctx.h(best_hash));
    kv2(
        "Difficulty",
        &diff_bits.map_or_else(|| "unknown".into(), |bits| format!("{bits} leading zeros")),
        &format!("(0x{})", &diff[..diff.len().min(16)]),
    );
    kv2(
        "Active UTXOs",
        &format!("{slots}"),
        &format!("({fill_pct}% of {capacity} slots, log={log_slots})"),
    );

    // Also fetch mempool size for quick overview
    if let Ok(mp) = rpc(ctx, "getMempoolSize", &[]).await {
        let n = mp.as_u64().unwrap_or(0);
        kv("Mempool", &format!("{n} pending tx(s)"));
    }

    Ok(())
}

async fn cmd_block(ctx: &Ctx<'_>, height: u64) -> anyhow::Result<()> {
    let result = rpc(ctx, "getBlock", &[height.into()])
        .await
        .context("getBlock")?;

    if ctx.json {
        return print_json(&result);
    }

    if result.is_null() {
        warn_msg(&format!(
            "Block {height} not available (the local complete-bundle reserve covers up to the last {RETAINED_BLOCK_SERVING_DEPTH} heights after warm-up)."
        ));
        return Ok(());
    }

    // The block is raw hex — show basic info
    let hex = result.as_str().unwrap_or("");
    section(&format!("Block #{height}"));
    kv(
        "Hex length",
        &format!("{} bytes ({} hex chars)", hex.len() / 2, hex.len()),
    );
    kv2(
        "Header (first 276B)",
        ctx.h(&hex[..hex.len().min(64)]),
        "(raw hex)",
    );
    println!();
    println!("  {} Use --json to get the full raw hex.", c!(DIM, "Tip:"));

    Ok(())
}

async fn cmd_header(ctx: &Ctx<'_>, height: u64) -> anyhow::Result<()> {
    let result = rpc(ctx, "getHeaderByHeight", &[height.into()])
        .await
        .context("getHeaderByHeight")?;

    if ctx.json {
        return print_json(&result);
    }

    if result.is_null() {
        warn_msg(&format!("No header found at height {height}."));
        return Ok(());
    }

    // For developers: print the raw hex
    let hex = result.as_str().unwrap_or("");
    section(&format!("Block header #{height}"));
    kv("Size", &format!("{} bytes (276)", hex.len() / 2));
    println!();
    // Print in 80-char rows for readability
    let hex_out = hex;
    for (i, chunk) in hex_out.as_bytes().chunks(80).enumerate() {
        let row = std::str::from_utf8(chunk).unwrap_or("");
        if i == 0 {
            println!("  {}", row);
        } else {
            println!("  {}{}{}", DIM, row, RST);
        }
    }

    Ok(())
}

async fn cmd_history_step_terminal(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getHistoryStepTerminal", &[])
        .await
        .context("getHistoryStepTerminal")?;

    if ctx.json {
        return print_json(&result);
    }

    if result.is_null() {
        warn_msg("No HistoryStep terminal is available at the finalized boundary.");
        return Ok(());
    }

    let hex = result.as_str().unwrap_or("");
    let bytes = hex.len() / 2;
    let kb = bytes as f64 / 1024.0;

    // Simple fingerprint: first 16 + last 16 chars of the hex
    let terminal_fingerprint = if hex.len() >= 32 {
        format!("{}…{}", &hex[..8], &hex[hex.len() - 8..])
    } else {
        hex[..hex.len().min(16)].to_string()
    };

    section("HistoryStep terminal");
    kv2(
        "Size",
        &format!("{bytes} bytes ({kb:.1} KB)"),
        "(constant-size history envelope)",
    );
    kv("Fingerprint", &terminal_fingerprint);
    println!();
    println!(
        "  {} The terminal binds the selected chain boundary for O(1)-history verification.",
        c!(DIM, "Note:")
    );

    Ok(())
}

async fn cmd_slot(ctx: &Ctx<'_>, index: u32) -> anyhow::Result<()> {
    let result = rpc(ctx, "getSlot", &[index.into()])
        .await
        .context("getSlot")?;

    if ctx.json {
        return print_json(&result);
    }

    let empty = result["empty"].as_bool().unwrap_or(true);
    let value = result["value"].as_u64().unwrap_or(0);
    let creation_id = result["creation_id"]
        .as_u64()
        .context("getSlot response missing creation_id")?;
    let owner = result["owner"].as_str().unwrap_or("?");

    section(&format!("Slot #{index}"));
    if empty {
        kv("Status", &c!(DIM, "empty (unspent / available)"));
    } else {
        kv("Status", &c!(GRN, "live UTXO"));
        kv2(
            "Value",
            &format!("{} {TICKER}", jetsam_str(value)),
            &format!("({value} μ{TICKER})"),
        );
        kv("Creation ID", &creation_id.to_string());
        kv("Owner", ctx.h(owner));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// New chain commands
// ---------------------------------------------------------------------------

async fn cmd_block_hash(ctx: &Ctx<'_>, height: u64) -> anyhow::Result<()> {
    let result = rpc(ctx, "getBlockHash", &[height.into()])
        .await
        .context("getBlockHash")?;
    if ctx.json {
        return print_json(&result);
    }
    match result.as_str() {
        Some(hash) => {
            section(&format!("Block #{height} hash"));
            kv("Hash", hash);
        }
        None => warn_msg(&format!("Block {height} not found")),
    }
    Ok(())
}

async fn cmd_block_header(ctx: &Ctx<'_>, height: u64) -> anyhow::Result<()> {
    let result = rpc(ctx, "getBlockHeader", &[height.into()])
        .await
        .context("getBlockHeader")?;
    if ctx.json {
        return print_json(&result);
    }
    if result.is_null() {
        warn_msg(&format!("Block {height} not found"));
        return Ok(());
    }
    section(&format!("Block header #{height}"));
    kv("hash", result["hash"].as_str().unwrap_or("?"));
    kv("prev_hash", result["prev_hash"].as_str().unwrap_or("?"));
    kv(
        "height",
        &result["height"].as_u64().unwrap_or(0).to_string(),
    );
    kv(
        "timestamp",
        &result["timestamp"].as_u64().unwrap_or(0).to_string(),
    );
    kv("miner", result["miner"].as_str().unwrap_or("?"));
    kv("state_root", result["state_root"].as_str().unwrap_or("?"));
    kv("tx_root", result["tx_root"].as_str().unwrap_or("?"));
    kv(
        "difficulty_target",
        result["difficulty_target"].as_str().unwrap_or("?"),
    );
    kv(
        "active_slot_count",
        &result["active_slot_count"]
            .as_u64()
            .unwrap_or(0)
            .to_string(),
    );
    kv(
        "log_slots",
        &result["log_slots"].as_u64().unwrap_or(0).to_string(),
    );
    kv(
        "alloc_counter",
        &result["alloc_counter"].as_u64().unwrap_or(0).to_string(),
    );
    Ok(())
}

async fn cmd_utxos_of(ctx: &Ctx<'_>, address: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "getSlotsByOwner", &[address.into()])
        .await
        .context("getSlotsByOwner")?;
    if ctx.json {
        return print_json(&result);
    }
    let slots = result.as_array().cloned().unwrap_or_default();
    section(&format!("UTXOs of {address}"));
    if slots.is_empty() {
        println!("  {}", c!(DIM, "(no UTXOs found for this address)"));
        return Ok(());
    }
    let total: u64 = slots.iter().map(|s| s["value"].as_u64().unwrap_or(0)).sum();
    separator(74);
    println!("  {:<12}  {:>20}  {:>14}", "slot", "creation id", TICKER);
    separator(74);
    for s in &slots {
        let slot = s["slot_index"].as_u64().unwrap_or(0);
        let creation_id = s["creation_id"]
            .as_u64()
            .context("getSlotsByOwner response missing creation_id")?;
        let value = s["value"].as_u64().unwrap_or(0);
        println!(
            "  {:<12}  {:>20}  {:>14}",
            slot,
            creation_id,
            jetsam_str(value)
        );
    }
    separator(74);
    println!(
        "  {:<12}  {:>20}  {:>14}  ({} UTXOs)",
        "TOTAL",
        "",
        jetsam_str(total),
        slots.len()
    );
    Ok(())
}

async fn cmd_tx(ctx: &Ctx<'_>, txhash: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "getTx", &[txhash.into()]).await.context("getTx")?;
    if ctx.json {
        return print_json(&result);
    }
    if result.is_null() {
        warn_msg(&format!("Transaction {txhash} not found (not confirmed)"));
        println!("  Use 'jetsam-cli mempool-tx {txhash}' to check if it is pending.");
        return Ok(());
    }
    section("Transaction");
    kv("tx_hash", result["tx_hash"].as_str().unwrap_or("?"));
    kv(
        "height",
        &result["height"].as_u64().unwrap_or(0).to_string(),
    );
    kv("block_hash", result["block_hash"].as_str().unwrap_or("?"));
    kv(
        "position",
        &result["tx_position"].as_u64().unwrap_or(0).to_string(),
    );
    Ok(())
}

async fn cmd_state(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getStateInfo", &[])
        .await
        .context("getStateInfo")?;
    if ctx.json {
        return print_json(&result);
    }

    let log_slots = result["log_slots"].as_u64().unwrap_or(0);
    let capacity = result["capacity"].as_u64().unwrap_or(0);
    let active = result["active_slots"].as_u64().unwrap_or(0);
    let fill_pct = result["fill_pct"].as_f64().unwrap_or(0.0);
    let headroom = result["slots_until_expand"].as_i64().unwrap_or(0);
    let trigger_pct = result["expand_trigger_pct"].as_u64().unwrap_or(75);
    let log_max = result["log_slots_max"].as_u64().unwrap_or(32);
    let size_human = result["state_size_human"].as_str().unwrap_or("?");

    section("UTXO state");
    kv2(
        "Slot space",
        &format!("2^{log_slots} = {capacity} slots"),
        &format!("(max 2^{log_max})"),
    );
    kv2(
        "Active UTXOs",
        &format!("{active}"),
        &format!("({fill_pct:.2}% full)"),
    );

    // Visual fill bar  [████████░░░░░░░░░░░░]  12.50%
    let bar_width = 30usize;
    let filled = ((fill_pct / 100.0) * bar_width as f64).round() as usize;
    let trigger_pos = ((trigger_pct as f64 / 100.0) * bar_width as f64).round() as usize;
    let bar: String = (0..bar_width)
        .map(|i| {
            if i < filled {
                '█'
            } else if i == trigger_pos.min(bar_width - 1) {
                '|'
            }
            // expansion marker
            else {
                '░'
            }
        })
        .collect();
    if is_tty() {
        println!(
            "  {CYN}{:<14}{RST} [{bar}] {fill_pct:.2}%  {DIM}(| = sample at {trigger_pct}%){RST}",
            "Fill",
            CYN = "\x1b[36m",
            RST = "\x1b[0m",
            DIM = "\x1b[2m"
        );
    } else {
        println!(
            "  {:<14} [{bar}] {fill_pct:.2}%  (| = sample at {trigger_pct}%)",
            "Fill"
        );
    }

    if headroom >= 0 {
        kv2(
            "Until expand",
            &format!("{headroom} slots"),
            &format!(
                "({:.2}% headroom)",
                headroom as f64 / capacity as f64 * 100.0
            ),
        );
    } else {
        kv(
            "Until expand",
            &c!(
                YLW,
                "ABOVE SAMPLE THRESHOLD (awaiting finalized-window majority)"
            ),
        );
    }
    kv("State storage", size_human);
    Ok(())
}

async fn cmd_mining(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getMiningInfo", &[])
        .await
        .context("getMiningInfo")?;
    if ctx.json {
        return print_json(&result);
    }
    let height = result["height"].as_u64().unwrap_or(0);
    let diff_bits = result["difficulty_bits"].as_u64().unwrap_or(0);
    let diff_target = result["difficulty_target"].as_str().unwrap_or("?");
    let reward_micro = result["block_reward_micro_jtm"].as_u64().unwrap_or(0);
    let active = result["active_slot_count"].as_u64().unwrap_or(0);
    let hashrate = result["pow_hashrate_hps"].as_f64();
    let hashes_total = result["pow_hashes_total"].as_u64().unwrap_or(0);
    let payout = result["payout_address"].as_str().unwrap_or("").to_string();
    section("Mining info");
    kv("Height", &height.to_string());
    kv2(
        "Difficulty",
        &format!("{diff_bits} leading zeros"),
        &format!("target: {diff_target}"),
    );
    kv2(
        "Block reward",
        &format!("{} {TICKER}/block", jetsam_str(reward_micro)),
        &format!("({reward_micro} \u{03bc}{TICKER})"),
    );
    kv("Active UTXOs", &active.to_string());
    if !payout.is_empty() {
        kv("Blocks paid to", &payout);
    }
    match hashrate {
        Some(hps) => kv2(
            "Your hashrate",
            &jetsam_miner::format_hashrate(hps),
            &format!("({hashes_total} hashes since start)"),
        ),
        None if hashes_total > 0 => kv2(
            "Your hashrate",
            "—",
            "(node is proving, not searching right now)",
        ),
        None => kv2(
            "Your hashrate",
            "—",
            "(this node is not mining — start it with --mode miner)",
        ),
    }
    Ok(())
}

async fn cmd_peers(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getPeerCount", &[])
        .await
        .context("getPeerCount")?;
    if ctx.json {
        return print_json(&result);
    }
    let n = result.as_u64().unwrap_or(0);
    section("Connected peers");
    kv("Count", &n.to_string());
    Ok(())
}

/// One peer in the merged adoption view.
#[derive(Debug, Clone)]
struct AdoptionPeer {
    /// `None` when the answering endpoint withheld identities: such a peer
    /// is counted, but cannot be recognised in another node's answer.
    peer_id: Option<String>,
    /// The freshest announcement among the answers that saw it.
    agent: String,
    /// Connected to at least one asked node.
    connected: bool,
    /// Its network profile was verified by at least one asked node.
    verified: bool,
    /// Seconds since the freshest sighting, on the clock of the node that
    /// made it.
    age_secs: u64,
}

/// Merge `getPeerVersions` answers by peer identity. A malformed answer is
/// an error, never a count.
fn merge_adoption_answers(answers: &[Value]) -> anyhow::Result<Vec<AdoptionPeer>> {
    let mut merged: Vec<AdoptionPeer> = Vec::new();
    let mut by_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (index, answer) in answers.iter().enumerate() {
        let answer = answer.get("result").unwrap_or(answer);
        let now = answer["now_unix"]
            .as_u64()
            .with_context(|| format!("answer {}: no now_unix", index + 1))?;
        let peers = answer["peers"]
            .as_array()
            .with_context(|| format!("answer {}: peers is not a list", index + 1))?;
        for peer in peers {
            let field = |name: &str| {
                anyhow::anyhow!("answer {}: a peer has no valid {name}", index + 1)
            };
            let peer_id = match &peer["peer_id"] {
                Value::Null => None,
                Value::String(id) => Some(id.clone()),
                _ => return Err(field("peer_id")),
            };
            let agent = jetsam_p2p::peer_versions::sanitize_agent_version(
                peer["agent_version"]
                    .as_str()
                    .ok_or_else(|| field("agent_version"))?,
            );
            let connected = peer["connected"]
                .as_bool()
                .ok_or_else(|| field("connected"))?;
            let verified = peer["profile_verified"]
                .as_bool()
                .ok_or_else(|| field("profile_verified"))?;
            let last_seen = peer["last_seen_unix"]
                .as_u64()
                .ok_or_else(|| field("last_seen_unix"))?;
            let sighting = AdoptionPeer {
                peer_id: peer_id.clone(),
                agent,
                connected,
                verified,
                age_secs: now.saturating_sub(last_seen),
            };
            let Some(id) = peer_id else {
                merged.push(sighting);
                continue;
            };
            match by_id.get(&id) {
                Some(&slot) => {
                    let known = &mut merged[slot];
                    known.connected |= sighting.connected;
                    known.verified |= sighting.verified;
                    if sighting.age_secs < known.age_secs {
                        known.age_secs = sighting.age_secs;
                        known.agent = sighting.agent;
                    }
                }
                None => {
                    by_id.insert(id, merged.len());
                    merged.push(sighting);
                }
            }
        }
    }
    Ok(merged)
}

/// Whether an announcement is `target`: a release (`1.6.0`, matching
/// `jetsam/1.6.0 …` exactly, not `1.6.0-dev`) or an exact announcement,
/// whole or up to a space.
fn agent_matches(agent: &str, target: &str) -> bool {
    if target.contains('/') {
        return agent == target
            || agent
                .strip_prefix(target)
                .is_some_and(|rest| rest.starts_with(' '));
    }
    agent
        .strip_prefix("jetsam/")
        .and_then(|rest| rest.split(' ').next())
        .is_some_and(|release| release == target)
}

#[derive(Debug, Default)]
struct AdoptionRow {
    agent: String,
    connected: usize,
    recent: usize,
}

#[derive(Debug, Default)]
struct AdoptionReport {
    /// Verified peers connected now, and those of them on the target.
    connected: usize,
    target_connected: usize,
    /// Verified peers connected now or seen within the window.
    recent: usize,
    target_recent: usize,
    /// Peers whose profile no asked node verified: not counted above.
    unverified: usize,
    /// Peers without an identity: one seen by two nodes counts twice.
    unmerged: usize,
    rows: Vec<AdoptionRow>,
}

fn adoption_report(
    peers: &[AdoptionPeer],
    target: Option<&str>,
    window_secs: u64,
) -> AdoptionReport {
    let mut report = AdoptionReport::default();
    let mut rows: std::collections::BTreeMap<&str, AdoptionRow> = Default::default();
    for peer in peers {
        if peer.peer_id.is_none() {
            report.unmerged += 1;
        }
        if !peer.verified {
            report.unverified += 1;
            continue;
        }
        let recent = peer.connected || peer.age_secs <= window_secs;
        if !recent {
            continue;
        }
        let on_target = target.is_some_and(|target| agent_matches(&peer.agent, target));
        let row = rows.entry(&peer.agent).or_insert_with(|| AdoptionRow {
            agent: peer.agent.clone(),
            ..AdoptionRow::default()
        });
        row.recent += 1;
        report.recent += 1;
        report.target_recent += usize::from(on_target);
        if peer.connected {
            row.connected += 1;
            report.connected += 1;
            report.target_connected += usize::from(on_target);
        }
    }
    report.rows = rows.into_values().collect();
    report.rows.sort_by(|a, b| {
        (b.recent, b.connected, &a.agent).cmp(&(a.recent, a.connected, &b.agent))
    });
    report
}

fn share(part: usize, whole: usize) -> String {
    if whole == 0 {
        format!("{part} / 0")
    } else {
        format!("{part} / {whole}  ({:.1} %)", 100.0 * part as f64 / whole as f64)
    }
}

async fn cmd_adoption(
    ctx: &Ctx<'_>,
    target: Option<&str>,
    window: u64,
    nodes: &[String],
    answer_files: &[std::path::PathBuf],
    offline: bool,
) -> anyhow::Result<()> {
    let mut endpoints: Vec<&str> = Vec::new();
    if !offline {
        endpoints.push(ctx.rpc);
    }
    endpoints.extend(nodes.iter().map(String::as_str));
    let mut answers = Vec::new();
    for endpoint in &endpoints {
        let node = Ctx {
            client: ctx.client,
            rpc: endpoint,
            json: ctx.json,
        };
        let answer = rpc(&node, "getPeerVersions", &[])
            .await
            .with_context(|| format!("getPeerVersions on {endpoint}"))?;
        answers.push(answer);
    }
    for path in answer_files {
        let bytes =
            std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let answer: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse {}", path.display()))?;
        answers.push(answer);
    }
    anyhow::ensure!(
        !answers.is_empty(),
        "nothing to count: --offline needs --node or --answer"
    );
    let peers = merge_adoption_answers(&answers)?;
    // By default, the share of peers running what the first node announces.
    let target = match target {
        Some(target) => Some(target.to_string()),
        None => answers
            .first()
            .map(|answer| answer.get("result").unwrap_or(answer))
            .and_then(|answer| answer["local_agent_version"].as_str())
            .map(jetsam_p2p::peer_versions::sanitize_agent_version),
    };
    let report = adoption_report(&peers, target.as_deref(), window);

    if ctx.json {
        let rows: Vec<Value> = report
            .rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "agent_version": row.agent,
                    "connected": row.connected,
                    "recent": row.recent,
                })
            })
            .collect();
        return print_json(&serde_json::json!({
            "answers": answers.len(),
            "target": target,
            "window_secs": window,
            "connected": report.connected,
            "target_connected": report.target_connected,
            "recent": report.recent,
            "target_recent": report.target_recent,
            "unverified": report.unverified,
            "unmerged": report.unmerged,
            "versions": rows,
        }));
    }

    section("Adoption (announced versions)");
    kv("Answers merged", &answers.len().to_string());
    kv("Target", target.as_deref().unwrap_or("none"));
    kv("Connected now", &share(report.target_connected, report.connected));
    kv(
        &format!("Seen in {window} s"),
        &share(report.target_recent, report.recent),
    );
    kv(
        "Unverified",
        &format!("{} (profile never verified: not counted)", report.unverified),
    );
    println!();
    println!("  {:<64} {:>9} {:>9}", "Version", "connected", "recent");
    separator(84);
    for row in &report.rows {
        println!("  {:<64} {:>9} {:>9}", row.agent, row.connected, row.recent);
    }
    println!();
    if report.unmerged > 0 && answers.len() > 1 {
        warn_msg(&format!(
            "{} peers came without an identity (public endpoint): one seen by two nodes is counted twice. Ask each node on loopback, or save its answer there and pass it with --answer.",
            report.unmerged
        ));
    }
    println!(
        "  A version is what a peer claims, nothing more; a release older than the announcement shows rust-libp2p/<version>. Hashrate is not measured here."
    );
    Ok(())
}

async fn cmd_estimate_fee(ctx: &Ctx<'_>, n_inputs: u32, n_outputs: u32) -> anyhow::Result<()> {
    let result = rpc(
        ctx,
        "estimateFeeDetailed",
        &[n_inputs.into(), n_outputs.into()],
    )
    .await
    .context("estimateFeeDetailed")?;
    if ctx.json {
        return print_json(&result);
    }
    let fee_micro = result["fee_micro_jtm"].as_u64().unwrap_or(0);
    let b = &result["breakdown"];
    section(&format!(
        "Fee estimate ({n_inputs} input(s), {n_outputs} output(s))"
    ));
    kv2(
        "Min relay fee",
        &format!("{} {TICKER}", jetsam_str(fee_micro)),
        &format!("({fee_micro} μ{TICKER})"),
    );
    kv(
        "Base",
        &format!("{} μ{TICKER}", b["base"].as_u64().unwrap_or(0)),
    );
    kv(
        "Inputs",
        &format!("{} μ{TICKER}", b["input"].as_u64().unwrap_or(0)),
    );
    kv(
        "Outputs",
        &format!("{} μ{TICKER}", b["output"].as_u64().unwrap_or(0)),
    );
    kv(
        "State growth burned",
        &format!("{} μ{TICKER}", b["state_growth"].as_u64().unwrap_or(0)),
    );
    kv(
        "Miner claimable",
        &format!("{} μ{TICKER}", b["miner_claimable"].as_u64().unwrap_or(0)),
    );
    println!();
    println!(
        "  {} output-centric: base + small input anti-DoS + output fee + burned net-new-state fee",
        c!(DIM, "Formula:")
    );
    Ok(())
}

async fn cmd_validate(ctx: &Ctx<'_>, address: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "validateAddress", &[address.into()])
        .await
        .context("validateAddress")?;
    if ctx.json {
        return print_json(&result);
    }
    let valid = result["valid"].as_bool().unwrap_or(false);
    section("Address validation");
    if valid {
        ok_msg("Valid address");
        kv("bech32m", result["bech32"].as_str().unwrap_or("?"));
        kv("hex", result["hex"].as_str().unwrap_or("?"));
    } else {
        let err = result["error"].as_str().unwrap_or("invalid");
        print_error(&format!("Invalid address: {err}"));
        bail!("invalid address");
    }
    Ok(())
}

async fn cmd_mempool_tx(ctx: &Ctx<'_>, txhash: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "getMempoolEntry", &[txhash.into()])
        .await
        .context("getMempoolEntry")?;
    if ctx.json {
        return print_json(&result);
    }
    if result.is_null() {
        warn_msg(&format!("Transaction {txhash} is not in the mempool."));
        println!("  It may have been confirmed. Use 'jetsam-cli tx {txhash}' to check.");
        return Ok(());
    }
    section("Mempool transaction");
    kv("tx_hash", result["tx_hash"].as_str().unwrap_or("?"));
    let fee = result["fee_micro_jtm"].as_u64().unwrap_or(0);
    kv2(
        "Fee",
        &format!("{} {TICKER}", jetsam_str(fee)),
        &format!("({fee} \u{03bc}{TICKER})"),
    );
    kv(
        "Inputs",
        &result["n_inputs"].as_u64().unwrap_or(0).to_string(),
    );
    kv(
        "Outputs",
        &result["n_outputs"].as_u64().unwrap_or(0).to_string(),
    );
    kv(
        "Physical pages",
        &result["page_count"].as_u64().unwrap_or(0).to_string(),
    );
    kv(
        "Minimum proof class",
        result["minimum_proof_class"].as_str().unwrap_or("?"),
    );
    kv(
        "Admitted at height",
        &result["admitted_height"].as_u64().unwrap_or(0).to_string(),
    );
    let has_authorization = result["has_authorization"].as_bool().unwrap_or(false);
    kv(
        "Authorization",
        if has_authorization {
            "attached"
        } else {
            "not attached"
        },
    );
    if result["requires_b255_miner"].as_bool().unwrap_or(false) {
        println!();
        warn_msg("Waiting for inclusion: this atomic PagedSpend requires a B255-qualified miner.");
    }
    Ok(())
}

async fn cmd_epoch(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getEpochAnchor", &[])
        .await
        .context("getEpochAnchor")?;

    if ctx.json {
        return print_json(&result);
    }

    let hash = result.as_str().unwrap_or("?");
    section("Epoch anchor");
    kv("Hash", ctx.h(hash));
    println!();
    println!(
        "  {} Exact transaction-epoch anchor accepted for the next block.",
        c!(DIM, "Note:")
    );

    Ok(())
}

async fn cmd_mempool(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "getMempoolInfo", &[])
        .await
        .context("getMempoolInfo")?;

    if ctx.json {
        return print_json(&result);
    }

    let size = result["size"].as_u64().unwrap_or(0);
    let fee_floor = result["fee_floor"].as_u64().unwrap_or(0);
    let txs = result["txs"].as_array().cloned().unwrap_or_default();

    section("Mempool");
    kv2("Pending", &size.to_string(), "transactions");
    kv2(
        "Fee floor",
        &format!("{} {TICKER}", jetsam_str(fee_floor)),
        &format!("({fee_floor} μ{TICKER} minimum)"),
    );

    if txs.is_empty() {
        println!();
        println!("  {}", c!(DIM, "(mempool is empty)"));
        return Ok(());
    }

    println!();
    separator(100);
    if is_tty() {
        println!(
            "  {}{:<20}  {:>12}  {:>5}  {:>3}→{:<3}  {:>5}  {}{}",
            BOLD, "tx hash", format!("fee (\u{03bc}{TICKER})"), "pages", "in", "out", "class", "proof", RST
        );
    } else {
        println!(
            "  {:<20}  {:>12}  {:>5}  {:>3}→{:<3}  {:>5}  {}",
            "tx hash", format!("fee (\u{03bc}{TICKER})"), "pages", "in", "out", "class", "proof"
        );
    }
    separator(100);

    let show = txs.len().min(20);
    for tx in txs.iter().take(show) {
        let hash = tx["tx_hash"].as_str().unwrap_or("?");
        let fee = tx["fee_micro_jtm"].as_u64().unwrap_or(0);
        let pages = tx["page_count"].as_u64().unwrap_or(0);
        let nin = tx["n_inputs"].as_u64().unwrap_or(0);
        let nout = tx["n_outputs"].as_u64().unwrap_or(0);
        let class = tx["minimum_proof_class"].as_str().unwrap_or("?");
        let authorization = if tx["has_authorization"].as_bool().unwrap_or(false) {
            c!(GRN, "✓")
        } else {
            c!(DIM, "·")
        };
        println!(
            "  {:<20}  {:>12}  {:>5}  {:>3}→{:<3}  {:>5}  {}",
            ctx.h(hash),
            fee,
            pages,
            nin,
            nout,
            class,
            authorization
        );
    }

    if txs.len() > show {
        println!("  {} {} more…", c!(DIM, "...and"), txs.len() - show);
    }
    if txs
        .iter()
        .any(|tx| tx["requires_b255_miner"].as_bool().unwrap_or(false))
    {
        println!();
        warn_msg("B255 entries are atomic and wait for a B255-qualified miner.");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Wallet commands
// ---------------------------------------------------------------------------

async fn cmd_address(
    ctx: &Ctx<'_>,
    new: bool,
    list: bool,
    index: Option<u32>,
    use_index: Option<u32>,
) -> anyhow::Result<()> {
    if let Some(idx) = use_index {
        let result = rpc(ctx, "walletSetActiveAddress", &[serde_json::json!(idx)])
            .await
            .context("walletSetActiveAddress")?;
        if ctx.json {
            return print_json(&result);
        }
        let addr = result["address"].as_str().unwrap_or("?");
        let bal = rpc(ctx, "walletGetBalance", &[])
            .await
            .ok()
            .and_then(|balance| balance["balance_eld"].as_f64())
            .unwrap_or(0.0);
        section(&format!("Active address switched [index={idx}]"));
        if is_tty() {
            println!("  {}{}{}", BOLD, addr, RST);
        } else {
            println!("  {}", addr);
        }
        println!("  Balance: {bal:.6} {TICKER}");
        println!();
        println!(
            "  {} Sends now spend from this address only; change returns here.",
            c!(DIM, "↑")
        );
        return Ok(());
    }
    if list {
        let result = rpc(ctx, "walletListAddresses", &[])
            .await
            .context("walletListAddresses")?;
        if ctx.json {
            return print_json(&result);
        }
        let addrs = result.as_array().cloned().unwrap_or_default();
        section("Wallet addresses");
        println!("  {:>7}  {}", "index", "address");
        separator(72);
        for a in &addrs {
            let idx = a["key_index"].as_u64().unwrap_or(0);
            let addr = a["address"].as_str().unwrap_or("");
            let active = a["is_active"].as_bool().unwrap_or(false);
            let marker = if active { "▶" } else { " " };
            println!("  {} {:>5}  {}", marker, idx, ctx.h(addr));
        }
        separator(72);
        println!("  {} locally generated address(es)", addrs.len());
        println!();
        println!(
            "  {}",
            c!(
                DIM,
                "▶ = active address (sends spend from it). Switch: 'address --use <index>'."
            )
        );
        println!(
            "  {}",
            c!(
                DIM,
                "To move funds between your addresses, activate the source and send normally."
            )
        );
    } else if new {
        let result = rpc(ctx, "walletNextAddress", &[])
            .await
            .context("walletNextAddress")?;
        if ctx.json {
            return print_json(&result);
        }
        let addr = result["address"].as_str().unwrap_or("");
        let idx = result["key_index"].as_u64().unwrap_or(0);
        section(&format!("New address [index={idx}]"));
        if is_tty() {
            println!("  {}{}{}", BOLD, addr, RST);
        } else {
            println!("  {}", addr);
        }
        println!();
        println!(
            "  {} The active address is unchanged. Activate this address with 'address --use {idx}'.",
            c!(DIM, "↑")
        );
    } else if let Some(idx) = index {
        let result = rpc(ctx, "walletGetAddress", &[serde_json::json!(idx)])
            .await
            .context("walletGetAddress")?;
        if ctx.json {
            return print_json(&result);
        }
        let addr = result.as_str().unwrap_or("?");
        section(&format!("Wallet address [index={idx}]"));
        if is_tty() {
            println!("  {}{}{}", BOLD, addr, RST);
        } else {
            println!("  {}", addr);
        }
    } else {
        // default: the ACTIVE address.
        let result = rpc(ctx, "walletActiveAddress", &[])
            .await
            .context("walletActiveAddress")?;
        if ctx.json {
            return print_json(&result);
        }
        let addr = result["address"].as_str().unwrap_or("?");
        let idx = result["key_index"].as_u64().unwrap_or(0);
        section(&format!("Active address [index={idx}]"));
        if is_tty() {
            println!("  {}{}{}", BOLD, addr, RST);
        } else {
            println!("  {}", addr);
        }
        println!();
        println!(
            "  {} Sends spend from this address; share it to receive {TICKER} here.",
            c!(DIM, "↑")
        );
        println!(
            "  {}Tip: 'address --new' for a fresh address, '--list' for all, '--use <i>' to switch.{}",
            DIM, RST
        );
    }
    Ok(())
}

/// The address this node actually pays its blocks to, when it is not the
/// wallet's own active address.
///
/// A node started with `--miner-address` credits an address whose key lives
/// elsewhere, so the local wallet stays empty forever. Reporting an empty
/// wallet without saying that sends the operator hunting for a wrong data
/// directory, which is the one thing that is not wrong.
async fn foreign_payout_address(ctx: &Ctx<'_>) -> Option<String> {
    let mining = rpc(ctx, "getMiningInfo", &[]).await.ok()?;
    let payout = mining["payout_address"].as_str()?.to_string();
    let wallet = rpc(ctx, "walletActiveAddress", &[]).await.ok()?;
    let active = wallet["address"].as_str()?;
    (payout != active).then_some(payout)
}

async fn cmd_balance(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "walletGetBalance", &[])
        .await
        .context("walletGetBalance")?;

    if ctx.json {
        return print_json(&result);
    }

    let micro = result["balance_micro_jtm"].as_u64().unwrap_or(0);
    let utxos = result["utxo_count"].as_u64().unwrap_or(0);
    let pending_out = result["pending_outbound_micro_jtm"].as_u64().unwrap_or(0);
    let spendable = result["spendable_micro_jtm"]
        .as_u64()
        .unwrap_or_else(|| micro.saturating_sub(pending_out));

    section("Wallet balance (active address)");
    if is_tty() {
        println!(
            "  {}Balance:{} {}{} {TICKER}{} {}({} \u{03bc}{TICKER}){}  {}({} UTXOs){}",
            CYN,
            RST,
            BOLD,
            jetsam_str(micro),
            RST,
            DIM,
            micro,
            RST,
            DIM,
            utxos,
            RST
        );
    } else {
        println!(
            "  Balance:           {} {TICKER}  ({micro} \u{03bc}{TICKER})  ({utxos} UTXOs)",
            jetsam_str(micro)
        );
    }

    if pending_out > 0 {
        if is_tty() {
            println!(
                "  {}Pending:{}  -{} {TICKER} outbound  {}({} \u{03bc}{TICKER} locked){}",
                YLW,
                RST,
                jetsam_str(pending_out),
                DIM,
                pending_out,
                RST
            );
            println!(
                "  {}Spendable:{} {}{} {TICKER}{}",
                CYN,
                RST,
                BOLD,
                jetsam_str(spendable),
                RST
            );
        } else {
            println!(
                "  Pending:           -{} {TICKER} outbound ({pending_out} \u{03bc}{TICKER} locked)",
                jetsam_str(pending_out)
            );
            println!("  Spendable:         {} {TICKER}", jetsam_str(spendable));
        }
    }

    if micro == 0 && utxos == 0 {
        println!();
        warn_msg("No UTXOs found on the active address.");
        if let Some(payout) = foreign_payout_address(ctx).await {
            println!("       This node mines to {payout},");
            println!("       which this wallet does not own — that is what --miner-address does.");
            println!(
                "       Those coins are spendable only from the wallet holding that key. Check them with:"
            );
            println!("         {}", c!(BOLD, format!("jetsam-cli utxos-of {payout}")));
        } else {
            println!(
                "       Run {} to reload the active address; use {} to switch accounts.",
                c!(BOLD, "'jetsam-cli scan'"),
                c!(BOLD, "'address --list'")
            );
        }
    }

    Ok(())
}

async fn cmd_utxos(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let result = rpc(ctx, "walletListUtxos", &[])
        .await
        .context("walletListUtxos")?;

    if ctx.json {
        return print_json(&result);
    }

    let utxos = result.as_array().cloned().unwrap_or_default();

    section("Wallet UTXOs");

    if utxos.is_empty() {
        println!(
            "  {}",
            c!(DIM, "(no UTXOs — run 'jetsam-cli scan' to discover)")
        );
        return Ok(());
    }

    // Compute total
    let total: u64 = utxos
        .iter()
        .map(|u| u["value_micro_jtm"].as_u64().unwrap_or(0))
        .sum();

    separator(96);
    if is_tty() {
        println!(
            "  {}{:<8}  {:>20}  {:>14}  {:>7}  {:>9}  {}{}",
            BOLD, "slot", "creation id", TICKER, "key", "at block", "address", RST
        );
    } else {
        println!(
            "  {:<8}  {:>20}  {:>14}  {:>7}  {:>9}  {}",
            "slot", "creation id", TICKER, "key", "at block", "address"
        );
    }
    separator(96);

    for u in &utxos {
        let slot = u["slot_index"].as_u64().unwrap_or(0);
        let creation_id = u["creation_id"]
            .as_u64()
            .context("walletListUtxos response missing creation_id")?;
        let micro = u["value_micro_jtm"].as_u64().unwrap_or(0);
        let key = u["key_index"].as_u64().unwrap_or(0);
        let height = u["confirmed_height"].as_u64().unwrap_or(0);
        let addr = u["address"].as_str().unwrap_or("?");
        println!(
            "  {:<8}  {:>20}  {:>14}  {:>7}  {:>9}  {}",
            slot,
            creation_id,
            jetsam_str(micro),
            key,
            height,
            ctx.h(addr)
        );
    }

    separator(96);
    if is_tty() {
        println!(
            "  {}{:<8}  {:>20}  {:>14}{}",
            BOLD,
            "TOTAL",
            "",
            jetsam_str(total),
            RST
        );
    } else {
        println!("  {:<8}  {:>20}  {:>14}", "TOTAL", "", jetsam_str(total));
    }

    Ok(())
}

async fn cmd_send(
    ctx: &Ctx<'_>,
    to: &str,
    amount: &str,
    fee: Option<&str>,
    dry_run: bool,
) -> anyhow::Result<()> {
    // --- Parse and validate inputs ---
    let amount_micro =
        parse_jetsam_amount(amount).with_context(|| format!("invalid amount {amount:?}"))?;

    if amount_micro == 0 {
        bail!("Amount cannot be zero.");
    }

    // Warn if the amount looks suspiciously large (> 1 000 000 JTM = 1e12 μJTM).
    // This catches the common mistake of passing μJTM to a JTM-denomination CLI.
    const MAX_WARN_MICRO: u64 = 1_000_000 * 1_000_000; // 1M coins in micro-units
    if amount_micro > MAX_WARN_MICRO {
        eprintln!(
            "⚠  Large amount: {} {TICKER} ({} μ{TICKER}). \
             Note: this CLI takes {TICKER}, not μ{TICKER}. Press Ctrl-C to cancel.",
            jetsam_str(amount_micro),
            amount_micro
        );
    }

    let fee_micro = match fee {
        Some(f) => parse_jetsam_amount(f).with_context(|| format!("invalid fee {f:?}"))?,
        None => 0, // auto
    };

    // `--fee` is in JTM, and an ordinary fee is a few thousandths of one. A
    // figure typed as if it were micro-JTM, or a stray zero, hands the whole
    // amount to a miner with no way back. Refuse anything absurd rather than
    // let a typo cost a hundred times the transfer.
    const FEE_SANITY_LIMIT_MICRO: u64 = 1_000_000; // 1 coin
    if fee_micro > FEE_SANITY_LIMIT_MICRO || (amount_micro > 0 && fee_micro > amount_micro) {
        bail!(
            "Refusing a fee of {} {TICKER}.\n\
             \tA fee is normally a few thousandths of one {TICKER}, and this one is \n\
             \t{}. Fees are paid to a miner and cannot be recovered.\n\
             \tOmit --fee to let the node price it, or pass a smaller figure.",
            jetsam_str(fee_micro),
            if fee_micro > FEE_SANITY_LIMIT_MICRO {
                format!("above the one-{TICKER} sanity limit")
            } else {
                format!("larger than the {} {TICKER} you are sending", jetsam_str(amount_micro))
            }
        );
    }

    // Validate address encoding before sending to the daemon.
    // Actual parsing/validation happens in the daemon; we just do a basic
    // sanity check to catch obvious typos before sending to RPC.
    let to_clean = to.trim();
    let looks_like_bech32 = looks_like_address(to_clean);
    if !looks_like_bech32 {
        bail!(
            "Invalid address format.\n\
             \tExpected: bech32m address (j1…)\n\
             \tGot:      {:?}\n\
             \tExample:  j1q9gnyj0z…",
            &to_clean[..to_clean.len().min(30)]
        );
    }

    if dry_run {
        let result = match rpc(
            ctx,
            "walletPlanSend",
            &[to_clean.into(), amount_micro.into(), fee_micro.into()],
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                if let Some(message) = input_limit_exceeded_message(&error) {
                    bail!(message);
                }
                return Err(error).context("walletPlanSend");
            }
        };
        if ctx.json {
            return print_json(&result);
        }
        section("Wallet send plan");
        kv("To", to_clean);
        kv2(
            "Amount",
            &format!("{} {TICKER}", jetsam_str(amount_micro)),
            &format!("({amount_micro} μ{TICKER})"),
        );
        let planned_fee = result["fee_micro_jtm"].as_u64().unwrap_or(0);
        kv2(
            "Fee",
            &format!("{} {TICKER}", jetsam_str(planned_fee)),
            &format!(
                "({planned_fee} μ{TICKER}){}",
                if fee.is_none() { " auto" } else { "" }
            ),
        );
        kv(
            "Inputs",
            &result["input_count"].as_u64().unwrap_or(0).to_string(),
        );
        kv(
            "Outputs",
            &result["output_count"].as_u64().unwrap_or(0).to_string(),
        );
        let change = result["change_micro_jtm"].as_u64().unwrap_or(0);
        kv2(
            "Change",
            &format!("{} {TICKER}", jetsam_str(change)),
            &format!("({change} μ{TICKER})"),
        );
        println!();
        println!(
            "  {} Dry run only; no proof was generated and nothing was submitted.",
            c!(DIM, "Note:")
        );
        return Ok(());
    }

    // --- Confirm interactively for large amounts ---
    if amount_micro >= 1_000_000_000 /* 1000 coins */ && is_tty() {
        print!(
            "  {} Send {}{} {TICKER}{} to {}{}{}? [y/N] ",
            c!(YLW, "⚠"),
            BOLD,
            jetsam_str(amount_micro),
            RST,
            CYN,
            to_clean,
            RST,
        );
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("  {}", c!(DIM, "Cancelled."));
            return Ok(());
        }
    }

    // --- Send ---
    let result = rpc(
        ctx,
        "walletSend",
        &[to_clean.into(), amount_micro.into(), fee_micro.into()],
    )
    .await;

    match result {
        Ok(r) if ctx.json => return print_json(&r),
        Ok(r) => {
            let actual_fee = r["fee_micro_jtm"].as_u64().unwrap_or(0);
            let auto_tag = if fee.is_none() { " (auto)" } else { "" };

            section("Transaction submitted");
            ok_msg(&format!("TX {}", ctx.h(r["txid"].as_str().unwrap_or("?"))));
            println!();
            kv("To", to_clean);
            kv2(
                "Amount",
                &format!("{} {TICKER}", jetsam_str(amount_micro)),
                &format!("({amount_micro} μ{TICKER})"),
            );
            kv2(
                "Fee",
                &format!("{} {TICKER}", jetsam_str(actual_fee)),
                &format!("({actual_fee} μ{TICKER}){auto_tag}"),
            );
            println!();
            println!(
                "  {} Submitted transaction is pending confirmation.",
                c!(DIM, "⏳")
            );
            println!(
                "  {} Use {} to check your balance after confirmation.",
                c!(DIM, "Tip:"),
                c!(BOLD, "'jetsam-cli balance'")
            );
        }
        Err(e) => {
            // Re-format common wallet errors into human language
            let msg = e.to_string();
            if let Some(message) = input_limit_exceeded_message(&e) {
                bail!(message);
            }
            let human = if msg.contains("Insufficient") || msg.contains("insufficient") {
                // Try to extract amounts
                format!(
                    "Insufficient funds.\n\
                     \t  Requested: {} {TICKER}  ({amount_micro} μ{TICKER})\n\
                     \t  Run 'jetsam-cli balance' to check your current balance.",
                    jetsam_str(amount_micro)
                )
            } else if msg.contains("no UTXO") || msg.contains("no utxo") {
                "No UTXOs available. Run 'jetsam-cli scan' to discover your coins.".into()
            } else if msg.contains("no empty slot hints") {
                "No empty output slots are currently available. This is usually transient; wait for the next block and retry."
                    .into()
            } else if msg.contains("output slot") || msg.contains("SlotConflict") {
                "Slot conflict: the output slot is occupied. This is transient — retry in a moment."
                    .into()
            } else if msg.contains("BelowMinFee") {
                format!(
                    "Fee is below the current network minimum. Retry without --fee for automatic fee selection.\n\tDetails: {msg}"
                )
            } else if msg.contains("proof") || msg.contains("prove") || msg.contains("task:") {
                format!(
                    "Proof generation failed. Retry once; if it repeats, save the logs and report it.\n\tDetails: {msg}"
                )
            } else {
                msg
            };
            bail!("{human}");
        }
    }

    Ok(())
}

async fn cmd_history(
    ctx: &Ctx<'_>,
    address_filter: Option<&str>,
    last: Option<usize>,
) -> anyhow::Result<()> {
    let result = rpc(ctx, "walletHistory", &[])
        .await
        .context("walletHistory")?;

    if ctx.json {
        return print_json(&result);
    }

    let entries = result.as_array().cloned().unwrap_or_default();

    // Apply address filter
    let mut filtered: Vec<&Value> = entries
        .iter()
        .filter(|e| {
            if let Some(filter) = address_filter {
                e["own_address"]
                    .as_str()
                    .map(|a| a == filter)
                    .unwrap_or(false)
                    || e["peer_address"]
                        .as_str()
                        .map(|a| a == filter)
                        .unwrap_or(false)
            } else {
                true
            }
        })
        .collect();

    // Apply --last N limit
    if let Some(n) = last {
        let len = filtered.len();
        if len > n {
            filtered = filtered[len - n..].to_vec();
        }
    }

    section("Transaction history");

    if filtered.is_empty() {
        println!("  {}", c!(DIM, "(no transactions yet)"));
        return Ok(());
    }

    separator(100);
    if is_tty() {
        println!(
            "  {}  {:<8}  {:<20}  {:>14}  {:<16}  {}{}",
            BOLD, "block", "dir", TICKER, "own[idx]", "counterparty", RST
        );
    } else {
        println!(
            "  {:<8}  {:<20}  {:>14}  {:<16}  {}",
            "block", "dir", TICKER, "own[idx]", "counterparty"
        );
    }
    separator(100);

    let totals = tally_history(&filtered);

    for e in &filtered {
        let height = e["height"].as_u64().unwrap_or(0);
        let micro = e["amount_micro_jtm"].as_u64().unwrap_or(0);
        let (sign, arrow, colour) = history_row_style(e);

        let own = e["own_address"].as_str().unwrap_or("");
        let own_idx = e["own_key_index"].as_u64();
        let own_display = if let Some(idx) = own_idx {
            let truncated = &own[..own.len().min(10)];
            format!("[{}]{}", idx, truncated)
        } else {
            String::new()
        };
        let peer = e["peer_address"].as_str().unwrap_or("\u{2014}");
        let amount_str = format!("{:>14}{}", jetsam_str(micro), sign);

        if is_tty() {
            println!(
                "  {:>8}  {:<20}  {}{}{}  {:<16}  {}",
                height,
                arrow,
                colour,
                amount_str,
                RST,
                own_display,
                ctx.h(peer)
            );
        } else {
            println!(
                "  {:>8}  {:<20}  {}  {:<16}  {}",
                height, arrow, amount_str, own_display, peer
            );
        }
    }

    separator(100);
    if is_tty() {
        println!(
            "  {}  {:<8}  {:<8}  {}{}  {}{}{}",
            BOLD,
            "",
            "",
            GRN,
            format!("+ {} {TICKER} received", jetsam_str(totals.received)),
            RED,
            format!("  - {} {TICKER} sent", jetsam_str(totals.sent)),
            RST
        );
    } else {
        println!(
            "  total: +{} received  -{} sent",
            jetsam_str(totals.received),
            jetsam_str(totals.sent)
        );
    }
    if let Some(line) = mined_total_line(&totals) {
        println!("  {}", c!(GRN, line));
    }
    if let Some(line) = orphaned_total_line(&totals) {
        println!("  {}", c!(YLW, line));
    }

    Ok(())
}

/// Totals shown under `jetsam-cli history`. Rewards of canonical blocks are
/// `mined`, apart from payments `received`; orphans are in no other total.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct HistoryTotals {
    received: u64,
    sent: u64,
    mined_blocks: u64,
    mined_micro: u64,
    orphaned_blocks: u64,
    orphaned_micro: u64,
}

/// A locally mined block that lost a reorganization: its reward does not
/// exist. Only an explicit `canonical: false` from the node counts; older
/// nodes do not send the field.
fn is_orphaned_coinbase(entry: &Value) -> bool {
    entry["is_coinbase"].as_bool() == Some(true) && entry["canonical"].as_bool() == Some(false)
}

/// A coinbase whose exact block the node reports on the canonical chain.
fn is_mined_coinbase(entry: &Value) -> bool {
    entry["is_coinbase"].as_bool() == Some(true) && entry["canonical"].as_bool() == Some(true)
}

/// Sign, direction label and colour of one history row.
fn history_row_style(entry: &Value) -> (&'static str, &'static str, &'static str) {
    let dir = entry["direction"].as_str().unwrap_or("?");
    if is_orphaned_coinbase(entry) {
        (" ", "orphaned \u{b7} no reward", YLW)
    } else if is_mined_coinbase(entry) {
        ("+", "mined", GRN)
    } else if dir == "received" || dir == "recv" {
        ("+", "\u{2190} recv", GRN)
    } else {
        ("-", "\u{2192} sent", RED)
    }
}

fn tally_history(entries: &[&Value]) -> HistoryTotals {
    let mut totals = HistoryTotals::default();
    for entry in entries {
        let micro = entry["amount_micro_jtm"].as_u64().unwrap_or(0);
        if is_orphaned_coinbase(entry) {
            totals.orphaned_blocks += 1;
            totals.orphaned_micro += micro;
        } else if is_mined_coinbase(entry) {
            totals.mined_blocks += 1;
            totals.mined_micro += micro;
        } else if entry["direction"].as_str() == Some("sent") {
            totals.sent += micro;
        } else {
            totals.received += micro;
        }
    }
    totals
}

fn mined_total_line(totals: &HistoryTotals) -> Option<String> {
    (totals.mined_blocks > 0).then(|| {
        format!(
            "mined: {} block{}, {} {TICKER} earned (canonical blocks only)",
            totals.mined_blocks,
            if totals.mined_blocks == 1 { "" } else { "s" },
            jetsam_str(totals.mined_micro)
        )
    })
}

fn orphaned_total_line(totals: &HistoryTotals) -> Option<String> {
    (totals.orphaned_blocks > 0).then(|| {
        format!(
            "orphaned \u{b7} no reward: {} block{}, {} {TICKER} \
             (lost a reorganization, never in the balance)",
            totals.orphaned_blocks,
            if totals.orphaned_blocks == 1 { "" } else { "s" },
            jetsam_str(totals.orphaned_micro)
        )
    })
}

async fn cmd_scan(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    if is_tty() {
        eprint!("  Reloading active address...");
        io::stderr().flush()?;
    }

    let result = rpc(ctx, "walletScan", &[]).await.context("walletScan")?;

    if is_tty() {
        eprintln!(" done.");
    }

    if ctx.json {
        return print_json(&result);
    }

    let found = result["found_utxos"].as_u64().unwrap_or(0);
    let balance_micro_jtm = result["balance_micro_jtm"].as_u64().unwrap_or(0);
    let active_index = result["active_index"].as_u64().unwrap_or(0);
    let snapshot_height = result["snapshot_height"].as_u64().unwrap_or(0);

    section("Active address reloaded");
    println!(
        "  Index {}  \u{2022}  Snapshot height {}  \u{2022}  Found {} UTXO(s)  \u{2022}  Balance: {} {TICKER}",
        active_index,
        snapshot_height,
        found,
        jetsam_str(balance_micro_jtm)
    );

    if found == 0 {
        println!();
        if let Some(payout) = foreign_payout_address(ctx).await {
            warn_msg("No UTXOs found — and this wallet is not the one being paid.");
            println!("       This node mines to {payout}, an address this wallet does not own.");
            println!(
                "       Scanning again cannot change that: the coins are credited to that key."
            );
        } else {
            warn_msg(
                "No UTXOs found. If you expect a balance, check that you're using the right data directory.",
            );
        }
    }

    Ok(())
}

async fn cmd_receipt(ctx: &Ctx<'_>, txhash: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "walletExportReceipt", &[txhash.into()])
        .await
        .context("walletExportReceipt")?;

    let hex = result
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("daemon returned non-string receipt"))?;

    // Raw hex to stdout so it can be redirected
    println!("{hex}");

    if is_tty() {
        // If outputting to terminal, also show a tip
        eprintln!();
        eprintln!(
            "  {} Redirect to a file: jetsam-cli receipt {} > receipt.hex",
            c!(DIM, "Tip:"),
            txhash
        );
        eprintln!(
            "  {} Verify:              jetsam-cli verify $(cat receipt.hex)",
            c!(DIM, "Tip:")
        );
    }

    Ok(())
}

async fn cmd_verify(ctx: &Ctx<'_>, receipt: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "verifyReceipt", &[receipt.into()])
        .await
        .context("verifyReceipt")?;

    if ctx.json {
        return print_json(&result);
    }

    let merkle_valid = result["merkle_valid"].as_bool().unwrap_or(false);
    let canonical = result["canonical"].as_bool().unwrap_or(false);
    let confirmed = result["confirmed"].as_bool().unwrap_or(false);
    let error = result["error"].as_str();
    let authenticated_summary = result["authenticated_summary"].as_object();

    section("Receipt verification");

    if confirmed {
        ok_msg("Receipt is VALID and canonical.");
        kv(
            "Merkle proof",
            if merkle_valid {
                "✓ valid"
            } else {
                "✗ invalid"
            },
        );
        kv(
            "On canonical chain",
            if canonical { "✓ yes" } else { "✗ no" },
        );
    } else {
        let reason = error.unwrap_or("receipt is not confirmed on the canonical chain");
        print_error(&format!("Receipt INVALID: {reason}"));
        kv(
            "Merkle proof",
            if merkle_valid {
                "✓ valid"
            } else {
                "✗ invalid"
            },
        );
        kv(
            "On canonical chain",
            if canonical {
                "✓ yes"
            } else {
                "✗ no (block may have been reorged)"
            },
        );
        bail!("Receipt verification failed");
    }

    if let Some(summary) = authenticated_summary {
        let txid = summary
            .get("txid")
            .and_then(|value| value.as_str())
            .unwrap_or("?");
        let height = summary
            .get("claimed_height")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let confirmed_unix = summary
            .get("confirmed_unix")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let fee = summary
            .get("fee_micro_jtm")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let inputs = summary
            .get("inputs")
            .and_then(|value| value.as_array())
            .map_or(0, Vec::len);
        let outputs = summary
            .get("outputs")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();

        println!();
        section("Authenticated payment");
        kv("Transaction", txid);
        kv2(
            "Block",
            &format!("#{height}"),
            &format!("unix {confirmed_unix}"),
        );
        kv("Fee", &format!("{} ({} μ{TICKER})", jetsam_str(fee), fee));
        kv("Inputs", &inputs.to_string());
        kv("Outputs", &outputs.len().to_string());
        for (index, output) in outputs.iter().enumerate() {
            let owner = output["owner"].as_str().unwrap_or("?");
            let amount = output["amount_micro_jtm"].as_u64().unwrap_or(0);
            let slot = output["slot_index"].as_u64().unwrap_or(0);
            kv2(
                &format!("  Output {}", index + 1),
                &format!("{} ({} μ{TICKER})", jetsam_str(amount), amount),
                &format!("to {owner}, slot {slot}"),
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Node control
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Delegation (v1.7, T3.8)
// ---------------------------------------------------------------------------

/// Unlock entries one call makes at most: the delegation objects one block
/// carries (and one pool holds).
const MAX_UNLOCK_PARTS: u64 = 32;

/// The entries an unlock of `amount` is split into (A1 m4): each payable by
/// one payout (at most `max_payable_entry`, from `getStakeInfo`) and at
/// least `min_unlock`, as even as possible.
fn split_unlock(amount: u64, max_payable_entry: u64, min_unlock: u64) -> anyhow::Result<Vec<u64>> {
    if amount < min_unlock {
        bail!(
            "an unlock is at least {} {TICKER}",
            jetsam_str(min_unlock)
        );
    }
    if max_payable_entry < min_unlock {
        bail!(
            "the stake vault cannot pay an entry of {} {TICKER} yet: try again later",
            jetsam_str(min_unlock)
        );
    }
    let parts = amount.div_ceil(max_payable_entry);
    if parts > MAX_UNLOCK_PARTS {
        bail!(
            "this unlock makes {parts} entries of at most {} {TICKER}: unlock at most {} {TICKER} per call",
            jetsam_str(max_payable_entry),
            jetsam_str(max_payable_entry.saturating_mul(MAX_UNLOCK_PARTS))
        );
    }
    let (base, extra) = (amount / parts, amount % parts);
    let split: Vec<u64> = (0..parts).map(|i| base + u64::from(i < extra)).collect();
    if split.iter().any(|part| *part < min_unlock) {
        bail!(
            "an unlock is split into entries of at least {} {TICKER}",
            jetsam_str(min_unlock)
        );
    }
    Ok(split)
}

/// Why stake objects are refused now (a `getStakeInfo` or `getRewardInfo`
/// reply), if they are.
fn inactive_reason(info: &Value) -> Option<String> {
    if info["delegation_active"].as_bool() == Some(true) {
        return None;
    }
    Some(
        info["inactive_reason"]
            .as_str()
            .unwrap_or(jetsam_rpc::guardian::DELEGATION_INACTIVE_MESSAGE)
            .to_string(),
    )
}

/// The object `walletBuildGuardianPayment` pays for a stake command and an
/// amount (µ units); `None` for the commands that pay nothing.
fn stake_request(command: &StakeCommand, amount: u64) -> Option<Value> {
    match command {
        StakeCommand::Lock { .. } => {
            Some(serde_json::json!({ "kind": "stake_lock", "amount": amount }))
        }
        StakeCommand::Unlock { .. } => {
            Some(serde_json::json!({ "kind": "stake_unlock", "amount": amount }))
        }
        StakeCommand::Claim { guardian, .. } => Some(serde_json::json!({
            "kind": "reward_claim",
            "account": if *guardian { "guardian" } else { "delegator" },
            "amount": amount,
        })),
        StakeCommand::Info { .. } | StakeCommand::Payout { .. } => None,
    }
}

async fn active_owner(ctx: &Ctx<'_>) -> anyhow::Result<String> {
    let wallet = rpc(ctx, "walletActiveAddress", &[])
        .await
        .context("walletActiveAddress")?;
    wallet["address"]
        .as_str()
        .map(str::to_string)
        .context("the daemon has no active wallet address")
}

/// Build the paying transaction of `object` with the daemon's wallet, then
/// hand the intent to its mempool.
async fn stake_submit(ctx: &Ctx<'_>, object: Value, fee: Option<&str>) -> anyhow::Result<Value> {
    let fee = fee.map(parse_jetsam_amount).transpose()?;
    let built = rpc(
        ctx,
        "walletBuildGuardianPayment",
        &[object, fee.map_or(Value::Null, Value::from)],
    )
    .await
    .context("walletBuildGuardianPayment")?;
    let intent = built["intent_hex"]
        .as_str()
        .context("walletBuildGuardianPayment returned no intent")?;
    let admitted = rpc(ctx, "submitGuardianObject", &[intent.into()])
        .await
        .context("submitGuardianObject")?;
    Ok(serde_json::json!({ "built": built, "admitted": admitted }))
}

async fn cmd_stake(ctx: &Ctx<'_>, command: &StakeCommand) -> anyhow::Result<()> {
    let (amount, fee) = match command {
        StakeCommand::Info { address } => return cmd_stake_info(ctx, address.as_deref()).await,
        StakeCommand::Payout { entry_id } => return cmd_stake_payout(ctx, *entry_id).await,
        StakeCommand::Lock { amount, fee }
        | StakeCommand::Unlock { amount, fee }
        | StakeCommand::Claim { amount, fee, .. } => (parse_jetsam_amount(amount)?, fee),
    };
    let owner = active_owner(ctx).await?;
    let info = rpc(ctx, "getStakeInfo", &[owner.clone().into()])
        .await
        .context("getStakeInfo")?;
    if let Some(reason) = inactive_reason(&info) {
        bail!("{reason}");
    }
    let parts = match command {
        StakeCommand::Unlock { .. } => split_unlock(
            amount,
            info["max_payable_entry"].as_u64().unwrap_or(0),
            info["min_unlock"].as_u64().unwrap_or(0),
        )?,
        _ => vec![amount],
    };
    if !matches!(command, StakeCommand::Lock { .. }) && !ctx.json {
        warn_msg("the fee is paid from another coin of the active address: keep a small one");
    }
    if parts.len() > 1 && !ctx.json {
        warn_msg(&format!(
            "split into {} entries, each payable by one payout",
            parts.len()
        ));
    }
    let mut results = Vec::new();
    for part in parts {
        let object = stake_request(command, part).expect("a paying command");
        let result = stake_submit(ctx, object, fee.as_deref()).await?;
        if !ctx.json {
            ok_msg(&format!(
                "{} {} {TICKER}: tx {} (fee {} {TICKER}), held for the next blocks",
                result["admitted"]["kind"].as_str().unwrap_or("object"),
                jetsam_str(part),
                result["built"]["txid"].as_str().unwrap_or("?"),
                jetsam_str(result["built"]["fee_micro_jtm"].as_u64().unwrap_or(0)),
            ));
        }
        results.push(result);
    }
    if ctx.json {
        return print_json(&Value::from(results));
    }
    Ok(())
}

async fn cmd_stake_info(ctx: &Ctx<'_>, address: Option<&str>) -> anyhow::Result<()> {
    let owner = match address {
        Some(address) => address.to_string(),
        None => active_owner(ctx).await?,
    };
    let stake = rpc(ctx, "getStakeInfo", &[owner.clone().into()])
        .await
        .context("getStakeInfo")?;
    let reward = rpc(ctx, "getRewardInfo", &[owner.clone().into()])
        .await
        .context("getRewardInfo")?;
    if ctx.json {
        return print_json(&serde_json::json!({ "stake": stake, "reward": reward }));
    }
    let amount = |value: &Value| format!("{} {TICKER}", jetsam_str(value.as_u64().unwrap_or(0)));
    section(&format!("Delegated stake of {owner}"));
    match inactive_reason(&stake) {
        Some(reason) => warn_msg(&reason),
        None => ok_msg("delegation active"),
    }
    kv("Active", &amount(&stake["active"]));
    kv("Activating", &amount(&stake["activating"]));
    kv("Reward balance", &amount(&reward["balance"]));
    if let Some(guardian) = reward["guardian_balance"].as_u64() {
        kv("Guardians' share", &amount(&Value::from(guardian)));
    }
    kv("Max payable entry", &amount(&stake["max_payable_entry"]));
    kv(
        "Next boundary",
        &reward["next_boundary"].as_u64().unwrap_or(0).to_string(),
    );
    let entries: Vec<&Value> = stake["entries"]
        .as_array()
        .into_iter()
        .chain(reward["entries"].as_array())
        .flatten()
        .collect();
    if !entries.is_empty() {
        section("Open entries");
        for entry in entries {
            println!(
                "  #{:<6} {:<17} {:>20}  due {}{}",
                entry["id"].as_u64().unwrap_or(0),
                entry["kind"].as_str().unwrap_or("?"),
                amount(&entry["amount"]),
                entry["due"].as_u64().unwrap_or(0),
                if entry["payable_now"].as_bool() == Some(true) {
                    "  (payable: jetsam-cli stake payout <id>)"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

async fn cmd_stake_payout(ctx: &Ctx<'_>, entry_id: u64) -> anyhow::Result<()> {
    let plan = rpc(ctx, "planGuardianPayout", &[entry_id.into()])
        .await
        .context("planGuardianPayout")?;
    if ctx.json {
        return print_json(&plan);
    }
    section(&format!("Payout of entry #{entry_id}"));
    kv("From pool", plan["pool"].as_str().unwrap_or("?"));
    kv(
        "Inputs",
        &plan["inputs"].as_array().map_or(0, Vec::len).to_string(),
    );
    for output in plan["outputs"].as_array().into_iter().flatten() {
        kv(
            "Output",
            &format!(
                "{} {TICKER} to {}",
                jetsam_str(output[1].as_u64().unwrap_or(0)),
                output[0].as_str().unwrap_or("?")
            ),
        );
    }
    kv(
        "Fee",
        &format!(
            "{} {TICKER} (taken out of the amount)",
            jetsam_str(plan["fee_micro_jtm"].as_u64().unwrap_or(0))
        ),
    );
    bail!(
        "this daemon's wallet does not build the payout transaction yet (it spends the pool \
         with its published secret): the plan above is what it will carry"
    )
}

// ---------------------------------------------------------------------------
// Bridge (v1.7): deposits for Base/EVM
// ---------------------------------------------------------------------------

/// A `--fee` above this is refused, as for `send`: a fee is a few thousandths
/// of one coin, paid to a miner, never recovered.
const BRIDGE_FEE_SANITY_LIMIT_MICRO: u64 = MICRO_PER_ELD;

/// How long `walletBridgeDeposit` may take: the node's wallet proves the
/// deposit, seconds on a calm machine, measured over the client's 60 s on a
/// loaded one (the call was then cut, and the node dropped the deposit).
const BRIDGE_DEPOSIT_RPC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// What to say when the deposit call got no answer (any error but a reply of
/// the node): its outcome is unknown, so it must be checked before a second
/// deposit is sent.
fn unknown_outcome_message(error: &anyhow::Error) -> Option<String> {
    if error.downcast_ref::<RpcCallError>().is_some() {
        return None;
    }
    Some(format!(
        "No answer from the node to the deposit ({error:#}). Its outcome is unknown: the node may \
         have submitted it, or dropped it. Check 'jetsam-cli mempool', 'jetsam-cli history' and \
         'jetsam-cli bridge status' before you deposit again: sending it again could deposit \
         twice."
    ))
}

/// Why this build sends no deposit at all, said before the node is asked
/// anything: its v1.7 height is not set (the public network and the test
/// network today). The vault is then an ordinary address whose spend secret
/// is published.
fn build_refusal(rules: &VaultRules, is_test_chain: bool) -> Option<String> {
    if rules.activation_height.is_some() && rules.deposits_from_height.is_some() {
        return None;
    }
    let build = if is_test_chain {
        "This test-network build"
    } else {
        "This mainnet build"
    };
    Some(format!(
        "{build} has no v1.7 activation height (V1_7_ACTIVATION_HEIGHT is None): the bridge is \
         not armed, and its vault is an ordinary address whose spend secret is published, so \
         anyone could take coins sent there. Deposits open with the release that arms v1.7."
    ))
}

/// Why the node's next block can carry no deposit (its `getVaultState`
/// reply, judged with this CLI's `rules`), if it cannot: this build not
/// armed, the node's v1.7 rules not active, an opening `F` other than this
/// CLI's, or the next block not above `F` yet.
fn deposits_closed_reason(state: &Value, rules: &VaultRules) -> Option<String> {
    if let Some(refusal) = build_refusal(rules, jetsam_chain::consensus::identity::IS_TEST_CHAIN) {
        return Some(refusal);
    }
    let (Some(tip), Some(opening)) = (state["tip_height"].as_u64(), rules.deposits_from_height)
    else {
        return Some("the node's getVaultState reply has no tip height".to_string());
    };
    if state["active"].as_bool() != Some(true) {
        return Some(format!(
            "the node's v1.7 rules are not active at its next block (its tip is {tip}): no \
             deposit is valid yet"
        ));
    }
    let node_opening = state["deposits_from_height"].as_u64();
    if node_opening != Some(opening) {
        return Some(format!(
            "the node and this CLI disagree about the bridge opening (node: {}, this CLI: \
             {opening}): use the jetsam-cli of the node's release",
            node_opening.map_or_else(|| "none".to_string(), |f| f.to_string())
        ));
    }
    if tip < opening {
        let wait = opening - tip;
        return Some(format!(
            "deposits are valid from block {} on; the node is at block {tip}: {wait} more block{} \
             to go",
            opening + 1,
            if wait == 1 { "" } else { "s" }
        ));
    }
    None
}

/// A deposit checked against the node's bridge before it is planned: open in
/// its next block, within the limits of one deposit (V4, the rule this build
/// carries) and within the vault's room (V8, as the node reports it). The
/// next block's height.
fn deposit_preflight(
    state: &Value,
    amount: u64,
    recipient: &[u8; 20],
    rules: &VaultRules,
) -> Result<u64, String> {
    if let Some(reason) = deposits_closed_reason(state, rules) {
        return Err(reason);
    }
    let next = state["tip_height"].as_u64().unwrap_or(0).saturating_add(1);
    jetsam_rpc::bridge::check_deposit(amount, recipient, next, rules).map_err(|e| e.to_string())?;
    let room = state["room"].as_u64().unwrap_or(0);
    if amount > room {
        return Err(format!(
            "the vault has room for {} {TICKER} more under its ceiling, this deposit is {} \
             {TICKER}: deposit less, or later",
            jetsam_str(room),
            jetsam_str(amount)
        ));
    }
    Ok(next)
}

/// The deposits `getVaultState` counted over its last 1 024 heights, and the
/// heights among them whose block the node no longer retains (not counted).
fn deposits_line(state: &Value) -> String {
    let counted = state["deposits_last_1024"].as_u64().unwrap_or(0);
    match state["deposit_heights_unread"].as_u64().unwrap_or(0) {
        0 => format!("{counted} in the last 1024 blocks"),
        unread => format!(
            "{counted} in the last 1024 blocks ({unread} of them no longer retained by the node: \
             not counted)"
        ),
    }
}

/// Only the word itself confirms a deposit.
fn is_confirmation(answer: &str) -> bool {
    answer.trim().eq_ignore_ascii_case("yes")
}

async fn cmd_bridge(ctx: &Ctx<'_>, command: &BridgeCommand) -> anyhow::Result<()> {
    match command {
        BridgeCommand::Status => cmd_bridge_status(ctx).await,
        BridgeCommand::Deposit {
            recipient,
            amount,
            fee,
            dry_run,
            yes,
        } => cmd_bridge_deposit(ctx, recipient, amount, fee.as_deref(), *dry_run, *yes).await,
    }
}

async fn cmd_bridge_status(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    let rules = VaultRules::current();
    let state = rpc(ctx, "getVaultState", &[])
        .await
        .context("getVaultState")?;
    let closed = deposits_closed_reason(&state, &rules);
    if ctx.json {
        return print_json(&serde_json::json!({
            "deposits_open": closed.is_none(),
            "closed_reason": closed,
            "this_cli": {
                "network": jetsam_chain::consensus::identity::CHAIN_NAME,
                "v1_7_activation_height": rules.activation_height,
                "deposits_from_height": rules.deposits_from_height,
                "min_deposit_micro": rules.min_deposit,
                "max_deposit_micro": rules.max_deposit,
                "vault": rules.vault.to_bech32(),
            },
            "node": state,
        }));
    }
    let coins = |value: &Value| format!("{} {TICKER}", jetsam_str(value.as_u64().unwrap_or(0)));
    let height =
        |value: Option<u64>| value.map_or_else(|| "not set".to_string(), |h| h.to_string());
    section("Bridge to Base/EVM (v1.7)");
    match &closed {
        None => ok_msg("deposits are open: the node's next block can carry one"),
        Some(reason) => warn_msg(&format!("deposits are closed: {reason}")),
    }
    println!();
    kv("This CLI", jetsam_chain::consensus::identity::CHAIN_NAME);
    kv("v1.7 height", &height(rules.activation_height));
    kv2(
        "Opening F",
        &height(rules.deposits_from_height),
        "(deposits valid in the blocks above it)",
    );
    kv(
        "One deposit",
        &format!(
            "{} to {} {TICKER}",
            jetsam_str(rules.min_deposit),
            jetsam_str(rules.max_deposit)
        ),
    );
    kv("Vault", &rules.vault.to_bech32());
    println!();
    kv("Node tip", &height(state["tip_height"].as_u64()));
    kv(
        "Node v1.7",
        if state["active"].as_bool() == Some(true) {
            "active at its next block"
        } else {
            "not active"
        },
    );
    kv(
        "Node opening F",
        &height(state["deposits_from_height"].as_u64()),
    );
    kv("Locked", &coins(&state["bridge_locked"]));
    kv2(
        "Room",
        &coins(&state["room"]),
        &format!("(ceiling {})", coins(&state["ceiling"])),
    );
    kv("Deposits", &deposits_line(&state));
    kv(
        "Releases paid",
        &state["released_count"].as_u64().unwrap_or(0).to_string(),
    );
    Ok(())
}

async fn cmd_bridge_deposit(
    ctx: &Ctx<'_>,
    recipient_text: &str,
    amount_text: &str,
    fee_text: Option<&str>,
    dry_run: bool,
    yes: bool,
) -> anyhow::Result<()> {
    use jetsam_rpc::bridge::{
        checksummed, parse_recipient, BridgeDepositPlan, BridgeDepositResult,
    };
    let rules = VaultRules::current();

    // --- What is checked before the node is asked anything ---
    let recipient = parse_recipient(recipient_text).map_err(|error| {
        anyhow::anyhow!("Invalid Base/EVM address {recipient_text:?}: {error}.")
    })?;
    let amount = parse_jetsam_amount(amount_text)
        .with_context(|| format!("invalid amount {amount_text:?}"))?;
    let fee = fee_text
        .map(|text| parse_jetsam_amount(text).with_context(|| format!("invalid fee {text:?}")))
        .transpose()?;
    if fee.is_some_and(|fee| fee > BRIDGE_FEE_SANITY_LIMIT_MICRO) {
        bail!(
            "Refusing a fee above one {TICKER}: a fee is normally a few thousandths of one {TICKER}, \
             paid to a miner and never recovered. Omit --fee to let the node price it."
        );
    }
    if let Some(refusal) = build_refusal(&rules, jetsam_chain::consensus::identity::IS_TEST_CHAIN) {
        bail!("{refusal} Nothing was sent.");
    }

    // --- The node's bridge, then the plan (no side effect) ---
    let state = rpc(ctx, "getVaultState", &[])
        .await
        .context("getVaultState")?;
    let next = deposit_preflight(&state, amount, &recipient, &rules)
        .map_err(|reason| anyhow::anyhow!("Deposit refused: {reason}. Nothing was sent."))?;
    let shown = checksummed(&recipient);
    let plan: BridgeDepositPlan = serde_json::from_value(
        rpc(
            ctx,
            "walletPlanBridgeDeposit",
            &[
                shown.clone().into(),
                amount.into(),
                fee.map_or(Value::Null, Value::from),
            ],
        )
        .await
        .context("walletPlanBridgeDeposit")?,
    )
    .context("walletPlanBridgeDeposit reply")?;
    if plan.total_micro_jtm > plan.spendable_micro_jtm {
        bail!(
            "Insufficient funds: the deposit and its fee take {} {TICKER}, the active address \
             {} can spend {} {TICKER}. Nothing was sent.",
            jetsam_str(plan.total_micro_jtm),
            plan.from,
            jetsam_str(plan.spendable_micro_jtm)
        );
    }
    if ctx.json && dry_run {
        return print_json(&serde_json::to_value(&plan)?);
    }
    if !ctx.json {
        section("Bridge deposit to Base/EVM");
        kv("From", &plan.from);
        kv("To (Base/EVM)", &plan.recipient);
        kv2(
            "Amount",
            &format!("{} {TICKER}", jetsam_str(amount)),
            &format!("({amount} \u{03bc}{TICKER})"),
        );
        kv2(
            "Fee",
            &format!("{} {TICKER}", jetsam_str(plan.fee_micro_jtm)),
            if fee.is_none() { "(auto)" } else { "" },
        );
        kv(
            "Total",
            &format!("{} {TICKER}", jetsam_str(plan.total_micro_jtm)),
        );
        kv("Vault", &plan.vault);
        kv2(
            "Next block",
            &next.to_string(),
            &format!("(deposits valid above {})", plan.deposits_from_height),
        );
        println!();
        warn_msg(
            "A deposit cannot be cancelled or sent elsewhere. Check the Base/EVM address above, \
             and use one you control (not an exchange's deposit address).",
        );
        println!(
            "  {} The bridge mints the same amount to it on Base/EVM: by the slow lane, at the \
             next checkpoint (about every 1 h 50) then about 24 h; or sooner, less a fee, if an \
             advancer advances it. Deposits left out beyond 22 a day wait; none is lost.",
            c!(DIM, "Note:")
        );
    }
    if dry_run {
        println!("  {} Dry run: nothing was built or sent.", c!(DIM, "Note:"));
        return Ok(());
    }

    // --- Explicit confirmation ---
    if !yes {
        if ctx.json || !io::stdin().is_terminal() {
            bail!("Confirmation required: run again with --yes to send this deposit. Nothing was sent.");
        }
        print!(
            "  Type {} to lock {} {TICKER} in the bridge vault for {}: ",
            c!(BOLD, "yes"),
            jetsam_str(amount),
            plan.recipient
        );
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if !is_confirmation(&answer) {
            println!("  {}", c!(DIM, "Cancelled: nothing was sent."));
            return Ok(());
        }
    }

    // --- Built, signed and submitted by the daemon's wallet, at the fee shown ---
    if !ctx.json {
        println!(
            "  {} The node's wallet builds and signs the deposit (it can take a minute)\u{2026}",
            c!(DIM, "\u{23f3}")
        );
    }
    let result = match rpc_with_timeout(
        ctx,
        "walletBridgeDeposit",
        &[
            plan.recipient.clone().into(),
            amount.into(),
            plan.fee_micro_jtm.into(),
        ],
        Some(BRIDGE_DEPOSIT_RPC_TIMEOUT),
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            if let Some(message) = input_limit_exceeded_message(&error) {
                bail!(message);
            }
            if let Some(message) = unknown_outcome_message(&error) {
                bail!(message);
            }
            return Err(error).context("walletBridgeDeposit: deposit not sent");
        }
    };
    if ctx.json {
        return print_json(&result);
    }
    let result: BridgeDepositResult =
        serde_json::from_value(result).context("walletBridgeDeposit reply")?;
    println!();
    section("Deposit submitted");
    ok_msg(&format!("TX {}", result.txid));
    kv("To (Base/EVM)", &result.recipient);
    kv(
        "Amount",
        &format!("{} {TICKER}", jetsam_str(result.amount_micro_jtm)),
    );
    kv(
        "Fee",
        &format!("{} {TICKER}", jetsam_str(result.fee_micro_jtm)),
    );
    println!();
    println!(
        "  {} Pending until a block carries it. Keep the txid: it names your deposit on the \
         bridge. Follow it with {}.",
        c!(DIM, "\u{23f3}"),
        c!(BOLD, "'jetsam-cli bridge status'")
    );
    Ok(())
}

async fn cmd_stop(ctx: &Ctx<'_>) -> anyhow::Result<()> {
    match rpc(ctx, "stop", &[]).await {
        Ok(r) => {
            let msg = r.as_str().unwrap_or("shutting down");
            ok_msg(&format!("Daemon is {msg}."));
        }
        Err(_) => {
            // Connection dropped = daemon already shutting down (expected)
            ok_msg("Daemon is shutting down.");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mining commands (external miner API)
// ---------------------------------------------------------------------------

async fn cmd_block_template(ctx: &Ctx<'_>, miner_addr: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "getBlockTemplate", &[miner_addr.into()])
        .await
        .context("getBlockTemplate")?;

    if ctx.json {
        return print_json(&result);
    }

    let height = result["height"].as_u64().unwrap_or(0);
    let n_txs = result["n_txs"].as_u64().unwrap_or(0);
    let claimable_fees = result["claimable_fees_micro_jtm"].as_u64().unwrap_or(0);
    let coinbase_value = result["coinbase_value_micro_jtm"].as_u64().unwrap_or(0);
    let template_id = result["template_id"].as_str().unwrap_or("");
    let expires_in_seconds = result["expires_in_seconds"].as_u64().unwrap_or(0);
    let nonce_field_index = result["nonce_field_index"].as_u64().unwrap_or(0);
    let difficulty_target = result["difficulty_target_hex"].as_str().unwrap_or("");
    let pow_fields = result["pow_fields_hex"].as_str().unwrap_or("");

    section("Block template");
    kv("Template ID", template_id);
    kv("Height", &height.to_string());
    kv("Txs in block", &n_txs.to_string());
    kv("Expires in", &format!("{expires_in_seconds} s"));
    kv("Nonce field", &nonce_field_index.to_string());
    kv("Difficulty", difficulty_target);
    kv("Claimable fees", &format!("{claimable_fees} μ{TICKER}"));
    kv("Coinbase value", &format!("{coinbase_value} μ{TICKER}"));
    if !pow_fields.is_empty() {
        kv2(
            "PoW fields",
            &format!("{}…", &pow_fields[..pow_fields.len().min(32)]),
            "(16 field elements, Poseidon2b PoW input)",
        );
    }
    println!();
    println!(
        "  {} Search the nonce, then run `jetsam-cli submit-block {template_id} <32-char-lowercase-nonce-hex>`.",
        c!(DIM, "PoW:")
    );
    if !pow_fields.is_empty() {
        println!("  {} {}", c!(DIM, "Field schedule hex:"), pow_fields);
    }

    Ok(())
}

async fn cmd_submit_block(ctx: &Ctx<'_>, template_id: &str, nonce_hex: &str) -> anyhow::Result<()> {
    let result = rpc(ctx, "submitBlock", &[template_id.into(), nonce_hex.into()])
        .await
        .context("submitBlock")?;

    if ctx.json {
        return print_json(&result);
    }

    let hash = result.as_str().unwrap_or("?");
    ok_msg(&format!("Block accepted: {}", ctx.h(hash)));

    Ok(())
}

// ---------------------------------------------------------------------------
// JSON-RPC transport
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RpcCallError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl std::fmt::Display for RpcCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RPC error ({}): {}", self.code, self.message)
    }
}

impl std::error::Error for RpcCallError {}

fn input_limit_exceeded_message(error: &anyhow::Error) -> Option<String> {
    let rpc = error.downcast_ref::<RpcCallError>()?;
    if rpc.code != i64::from(jetsam_rpc::types::WALLET_INPUT_LIMIT_EXCEEDED_CODE) {
        return None;
    }
    let max_inputs = rpc
        .data
        .as_ref()
        .and_then(|data| data.get("max_inputs"))
        .and_then(Value::as_u64)
        .unwrap_or(8);
    Some(format!(
        "Payment cannot be created: it requires more than {max_inputs} inputs."
    ))
}

async fn rpc(ctx: &Ctx<'_>, method: &str, params: &[Value]) -> anyhow::Result<Value> {
    rpc_with_timeout(ctx, method, params, None).await
}

/// [`rpc`] with a timeout of its own instead of the client's 60 s.
async fn rpc_with_timeout(
    ctx: &Ctx<'_>,
    method: &str,
    params: &[Value],
    timeout: Option<std::time::Duration>,
) -> anyhow::Result<Value> {
    let method_full = format!("jetsam_{method}");
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id":      1,
        "method":  method_full,
        "params":  params,
    });

    let mut request = ctx.client.post(ctx.rpc).json(&body);
    if let Some(timeout) = timeout {
        request = request.timeout(timeout);
    }
    let resp: Value = request
        .send()
        .await
        .map_err(|e| {
            if e.is_connect() || e.is_timeout() {
                anyhow::anyhow!(
                    "Node is not responding.\n\
                     \tIs the jetsam daemon running?  Try: jetsam --mode miner\n\
                     \tRPC endpoint: {}\n\
                     \tOverride with --rpc <URL> or JETSAM_RPC env var",
                    ctx.rpc
                )
            } else {
                anyhow::anyhow!("HTTP error: {e}")
            }
        })?
        .json()
        .await
        .with_context(|| format!("decode JSON-RPC response for {method}"))?;

    if let Some(err) = resp.get("error") {
        let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(anyhow::Error::new(RpcCallError {
            code,
            message: message.to_string(),
            data: err.get("data").cloned(),
        }));
    }

    Ok(resp["result"].clone())
}

fn print_json(v: &Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

/// Cheap client-side sanity check before the daemon does the real parsing.
///
/// The prefix is derived from the protocol constant, never written out here:
/// a hard-coded prefix is exactly how the upstream `o1` check survived this
/// chain's rename and rejected every valid address it was handed.
fn looks_like_address(candidate: &str) -> bool {
    let prefix = format!("{}1", jetsam_chain::consensus::identity::ADDRESS_HRP);
    candidate
        .trim()
        .to_ascii_lowercase()
        .starts_with(&prefix.to_ascii_lowercase())
}

#[cfg(test)]
mod adoption_tests {
    use super::{adoption_report, agent_matches, merge_adoption_answers};
    use serde_json::{json, Value};

    fn answer(now_unix: u64, peers: Value) -> Value {
        json!({
            "local_agent_version": "jetsam/1.6.0 (v1.6=5760)",
            "now_unix": now_unix,
            "peers": peers,
        })
    }

    fn peer(id: Value, agent: &str, connected: bool, verified: bool, last_seen: u64) -> Value {
        json!({
            "peer_id": id,
            "agent_version": agent,
            "connected": connected,
            "profile_verified": verified,
            "first_seen_unix": 1,
            "last_seen_unix": last_seen,
        })
    }

    #[test]
    fn two_seeds_count_a_shared_peer_once() {
        let first = answer(
            1_000,
            json!([
                peer(json!("P1"), "jetsam/1.6.0 (v1.6=5760)", true, true, 990),
                peer(json!("P2"), "rust-libp2p/0.46.0", false, true, 400),
                peer(json!("P5"), "rust-libp2p/0.46.0", false, true, 10),
            ]),
        );
        // Another node, another clock.
        let second = answer(
            50_000,
            json!([
                peer(json!("P1"), "jetsam/1.6.0 (v1.6=5760)", false, true, 49_000),
                peer(json!("P3"), "jetsam/1.6.0-dev (v1.6=4800)", true, true, 49_990),
                // Never completed the profile handshake: not on this chain.
                peer(json!("P4"), "jetsam/1.6.0 (v1.6=5760)", true, false, 49_990),
            ]),
        );
        let peers = merge_adoption_answers(&[first, second]).unwrap();
        assert_eq!(peers.len(), 5);
        let shared = peers
            .iter()
            .find(|peer| peer.peer_id.as_deref() == Some("P1"))
            .unwrap();
        assert!(shared.connected && shared.verified);
        assert_eq!(shared.age_secs, 10);

        let report = adoption_report(&peers, Some("1.6.0"), 900);
        // Connected and verified: P1, P3. Seen within 15 minutes: P1, P3, P2
        // (P5 left 990 s ago).
        assert_eq!((report.target_connected, report.connected), (1, 2));
        assert_eq!((report.target_recent, report.recent), (1, 3));
        assert_eq!(report.unverified, 1);
        assert_eq!(report.unmerged, 0);
        let rows: Vec<(&str, usize, usize)> = report
            .rows
            .iter()
            .map(|row| (row.agent.as_str(), row.connected, row.recent))
            .collect();
        assert_eq!(
            rows,
            vec![
                ("jetsam/1.6.0 (v1.6=5760)", 1, 1),
                ("jetsam/1.6.0-dev (v1.6=4800)", 1, 1),
                ("rust-libp2p/0.46.0", 0, 1),
            ]
        );
        // No target: the same counts, no share.
        let untargeted = adoption_report(&peers, None, 900);
        assert_eq!((untargeted.target_connected, untargeted.target_recent), (0, 0));
        assert_eq!((untargeted.connected, untargeted.recent), (2, 3));
    }

    #[test]
    fn a_target_is_a_release_or_an_exact_agent_prefix() {
        assert!(agent_matches("jetsam/1.6.0 (v1.6=5760)", "1.6.0"));
        assert!(!agent_matches("jetsam/1.6.0-dev (v1.6=4800)", "1.6.0"));
        assert!(!agent_matches("rust-libp2p/0.46.0", "0.46.0"));
        assert!(agent_matches(
            "jetsam/1.6.0 (v1.6=5760)",
            "jetsam/1.6.0 (v1.6=5760)"
        ));
        assert!(!agent_matches(
            "jetsam/1.6.0 (v1.6=6720)",
            "jetsam/1.6.0 (v1.6=5760)"
        ));
    }

    #[test]
    fn answers_without_identities_are_counted_but_not_merged() {
        let hidden = |now| {
            answer(
                now,
                json!([peer(Value::Null, "jetsam/1.6.0 (v1.6=5760)", true, true, now)]),
            )
        };
        let peers = merge_adoption_answers(&[hidden(10), hidden(20)]).unwrap();
        assert_eq!(peers.len(), 2);
        let report = adoption_report(&peers, Some("1.6.0"), 60);
        assert_eq!((report.target_connected, report.connected), (2, 2));
        assert_eq!(report.unmerged, 2);
    }

    #[test]
    fn a_malformed_answer_is_an_error_not_a_count() {
        assert!(merge_adoption_answers(&[json!({"peers": "nope"})]).is_err());
        assert!(merge_adoption_answers(&[answer(1, json!([{"peer_id": "P1"}]))]).is_err());
    }
}

#[cfg(test)]
mod address_tests {
    use super::looks_like_address;

    #[test]
    fn a_canonical_address_is_accepted_and_the_upstream_prefix_is_not() {
        // Real addresses from this network. The prefix is read from the
        // protocol constant for the same reason `looks_like_address` reads it:
        // the test chain's is `tj`, so a literal pins one of the two networks
        // and makes the other one's addresses look foreign.
        let hrp = jetsam_chain::consensus::identity::ADDRESS_HRP;
        assert!(looks_like_address(&format!(
            "{hrp}1snyeeyhqrkyn303q3ctwn0yzk4aj4f3c64qtgu2etsj6nn6ludjq5zt8nf"
        )));
        assert!(looks_like_address(&format!(
            "{hrp}19fe83j2l7ukelk63477jdhlr2prsmxytqu8lzu8q3hhfhegzxs6qefc03u"
        )));
        // Leading and trailing whitespace is a paste artefact, not a typo.
        assert!(looks_like_address(&format!(
            "  {hrp}1snyeeyhqrkyn303q3ctwn0yzk4aj4f3c64qtgu2etsj6nn6ludjq5zt8nf\n"
        )));
        // Upper case is still the same address.
        assert!(looks_like_address(&format!(
            "{}1SNYEEYHQRKYN303Q3CTWN0YZK4AJ4F3C64QTGU2ETSJ6NN6LUDJQ5ZT8NF",
            hrp.to_ascii_uppercase()
        )));
        // The upstream prefix is a different network's address.
        assert!(!looks_like_address(
            "o1q9p2w4t8k3ux7c5n0r6dmzfae9hj2ls4v8y6c3b7n5q2wk0t9xp"
        ));
        assert!(!looks_like_address("not-an-address"));
        assert!(!looks_like_address(""));
    }
}

#[cfg(test)]
mod amount_tests {
    use super::{input_limit_exceeded_message, jetsam_str, parse_jetsam_amount, RpcCallError};

    #[test]
    fn decimal_amounts_are_exact_and_checked() {
        assert_eq!(parse_jetsam_amount("10.5").unwrap(), 10_500_000);
        assert_eq!(parse_jetsam_amount(".000001").unwrap(), 1);
        assert_eq!(parse_jetsam_amount("1.").unwrap(), 1_000_000);
        assert_eq!(jetsam_str(u64::MAX), "18446744073709.551615");
        assert!(parse_jetsam_amount("0.0000001").is_err());
        assert!(parse_jetsam_amount("NaN").is_err());
        assert!(parse_jetsam_amount("inf").is_err());
        assert!(parse_jetsam_amount("18446744073710").is_err());
    }

    #[test]
    fn input_limit_error_is_concise_and_uses_the_structured_limit() {
        let error = anyhow::Error::new(RpcCallError {
            code: i64::from(jetsam_rpc::types::WALLET_INPUT_LIMIT_EXCEEDED_CODE),
            message: jetsam_rpc::types::WALLET_INPUT_LIMIT_EXCEEDED_MESSAGE.to_string(),
            data: Some(serde_json::json!({ "max_inputs": 8 })),
        });
        assert_eq!(
            input_limit_exceeded_message(&error).as_deref(),
            Some("Payment cannot be created: it requires more than 8 inputs.")
        );
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::DEFAULT_RPC_URL;

    /// The advertised endpoint and the daemon's own port must agree.
    ///
    /// `clap` needs a literal for `default_value`, so `DEFAULT_RPC_URL`
    /// restates a port that `identity` already owns. Two declarations of one
    /// truth drift apart in silence unless something reads both: a testnet
    /// client pointed at 9701 says "connection refused", and the reader goes
    /// looking for a firewall instead of a port.
    #[test]
    fn the_default_endpoint_matches_this_build() {
        let expected = format!(
            "http://127.0.0.1:{}",
            jetsam_chain::consensus::identity::DEFAULT_RPC_PORT
        );
        assert_eq!(DEFAULT_RPC_URL, expected);
    }
}

#[cfg(test)]
mod history_tests {
    use super::{
        history_row_style, mined_total_line, orphaned_total_line, tally_history, HistoryTotals, GRN,
    };
    use jetsam_chain::consensus::identity::TICKER;
    use serde_json::{json, Value};

    fn coinbase(height: u64, micro: u64, canonical: Option<bool>) -> Value {
        let mut entry = json!({
            "tx_hash": format!("{height:064x}"),
            "height": height,
            "direction": "received",
            "is_coinbase": true,
            "amount_micro_jtm": micro,
        });
        if let Some(canonical) = canonical {
            entry["canonical"] = json!(canonical);
            entry["confirmations"] = json!(if canonical { 1 } else { 0 });
        }
        entry
    }

    fn payment(micro: u64) -> Value {
        json!({
            "tx_hash": format!("{micro:064x}"),
            "height": 12,
            "direction": "received",
            "is_coinbase": false,
            "amount_micro_jtm": micro,
        })
    }

    /// The reported case: three locally mined blocks around 29344 all lost a
    /// reorganization, the balance is zero, and the history summed them as
    /// "+180.03 JTM received".
    #[test]
    fn orphaned_coinbases_are_not_received_money() {
        let entries = [
            coinbase(29_342, 60_010_000, Some(false)),
            coinbase(29_343, 60_010_000, Some(false)),
            coinbase(29_344, 60_010_000, Some(false)),
        ];
        let refs = entries.iter().collect::<Vec<_>>();
        let totals = tally_history(&refs);
        assert_eq!(totals.received, 0, "orphaned rewards counted as received");
        assert_eq!((totals.mined_blocks, totals.mined_micro), (0, 0));
        assert_eq!(totals.orphaned_blocks, 3);
        assert_eq!(totals.orphaned_micro, 180_030_000);
        assert_eq!(
            orphaned_total_line(&totals),
            Some(format!(
                "orphaned \u{b7} no reward: 3 blocks, 180.030000 {TICKER} \
                 (lost a reorganization, never in the balance)"
            ))
        );
        assert_eq!(mined_total_line(&totals), None);
    }

    #[test]
    fn an_orphaned_coinbase_row_is_labelled_and_not_green() {
        let (sign, label, colour) = history_row_style(&coinbase(29_344, 60_010_000, Some(false)));
        assert_eq!(label, "orphaned \u{b7} no reward");
        assert_ne!(sign, "+");
        assert_ne!(colour, GRN);
    }

    /// A coinbase whose block is canonical is a mined reward: labelled
    /// "mined", in its own total, apart from payments received.
    #[test]
    fn a_canonical_coinbase_is_mined_apart_from_payments() {
        let mined = coinbase(29_345, 60_010_000, Some(true));
        let paid = payment(2_000_000);
        assert_eq!(history_row_style(&mined), ("+", "mined", GRN));
        assert_eq!(history_row_style(&paid), ("+", "\u{2190} recv", GRN));
        let totals = tally_history(&[&mined, &paid]);
        assert_eq!(
            totals,
            HistoryTotals {
                received: 2_000_000,
                sent: 0,
                mined_blocks: 1,
                mined_micro: 60_010_000,
                orphaned_blocks: 0,
                orphaned_micro: 0,
            }
        );
        assert_eq!(
            mined_total_line(&totals),
            Some(format!(
                "mined: 1 block, 60.010000 {TICKER} earned (canonical blocks only)"
            ))
        );
        assert_eq!(orphaned_total_line(&totals), None);
    }

    #[test]
    fn no_orphan_line_without_orphans() {
        let entry = coinbase(10, 50_000_000, Some(true));
        assert_eq!(orphaned_total_line(&tally_history(&[&entry])), None);
    }

    /// An entry from an older node carries no status: the history cannot tell
    /// mined from orphaned and keeps the old display.
    #[test]
    fn a_coinbase_without_status_keeps_the_old_display() {
        let legacy = coinbase(29_346, 60_010_000, None);
        assert_eq!(history_row_style(&legacy), ("+", "\u{2190} recv", GRN));
        let totals = tally_history(&[&legacy]);
        assert_eq!(totals.received, 60_010_000);
        assert_eq!((totals.mined_blocks, totals.orphaned_blocks), (0, 0));
    }

    /// The node recomputes `canonical` on every call: the same block moves
    /// from mined to orphaned when another block wins its height, and back
    /// when the chain returns to it. The totals follow.
    #[test]
    fn a_reorganization_moves_a_coinbase_between_mined_and_orphaned() {
        let mined = (1, 60_010_000, 0, 0);
        let orphaned = (0, 0, 1, 60_010_000);
        for (canonical, expected) in [(true, mined), (false, orphaned), (true, mined)] {
            let entry = coinbase(29_344, 60_010_000, Some(canonical));
            let totals = tally_history(&[&entry]);
            assert_eq!(
                (
                    totals.mined_blocks,
                    totals.mined_micro,
                    totals.orphaned_blocks,
                    totals.orphaned_micro
                ),
                expected,
                "canonical={canonical}"
            );
            assert_eq!(totals.received, 0);
            let label = history_row_style(&entry).1;
            assert_eq!(
                label,
                if canonical {
                    "mined"
                } else {
                    "orphaned \u{b7} no reward"
                }
            );
        }
    }
}

#[cfg(test)]
mod ticker_tests {
    use super::Cli;
    use clap::CommandFactory;
    use jetsam_chain::consensus::identity::{SUBUNIT_NAME, TICKER};

    /// The words of `text` that name a ticker other than this chain's (its
    /// coin or its subunit): the test chain's CLI printed `JTM`, the public
    /// network's ticker, beside test coins worth nothing.
    fn foreign_tickers(text: &str) -> Vec<String> {
        let coin = ["J", "TM"].concat();
        text.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|word| word.contains(&coin))
            .filter(|word| *word != TICKER && *word != SUBUNIT_NAME)
            .map(str::to_string)
            .collect()
    }

    fn every_help(command: &mut clap::Command, out: &mut String) {
        out.push_str(&command.render_long_help().to_string());
        for sub in command.get_subcommands_mut() {
            every_help(sub, out);
        }
    }

    /// Every help page names this chain's ticker (`JTMT` on the test chain).
    #[test]
    fn the_help_names_this_chains_ticker() {
        let mut command = Cli::command();
        command.build();
        let mut help = String::new();
        every_help(&mut command, &mut help);
        assert!(help.contains(TICKER), "the help names a ticker");
        assert_eq!(foreign_tickers(&help), Vec::<String>::new());
    }

    /// Nothing this CLI prints spells a ticker out: every amount, label and
    /// message takes it from `identity::TICKER` (the source above the tests,
    /// comments aside; the help is checked above).
    #[test]
    fn printed_text_takes_the_ticker_from_the_profile() {
        let source = include_str!("jetsam_cli.rs");
        let code = source
            .split(&["#[cfg", "(test)]"].concat())
            .next()
            .unwrap();
        let spelled: Vec<String> = code
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim_start().starts_with("//"))
            .filter(|(_, line)| line.contains(&["J", "TM"].concat()))
            .map(|(index, line)| format!("{}: {}", index + 1, line.trim()))
            .collect();
        assert_eq!(spelled, Vec::<String>::new());
    }
}

#[cfg(test)]
mod stake_tests {
    use super::{inactive_reason, split_unlock, stake_request, Cli, Command, StakeCommand};
    use clap::Parser;
    use jetsam_rpc::guardian::{GuardianPaymentObject, DELEGATION_INACTIVE_MESSAGE};

    const ONE: u64 = 1_000_000;

    /// A1 m4: an unlock larger than one payout can pay is split into
    /// entries each payable by one payout and each at least the minimum, as
    /// even as possible; one that fits stays whole.
    #[test]
    fn an_unlock_is_split_into_payable_entries() {
        assert_eq!(split_unlock(25 * ONE, 100 * ONE, ONE).unwrap(), vec![25 * ONE]);
        assert_eq!(
            split_unlock(25 * ONE, 10 * ONE, ONE).unwrap(),
            vec![8_333_334, 8_333_333, 8_333_333]
        );
        let parts = split_unlock(1_000 * ONE + 7, 300 * ONE, ONE).unwrap();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts.iter().sum::<u64>(), 1_000 * ONE + 7);
        assert!(parts.iter().all(|part| *part <= 300 * ONE && *part >= ONE));
        assert_eq!(split_unlock(20 * ONE, 20 * ONE, ONE).unwrap(), vec![20 * ONE]);
        assert!(split_unlock(ONE - 1, 100 * ONE, ONE).is_err(), "below the minimum");
        assert!(
            split_unlock(3 * ONE, 2 * ONE, 2 * ONE).is_err(),
            "two entries of 1.5 each, under a minimum of 2"
        );
        assert!(split_unlock(5 * ONE, ONE - 1, ONE).is_err(), "nothing payable yet");
        assert!(
            split_unlock(33 * ONE, ONE, ONE).is_err(),
            "more entries than a block carries: in several calls"
        );
        assert_eq!(split_unlock(32 * ONE, ONE, ONE).unwrap().len(), 32);
    }

    /// Before activation every stake command says why, in the words of T1.1
    /// §10 (height J_D, or the pot not yet funded by both funds in one block).
    #[test]
    fn an_inactive_delegation_says_why() {
        let inactive = serde_json::json!({
            "delegation_active": false,
            "inactive_reason": DELEGATION_INACTIVE_MESSAGE,
        });
        let reason = inactive_reason(&inactive).expect("inactive");
        assert!(reason.contains("J_D") && reason.contains("both development funds"));
        let silent = serde_json::json!({ "delegation_active": false });
        assert_eq!(inactive_reason(&silent).as_deref(), Some(DELEGATION_INACTIVE_MESSAGE));
        assert_eq!(inactive_reason(&serde_json::json!({ "delegation_active": true })), None);
    }

    /// The subcommands, and the object each one asks the daemon to pay for,
    /// in the shape `walletBuildGuardianPayment` reads.
    #[test]
    fn each_stake_command_asks_for_its_object() {
        let parse = |args: &[&str]| match Cli::try_parse_from(args).expect("parses").cmd {
            Command::Stake(command) => command,
            _ => panic!("not a stake command"),
        };
        let object = |command: &StakeCommand, amount: u64| -> GuardianPaymentObject {
            serde_json::from_value(stake_request(command, amount).expect("a request"))
                .expect("the RPC's shape")
        };
        let lock = parse(&["jetsam-cli", "stake", "lock", "12.5"]);
        assert_eq!(
            object(&lock, 12_500_000),
            GuardianPaymentObject::StakeLock { amount: 12_500_000 }
        );
        let unlock = parse(&["jetsam-cli", "stake", "unlock", "3", "--fee", "0.06"]);
        assert!(matches!(&unlock, StakeCommand::Unlock { fee: Some(fee), .. } if fee == "0.06"));
        assert_eq!(
            object(&unlock, 3 * ONE),
            GuardianPaymentObject::StakeUnlock { amount: 3 * ONE }
        );
        assert_eq!(
            object(&parse(&["jetsam-cli", "stake", "claim", "2"]), 2 * ONE),
            GuardianPaymentObject::RewardClaim {
                account: "delegator".into(),
                amount: 2 * ONE
            }
        );
        assert_eq!(
            object(&parse(&["jetsam-cli", "stake", "claim", "2", "--guardian"]), 2 * ONE),
            GuardianPaymentObject::RewardClaim {
                account: "guardian".into(),
                amount: 2 * ONE
            }
        );
        let payout = parse(&["jetsam-cli", "stake", "payout", "7"]);
        assert!(matches!(payout, StakeCommand::Payout { entry_id: 7 }));
        assert!(stake_request(&payout, 0).is_none(), "a payout needs no key");
        assert!(matches!(
            parse(&["jetsam-cli", "stake", "info"]),
            StakeCommand::Info { address: None }
        ));
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::{
        build_refusal, deposit_preflight, deposits_closed_reason, deposits_line, is_confirmation,
        unknown_outcome_message, BridgeCommand, Cli, Command, RpcCallError,
        BRIDGE_DEPOSIT_RPC_TIMEOUT,
    };
    use clap::{CommandFactory, Parser};
    use jetsam_chain::consensus::vault::{
        bridge_deposits_from_height, VaultRules, MAX_DEPOSIT_MICRO, MIN_DEPOSIT_MICRO,
    };
    use serde_json::json;

    const ONE: u64 = 1_000_000;
    const RECIPIENT: [u8; 20] = [0x5a; 20];
    const ADDRESS: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";

    fn armed(j: u64) -> VaultRules {
        let mut rules = VaultRules::current();
        rules.activation_height = Some(j);
        rules.deposits_from_height = bridge_deposits_from_height(Some(j));
        rules
    }

    fn unarmed() -> VaultRules {
        let mut rules = VaultRules::current();
        rules.activation_height = None;
        rules.deposits_from_height = None;
        rules
    }

    /// A `getVaultState` reply at `tip` with the opening `f` and `room` left.
    fn state(active: bool, tip: u64, f: Option<u64>, room: u64) -> serde_json::Value {
        json!({
            "active": active, "tip_height": tip, "bridge_locked": 10_000 * ONE - room,
            "ceiling": 10_000 * ONE, "room": room, "floor": 95, "deposits_from_height": f,
            "released_count": 0, "released_in_window": 0, "window_blocks": 480,
            "window_cap": 2_500 * ONE, "deposits_by_height": [], "deposits_last_1024": 0,
            "held_releases": 0, "deposit_heights_unread": 0,
        })
    }

    #[test]
    fn the_bridge_commands_parse() {
        let parse = |args: &[&str]| match Cli::try_parse_from(args).expect("parses").cmd {
            Command::Bridge(command) => command,
            _ => panic!("not a bridge command"),
        };
        match parse(&["jetsam-cli", "bridge", "deposit", ADDRESS, "12.5"]) {
            BridgeCommand::Deposit {
                recipient,
                amount,
                fee,
                dry_run,
                yes,
            } => {
                assert_eq!(recipient, ADDRESS);
                assert_eq!(amount, "12.5");
                assert_eq!(fee, None);
                assert!(
                    !dry_run && !yes,
                    "neither a dry run nor confirmed by default"
                );
            }
            _ => panic!("not a deposit"),
        }
        assert!(matches!(
            parse(&[
                "jetsam-cli", "bridge", "deposit", ADDRESS, "3", "--fee", "0.05", "--yes",
                "--dry-run"
            ]),
            BridgeCommand::Deposit { fee: Some(fee), yes: true, dry_run: true, .. } if fee == "0.05"
        ));
        assert!(matches!(
            parse(&["jetsam-cli", "bridge", "status"]),
            BridgeCommand::Status
        ));
        assert!(
            Cli::try_parse_from(["jetsam-cli", "bridge", "deposit", ADDRESS]).is_err(),
            "an amount is required"
        );
    }

    /// While the build's v1.7 height is none the CLI sends nothing and says
    /// why, before it asks the node anything; on the public network it says
    /// so in as many words.
    #[test]
    fn this_build_refuses_every_deposit_while_its_v1_7_height_is_none() {
        let mainnet = build_refusal(&unarmed(), false).expect("refused");
        assert!(mainnet.contains("mainnet"), "{mainnet}");
        assert!(mainnet.contains("not armed"), "{mainnet}");
        assert!(
            !mainnet.contains("Nothing was sent"),
            "a reason, read by bridge status too: {mainnet}"
        );
        let testnet = build_refusal(&unarmed(), true).expect("refused");
        assert!(!testnet.contains("mainnet"), "{testnet}");
        assert!(testnet.contains("not armed"), "{testnet}");
        assert_eq!(build_refusal(&armed(9_600), false), None);
        assert_eq!(build_refusal(&armed(9_600), true), None);
        let current = build_refusal(
            &VaultRules::current(),
            jetsam_chain::consensus::identity::IS_TEST_CHAIN,
        );
        assert_eq!(
            current.is_some(),
            jetsam_chain::consensus::params::V1_7_ACTIVATION_HEIGHT.is_none()
        );
    }

    /// The node is asked before anything is planned: not active at its next
    /// block, an opening other than this CLI's, or not open yet — each said
    /// in clear; open from the block after F.
    #[test]
    fn deposits_are_open_only_when_the_node_and_this_cli_agree() {
        let rules = armed(9_600);
        let f = rules.deposits_from_height.unwrap();
        let room = 10_000 * ONE;
        assert_eq!(
            deposits_closed_reason(&state(true, f, Some(f), room), &rules),
            None
        );
        assert_eq!(
            deposits_closed_reason(&state(true, f + 500, Some(f), room), &rules),
            None
        );
        let not_yet = deposits_closed_reason(&state(true, f - 1, Some(f), room), &rules)
            .expect("closed at F");
        assert!(
            not_yet.contains(&format!("{}", f + 1)) && not_yet.contains("1 more block"),
            "{not_yet}"
        );
        let below_j = deposits_closed_reason(&state(false, 100, Some(f), room), &rules)
            .expect("closed below J");
        assert!(below_j.contains("not active"), "{below_j}");
        let other = deposits_closed_reason(&state(true, f + 5, Some(f + 960), room), &rules)
            .expect("disagree");
        assert!(other.contains("disagree"), "{other}");
        let none = deposits_closed_reason(&state(true, f + 5, None, room), &rules)
            .expect("the node has no F");
        assert!(
            none.contains("disagree") || none.contains("not armed"),
            "{none}"
        );
        let this_build = deposits_closed_reason(&state(true, f + 5, Some(f), room), &unarmed())
            .expect("this build is not armed");
        assert!(this_build.contains("not armed"), "{this_build}");
    }

    /// The amount: within the limits of one deposit and within the vault's
    /// room (V8, as the node reports it); the next block's height otherwise.
    #[test]
    fn a_deposit_is_checked_against_the_limits_and_the_room_before_it_is_planned() {
        let rules = armed(9_600);
        let f = rules.deposits_from_height.unwrap();
        let open = state(true, f + 3, Some(f), 50 * ONE);
        assert_eq!(
            deposit_preflight(&open, 50 * ONE, &RECIPIENT, &rules),
            Ok(f + 4)
        );
        let full = deposit_preflight(&open, 50 * ONE + 1, &RECIPIENT, &rules).unwrap_err();
        assert!(full.contains("room"), "{full}");
        let wide = state(true, f + 3, Some(f), 10_000 * ONE);
        let small =
            deposit_preflight(&wide, MIN_DEPOSIT_MICRO - 1, &RECIPIENT, &rules).unwrap_err();
        assert!(small.contains("minimum"), "{small}");
        let large =
            deposit_preflight(&wide, MAX_DEPOSIT_MICRO + 1, &RECIPIENT, &rules).unwrap_err();
        assert!(large.contains("maximum"), "{large}");
        let zero = deposit_preflight(&wide, 10 * ONE, &[0u8; 20], &rules).unwrap_err();
        assert!(zero.contains("zero"), "{zero}");
        let closed = deposit_preflight(
            &state(true, f - 1, Some(f), 50 * ONE),
            5 * ONE,
            &RECIPIENT,
            &rules,
        );
        assert!(closed.is_err());
    }

    /// `bridge status` counts the deposits of the heights the node still
    /// retains, and says how many it no longer does (the rehearsal's node
    /// counted 0 over 1 024 blocks with 308 locked: 845 heights unread).
    #[test]
    fn the_status_says_which_deposits_it_did_not_count() {
        let mut node = state(true, 986, Some(128), 9_692 * ONE);
        node["deposits_last_1024"] = json!(0);
        node["deposit_heights_unread"] = json!(845);
        let line = deposits_line(&node);
        assert!(
            line.contains("845") && line.contains("not counted"),
            "{line}"
        );
        node["deposits_last_1024"] = json!(3);
        node["deposit_heights_unread"] = json!(0);
        assert_eq!(deposits_line(&node), "3 in the last 1024 blocks");
    }

    /// The deposit call proves the deposit in the node's wallet: measured
    /// over the client's 60 s on this machine at load 560 (the call was cut,
    /// the node dropped the deposit). It gets ten minutes, and a call that
    /// still gets no answer is said to have an unknown outcome — never "the
    /// node is not responding" alone, which invites sending a second deposit.
    /// A refusal of the node is a refusal.
    #[test]
    fn a_deposit_without_an_answer_has_an_unknown_outcome() {
        assert!(BRIDGE_DEPOSIT_RPC_TIMEOUT >= std::time::Duration::from_secs(600));
        let cut = anyhow::anyhow!("Node is not responding.\n\tIs the jetsam daemon running?");
        let message = unknown_outcome_message(&cut).expect("no answer");
        assert!(
            message.contains("unknown")
                && message.contains("twice")
                && message.contains("bridge status"),
            "{message}"
        );
        let refused = anyhow::Error::new(RpcCallError {
            code: -32000,
            message: "deposit refused by the mempool: BelowMinFee".into(),
            data: None,
        });
        assert_eq!(unknown_outcome_message(&refused), None);
    }

    /// Only the word itself confirms a deposit: not `y`, not an empty line.
    #[test]
    fn only_yes_confirms_a_deposit() {
        for answer in ["yes", "YES", "Yes", " yes\n"] {
            assert!(is_confirmation(answer), "{answer:?}");
        }
        for answer in ["", "\n", "y", "Y", "no", "oui", "yes please", "ye"] {
            assert!(!is_confirmation(answer), "{answer:?}");
        }
    }

    /// Public text: the bridge's help and messages say Base/EVM, never the
    /// network's name nor the two-letter name of the proof technique.
    #[test]
    fn the_bridge_texts_say_base_evm_only() {
        let mut command = Cli::command();
        command.build();
        let bridge = command
            .find_subcommand_mut("bridge")
            .expect("a bridge command");
        let mut help = bridge.render_long_help().to_string();
        for sub in bridge.get_subcommands_mut() {
            help.push_str(&sub.render_long_help().to_string());
        }
        assert!(help.contains("Base/EVM"), "{help}");
        let source = include_str!("jetsam_cli.rs");
        let code = source.split(&["#[cfg", "(test)]"].concat()).next().unwrap();
        let network = ["ether", "eum"].concat();
        let technique = ["z", "k"].concat();
        for text in [help.as_str(), code] {
            let lower = text.to_ascii_lowercase();
            assert!(!lower.contains(&network), "the network's name");
            assert!(
                !lower
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|word| word == technique),
                "the technique's name"
            );
        }
        let refusal = build_refusal(&unarmed(), false).unwrap();
        assert!(!refusal.to_ascii_lowercase().contains(&network));
    }
}
