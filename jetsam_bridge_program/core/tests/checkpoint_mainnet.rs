// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The checkpoint relation with its v3 G6 on REAL mainnet data, natively: the 12 headers 44 570 to
//! 44 581 on the checkpoint at 44 569 (the v2 prototype's input `main-v2-12`), plus the whole real
//! blocks of its deposit window `]44 561 ; 44 573]` it kept (44 564 to 44 573: 89 pages, read from the
//! public RPC on 26/10/2026), as `tests/fixtures/checkpoint-mainnet-44570-44581.bin` (input `JSPVIN04`).
//!
//! No deposit exists on the mainnet, so the public values must be exactly those the v2 guest committed
//! for the same headers in the SP1 executor (`.pv.hex`): reading whole blocks adds nothing and changes
//! nothing, and a block not given changes nothing. Each block refuses alone when it is not the body its
//! header commits to, or outside the window, or out of order.

use jetsam_bridge_program::keccak::keccak_f_soft;
use jetsam_bridge_program::{deposit, rules, spv};
use tw_core::towerwalk::CELLS;

const INPUT: &[u8] = include_bytes!("fixtures/checkpoint-mainnet-44570-44581.bin");
const PV_HEX: &str = include_str!("fixtures/checkpoint-mainnet-44570-44581.pv.hex");

/// The fixture split: everything before the deposit blocks, then `(height, pages)` of each block.
fn split() -> (Vec<u8>, Vec<(u64, Vec<Vec<u8>>)>) {
    let mut o = 8 + 8 + 8 + 32 + 32 + 8 + 32 * 32 + 32 + 32 * 32 + 1;
    let u32_at = |o: usize| u32::from_le_bytes(INPUT[o..o + 4].try_into().unwrap()) as usize;
    o += 4 + u32_at(o) * 212;
    o += 4 + u32_at(o) * 212;
    let head = INPUT[..o].to_vec();
    let n_blk = u32_at(o);
    o += 4;
    let mut blocks = Vec::new();
    for _ in 0..n_blk {
        let height = u64::from_le_bytes(INPUT[o..o + 8].try_into().unwrap());
        let n = u32_at(o + 8);
        o += 12;
        blocks.push((height, (0..n).map(|i| INPUT[o + i * 323..o + (i + 1) * 323].to_vec()).collect()));
        o += n * 323;
    }
    assert_eq!(o, INPUT.len());
    (head, blocks)
}

fn input(head: &[u8], blocks: &[(u64, Vec<Vec<u8>>)]) -> Vec<u8> {
    let mut v = head.to_vec();
    v.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for (h, pages) in blocks {
        v.extend_from_slice(&h.to_le_bytes());
        v.extend_from_slice(&(pages.len() as u32).to_le_bytes());
        pages.iter().for_each(|p| v.extend_from_slice(p));
    }
    v
}

/// The public values, or the message of the rule that refused.
fn run(input: &[u8]) -> Result<Vec<u8>, String> {
    std::panic::catch_unwind(|| {
        let mut pad = vec![0u64; CELLS];
        let out = spv::run_with(input, keccak_f_soft, &mut pad, &rules::PRODUCTION, &deposit::DEPOSIT_RULE);
        assert!(out.new_deposits.is_empty(), "no deposit on the mainnet");
        out.public_values.to_vec()
    })
    .map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn whole_real_blocks_leave_the_v2_public_values_unchanged() {
    std::panic::set_hook(Box::new(|_| {}));
    let (head, blocks) = split();
    assert_eq!(blocks.iter().map(|(h, _)| *h).collect::<Vec<_>>(), (44_564..=44_573).collect::<Vec<_>>());
    assert_eq!(blocks.iter().map(|(_, p)| p.len()).sum::<usize>(), 89);
    let pv = PV_HEX.trim();
    assert_eq!(run(INPUT).map(|v| hex(&v)).as_deref(), Ok(pv), "the ten blocks");
    assert_eq!(run(&input(&head, &[])).map(|v| hex(&v)).as_deref(), Ok(pv), "no block");
    assert_eq!(run(&input(&head, &blocks[3..5])).map(|v| hex(&v)).as_deref(), Ok(pv), "two blocks");
    let _ = std::panic::take_hook();
}

#[test]
fn a_block_that_is_not_its_headers_or_out_of_the_window_or_order_refuses() {
    std::panic::set_hook(Box::new(|_| {}));
    let (head, blocks) = split();
    let refuse = |b: &[(u64, Vec<Vec<u8>>)]| run(&input(&head, b)).unwrap_err();
    // a page changed, a page left out, a page added
    let mut b = blocks.clone();
    b[6].1[8][100] ^= 1;
    assert_eq!(refuse(&b), "tx_root");
    let mut b = blocks.clone();
    b[6].1.remove(0);
    assert!(refuse(&b).starts_with("block body") || refuse(&b) == "tx_root", "{}", refuse(&b));
    let mut b = blocks.clone();
    let extra = b[2].1[0].clone();
    b[3].1.push(extra);
    assert!(refuse(&b).starts_with("block body"), "{}", refuse(&b));
    // the pages of one block under the height of another
    let mut b = blocks.clone();
    b[0].0 = 44_565;
    b.remove(1);
    assert_eq!(refuse(&b), "tx_root");
    // the window ]prev.height - 8 ; height - 8] = ]44 561 ; 44 573]
    assert_eq!(refuse(&[(44_561, blocks[0].1.clone())]), "deposit block not final in window");
    assert_eq!(refuse(&[(44_574, blocks[0].1.clone())]), "deposit block not final in window");
    assert_eq!(refuse(&[(44_562, blocks[0].1.clone())]), "tx_root", "inside the window, another block's pages");
    // chain order, each block once
    assert_eq!(refuse(&[blocks[2].clone(), blocks[1].clone()]), "deposit block order");
    assert_eq!(refuse(&[blocks[2].clone(), blocks[2].clone()]), "deposit block order");
    let _ = std::panic::take_hook();
}
