// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The program's hard checkpoints of the public network's v1.6 crossing (final audit F1): blocks 46 079 and
//! 46 080, pinned by the node since the v1.6.1 release (`params::HARD_CHECKPOINTS`), are pinned by the program
//! too (`rules::MAINNET`, step 0 of every new header in `spv.rs`), so a proof can never carry the branch a
//! node left on v1.5.1 went on mining from 46 080'. The raw headers are those the four mainnet seeds served on
//! 2026-10-30 (`jetsam_getHeaderByHeight`, the bytes of the v1.6.1 release's crossing test), hashed here by the
//! program's own `block_id`.

use jetsam_bridge_program::hdr::{block_id, Hdr, HEADER_WIRE_SIZE};
use jetsam_bridge_program::rules;

const CROSSING: [&str; 3] = [
    "3f3578cad1c7d8fb40f247ca4554de26322b06043310edf560a3a5fbde88633b0577d4b4374fbae064aeff79888a07fc02c27362706f10643f78d7b51de79d6e72cd2a5cd9127f0e41c8adbc59cd7d351435255fbed884e1e3f66c28318a8b4ba8c0e36a00000000ffb30000000000008f2bdf39af91926de8204283e0a01a661ef9292bb30792adb54295c6faa0607dc13602000000c00055586517aa06000007d9df5f3824cefb090b780852676292bf3bb3a537faa05e1abbdaed0f00000018000000b8e7000000000000a1d0010000000000",
    "cf00a7d007821cd38488ec025a78a456bfaf7f6958595606798e0de8a852a6934bcb98ca2bc921f4818762f135f9e29127e30601162271d438bbe708754c641c14414e912da7410ec1e25821cca329cf2791314ea4e6df62b83232a92156dd5830c1e36a0000000000b40000000000002f3c1654dd1288326987fa6cc82f8108b84ae31beb23e3907457cff6bc9de442174b010000007006f121269bf1f15403cbe1eb41a33bafc0c07dfeb3c3436a457006f79603ed262a8205beb90d00000018000000bbe7000000000000a4d0010000000000",
    "a2a91f26d8a14a7346f13df111851d5c703d3268a52f42ce9137dee818db585f46d2a82afc599d1303c63526bd61cb50288e9598bcf75e1802f7871a7ed68d6f4391ba49da6e7e7205303691bd34dba30e42ca3367deb84a7c7e25ea95c798fdb0c1e36a0000000001b400000000000005e12595e43723f0f90d3907f57276acd0c005bbafb1072ea0deb686e06224b3178a7038070000008005940da3091013c701c4dda1c8fdca2673be3b89e3b13b8dda1e86bb1968749669a1880d00000018000000bce7000000000000a5d0010000000000",
];

fn header(hex: &str) -> [u8; HEADER_WIRE_SIZE] {
    let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
    bytes.try_into().expect("a 212-byte wire header")
}

#[test]
fn the_public_networks_v1_6_crossing_blocks_are_pinned() {
    let [before, at, after] = CROSSING.map(header);
    assert_eq!((Hdr(&before).height(), Hdr(&at).height(), Hdr(&after).height()), (46_079, 46_080, 46_081));
    assert_eq!(Hdr(&at).prev(), block_id(&before), "one chain");
    assert_eq!(Hdr(&after).prev(), block_id(&at), "one chain");
    assert_eq!(rules::MAINNET.v1_6_activation, Some(46_080));
    assert_eq!(rules::MAINNET.hard_checkpoint(46_079), Some(block_id(&before)));
    assert_eq!(rules::MAINNET.hard_checkpoint(46_080), Some(block_id(&at)));
    assert!(rules::MAINNET.pins.windows(2).all(|p| p[0].0 < p[1].0), "pins in height order");
}

/// The other branch's block at 46 080, on the same 46 079: another id, so step 0 refuses it.
#[test]
fn a_header_at_46080_other_than_the_pinned_block_is_not_the_pin() {
    let mut theirs = header(CROSSING[1]);
    theirs[144] ^= 1; // the nonce (wire offset 144)
    assert_eq!(Hdr(&theirs).prev(), Hdr(&header(CROSSING[1])).prev(), "the same parent");
    assert_ne!(block_id(&theirs), block_id(&header(CROSSING[1])));
    let pin = rules::MAINNET.hard_checkpoint(46_080).expect("46 080 is pinned");
    assert_ne!(block_id(&theirs), pin);
    assert_eq!(rules::MAINNET.hard_checkpoint(46_081), None, "only the crossing pair");
}
