use super::*;

fn rep(b: u8) -> [u8; 32] {
    [b; 32]
}

fn be(x: u64) -> [u8; 32] {
    w64(x)
}

/// The checkpoint of the vectors computed with `cast abi-encode` / `cast keccak`
/// (foundry 1.4.4) on 2026-10-31, independently of this crate.
fn vector_checkpoint() -> Checkpoint {
    Checkpoint {
        height: 130,
        cum_work: be(0x1234),
        headers_digest: rep(0x11),
        deposit_root: rep(0x22),
        deposit_count: 3,
        anchor_root: rep(0x33),
        locked_height: 122,
        tip_block_id: rep(0x44),
        locked_block_id: rep(0x55),
        locked_cum_work: be(0x1200),
    }
}

#[test]
fn checkpoint_id_matches_the_contract_encoding() {
    assert_eq!(
        hex::encode(vector_checkpoint().id()),
        "2df73af4b55f7e2de67105a20ca49a24f1614695999835c74c921f120911cc41"
    );
}

/// A5-4 and the re-audit's N1: 23 words, the bridge's tip block (0x88…) and `tip_on_branch`, then the
/// bridge tip's locked block (0x99…) and `locked_on_branch` last. Vectors: `cast abi-encode` of the 23
/// static values then `cast keccak` (foundry 1.4.4), 2026-11-05 (the same method gives the 21-word vector
/// of A5-4, f6ba8aeb…, again).
#[test]
fn public_values_are_736_bytes_and_match_the_contract_encoding() {
    let mn = Mn {
        tip_block_work: be(16),
        fork_base_timestamp: 1_700_000_000,
        fork_base_work: be(32),
        segment_work: be(48),
        max_timestamp: 1_700_000_500,
        tip_on_branch: true,
        locked_on_branch: true,
    };
    let pv = public_values(&rep(0x66), &vector_checkpoint(), 2, 101, &rep(0x88), &rep(0x99), &mn);
    assert_eq!(pv.len(), 736);
    assert_eq!(pv[608..640], rep(0x88));
    assert_eq!(pv[640..672], be(1));
    assert_eq!(pv[672..704], rep(0x99));
    assert_eq!(pv[704..736], be(1));
    assert_eq!(
        hex::encode(keccak256(&pv)),
        "f7f6738603751a4880913ee71517ce0bc3652a020a8f94b42901d6b6f75c9696"
    );
    assert_eq!(
        hex::encode(mock_proof(&pv)),
        "c9a197947d17015c118a046e21c114a333842ac602e5a894eca78a84631c6734"
    );
    let off = public_values(&rep(0x66), &vector_checkpoint(), 2, 101, &rep(0x88), &rep(0x99), &Mn { tip_on_branch: false, ..mn });
    assert_eq!((off[..640] == pv[..640], off[640..672] == [0u8; 32], off[672..] == pv[672..]), (true, true, true));
    let unlocked = public_values(&rep(0x66), &vector_checkpoint(), 2, 101, &rep(0x88), &rep(0x99), &Mn { locked_on_branch: false, ..mn });
    assert_eq!(hex::encode(keccak256(&unlocked)), "6c8de2aa7d907888213924c12bd1e1bb45e277551180588aaf3345b25e2482e4");
    assert_eq!((unlocked[..704] == pv[..704], unlocked[704..736] == [0u8; 32]), (true, true));
}

#[test]
fn deposit_leaf_matches_the_contract_encoding() {
    let mut recipient = [0u8; 20];
    recipient[19] = 0xa1;
    let d = Deposit {
        index: 5,
        recipient,
        amount: 100_000_000,
        jetsam_tx_id: rep(0x77),
    };
    assert_eq!(
        hex::encode(deposit_leaf(&d)),
        "40b87546a824153aa241eba98cd8cf73c2e21747d8ee55e2b05772b08515f8a3"
    );
}

#[test]
fn cosign_statement_describes_locked_block_and_anchor_ends_at_it() {
    let mut bridge = [0u8; 20];
    bridge[0] = 0xee;
    bridge[1] = 0x8b;
    bridge[2] = 0x5f;
    bridge[3] = 0xca;
    bridge[18] = 0x3e;
    bridge[19] = 0x2f;
    let domain = Domain {
        network: rep(0x88),
        l1_chain_id: 31_337,
        l1_bridge: bridge,
    };
    let cp = vector_checkpoint();
    let s = cp.statement(&domain, 1_700_003_600);
    assert_eq!(s.height, cp.locked_height, "the locked block, never the tip");
    assert_eq!(s.block_id, cp.locked_block_id);
    assert_eq!(s.cum_work, cp.locked_cum_work);
    assert_eq!(s.headers_digest, cp.headers_digest);
    assert_eq!(s.anchor_root, cp.anchor_root);
    assert_eq!(s.deposit_root, cp.deposit_root);
    assert_eq!(s.deposit_count, cp.deposit_count);
    assert!(domain.matches(&s));
    // the bytes the contract builds (cast abi-encode of the 13 words)
    assert_eq!(
        hex::encode(s.message()),
        "3ac121013ff196d0b772bc0128ec556a2c956778d93f7280c8c28a3f14003c60"
    );
}
