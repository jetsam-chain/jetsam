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

#[test]
fn public_values_are_608_bytes_and_match_the_contract_encoding() {
    let mn = Mn {
        tip_block_work: be(16),
        fork_base_timestamp: 1_700_000_000,
        fork_base_work: be(32),
        segment_work: be(48),
        max_timestamp: 1_700_000_500,
    };
    let pv = public_values(&rep(0x66), &vector_checkpoint(), 2, 101, &mn);
    assert_eq!(pv.len(), 608);
    assert_eq!(
        hex::encode(keccak256(&pv)),
        "51cf9a458094b543441134a8b17abe7e11594ff324dbf3a023ff992565d8c260"
    );
    assert_eq!(
        hex::encode(mock_proof(&pv)),
        "0ecc01b9aba9e13b336daa5024f4013cff08d5e4653411ab9cd7ff1bb49667de"
    );
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
