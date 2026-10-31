use super::*;
use crate::checkpoint::w64;

fn rep(b: u8) -> [u8; 32] {
    [b; 32]
}

fn cp() -> Checkpoint {
    Checkpoint {
        height: 130,
        cum_work: w64(4660),
        headers_digest: rep(0x11),
        deposit_root: rep(0x22),
        deposit_count: 3,
        anchor_root: rep(0x33),
        locked_height: 122,
        tip_block_id: rep(0x44),
        locked_block_id: rep(0x55),
        locked_cum_work: w64(4608),
    }
}

fn mn() -> Mn {
    Mn { tip_block_work: w64(16), fork_base_timestamp: 1_700_000_000, fork_base_work: w64(32), segment_work: w64(48), max_timestamp: 1_700_000_500 }
}

fn k(data: &[u8]) -> String {
    format!("0x{}", hex::encode(keccak256(data)))
}

/// Vectors: `cast calldata … | cast keccak` (foundry 1.4.4), 2026-10-31.
#[test]
fn call_data_matches_cast() {
    let sig = [0x01u8; 256];
    assert_eq!(k(&finalize_fast(&cp(), 1_700_003_600, &sig)), "0xbacc1b0bffa6f486f9b7171e9093da971bbc5d33ad8143f12a84c893f4b812f2");
    assert_eq!(k(&init_genesis(&cp(), 1_700_003_600, &sig)), "0x8259e4e8f58e5ec3f0aab1a98049a7f48cb7e649dcb5b3e3feffc697a7547d0d");
    assert_eq!(k(&submit_checkpoint(&rep(0x66), &cp(), 2, &mn(), &rep(0x77))), "0xede642940001a7f2e8aab237d97a9b7370a14ebef378511f0b3193b89f60898c");
    assert_eq!(
        k(&submit_and_finalize_fast(&rep(0x66), &cp(), 2, &mn(), &rep(0x77), 1_700_003_600, &sig)),
        "0x0e0c963853ddb8e7f3431c50c0a69dc8dd8299571e697e6dcb664712fcbc1b35"
    );
    let mut recipient = [0u8; 20];
    recipient[19] = 0xa1;
    let d = Deposit { index: 5, recipient, amount: 100_000_000, jetsam_tx_id: rep(0x77) };
    assert_eq!(k(&claim_deposit(&d, &rep(0x22), &[rep(0x99); 32])), "0x1240465a89adf79f3500a1e98acd0603de33d94a3d3276303a0fee55c717298a");
    assert_eq!(k(&prove_equivocation(&[1, 2], &[3, 4, 5])), "0xd8d63462be65030b631fc8509b0a39f2ae1eb09d7a65a1680af3bbde414801a3");
    assert_eq!(k(&burn(30_000_000, &rep(0x44))), "0xe7217e0d32d486503751b64fceaa7b8cb8edf546b2732245b1eec004e33fc104");
}
