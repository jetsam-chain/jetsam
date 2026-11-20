//! D176: the deposits of a block of the return over Base are read with the node's rule as the node applies it.
//!
//! Block 9 155 of the public test chain (fixture: its bytes as the three test nodes served them by `getBlock`,
//! 2026-11-11) carries the fast release of burn 3: its release group pays the reserve that advanced the burn (the
//! advance's creditor, not the burn's owner) and carries the fast lane's marker (`fast_release_marker`, committed to
//! the attestation's Base block), not the slow release's. The nodes accepted it; the relayer and the robot read it
//! with the slow lane's marker and the burn's owner as payee, refused it (`ReleasesDoNotMatchMarkers`) and took it
//! for a missing body. It holds no deposit (`getDepositBlock` does not keep it, and the vault's balance after it is
//! the deposits less the releases).

use jetsam_chain::consensus::vault::VaultRules;
use jetsam_guardian::fill::deposits_of_block;

const BLOCK_9155: &[u8] = include_bytes!("fixtures/testnet-block-9155-fast-release.bin");
const BLOCK_9155_ID: &str = "2cc2fbf3210ff9dbf7053341b42a3cc556330efbd800f9568e77ba9dbda56659";

#[test]
fn a_fast_release_block_of_the_public_test_chain_is_read_and_holds_no_deposit() {
    let rules = VaultRules::current();
    if !jetsam_chain::consensus::vault::TESTNET_PROFILE || !rules.active_at(9_155) {
        // a block of the public test chain is read with its rule only: the default build has no v1.7 clock at this
        // height, and the rehearsal's clock (from 96) reads its own markers, not the public test chain's
        eprintln!("skipped: this build's v1.7 clock is not the public test chain's");
        return;
    }
    let id: [u8; 32] = hex::decode(BLOCK_9155_ID).unwrap().try_into().unwrap();
    let deposits = deposits_of_block(BLOCK_9155, &id, &rules).expect("the node accepted block 9155: its deposits are readable");
    assert!(deposits.is_empty(), "block 9155 holds no deposit, read {deposits:?}");
}
