// SPDX-License-Identifier: Apache-2.0
//! SP1 6.10.0 guest (standard SP1, no custom chip): the bridge DEPOSIT-CLAIM program v3 (individual
//! path). Proves that the group `group` of the Jetsam block at `height`, given whole, is a vault deposit
//! under V4 — the node's own rule (`jetsam_vault_rule`) — with the recipient written in its marker, and
//! commits the 192-byte public values of `JetsamBridge.depositClaimValues`. The contract checks that
//! this block is anchored by a final checkpoint.
#![no_main]
sp1_zkvm::entrypoint!(main);

fn keccak_permute(st: &mut [u64; 25]) {
    sp1_zkvm::syscalls::syscall_keccak_permute(st as *mut [u64; 25]);
}

pub fn main() {
    let input = sp1_zkvm::io::read_vec();
    let out = spv_core::claim::run_claim(&input, keccak_permute);
    sp1_zkvm::io::commit_slice(&out.public_values);
}
