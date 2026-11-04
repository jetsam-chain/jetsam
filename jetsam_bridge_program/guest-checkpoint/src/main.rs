// SPDX-License-Identifier: Apache-2.0
//! SP1 6.10.0 guest (standard SP1, no custom chip): the bridge CHECKPOINT program. Verifies a Jetsam
//! header segment, the block anchors it makes final and the vault deposits of whole blocks under V4 —
//! the node's own rule (`jetsam_vault_rule`, G6 = V4) — against the previous bridge checkpoint, and
//! commits the 320-byte public values of the v2 frame (the v3 frame is not written yet, see
//! `jetsam_bridge_program::spv`).
#![no_main]
sp1_zkvm::entrypoint!(main);

use tw_core::towerwalk::CELLS;

static mut PAD: [u64; CELLS] = [0u64; CELLS];

fn keccak_permute(st: &mut [u64; 25]) {
    sp1_zkvm::syscalls::syscall_keccak_permute(st as *mut [u64; 25]);
}

#[allow(static_mut_refs)]
pub fn main() {
    let input = sp1_zkvm::io::read_vec();
    let out = spv_core::spv::run(&input, keccak_permute, unsafe { &mut PAD[..] });
    sp1_zkvm::io::commit_slice(&out.public_values);
}
