use super::*;

#[test]
fn memory_is_locked_only_when_no_later_allocation_can_be_refused() {
    let cap_ipc_lock = 1u64 << 14;
    assert!(may_lock_all(libc::RLIM_INFINITY, 0), "unlimited");
    assert!(may_lock_all(8 << 20, cap_ipc_lock), "CAP_IPC_LOCK lifts the limit");
    assert!(may_lock_all(0, 0x1ff_ffff_ffff), "root's full set");
    assert!(!may_lock_all(8 << 20, 0), "8 MiB without the capability: a growing robot would abort");
    assert!(!may_lock_all(8 << 20, !cap_ipc_lock), "every capability but CAP_IPC_LOCK");
}

// the hardening itself runs in a child process: `tests/fill_start.rs` (no unit test spawns a process)
