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

/// E-2 (audit 2026-11-10): a process holding a hot key starts only with its memory locked, unless the operator's
/// explicit flag (an ALERT then); without a hot key, nothing changes.
#[test]
fn a_hot_key_is_held_in_locked_memory_only_unless_the_operator_says_so() {
    let locked = Hardened { dumpable_off: true, core_off: true, memory_locked: Ok(()) };
    let unlocked = Hardened { memory_locked: Err("RLIMIT_MEMLOCK is 8388608 bytes without CAP_IPC_LOCK".into()), ..locked.clone() };
    assert_eq!(memory_gate(&locked, true, false).unwrap(), None);
    assert_eq!(memory_gate(&locked, true, true).unwrap(), None);
    assert_eq!(memory_gate(&unlocked, false, false).unwrap(), None, "no hot key");
    let e = memory_gate(&unlocked, true, false).unwrap_err().to_string();
    assert!(e.contains("holds a hot key") && e.contains("its memory is not locked (RLIMIT_MEMLOCK") && e.contains(ALLOW_UNLOCKED_MEMORY), "{e}");
    let alert = memory_gate(&unlocked, true, true).unwrap().unwrap();
    assert!(alert.starts_with("ALERT MEMORY_NOT_LOCKED (RLIMIT_MEMLOCK") && alert.contains(ALLOW_UNLOCKED_MEMORY), "{alert}");
    assert_eq!(ALLOW_UNLOCKED_MEMORY, "--allow-unlocked-memory");
}
