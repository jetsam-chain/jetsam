// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A5-15: the hardening of a process that holds a secret in memory (the robots' hot keys): no core dump,
//! not readable by a debugger of the same user (`PR_SET_DUMPABLE = 0`), memory locked out of the swap when
//! that can never refuse a later allocation. As `jg-sign` does for its share (`jg-sign.rs`, `harden`).
//! E-2 (audit 2026-11-10): a robot that holds its hot key does not start with its memory not locked
//! ([`memory_gate`]), unless the operator says so ([`ALLOW_UNLOCKED_MEMORY`], an ALERT).

/// What [`harden`] obtained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hardened {
    /// `prctl(PR_SET_DUMPABLE, 0)` succeeded.
    pub dumpable_off: bool,
    /// `RLIMIT_CORE` is 0 (soft and hard).
    pub core_off: bool,
    /// `mlockall(MCL_CURRENT | MCL_FUTURE)`, or why not.
    pub memory_locked: Result<(), String>,
}

/// Whether `mlockall(MCL_FUTURE)` can never make a later allocation fail: the soft `RLIMIT_MEMLOCK` is
/// unlimited, or the process holds `CAP_IPC_LOCK` (bit 14 of `CapEff`), which lifts that limit. Under a
/// bounded limit a growing robot would abort on its first allocation above it: then the memory is not
/// locked (a warning), and the swap must be off or encrypted on that host.
pub fn may_lock_all(memlock_soft: u64, cap_eff: u64) -> bool {
    memlock_soft == libc::RLIM_INFINITY || cap_eff & (1 << 14) != 0
}

/// `CapEff` of this process (`/proc/self/status`), 0 when unreadable.
fn cap_eff() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("CapEff:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|h| u64::from_str_radix(h, 16).ok()))
        .unwrap_or(0)
}

/// E-2 (audit 2026-11-10): the command-line flag that starts a program holding a hot key with its memory not locked
/// (an ALERT then).
pub const ALLOW_UNLOCKED_MEMORY: &str = "--allow-unlocked-memory";

/// E-2: a process that holds a hot key (`holds_hot_key`) starts only with its memory locked (`h.memory_locked`):
/// an error otherwise (it does not start), unless `allow_unlocked` (the operator's [`ALLOW_UNLOCKED_MEMORY`]): then
/// `Some` ALERT line to log. `None`: nothing to say (memory locked, or no hot key).
pub fn memory_gate(h: &Hardened, holds_hot_key: bool, allow_unlocked: bool) -> anyhow::Result<Option<String>> {
    match &h.memory_locked {
        Err(why) if holds_hot_key && allow_unlocked => Ok(Some(format!(
            "ALERT MEMORY_NOT_LOCKED ({why}): started with {ALLOW_UNLOCKED_MEMORY}, the hot key may be written to swap (keep the swap off or encrypted)"
        ))),
        Err(why) if holds_hot_key => anyhow::bail!(
            "this process holds a hot key and its memory is not locked ({why}): raise RLIMIT_MEMLOCK to unlimited \
             (LimitMEMLOCK=infinity, or memlock unlimited in limits.conf) or grant CAP_IPC_LOCK; {ALLOW_UNLOCKED_MEMORY} \
             starts it anyway, with an ALERT"
        ),
        _ => Ok(None),
    }
}

/// Harden this process (see the module).
pub fn harden() -> Hardened {
    // SAFETY: prctl, setrlimit, getrlimit and mlockall on this process, with valid arguments.
    unsafe {
        let dumpable_off = libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) == 0;
        let zero = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        let core_off = libc::setrlimit(libc::RLIMIT_CORE, &zero) == 0;
        let mut memlock = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        let soft = if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut memlock) == 0 { memlock.rlim_cur } else { 0 };
        let memory_locked = if !may_lock_all(soft, cap_eff()) {
            Err(format!("RLIMIT_MEMLOCK is {soft} bytes without CAP_IPC_LOCK: memory not locked (keep the swap off or encrypted)"))
        } else if libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) == 0 {
            Ok(())
        } else {
            Err(format!("mlockall refused: {}", std::io::Error::last_os_error()))
        };
        Hardened { dumpable_off, core_off, memory_locked }
    }
}

#[cfg(test)]
#[path = "harden_tests.rs"]
mod tests;
