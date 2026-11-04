// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Why a test that forks lives in a test binary of its own, never with the library's unit tests.
//!
//! The signer journal refuses a second instance by a `flock` on `<dir>/lock` (`Journal::open`). A `flock`
//! belongs to the OPEN FILE DESCRIPTION, and a forked child shares every description of its parent until
//! its `exec` closes the copy (the descriptor is `O_CLOEXEC`). So while ANY thread of a process forks, a
//! journal another thread of the same process drops and reopens is still locked by the child: "locked by
//! another signer instance". `cargo test` runs the library's unit tests as threads of one process: one
//! test that spawned a child process (`setpriv`, the SO_PEERCRED test of `ipc`) made the journal tests
//! fail at random. Test binaries run one after another, so a test that forks is safe in its own file.
//!
//! This binary holds this one test only: its own fork is the only one.

use std::os::unix::process::CommandExt;
use std::sync::mpsc;
use std::time::Duration;

use jetsam_guardian::journal::Journal;

#[test]
fn a_forked_child_holds_the_journal_lock_until_it_execs() {
    let dir = tempfile::tempdir().unwrap();
    let j = Journal::open(dir.path()).unwrap();
    // a thread forks now; the child waits 2 s before its exec (`pre_exec` runs in the child, between them)
    let (forked, at_fork) = mpsc::channel();
    let spawner = std::thread::spawn(move || {
        let mut cmd = std::process::Command::new("true");
        unsafe {
            cmd.pre_exec(|| {
                std::thread::sleep(Duration::from_secs(2));
                Ok(())
            });
        }
        forked.send(()).unwrap();
        // returns once the child has exec'ed (std waits for the exec on a close-on-exec pipe)
        cmd.status().unwrap()
    });
    at_fork.recv().unwrap();
    std::thread::sleep(Duration::from_millis(500));
    // the child (not exec'ed yet) shares the lock's open file description: dropping ours keeps it locked
    drop(j);
    let e = Journal::open(dir.path()).err().expect("still locked by the child of this very process");
    assert!(e.to_string().contains("locked by another signer instance"), "{e}");
    assert!(spawner.join().unwrap().success());
    // the child's exec closed its copy: the lock is free
    assert!(Journal::open(dir.path()).is_ok(), "released at the child's exec");
}
