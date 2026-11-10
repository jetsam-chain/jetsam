// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The RPC cookie (audit 2026-11-10, R-1, R-2).
//!
//! The node's wallet methods, `stop` and `registerClient` answer only to
//! `Authorization: Bearer <cookie>`: anything on this machine could otherwise
//! spend the wallet and stop the node through the loopback RPC, and the
//! `--mining-key` bearer handed to a pool's miners opened the same methods.
//! The cookie is new at every start and written in the data directory,
//! readable by the node's user only (0600 on Unix; on Windows the file keeps
//! the ACL of its directory), as bitcoind's `.cookie`: whoever can read it
//! can already read `wallet.key` beside it.

use std::io::Write;
use std::path::{Path, PathBuf};

use rand::RngCore;

/// The cookie's file name in the data directory.
pub const RPC_COOKIE_FILE: &str = "rpc.cookie";

/// The cookie file of the node whose data directory is `data_dir`.
pub fn path_in(data_dir: &Path) -> PathBuf {
    data_dir.join(RPC_COOKIE_FILE)
}

/// The cookie file of a node on its default data directory (`~/.jetsam/data`,
/// the node's `config::default_data_dir` with `~` expanded as the node
/// expands it), for a client on the same machine.
pub fn default_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(
        PathBuf::from(home)
            .join(format!(
                ".{}",
                jetsam_chain::consensus::identity::DATA_DIR_NAME
            ))
            .join("data")
            .join(RPC_COOKIE_FILE),
    )
}

/// Write a new cookie (256 random bits, in hex) in `data_dir`, replacing the
/// previous one, and return it. Written beside it first, created by this call
/// only (never through a file or link already there), then renamed over it.
pub fn write_new(data_dir: &Path) -> std::io::Result<String> {
    let mut secret = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut secret);
    let cookie = hex::encode(secret);
    let staging = data_dir.join(format!("{RPC_COOKIE_FILE}.new"));
    match std::fs::remove_file(&staging) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&staging)?;
    file.write_all(cookie.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&staging, path_in(data_dir))?;
    Ok(cookie)
}

/// The cookie in the file at `path` (surrounding whitespace ignored); an
/// empty file is no cookie.
pub fn read(path: &Path) -> std::io::Result<String> {
    let cookie = std::fs::read_to_string(path)?.trim().to_owned();
    if cookie.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} is empty", path.display()),
        ));
    }
    Ok(cookie)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new cookie at every start, 256 bits in hex, readable back, and on
    /// Unix readable by this user only.
    #[test]
    fn a_new_cookie_is_written_at_every_start_for_this_user_only() {
        let directory = tempfile::tempdir().unwrap();
        let first = write_new(directory.path()).unwrap();
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(read(&path_in(directory.path())).unwrap(), first);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path_in(directory.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        // A stale staging file (a crash between write and rename) is replaced,
        // never written through.
        std::fs::write(directory.path().join("rpc.cookie.new"), "stale").unwrap();
        let second = write_new(directory.path()).unwrap();
        assert_ne!(second, first, "the cookie changes at every start");
        assert_eq!(read(&path_in(directory.path())).unwrap(), second);
        assert!(!directory.path().join("rpc.cookie.new").exists());
    }

    /// A client finds the cookie of a node on its default data directory.
    #[test]
    fn the_default_cookie_is_in_the_default_data_directory() {
        let path = default_path().expect("HOME is set");
        let expected = std::path::Path::new(&format!(
            ".{}",
            jetsam_chain::consensus::identity::DATA_DIR_NAME
        ))
        .join("data")
        .join("rpc.cookie");
        assert!(path.ends_with(&expected), "{}", path.display());
    }

    #[test]
    fn a_cookie_is_read_without_its_whitespace_and_an_empty_file_is_none() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cookie");
        std::fs::write(&path, "abc123\n").unwrap();
        assert_eq!(read(&path).unwrap(), "abc123");
        std::fs::write(&path, " \n").unwrap();
        assert!(read(&path).is_err());
        assert!(read(&directory.path().join("missing")).is_err());
    }
}
