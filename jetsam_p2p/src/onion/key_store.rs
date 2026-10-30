// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The persisted onion service key.
//!
//! Tor generates the key on the first `ADD_ONION NEW:ED25519-V3` and hands it
//! back once; it is written here, owner-only, so that every later start
//! publishes the same `.onion` address. The file is never rotated or
//! overwritten: a corrupt, foreign-owned, symlinked or group/world-readable
//! file stops the onion service instead of silently changing its address.
//!
//! Format (text, bounded):
//! ```text
//! JETSAM-ONION-KEYv1
//! <56-character service ID>
//! ED25519-V3:<base64 of the 64-byte expanded secret key>
//! ```

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use super::address::OnionServiceId;
use super::OnionError;

pub const ONION_KEY_FILE: &str = "onion_service_v3.key";
const MAGIC_LINE: &str = "JETSAM-ONION-KEYv1";
const MAX_KEY_FILE_BYTES: usize = 512;
pub(crate) const KEY_TYPE_PREFIX: &str = "ED25519-V3:";

/// A service key as Tor understands it, with the service ID it yields.
pub struct OnionKey {
    blob: String,
    service_id: OnionServiceId,
}

impl OnionKey {
    /// `key_blob` is `ED25519-V3:<base64>` exactly as Tor wrote it.
    pub fn new(key_blob: &str, service_id: OnionServiceId) -> Result<Self, OnionError> {
        if !is_valid_key_blob(key_blob) {
            return Err(OnionError::KeyStore(
                "the service key is not an ED25519-V3 base64 key".into(),
            ));
        }
        Ok(Self {
            blob: key_blob.to_string(),
            service_id,
        })
    }

    pub fn key_blob(&self) -> &str {
        &self.blob
    }

    pub fn service_id(&self) -> &OnionServiceId {
        &self.service_id
    }
}

impl Drop for OnionKey {
    fn drop(&mut self) {
        let mut bytes = std::mem::take(&mut self.blob).into_bytes();
        bytes.fill(0);
    }
}

impl fmt::Debug for OnionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnionKey")
            .field("service_id", &self.service_id)
            .field("blob", &"<redacted>")
            .finish()
    }
}

/// `ED25519-V3:` followed by the base64 of 64 bytes (86 symbols, optionally
/// padded with `==`): nothing that could break a control command line.
pub(crate) fn is_valid_key_blob(blob: &str) -> bool {
    let Some(body) = blob.strip_prefix(KEY_TYPE_PREFIX) else {
        return false;
    };
    let symbols = body.strip_suffix("==").unwrap_or(body);
    symbols.len() == BASE64_KEY_SYMBOLS
        && symbols
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/')
}

const BASE64_KEY_SYMBOLS: usize = 86;

fn store_error(path: &Path, what: &str, error: impl fmt::Display) -> OnionError {
    OnionError::KeyStore(format!("{what} {}: {error}", path.display()))
}

/// The stored key, `Ok(None)` when none was ever stored.
pub fn load(data_dir: &Path) -> Result<Option<OnionKey>, OnionError> {
    let path = data_dir.join(ONION_KEY_FILE);
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let file = match options.open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(store_error(&path, "open", error)),
    };
    validate_metadata(&path, &file)?;

    let mut bytes = Vec::with_capacity(MAX_KEY_FILE_BYTES + 1);
    let read = file
        .take((MAX_KEY_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes);
    if let Err(error) = read {
        bytes.fill(0);
        return Err(store_error(&path, "read", error));
    }
    let parsed = parse_key_file(&bytes)
        .map_err(|reason| store_error(&path, "refusing the onion service key", reason));
    bytes.fill(0);
    parsed.map(Some)
}

fn parse_key_file(bytes: &[u8]) -> Result<OnionKey, &'static str> {
    if bytes.len() > MAX_KEY_FILE_BYTES {
        return Err("file exceeds the bounded format");
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "file is not text")?;
    let body = text.strip_suffix('\n').ok_or("file does not end a line")?;
    let mut lines = body.split('\n');
    let (Some(magic), Some(service_id), Some(blob), None) =
        (lines.next(), lines.next(), lines.next(), lines.next())
    else {
        return Err("file does not hold exactly three lines");
    };
    if magic != MAGIC_LINE {
        return Err("invalid format marker");
    }
    let service_id = OnionServiceId::parse(service_id).map_err(|_| "invalid service ID")?;
    OnionKey::new(blob, service_id).map_err(|_| "invalid key")
}

/// Store a key Tor has just generated. Never overwrites an existing file.
pub fn store_new(data_dir: &Path, key: &OnionKey) -> Result<(), OnionError> {
    std::fs::create_dir_all(data_dir)
        .map_err(|error| store_error(data_dir, "create directory", error))?;
    let path = data_dir.join(ONION_KEY_FILE);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let mut file = options
        .open(&path)
        .map_err(|error| store_error(&path, "create (never overwritten)", error))?;
    let mut contents = format!(
        "{MAGIC_LINE}\n{}\n{}\n",
        key.service_id().as_string(),
        key.key_blob()
    )
    .into_bytes();
    let written = file.write_all(&contents).and_then(|()| file.sync_all());
    contents.fill(0);
    written.map_err(|error| store_error(&path, "persist", error))?;
    #[cfg(unix)]
    File::open(data_dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| store_error(data_dir, "sync directory", error))?;
    Ok(())
}

fn validate_metadata(path: &Path, file: &File) -> Result<(), OnionError> {
    let metadata = file
        .metadata()
        .map_err(|error| store_error(path, "stat", error))?;
    if !metadata.is_file() {
        return Err(store_error(path, "refusing", "not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let mode = metadata.permissions().mode() & 0o777;
        if mode != 0o600 {
            return Err(store_error(
                path,
                "refusing",
                format!("insecure permissions {mode:o}, expected 600"),
            ));
        }
        // SAFETY: geteuid has no preconditions and does not access pointers.
        let expected_owner = unsafe { libc::geteuid() };
        if metadata.uid() != expected_owner {
            return Err(store_error(
                path,
                "refusing",
                format!("owned by uid {}, expected {expected_owner}", metadata.uid()),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";
    // base64 of 64 synthetic bytes ((i * 37 + 11) mod 256), not a real key.
    pub(crate) const KEY_BLOB: &str = "ED25519-V3:CzBVep/E6Q4zWH2ix+wRNluApcrvFDleg6jN8hc8YYar0PUaP2SJrtP4HUJnjLHW+yBFao+02f4jSG2St9wBJg==";

    pub(crate) fn sample_key() -> OnionKey {
        OnionKey::new(KEY_BLOB, OnionServiceId::parse(SYNTHETIC).unwrap()).unwrap()
    }

    #[cfg(unix)]
    fn mode_of(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn onion_key_blob_shape_is_strict() {
        assert!(is_valid_key_blob(KEY_BLOB));
        let unpadded = KEY_BLOB.trim_end_matches('=');
        assert!(is_valid_key_blob(unpadded));
        let body = &KEY_BLOB[KEY_TYPE_PREFIX.len()..];
        for bad in [
            String::new(),
            body.to_string(),
            format!("RSA1024:{body}"),
            format!("{KEY_TYPE_PREFIX}{}", &body[1..]),
            format!("{KEY_TYPE_PREFIX}{body}="),
            format!("{KEY_TYPE_PREFIX}{body}\r\nDEL_ONION x"),
            format!("{KEY_TYPE_PREFIX}{} ", &body[..87]),
            format!("{KEY_TYPE_PREFIX}{}-=", &body[..85]),
            format!("{KEY_TYPE_PREFIX}={}", &body[..87]),
        ] {
            assert!(!is_valid_key_blob(&bad), "{bad:?} accepted");
        }
        assert!(OnionKey::new(
            "ED25519-V3:short",
            OnionServiceId::parse(SYNTHETIC).unwrap()
        )
        .is_err());
        assert!(!format!("{:?}", sample_key()).contains(body));
    }

    #[test]
    fn onion_key_is_stored_owner_only_and_reloaded_identically() {
        let directory = tempfile::tempdir().unwrap();
        assert!(load(directory.path()).unwrap().is_none());
        store_new(directory.path(), &sample_key()).unwrap();
        let path = directory.path().join(ONION_KEY_FILE);
        #[cfg(unix)]
        assert_eq!(mode_of(&path), 0o600);
        let loaded = load(directory.path()).unwrap().unwrap();
        assert_eq!(loaded.key_blob(), KEY_BLOB);
        assert_eq!(loaded.service_id().as_string(), SYNTHETIC);
        // Never rotated: a second store fails and leaves the first key.
        let before = std::fs::read(&path).unwrap();
        assert!(store_new(directory.path(), &sample_key()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn onion_key_store_creates_the_data_directory() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("a").join("b");
        store_new(&nested, &sample_key()).unwrap();
        assert!(load(&nested).unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn onion_key_load_fails_closed_on_permissions_symlinks_and_corruption() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        store_new(directory.path(), &sample_key()).unwrap();
        let path = directory.path().join(ONION_KEY_FILE);
        let good = std::fs::read_to_string(&path).unwrap();

        for mode in [0o640, 0o604, 0o644, 0o700] {
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            assert!(load(directory.path()).is_err(), "mode {mode:o} accepted");
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(load(directory.path()).is_ok());

        let write = |text: &str| {
            std::fs::write(&path, text).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        };
        let lines: Vec<&str> = good.lines().collect();
        let bad_files = [
            String::new(),
            good.replace(MAGIC_LINE, "JETSAM-ONION-KEYv2"),
            good.replace(
                SYNTHETIC,
                "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3iead",
            ),
            format!("{}\n{}\n{}\r\nDEL_ONION x\n", lines[0], lines[1], lines[2]),
            format!("{}\n{}\n", lines[0], lines[1]),
            format!("{good}extra\n"),
            format!("{good}{}", "x".repeat(MAX_KEY_FILE_BYTES)),
        ];
        for text in bad_files {
            write(&text);
            assert!(load(directory.path()).is_err(), "{text:?} accepted");
        }

        std::fs::remove_file(&path).unwrap();
        let target = directory.path().join("elsewhere.key");
        std::fs::write(&target, &good).unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(load(directory.path()).is_err());
        assert!(store_new(directory.path(), &sample_key()).is_err());

        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(load(directory.path()).is_err());
    }
}
