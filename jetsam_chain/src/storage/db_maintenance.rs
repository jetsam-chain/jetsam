// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Operator maintenance of the chain database file.
//!
//! Two things live here, both outside consensus:
//!
//! * the offline compaction behind `jetsam --compact-db`: a compact copy of
//!   `mdbx.dat` written to an ordinary file of the same directory, checked
//!   table by table against the source, then put in place by renames, the
//!   previous file being kept as `mdbx.dat.old`;
//! * the health figures of the file (its size, the data it really holds, the
//!   free pages its GC keeps) and of its volume, behind
//!   `getChainInfo.storage` and the node's disk warning.
//!
//! Why: on the mainnet seeds in November 2026 `mdbx.dat` reached 11.6 GB for
//! about 95 MB of live data. Pages freed by the rotation of the large state
//! segments were never reused, the file grew until the disk was full, and
//! nothing said so before the node stopped. A compact copy brought it back to
//! about 200 MB. The copy must go to an ordinary file: a compact copy written
//! through a pipe came out corrupted above about 45 MiB.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use libmdbx::{Database, DatabaseOptions, Mode, NoWriteMap};

/// The database file in a data directory.
pub const DATA_FILE: &str = "mdbx.dat";
/// Its MDBX lock file.
pub const LOCK_FILE: &str = "mdbx.lck";
/// Where the compact copy is written before it is checked and put in place.
pub const COMPACT_FILE: &str = "mdbx.dat.compact";
/// The previous database file, kept after a compaction until the operator
/// removes it (`jetsam --compact-db-remove-old`).
pub const OLD_DATA_FILE: &str = "mdbx.dat.old";
/// The previous lock file, kept beside it.
pub const OLD_LOCK_FILE: &str = "mdbx.lck.old";

/// A disk more used than this (as `df` counts it) is worth a warning.
pub const DISK_USED_WARN_PERCENT: u64 = 80;
/// Free pages held by the GC above this share of the file are worth a
/// warning: the file is mostly empty and a compaction would give it back.
pub const GC_FREE_WARN_PERCENT: u64 = 50;
/// The same warning is written at most once in this interval.
pub const HEALTH_WARNING_EVERY: Duration = Duration::from_secs(3_600);

const MAX_TABLES: u64 = 64;

/// Why a compaction did not happen. In every case the database is as it was.
#[derive(Debug)]
pub enum CompactError {
    /// There is no `mdbx.dat` in the directory.
    NoDatabase(PathBuf),
    /// Another process (normally the node) has the database open.
    InUse(PathBuf),
    /// A file a compaction would create or overwrite is already there.
    Leftover(PathBuf),
    /// The volume has less room than the copy needs.
    NoSpace { needed: u64, available: u64 },
    /// The copy does not hold exactly what the source holds.
    Mismatch(String),
    Mdbx(String),
    Io(String),
}

impl std::fmt::Display for CompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDatabase(path) => write!(f, "no chain database at {}", path.display()),
            Self::InUse(path) => write!(
                f,
                "the database {} is open in another process: stop the node first",
                path.display()
            ),
            Self::Leftover(path) => write!(
                f,
                "{} already exists: remove or move it first (after a validated compaction, \
                 `jetsam --compact-db-remove-old` deletes the previous file)",
                path.display()
            ),
            Self::NoSpace { needed, available } => write!(
                f,
                "not enough free space for the compact copy: {needed} bytes needed, \
                 {available} available"
            ),
            Self::Mismatch(what) => write!(f, "the compact copy differs from the source: {what}"),
            Self::Mdbx(e) => write!(f, "mdbx: {e}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompactError {}

/// What a compaction did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactReport {
    pub before_bytes: u64,
    pub after_bytes: u64,
    pub tables: usize,
    pub entries: u64,
    /// The previous file, kept.
    pub old_file: PathBuf,
}

/// The logical content of one table: its entry count, its key and value
/// bytes, and a FNV-1a 64 digest of every key and value with its length (the
/// digest of the DATAVOL `dbhash` tool).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFingerprint {
    pub name: String,
    pub entries: u64,
    pub bytes: u64,
    pub digest: u64,
}

/// The health of a database file and of its volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DbHealth {
    /// Size of `mdbx.dat` on disk.
    pub file_bytes: u64,
    /// Pages allocated up to the last used page.
    pub allocated_bytes: u64,
    /// Allocated pages that hold data (allocated minus GC free pages).
    pub live_bytes: u64,
    /// Free pages kept by the GC for reuse.
    pub gc_free_bytes: u64,
    /// The volume holding the file, as `df` counts it (unix only).
    pub disk_total_bytes: Option<u64>,
    pub disk_used_bytes: Option<u64>,
    pub disk_available_bytes: Option<u64>,
}

impl DbHealth {
    /// Used share of the volume, as `df` shows it (`used / (used + available)`).
    pub fn disk_used_percent(&self) -> Option<u64> {
        let used = self.disk_used_bytes?;
        let available = self.disk_available_bytes?;
        let base = used.checked_add(available)?;
        if base == 0 {
            return None;
        }
        // Rounded up, as `df` does.
        Some(((used as u128 * 100).div_ceil(base as u128)) as u64)
    }

    /// Share of the file held by GC free pages.
    pub fn gc_free_percent(&self) -> u64 {
        if self.file_bytes == 0 {
            return 0;
        }
        (self.gc_free_bytes as u128 * 100 / self.file_bytes as u128) as u64
    }

    /// The operator warning these figures call for, if any.
    pub fn warning(&self) -> Option<String> {
        let disk = self
            .disk_used_percent()
            .filter(|used| *used > DISK_USED_WARN_PERCENT);
        let gc = Some(self.gc_free_percent()).filter(|gc| *gc > GC_FREE_WARN_PERCENT);
        if disk.is_none() && gc.is_none() {
            return None;
        }
        let mut causes = Vec::new();
        if let Some(used) = disk {
            causes.push(format!(
                "the data volume is {used}% used ({} bytes available)",
                self.disk_available_bytes.unwrap_or(0)
            ));
        }
        if let Some(gc) = gc {
            causes.push(format!(
                "{gc}% of the database file is free pages ({} of {} bytes, {} bytes of live data)",
                self.gc_free_bytes, self.file_bytes, self.live_bytes
            ));
        }
        Some(format!(
            "database storage: {}; stop the node and run `jetsam --compact-db --data-dir <dir>` \
             to give the free pages back to the disk",
            causes.join(" and ")
        ))
    }
}

/// At most one health warning per [`HEALTH_WARNING_EVERY`].
#[derive(Debug, Default)]
pub struct HealthWarnings {
    last: Option<Instant>,
}

impl HealthWarnings {
    pub fn new() -> Self {
        Self::default()
    }

    /// The warning to log now, if `health` calls for one and none was logged
    /// in the last [`HEALTH_WARNING_EVERY`].
    pub fn due(&mut self, health: &DbHealth, now: Instant) -> Option<String> {
        let warning = health.warning()?;
        if let Some(last) = self.last {
            if now.saturating_duration_since(last) < HEALTH_WARNING_EVERY {
                return None;
            }
        }
        self.last = Some(now);
        Some(warning)
    }
}

/// Health of the open database `db` whose file is `data_file`.
pub fn health_of(db: &Database<NoWriteMap>, data_file: &Path) -> Result<DbHealth, CompactError> {
    let file_bytes = std::fs::metadata(data_file)
        .map_err(|e| io_error("read the size of", data_file, e))?
        .len();
    let page_size = u64::from(db.stat().map_err(mdbx_error)?.page_size());
    let allocated_pages = db.info().map_err(mdbx_error)?.last_pgno() as u64 + 1;
    let gc_free_pages = db.freelist().map_err(mdbx_error)? as u64;
    let allocated_bytes = allocated_pages * page_size;
    let gc_free_bytes = (gc_free_pages * page_size).min(allocated_bytes);
    let disk = data_file.parent().and_then(volume_space);
    Ok(DbHealth {
        file_bytes,
        allocated_bytes,
        live_bytes: allocated_bytes - gc_free_bytes,
        gc_free_bytes,
        disk_total_bytes: disk.map(|d| d.total),
        disk_used_bytes: disk.map(|d| d.used),
        disk_available_bytes: disk.map(|d| d.available),
    })
}

#[derive(Debug, Clone, Copy)]
struct VolumeSpace {
    total: u64,
    used: u64,
    available: u64,
}

/// The volume holding `dir`, as `df` counts it: `available` is what an
/// unprivileged process may still write.
#[cfg(unix)]
fn volume_space(dir: &Path) -> Option<VolumeSpace> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a valid NUL-terminated string and `stat` a valid,
    // writable statvfs for the duration of the call.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let fragment = stat.f_frsize as u64;
    let total = stat.f_blocks as u64 * fragment;
    let free = stat.f_bfree as u64 * fragment;
    Some(VolumeSpace {
        total,
        used: total.saturating_sub(free),
        available: stat.f_bavail as u64 * fragment,
    })
}

#[cfg(not(unix))]
fn volume_space(_dir: &Path) -> Option<VolumeSpace> {
    None
}

fn mdbx_error(error: libmdbx::Error) -> CompactError {
    CompactError::Mdbx(error.to_string())
}

fn io_error(action: &str, path: &Path, error: std::io::Error) -> CompactError {
    CompactError::Io(format!("{action} {}: {error}", path.display()))
}

/// FNV-1a 64 of `bytes` preceded by their length (8 bytes, little endian),
/// as the DATAVOL `dbhash` tool computes it.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 1_099_511_628_211;
    for byte in (bytes.len() as u64).to_le_bytes().into_iter().chain(bytes.iter().copied()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;

fn fingerprint_of(db: &Database<NoWriteMap>) -> Result<Vec<TableFingerprint>, CompactError> {
    let txn = db.begin_ro_txn().map_err(mdbx_error)?;
    let main = txn.open_table(None).map_err(mdbx_error)?;
    let mut names = Vec::new();
    {
        let mut cursor = txn.cursor(&main).map_err(mdbx_error)?;
        let mut item: Option<(Vec<u8>, ())> = cursor.first().map_err(mdbx_error)?;
        while let Some((name, ())) = item {
            names.push(String::from_utf8(name).map_err(|_| {
                CompactError::Mismatch("a table name is not UTF-8".into())
            })?);
            item = cursor.next().map_err(mdbx_error)?;
        }
    }
    let mut tables = Vec::with_capacity(names.len());
    for name in names {
        let table = txn.open_table(Some(&name)).map_err(mdbx_error)?;
        let mut cursor = txn.cursor(&table).map_err(mdbx_error)?;
        let (mut entries, mut bytes, mut digest) = (0u64, 0u64, FNV_OFFSET);
        let mut item: Option<(std::borrow::Cow<'_, [u8]>, std::borrow::Cow<'_, [u8]>)> =
            cursor.first().map_err(mdbx_error)?;
        while let Some((key, value)) = item {
            digest = fnv1a(fnv1a(digest, &key), &value);
            entries += 1;
            bytes += (key.len() + value.len()) as u64;
            item = cursor.next().map_err(mdbx_error)?;
        }
        tables.push(TableFingerprint {
            name,
            entries,
            bytes,
            digest,
        });
    }
    Ok(tables)
}

/// Logical content of every named table of the database in `path` (a data
/// directory, or a database file when `no_sub_dir`), opened read-only and
/// exclusively.
pub fn database_fingerprint(
    path: &Path,
    no_sub_dir: bool,
) -> Result<Vec<TableFingerprint>, CompactError> {
    fingerprint_of(&open_exclusive_read_only(path, no_sub_dir)?)
}

/// Compact the stopped database of `data_dir`. See the module documentation.
pub fn compact_database(data_dir: &Path) -> Result<CompactReport, CompactError> {
    let data = data_dir.join(DATA_FILE);
    let lock = data_dir.join(LOCK_FILE);
    let compact = data_dir.join(COMPACT_FILE);
    let old_data = data_dir.join(OLD_DATA_FILE);
    let old_lock = data_dir.join(OLD_LOCK_FILE);
    if !data.is_file() {
        return Err(CompactError::NoDatabase(data));
    }
    for leftover in [&compact, &old_data, &old_lock] {
        if std::fs::symlink_metadata(leftover).is_ok() {
            return Err(CompactError::Leftover(leftover.clone()));
        }
    }
    let original = std::fs::metadata(&data).map_err(|e| io_error("read", &data, e))?;
    let before_bytes = original.len();

    // Exclusive: refused while the node (or anything else) has it open, and
    // nothing can write to it until this process is done.
    let source = open_exclusive_read_only(data_dir, false).map_err(|error| match error {
        CompactError::InUse(_) => CompactError::InUse(data.clone()),
        other => other,
    })?;
    let expected = fingerprint_of(&source)?;
    let health = health_of(&source, &data)?;
    if let Some(available) = health.disk_available_bytes {
        // The copy holds the live pages; twice that plus a growth step
        // leaves room for the B-tree rebuilt around them.
        let needed = health.live_bytes.saturating_mul(2) + 64 * 1024 * 1024;
        if available < needed {
            return Err(CompactError::NoSpace { needed, available });
        }
    }

    let result = (|| {
        copy_compact(&source, &compact)?;
        let copy = open_exclusive_read_only(&compact, true)?;
        let found = fingerprint_of(&copy)?;
        drop(copy);
        if found != expected {
            let what = expected
                .iter()
                .zip(found.iter())
                .find(|(a, b)| a != b)
                .map(|(a, b)| format!("table {}: {a:?} became {b:?}", a.name))
                .unwrap_or_else(|| {
                    format!("{} tables became {}", expected.len(), found.len())
                });
            return Err(CompactError::Mismatch(what));
        }
        keep_owner_and_mode(&original, &compact)?;
        sync_file(&compact)?;
        Ok(())
    })();
    drop(source);
    // The copy opened without a subdirectory may leave a lock file beside it.
    let _ = std::fs::remove_file(data_dir.join(format!("{COMPACT_FILE}-lck")));
    if let Err(error) = result {
        let _ = std::fs::remove_file(&compact);
        return Err(error);
    }
    let after_bytes = std::fs::metadata(&compact)
        .map_err(|e| io_error("read", &compact, e))?
        .len();

    // Renames only, on one filesystem: each step is undone if a later one
    // fails, so a failure leaves the directory as it was.
    let lock_moved = if lock.exists() {
        if let Err(e) = std::fs::rename(&lock, &old_lock) {
            let _ = std::fs::remove_file(&compact);
            return Err(io_error("move aside", &lock, e));
        }
        true
    } else {
        false
    };
    let undo_lock = || {
        if lock_moved {
            let _ = std::fs::rename(&old_lock, &lock);
        }
    };
    if let Err(e) = std::fs::rename(&data, &old_data) {
        undo_lock();
        let _ = std::fs::remove_file(&compact);
        return Err(io_error("move aside", &data, e));
    }
    if let Err(e) = std::fs::rename(&compact, &data) {
        let _ = std::fs::rename(&old_data, &data);
        undo_lock();
        let _ = std::fs::remove_file(&compact);
        return Err(io_error("put in place", &compact, e));
    }
    sync_directory(data_dir);
    Ok(CompactReport {
        before_bytes,
        after_bytes,
        tables: expected.len(),
        entries: expected.iter().map(|t| t.entries).sum(),
        old_file: old_data,
    })
}

/// `mdbx_env_copy(MDBX_CP_COMPACT)` of `source` to the new file `dest`.
fn copy_compact(source: &Database<NoWriteMap>, dest: &Path) -> Result<(), CompactError> {
    let path = dest
        .to_str()
        .and_then(|p| std::ffi::CString::new(p).ok())
        .ok_or_else(|| CompactError::Io(format!("unusable path {}", dest.display())))?;
    // SAFETY: the environment is open for the whole call (borrowed from
    // `source`), `path` is NUL-terminated, and libmdbx creates `dest`
    // itself (it must not exist).
    let rc = unsafe { mdbx_sys::mdbx_env_copy(source.ptr().0, path.as_ptr(), mdbx_sys::MDBX_CP_COMPACT) };
    if rc != 0 {
        return Err(CompactError::Mdbx(format!(
            "compact copy to {}: {}",
            dest.display(),
            libmdbx::Error::from_err_code(rc)
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn keep_owner_and_mode(original: &std::fs::Metadata, file: &Path) -> Result<(), CompactError> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let current = std::fs::metadata(file).map_err(|e| io_error("read", file, e))?;
    if (current.uid(), current.gid()) != (original.uid(), original.gid()) {
        std::os::unix::fs::chown(file, Some(original.uid()), Some(original.gid()))
            .map_err(|e| io_error("give the owner of the database to", file, e))?;
    }
    std::fs::set_permissions(file, std::fs::Permissions::from_mode(original.mode() & 0o7777))
        .map_err(|e| io_error("give the mode of the database to", file, e))
}

#[cfg(not(unix))]
fn keep_owner_and_mode(_original: &std::fs::Metadata, _file: &Path) -> Result<(), CompactError> {
    Ok(())
}

fn sync_file(file: &Path) -> Result<(), CompactError> {
    std::fs::File::open(file)
        .and_then(|f| f.sync_all())
        .map_err(|e| io_error("sync", file, e))
}

fn sync_directory(dir: &Path) {
    #[cfg(unix)]
    if let Ok(dir) = std::fs::File::open(dir) {
        let _ = dir.sync_all();
    }
    #[cfg(not(unix))]
    let _ = dir;
}

/// Delete the previous database file kept by a compaction (and its lock
/// file). Returns the bytes freed.
pub fn remove_old_database(data_dir: &Path) -> Result<u64, CompactError> {
    let old_data = data_dir.join(OLD_DATA_FILE);
    let metadata = match std::fs::symlink_metadata(&old_data) {
        Ok(metadata) if metadata.is_file() => metadata,
        _ => return Err(CompactError::NoDatabase(old_data)),
    };
    if !data_dir.join(DATA_FILE).is_file() {
        // Without a current database the old file is the only copy left.
        return Err(CompactError::NoDatabase(data_dir.join(DATA_FILE)));
    }
    std::fs::remove_file(&old_data).map_err(|e| io_error("delete", &old_data, e))?;
    let old_lock = data_dir.join(OLD_LOCK_FILE);
    if old_lock.exists() {
        std::fs::remove_file(&old_lock).map_err(|e| io_error("delete", &old_lock, e))?;
    }
    sync_directory(data_dir);
    Ok(metadata.len())
}

fn open_exclusive_read_only(
    path: &Path,
    no_sub_dir: bool,
) -> Result<Database<NoWriteMap>, CompactError> {
    Database::<NoWriteMap>::open_with_options(
        path,
        DatabaseOptions {
            mode: Mode::ReadOnly,
            exclusive: true,
            no_sub_dir,
            max_tables: Some(MAX_TABLES),
            ..Default::default()
        },
    )
    .map_err(|error| match error {
        // MDBX_BUSY, or the whole-file lock refused because another process
        // holds its own lock on the file (EACCES, EAGAIN, EBUSY on Linux).
        libmdbx::Error::Busy | libmdbx::Error::Other(11 | 16) => {
            CompactError::InUse(path.to_path_buf())
        }
        libmdbx::Error::Access => {
            let file = if no_sub_dir {
                path.to_path_buf()
            } else {
                path.join(DATA_FILE)
            };
            // EACCES is also a plain permission problem: say which it is.
            match std::fs::File::open(&file) {
                Err(e) => io_error("open", &file, e),
                Ok(_) => CompactError::InUse(path.to_path_buf()),
            }
        }
        other => CompactError::Mdbx(other.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libmdbx::{ReadWriteOptions, TableFlags, WriteFlags};

    const MIB: usize = 1024 * 1024;

    fn open_rw(dir: &Path) -> Database<NoWriteMap> {
        Database::<NoWriteMap>::open_with_options(
            dir,
            DatabaseOptions {
                max_tables: Some(MAX_TABLES),
                mode: Mode::ReadWrite(ReadWriteOptions {
                    min_size: Some(4 * MIB as isize),
                    max_size: Some(1024 * MIB as isize),
                    growth_step: Some(16 * MIB as isize),
                    ..Default::default()
                }),
                liforeclaim: true,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// A database of 96 one-MiB values in `big` and 1 000 small ones in
    /// `small`, of which 92 big values were then deleted: the file keeps
    /// about 92 MiB of free pages its GC holds.
    fn database_with_holes(dir: &Path) {
        let db = open_rw(dir);
        let txn = db.begin_rw_txn().unwrap();
        let big = txn.create_table(Some("big"), TableFlags::empty()).unwrap();
        for i in 0u32..96 {
            let value = vec![(i % 251) as u8; MIB];
            txn.put(&big, i.to_be_bytes(), value, WriteFlags::empty()).unwrap();
        }
        let small = txn.create_table(Some("small"), TableFlags::empty()).unwrap();
        for i in 0u32..1_000 {
            txn.put(&small, i.to_be_bytes(), i.to_le_bytes(), WriteFlags::empty())
                .unwrap();
        }
        txn.commit().unwrap();
        let txn = db.begin_rw_txn().unwrap();
        let big = txn.open_table(Some("big")).unwrap();
        for i in 4u32..96 {
            txn.del(&big, i.to_be_bytes(), None).unwrap();
        }
        txn.commit().unwrap();
    }

    fn file_len(path: &Path) -> u64 {
        std::fs::metadata(path).unwrap().len()
    }

    #[test]
    fn compaction_shrinks_a_database_with_holes_keeps_its_content_and_the_old_file() {
        let dir = tempfile::tempdir().unwrap();
        database_with_holes(dir.path());
        let data = dir.path().join(DATA_FILE);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o604)).unwrap();
        }
        let before_len = file_len(&data);
        let before = database_fingerprint(dir.path(), false).unwrap();
        assert_eq!(before.len(), 2, "{before:?}");

        let report = compact_database(dir.path()).unwrap();

        assert_eq!(report.before_bytes, before_len);
        assert_eq!(report.after_bytes, file_len(&data));
        assert!(
            report.after_bytes * 3 < report.before_bytes,
            "not compacted: {report:?}"
        );
        assert_eq!(report.tables, 2);
        assert_eq!(report.entries, 4 + 1_000);
        assert_eq!(report.old_file, dir.path().join(OLD_DATA_FILE));
        // The previous file is kept, untouched.
        assert_eq!(file_len(&dir.path().join(OLD_DATA_FILE)), before_len);
        assert!(!dir.path().join(COMPACT_FILE).exists());
        // Same content, table by table.
        assert_eq!(database_fingerprint(dir.path(), false).unwrap(), before);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let old = std::fs::metadata(dir.path().join(OLD_DATA_FILE)).unwrap();
            let new = std::fs::metadata(&data).unwrap();
            assert_eq!(new.mode() & 0o7777, 0o604);
            assert_eq!((new.uid(), new.gid()), (old.uid(), old.gid()));
        }
        // The compacted base opens read-write and serves its data.
        let db = open_rw(dir.path());
        let txn = db.begin_ro_txn().unwrap();
        let big = txn.open_table(Some("big")).unwrap();
        let value: Option<Vec<u8>> = txn.get(&big, &3u32.to_be_bytes()).unwrap();
        assert_eq!(value, Some(vec![3u8; MIB]));
    }

    #[test]
    fn compaction_of_the_node_store_keeps_every_table() {
        let dir = tempfile::tempdir().unwrap();
        drop(crate::storage::MdbxStore::open(dir.path()).unwrap());
        let before = database_fingerprint(dir.path(), false).unwrap();
        compact_database(dir.path()).unwrap();
        assert_eq!(database_fingerprint(dir.path(), false).unwrap(), before);
        drop(crate::storage::MdbxStore::open(dir.path()).unwrap());
    }

    #[test]
    fn compaction_without_a_database_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            compact_database(dir.path()),
            Err(CompactError::NoDatabase(_))
        ));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn compaction_refuses_leftovers_and_touches_nothing() {
        for leftover in [COMPACT_FILE, OLD_DATA_FILE, OLD_LOCK_FILE] {
            let dir = tempfile::tempdir().unwrap();
            database_with_holes(dir.path());
            std::fs::write(dir.path().join(leftover), b"previous").unwrap();
            let data = dir.path().join(DATA_FILE);
            let before = std::fs::read(&data).unwrap();
            let error = compact_database(dir.path()).unwrap_err();
            assert!(matches!(error, CompactError::Leftover(_)), "{leftover}: {error}");
            assert_eq!(std::fs::read(&data).unwrap(), before, "{leftover}");
            assert_eq!(std::fs::read(dir.path().join(leftover)).unwrap(), b"previous");
        }
    }

    #[test]
    fn compaction_refuses_a_database_open_in_this_process() {
        let dir = tempfile::tempdir().unwrap();
        database_with_holes(dir.path());
        let held = open_rw(dir.path());
        let before_len = file_len(&dir.path().join(DATA_FILE));
        let error = compact_database(dir.path()).unwrap_err();
        assert!(matches!(error, CompactError::InUse(_)), "{error}");
        drop(held);
        assert_eq!(file_len(&dir.path().join(DATA_FILE)), before_len);
        assert!(!dir.path().join(COMPACT_FILE).exists());
        assert!(!dir.path().join(OLD_DATA_FILE).exists());
    }

    const HOLD_ENV: &str = "JETSAM_DB_MAINTENANCE_HOLD_DIR";

    /// Helper run in a child process by the next test: holds the store of
    /// `$JETSAM_DB_MAINTENANCE_HOLD_DIR` open, as a running node does, until
    /// `release` appears there.
    #[test]
    #[ignore = "child process of compaction_refuses_a_database_open_in_another_process"]
    fn hold_database_open_helper() {
        let Some(dir) = std::env::var_os(HOLD_ENV).map(PathBuf::from) else {
            return;
        };
        let store = crate::storage::MdbxStore::open(&dir).unwrap();
        std::fs::write(dir.join("ready"), b"").unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        while !dir.join("release").exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        drop(store);
    }

    #[test]
    fn compaction_refuses_a_database_open_in_another_process() {
        let dir = tempfile::tempdir().unwrap();
        database_with_holes(dir.path());
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "storage::db_maintenance::tests::hold_database_open_helper",
                "--exact",
                "--ignored",
                "--nocapture",
            ])
            .env(HOLD_ENV, dir.path())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(60);
        while !dir.path().join("ready").exists() {
            assert!(Instant::now() < deadline, "the holder never opened the store");
            std::thread::sleep(Duration::from_millis(20));
        }
        let before = std::fs::read(dir.path().join(DATA_FILE)).unwrap();
        let result = compact_database(dir.path());
        std::fs::write(dir.path().join("release"), b"").unwrap();
        assert!(child.wait().unwrap().success());
        let error = result.unwrap_err();
        assert!(matches!(error, CompactError::InUse(_)), "{error}");
        assert_eq!(std::fs::read(dir.path().join(DATA_FILE)).unwrap(), before);
        assert!(!dir.path().join(COMPACT_FILE).exists());
        assert!(!dir.path().join(OLD_DATA_FILE).exists());
    }

    #[test]
    fn removing_the_old_database_deletes_only_the_previous_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            remove_old_database(dir.path()),
            Err(CompactError::NoDatabase(_))
        ));
        database_with_holes(dir.path());
        compact_database(dir.path()).unwrap();
        let old_len = file_len(&dir.path().join(OLD_DATA_FILE));
        let fingerprint = database_fingerprint(dir.path(), false).unwrap();
        assert_eq!(remove_old_database(dir.path()).unwrap(), old_len);
        assert!(!dir.path().join(OLD_DATA_FILE).exists());
        assert!(!dir.path().join(OLD_LOCK_FILE).exists());
        assert_eq!(database_fingerprint(dir.path(), false).unwrap(), fingerprint);
    }

    #[test]
    fn health_reports_the_file_its_live_data_and_its_gc_free_pages() {
        let dir = tempfile::tempdir().unwrap();
        database_with_holes(dir.path());
        let data = dir.path().join(DATA_FILE);
        let health = {
            let db = open_rw(dir.path());
            health_of(&db, &data).unwrap()
        };
        assert_eq!(health.file_bytes, file_len(&data));
        assert!(health.allocated_bytes <= health.file_bytes);
        assert_eq!(health.live_bytes + health.gc_free_bytes, health.allocated_bytes);
        assert!(health.gc_free_bytes >= 40 * MIB as u64, "{health:?}");
        assert!(health.live_bytes < 8 * MIB as u64, "{health:?}");
        #[cfg(unix)]
        {
            let total = health.disk_total_bytes.unwrap();
            let used = health.disk_used_bytes.unwrap();
            let available = health.disk_available_bytes.unwrap();
            assert!(total > 0 && used <= total && available <= total, "{health:?}");
            assert!(health.disk_used_percent().unwrap() <= 100);
        }
        compact_database(dir.path()).unwrap();
        let db = open_rw(dir.path());
        let after = health_of(&db, &data).unwrap();
        assert!(after.gc_free_bytes < 1 * MIB as u64, "{after:?}");
        assert!(after.file_bytes < health.file_bytes);
    }

    #[test]
    fn the_store_reports_its_health() {
        let dir = tempfile::tempdir().unwrap();
        let store = crate::storage::MdbxStore::open(dir.path()).unwrap();
        let health = store.database_health().unwrap();
        assert_eq!(health.file_bytes, file_len(&dir.path().join(DATA_FILE)));
        assert!(health.live_bytes > 0);
        let cached = store
            .database_health_cached(Duration::from_secs(60))
            .unwrap();
        assert_eq!(cached.file_bytes, health.file_bytes);
    }

    fn health(file: u64, gc: u64, disk_used: u64, disk_available: u64) -> DbHealth {
        DbHealth {
            file_bytes: file,
            allocated_bytes: file,
            live_bytes: file - gc,
            gc_free_bytes: gc,
            disk_total_bytes: Some(disk_used + disk_available + 5),
            disk_used_bytes: Some(disk_used),
            disk_available_bytes: Some(disk_available),
        }
    }

    #[test]
    fn health_warns_above_80_percent_of_disk_or_50_percent_of_gc() {
        let quiet = health(1_000, 500, 80, 20);
        assert_eq!(quiet.disk_used_percent(), Some(80));
        assert_eq!(quiet.gc_free_percent(), 50);
        assert_eq!(quiet.warning(), None);

        let full = health(1_000, 100, 81, 19);
        let warning = full.warning().unwrap();
        assert!(warning.contains("81%"), "{warning}");
        assert!(warning.contains("--compact-db"), "{warning}");

        let empty_file = health(1_000, 510, 10, 90);
        let warning = empty_file.warning().unwrap();
        assert!(warning.contains("51%"), "{warning}");
        assert!(warning.contains("--compact-db"), "{warning}");

        let no_disk = DbHealth {
            disk_total_bytes: None,
            disk_used_bytes: None,
            disk_available_bytes: None,
            ..quiet
        };
        assert_eq!(no_disk.disk_used_percent(), None);
        assert_eq!(no_disk.warning(), None);
    }

    #[test]
    fn health_warnings_are_due_at_most_once_an_hour() {
        let start = Instant::now();
        let bad = health(1_000, 900, 10, 90);
        let good = health(1_000, 0, 10, 90);
        let mut warnings = HealthWarnings::new();
        assert_eq!(warnings.due(&good, start), None);
        assert!(warnings.due(&bad, start).is_some());
        assert_eq!(warnings.due(&bad, start + Duration::from_secs(60)), None);
        assert_eq!(
            warnings.due(&bad, start + HEALTH_WARNING_EVERY - Duration::from_secs(1)),
            None
        );
        assert!(warnings.due(&bad, start + HEALTH_WARNING_EVERY).is_some());
    }
}
