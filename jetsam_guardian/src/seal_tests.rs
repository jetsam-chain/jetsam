use super::*;

#[test]
fn share_ciphertext_opens_only_with_its_machine_key() {
    let a = MachineKey::from_secret([1u8; 32]);
    let b = MachineKey::from_secret([2u8; 32]);
    let payload = share_payload(1, 3, &[0x42u8; 32]);
    let blob = seal_to(&a.public, &payload, &share_aad(1, 3)).unwrap();
    assert_eq!(&open_sealed(&a, &blob, &share_aad(1, 3)).unwrap()[..], &payload[..]);
    assert!(open_sealed(&b, &blob, &share_aad(1, 3)).is_err(), "another machine key");
    assert!(open_sealed(&a, &blob, &share_aad(1, 4)).is_err(), "another share's aad");
    let mut tampered = blob.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(open_sealed(&a, &tampered, &share_aad(1, 3)).is_err());
    // the scalar is not in the clear in the blob
    assert!(!blob.windows(32).any(|w| w == [0x42u8; 32]));
}

#[test]
fn machine_key_opens_only_with_its_passphrase() {
    let k = MachineKey::generate().unwrap();
    let blob = seal_machine_key(&k, b"correct horse", KdfCost::TEST).unwrap();
    let opened = open_machine_key(&blob, b"correct horse").unwrap();
    assert_eq!(opened.public, k.public);
    assert!(open_machine_key(&blob, b"wrong").is_err());
    assert!(open_machine_key(&blob[..blob.len() - 1], b"correct horse").is_err());
    assert_ne!(MachineKey::generate().unwrap().public, k.public);
}

/// M-12 (external audit of the robots): a secret written at rest is this user's alone (0600, whatever the umask),
/// never over a file already there, never through a symbolic link.
#[test]
fn a_secret_file_is_created_0600_and_never_over_another() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("feed.key");
    // SAFETY: the umask of this test process, set and restored around one call
    let old = unsafe { libc::umask(0o022) };
    let written = write_new_private(&path, b"secret");
    unsafe { libc::umask(old) };
    written.unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    assert_eq!(std::fs::read(&path).unwrap(), b"secret");
    assert!(write_new_private(&path, b"other").is_err(), "never over a file already there");
    assert_eq!(std::fs::read(&path).unwrap(), b"secret");
    let link = dir.path().join("link.key");
    std::os::unix::fs::symlink(dir.path().join("target"), &link).unwrap();
    assert!(write_new_private(&link, b"x").is_err(), "never through a symbolic link");
    assert!(!dir.path().join("target").exists());
}

/// M-12: a machine key sealed with the TEST cost opens only on the private rehearsal; the production cost
/// everywhere.
#[test]
fn a_test_cost_machine_key_is_refused_off_the_private_rehearsal() {
    let k = MachineKey::from_secret([3u8; 32]);
    let test = seal_machine_key(&k, b"p", KdfCost::TEST).unwrap();
    assert_eq!(sealed_cost(&test).unwrap(), KdfCost::TEST);
    assert!(check_machine_key_cost(&test, crate::Profile::PrivateRehearsal).is_ok());
    for p in [crate::Profile::Mainnet, crate::Profile::PublicTestnet, crate::Profile::OtherNetwork] {
        let e = check_machine_key_cost(&test, p).unwrap_err().to_string();
        assert!(e.contains("TEST") || e.contains("test"), "{p:?}: {e}");
    }
    // the production header (not derived here: 256 MiB) is taken everywhere
    let mut prod = test.clone();
    prod[5..9].copy_from_slice(&KdfCost::PRODUCTION.m_kib.to_be_bytes());
    prod[9..13].copy_from_slice(&KdfCost::PRODUCTION.t.to_be_bytes());
    assert!(check_machine_key_cost(&prod, crate::Profile::Mainnet).is_ok());
    assert!(sealed_cost(&test[..10]).is_err());
}
