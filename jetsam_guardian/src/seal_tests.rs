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
