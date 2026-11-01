use super::*;

fn genesis_body() -> GenesisBody {
    GenesisBody {
        index: 1,
        generation: 1,
        group_key: "00".repeat(48),
        commitments_digest: "00".repeat(32),
        network: "11".repeat(32),
        l1_chain_id: 31_337,
        l1_bridge: "22".repeat(20),
        h0_height: 0,
        h0_id: "33".repeat(32),
        h0_work: format!("01{}", "00".repeat(31)),
        created_unix: 1,
    }
}

#[test]
fn journal_replays_its_chain_and_its_signatures() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut j = Journal::open(dir.path()).unwrap();
        assert!(j.genesis.is_none());
        j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
        let ext = ExtendBody {
            from: 1,
            entries: vec![("44".repeat(32), "55".repeat(32), format!("02{}", "00".repeat(31)))],
        };
        j.append(RecordType::Extend, serde_json::to_vec(&ext).unwrap()).unwrap();
        let signed = SignedBody {
            statement_hash: "66".repeat(32),
            partial: "77".repeat(96),
            at_unix: 5,
        };
        j.append(RecordType::Signed, serde_json::to_vec(&signed).unwrap()).unwrap();
        assert_eq!(j.t_max(), Some(1));
    }
    let j = Journal::open(dir.path()).unwrap();
    assert_eq!(j.genesis, Some(genesis_body()));
    assert_eq!(j.t_max(), Some(1));
    assert_eq!(j.entry(1).unwrap().id, [0x44u8; 32]);
    assert_eq!(j.entry(0).unwrap().id, [0x33u8; 32]);
    assert_eq!(j.signed.get(&[0x66u8; 32]), Some(&[0x77u8; 96]));
    assert_eq!(j.head().0, 3);
    assert!(j.halted.is_none());
}

#[test]
fn signer_truncated_tail_is_discarded_only_before_release() {
    let dir = tempfile::tempdir().unwrap();
    let path = {
        let mut j = Journal::open(dir.path()).unwrap();
        j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
        j.path().to_path_buf()
    };
    let good = std::fs::metadata(&path).unwrap().len();
    // an interrupted write: a record cut in the middle
    let (record, _) = encode_record(&[9u8; 32], RecordType::Intent, b"{\"x\":1}");
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    f.write_all(&record[..record.len() / 2]).unwrap();
    drop(f);
    let j = Journal::open(dir.path()).unwrap();
    assert!(j.halted.is_none(), "a cut tail is not a broken journal");
    assert_eq!(j.discarded_tail as u64, (record.len() / 2) as u64);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), good, "the tail is cut off the file");
    assert_eq!(j.head().0, 1);
}

#[test]
fn signer_halts_on_broken_hash_chain() {
    let dir = tempfile::tempdir().unwrap();
    let path = {
        let mut j = Journal::open(dir.path()).unwrap();
        j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
        j.append(RecordType::Published, b"{}".to_vec()).unwrap();
        j.append(RecordType::Published, b"{}".to_vec()).unwrap();
        j.path().to_path_buf()
    };
    let mut bytes = std::fs::read(&path).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let j = Journal::open(dir.path()).unwrap();
    assert!(j.halted.as_deref().is_some_and(|c| c.starts_with("BROKEN_JOURNAL")), "{:?}", j.halted);
}

#[test]
fn halt_persists_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut j = Journal::open(dir.path()).unwrap();
        j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
        j.halt("HALT_DEEP_DIVERGENCE", "test", 7).unwrap();
    }
    let j = Journal::open(dir.path()).unwrap();
    assert_eq!(j.halted.as_deref(), Some("HALT_DEEP_DIVERGENCE"));
}

#[test]
fn a_failed_write_is_an_error_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut j = Journal::open(dir.path()).unwrap();
    j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
    let before = j.head();
    j.fail_next_write = true;
    assert!(j.append(RecordType::Published, b"{}".to_vec()).is_err());
    assert_eq!(j.head(), before);
    drop(j);
    assert_eq!(Journal::open(dir.path()).unwrap().head(), before);
}

#[test]
fn signer_detects_journal_behind_published_head() {
    let dir = tempfile::tempdir().unwrap();
    let mut j = Journal::open(dir.path()).unwrap();
    j.init_genesis(&genesis_body(), [0u8; 32]).unwrap();
    j.append(RecordType::Published, b"{}".to_vec()).unwrap();
    j.append(RecordType::Published, b"{}".to_vec()).unwrap();
    let (n, head) = j.head();
    assert_eq!(n, 3);
    assert_eq!(j.check_published_head(3, &head), None);
    let h1 = j.record_hash(1).unwrap();
    assert_eq!(j.check_published_head(1, &h1), None, "an older published head on this chain");
    assert_eq!(j.check_published_head(4, &[7u8; 32]).as_deref(), Some("JOURNAL_BEHIND_PUBLISHED_HEAD"));
    assert_eq!(j.check_published_head(2, &[7u8; 32]).as_deref(), Some("JOURNAL_DIVERGES_FROM_PUBLISHED_HEAD"));
    drop(j);
    // the hashes are rebuilt by the replay
    let r = Journal::replay_readonly(dir.path()).unwrap();
    assert_eq!(r.record_hash(1), Some(h1));
    assert_eq!(r.check_published_head(3, &head), None);
}
