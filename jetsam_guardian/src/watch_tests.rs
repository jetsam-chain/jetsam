use super::*;

fn watch_toml(network: &str, chain: u64, extra: &str) -> String {
    // E-3: the mainnet's watcher reads a second L1 endpoint (none given here: the one of each test, or the default one)
    let second = if network == hex::encode(crate::MAINNET_NETWORK) && !extra.contains("l1_rpc_second") { "l1_rpc_second = \"http://127.0.0.1:8547\"\n" } else { "" };
    let extra = format!("{second}{extra}");
    format!(
        "network = \"{network}\"\nl1_chain_id = {chain}\nl1_bridge = \"0x1111111111111111111111111111111111111111\"\nl1_rpc = \"http://127.0.0.1:8545\"\n\
         state_dir = \"/tmp/w\"\npause_command = [\"/opt/bridge-pause.sh\", \"--send\"]\nalert_command = [\"/opt/alert\", \"{{event}}\", \"{{detail}}\"]\n{extra}\n\
         [[nodes]]\nname = \"a\"\nrpc = \"http://127.0.0.1:19711\"\n[[nodes]]\nname = \"b\"\nrpc = \"http://127.0.0.1:19721\"\n"
    )
}

/// A-2: `watch.toml`'s defaults (an alert after 6 600 s without a point, a poll a minute, two nodes, 1 200 s per command)
/// and what does not start: an alert after `W_STALL` (the contract stalled first) or never, a poll as long as the alert,
/// a command without a bound or without a program, fewer than two nodes off the rehearsal, the clearnet off it.
#[test]
fn a2_watch_toml_defaults_and_what_does_not_start() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let c = WatchConfig::parse(&watch_toml(&mainnet, 1, "")).unwrap();
    assert_eq!((c.alert_after_secs, c.poll_secs, c.min_nodes, c.command_timeout_secs), (6_600, 60, 2, 1_200));
    assert_eq!(c.pause_command, vec!["/opt/bridge-pause.sh".to_string(), "--send".to_string()]);
    let e = |extra: &str| WatchConfig::parse(&watch_toml(&mainnet, 1, extra)).unwrap_err().to_string();
    assert!(WatchConfig::parse(&watch_toml(&mainnet, 1, "alert_after_secs = 7200\n")).is_ok());
    assert!(e("alert_after_secs = 7201\n").contains("W_STALL"), "after the stall");
    assert!(e("alert_after_secs = 0\n").contains("alert_after_secs"));
    assert!(e("poll_secs = 0\n").contains("poll_secs"));
    assert!(e("poll_secs = 6600\n").contains("poll_secs"), "a poll as long as the alert's delay");
    assert!(e("command_timeout_secs = 0\n").contains("command_timeout_secs"));
    assert!(e("min_nodes = 1\n").contains("floor"));
    assert!(e("min_nodes = 3\n").contains("nodes configured"));
    assert!(e("[egress]\nmode = \"clearnet\"\n").contains("clearnet"));
    assert!(WatchConfig::parse(&watch_toml(&mainnet, 1, "bogus = 1\n")).is_err(), "unknown field");
    let text = watch_toml(&mainnet, 1, "");
    let no_pause = text.replace("pause_command = [\"/opt/bridge-pause.sh\", \"--send\"]", "pause_command = []");
    assert!(WatchConfig::parse(&no_pause).unwrap_err().to_string().contains("pause_command"));
    let no_alert = text.replace("alert_command = [\"/opt/alert\", \"{event}\", \"{detail}\"]", "alert_command = []");
    assert!(WatchConfig::parse(&no_alert).unwrap_err().to_string().contains("alert_command"));
    let twice = text.replace("http://127.0.0.1:19721", "http://127.0.0.1:19711");
    assert!(WatchConfig::parse(&twice).unwrap_err().to_string().contains("twice"));
    let off_loopback = text.replace("http://127.0.0.1:19711", "http://10.1.2.3:19711");
    assert!(WatchConfig::parse(&off_loopback).is_err(), "a node off the loopback and not .onion");
}

/// A-2: the contract's tip locks a block off this watcher's chain only when `min_nodes` nodes hold ANOTHER block at its
/// height (and fewer than `min_nodes` hold it); on it when `min_nodes` hold it (and fewer hold another); a node behind,
/// silent or failing is no witness either way; nodes that disagree decide nothing.
#[test]
fn a2_the_tips_locked_block_is_off_our_chain_only_when_min_nodes_nodes_hold_another_block_there() {
    use Verdict::*;
    let ours = [1u8; 32];
    let (o, x) = (Some(ours), Some([2u8; 32]));
    assert_eq!(locked_verdict(&ours, &[o, o], 2), OnOurChain);
    assert_eq!(locked_verdict(&ours, &[x, x], 2), OffOurChain);
    assert_eq!(locked_verdict(&ours, &[x, None], 2), Unknown, "a node behind or silent is no witness");
    assert_eq!(locked_verdict(&ours, &[o, None], 2), Unknown);
    assert_eq!(locked_verdict(&ours, &[o, x], 2), Unknown, "the nodes disagree");
    assert_eq!(locked_verdict(&ours, &[x, x, o], 2), OffOurChain, "two of three");
    assert_eq!(locked_verdict(&ours, &[o, o, x], 2), OnOurChain);
    assert_eq!(locked_verdict(&ours, &[o, o, x, x], 2), Unknown, "two against two");
    assert_eq!(locked_verdict(&ours, &[x], 1), OffOurChain, "the rehearsal's single node");
}

fn tip(id: u8, accepted_at: u64) -> Tip {
    Tip { id: [id; 32], locked_height: 100, locked_block: [0xab; 32], accepted_at, paused_until: 0 }
}

fn events(actions: &[Action]) -> Vec<String> {
    actions
        .iter()
        .map(|a| match a {
            Action::Pause => "pause".to_string(),
            Action::Alert { event, .. } => event.clone(),
        })
        .collect()
}

/// A-2: an incident opens when the tip locks a block off the chain and lasts until it locks one of the chain again (an
/// unknown verdict keeps it, a new tip off the chain stays in it): ONE pause per incident; a pause that fails is tried
/// again `PAUSE_RETRY_SECS` later, `PAUSE_ATTEMPTS` times at most, each outcome alerted; then a new incident pauses again.
#[test]
fn a2_one_pause_per_incident_tried_again_after_a_failure_within_bounds() {
    let mut s = WatchState::default();
    let t = tip(1, 1_000);
    assert!(decide(&mut s, &t, Verdict::OnOurChain, 1_100, 10, 6_600).is_empty(), "on the chain: nothing");
    assert_eq!(events(&decide(&mut s, &t, Verdict::OffOurChain, 1_100, 50, 6_600)), vec!["pause"]);
    pause_started(&mut s, 50);
    assert_eq!(events(&[pause_ended(&mut s, true, "exit status 0")]), vec!["PAUSED"]);
    for (v, at) in [(Verdict::OffOurChain, 60), (Verdict::Unknown, 400), (Verdict::OffOurChain, 10_000)] {
        assert!(decide(&mut s, &t, v, 1_200, at, 6_600).is_empty(), "the same incident: no pause again ({v:?} at {at})");
    }
    assert!(decide(&mut s, &tip(2, 1_300), Verdict::OffOurChain, 1_400, 10_010, 6_600).is_empty(), "a new tip off the chain: the same incident");
    assert!(s.incident.is_some());
    assert!(decide(&mut s, &tip(3, 1_500), Verdict::OnOurChain, 1_600, 10_020, 6_600).is_empty());
    assert!(s.incident.is_none(), "back on the chain: over");
    // a new incident whose pause fails three times
    assert_eq!(events(&decide(&mut s, &t, Verdict::OffOurChain, 1_700, 20_000, 6_600)), vec!["pause"]);
    pause_started(&mut s, 20_000);
    assert_eq!(events(&[pause_ended(&mut s, false, "exit status 3")]), vec!["PAUSE_FAILED"]);
    assert!(decide(&mut s, &t, Verdict::OffOurChain, 1_700, 20_000 + PAUSE_RETRY_SECS - 1, 6_600).is_empty(), "not before PAUSE_RETRY_SECS");
    assert_eq!(events(&decide(&mut s, &t, Verdict::OffOurChain, 1_700, 20_000 + PAUSE_RETRY_SECS, 6_600)), vec!["pause"]);
    pause_started(&mut s, 20_000 + PAUSE_RETRY_SECS);
    assert_eq!(events(&[pause_ended(&mut s, false, "timed out")]), vec!["PAUSE_FAILED"]);
    assert_eq!(events(&decide(&mut s, &t, Verdict::OffOurChain, 1_700, 20_000 + 2 * PAUSE_RETRY_SECS, 6_600)), vec!["pause"]);
    pause_started(&mut s, 20_000 + 2 * PAUSE_RETRY_SECS);
    assert_eq!(PAUSE_ATTEMPTS, 3);
    assert_eq!(events(&[pause_ended(&mut s, false, "exit status 4")]), vec!["PAUSE_GAVE_UP"]);
    assert!(decide(&mut s, &t, Verdict::OffOurChain, 1_700, 90_000, 6_600).is_empty(), "never past PAUSE_ATTEMPTS");
    // the state survives a restart (serialized), the incident with it
    let back: WatchState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.incident.as_ref().map(|i| i.attempts), Some(PAUSE_ATTEMPTS));
}

/// A-2: the bridge already frozen (`pausedUntil` after the L1 clock) when an incident opens or a run is due (an operator's
/// pause; a run whose outcome was lost: sent, then timed out, or the watcher stopped while it ran): no run (it would
/// charge 48 h of the budget for nothing), the incident counts as paused, said once. A freeze that ended is none.
#[test]
fn a2_no_pause_run_while_the_bridge_is_already_frozen() {
    let mut s = WatchState::default();
    let frozen = Tip { paused_until: 5_000, ..tip(1, 1_000) };
    assert_eq!(events(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_100, 10, 6_600)), vec!["ALREADY_PAUSED"]);
    assert!(s.incident.as_ref().is_some_and(|i| i.paused && i.attempts == 0), "{:?}", s.incident);
    assert!(decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 400, 6_600).is_empty(), "said once");
    // a run that failed (its outcome lost), the bridge frozen meanwhile: not run again
    let mut s = WatchState::default();
    assert_eq!(events(&decide(&mut s, &tip(1, 1_000), Verdict::OffOurChain, 1_100, 10, 6_600)), vec!["pause"]);
    pause_started(&mut s, 10);
    pause_ended(&mut s, false, "timed out");
    assert_eq!(events(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 10 + PAUSE_RETRY_SECS, 6_600)), vec!["ALREADY_PAUSED"]);
    assert!(decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 10 + 2 * PAUSE_RETRY_SECS, 6_600).is_empty());
    // `paused()` is `now < pausedUntil`: a freeze ending now is over
    let mut s = WatchState::default();
    let ended = Tip { paused_until: 1_100, ..tip(1, 1_000) };
    assert_eq!(events(&decide(&mut s, &ended, Verdict::OffOurChain, 1_100, 10, 6_600)), vec!["pause"]);
}

/// A-2: no new point for MORE than `alert_after_secs` on the L1 clock: one alert for that point (never a pause), none
/// at exactly that age, a new point re-arms it.
#[test]
fn a2_the_alert_for_a_late_point_is_raised_once_per_point() {
    let mut s = WatchState::default();
    assert!(decide(&mut s, &tip(1, 1_000), Verdict::OnOurChain, 1_000 + 6_600, 5, 6_600).is_empty(), "exactly 6 600 s: not more");
    assert_eq!(events(&decide(&mut s, &tip(1, 1_000), Verdict::OnOurChain, 1_000 + 6_601, 6, 6_600)), vec!["NO_NEW_POINT"]);
    assert!(decide(&mut s, &tip(1, 1_000), Verdict::OnOurChain, 1_000 + 9_000, 7, 6_600).is_empty(), "once per point");
    assert!(decide(&mut s, &tip(2, 9_000), Verdict::OnOurChain, 9_100, 8, 6_600).is_empty(), "a new point, young");
    assert_eq!(events(&decide(&mut s, &tip(2, 9_000), Verdict::Unknown, 9_000 + 7_000, 9, 6_600)), vec!["NO_NEW_POINT"], "re-armed");
    // a late point AND a tip off the chain: both
    let mut s = WatchState::default();
    assert_eq!(events(&decide(&mut s, &tip(1, 1_000), Verdict::OffOurChain, 9_000, 1, 6_600)), vec!["pause", "NO_NEW_POINT"]);
}

/// A-2: a watcher that could read nothing (the L1 or its nodes) for more than `alert_after_secs` of its own clock says so
/// once, through the alert command: it cannot see a point, so it cannot say none is late.
#[test]
fn a2_a_watcher_that_reads_nothing_for_alert_after_secs_says_so_once() {
    let mut alerted = false;
    assert!(blind_alert(1_000, 1_000 + 6_600, 6_600, &mut alerted).is_none());
    assert_eq!(events(&blind_alert(1_000, 1_000 + 6_601, 6_600, &mut alerted).into_iter().collect::<Vec<_>>()), vec!["WATCHER_BLIND"]);
    assert!(alerted);
    assert!(blind_alert(1_000, 1_000 + 9_000, 6_600, &mut alerted).is_none(), "once");
}

/// A-2: the operator's commands are argv lists, never given to a shell (`$(…)` stays as written), their placeholders
/// replaced, and each one ends within its timeout: past it, its whole process group is killed (a child it started
/// included) and the run is an error.
#[test]
fn a2_commands_are_argv_never_a_shell_and_bounded_by_their_timeout() {
    let argv = substitute(&["/opt/alert".to_string(), "{event}".to_string(), "at {detail} now".to_string()], &[("{event}", "PAUSED"), ("{detail}", "L=5")]);
    assert_eq!(argv, vec!["/opt/alert", "PAUSED", "at L=5 now"]);
    // through a shell, "$(echo x)" would be "x"
    let st = run_command(&["/usr/bin/test".to_string(), "$(echo x)".to_string(), "=".to_string(), "x".to_string()], Duration::from_secs(10)).unwrap();
    assert!(!st.success(), "argv given to a shell");
    assert!(run_command(&["/bin/true".to_string()], Duration::from_secs(10)).unwrap().success());
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("child.pid");
    let script = format!("sleep 30 & echo $! > {}; wait", pid_file.display());
    let start = std::time::Instant::now();
    let e = run_command(&["/bin/sh".to_string(), "-c".to_string(), script], Duration::from_secs(1)).unwrap_err().to_string();
    assert!(e.contains("timed out after 1 s"), "{e}");
    assert!(start.elapsed() < Duration::from_secs(10));
    let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let alive = std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| !s.contains(") Z "));
    assert!(!alive, "the command's own child {pid} outlived its timeout");
    assert!(run_command(&[], Duration::from_secs(1)).is_err(), "no program");
}

// ---------------------------------------------------------------- GU-1, GU-2 (audit n°2)

/// GU-1: nodes that answer "no block at that height" (the locked height is above their tip) are witnesses of that:
/// `min_nodes` of them make the tip's locked block above our tip; a node silent is still no witness.
#[test]
fn gu1_a_locked_block_above_the_tip_of_min_nodes_nodes_is_seen() {
    use Verdict::*;
    let ours = [1u8; 32];
    let (o, x, none) = (Some(Some(ours)), Some(Some([2u8; 32])), Some(None));
    assert_eq!(locked_verdict_of(&ours, &[none, none], 2), AboveOurTip);
    assert_eq!(locked_verdict_of(&ours, &[none, None], 2), Unknown, "a silent node is no witness");
    assert_eq!(locked_verdict_of(&ours, &[none, o], 2), Unknown);
    assert_eq!(locked_verdict_of(&ours, &[none, none, o], 2), AboveOurTip, "two of three hold nothing there");
    assert_eq!(locked_verdict_of(&ours, &[o, o], 2), OnOurChain);
    assert_eq!(locked_verdict_of(&ours, &[x, x], 2), OffOurChain);
    assert_eq!(locked_verdict_of(&ours, &[o, o, none, none], 2), OnOurChain, "two hold it: on our chain");
}

/// GU-1 (audit test): a forged point locks "honest tip + 1 000" (or a 55 % pool's private branch passed the honest
/// tip): both nodes answer nothing at that height, and the contract's tip changes at every round. Today: nothing.
/// Expected: once it lasts `ABOVE_TIP_PAUSE_SECS`, an incident and its pause (once); back on our chain: over.
#[test]
fn gu1_a_tip_that_locks_above_our_nodes_tip_is_paused_once_it_lasts() {
    let mut s = WatchState::default();
    let answers = [Some(None), Some(None)];
    let mut paused_at = None;
    for round in 0..8u64 {
        let now = 1_000 + round * 300;
        let t = Tip { locked_height: 20_000 + round, ..tip(10 + round as u8, 5_000 + round) };
        let v = locked_verdict_of(&t.locked_block, &answers, 2);
        let actions = decide(&mut s, &t, v, 5_100 + round, now, 6_600);
        if events(&actions).contains(&"pause".to_string()) {
            assert!(paused_at.is_none(), "one pause");
            paused_at = Some(now);
            pause_started(&mut s, now);
            pause_ended(&mut s, true, "exit status 0");
        }
    }
    assert_eq!(paused_at, Some(1_000 + ABOVE_TIP_PAUSE_SECS));
    assert!(s.incident.is_some());
    // the nodes catch up and hold it: over, and the clock starts again
    let t = tip(30, 9_000);
    assert!(decide(&mut s, &t, Verdict::OnOurChain, 9_100, 9_000, 6_600).is_empty());
    assert!(s.incident.is_none() && s.above_since.is_none());
    // a short stretch above (a node restarting): no pause
    let mut s = WatchState::default();
    assert!(decide(&mut s, &t, Verdict::AboveOurTip, 9_100, 100, 6_600).is_empty());
    assert!(decide(&mut s, &t, Verdict::AboveOurTip, 9_100, 100 + ABOVE_TIP_PAUSE_SECS - 1, 6_600).is_empty());
    assert!(decide(&mut s, &t, Verdict::OnOurChain, 9_100, 100 + ABOVE_TIP_PAUSE_SECS, 6_600).is_empty());
    assert!(decide(&mut s, &t, Verdict::AboveOurTip, 9_100, 100 + ABOVE_TIP_PAUSE_SECS + 1, 6_600).is_empty(), "the clock started again");
}

/// GU-1, GU-2: an unknown verdict that lasts more than `alert_after_secs` (a node of two down, nodes that disagree, two
/// L1 sources that disagree) is alerted, once per stretch; a verdict again re-arms it.
#[test]
fn gu2_an_unknown_verdict_that_lasts_is_alerted_once() {
    let mut s = WatchState::default();
    let t = tip(1, 1_000);
    assert!(decide(&mut s, &t, Verdict::Unknown, 1_100, 10, 6_600).is_empty());
    assert!(decide(&mut s, &t, Verdict::Unknown, 1_100, 10 + 6_600, 6_600).is_empty(), "not more than alert_after_secs");
    assert_eq!(events(&decide(&mut s, &t, Verdict::Unknown, 1_100, 10 + 6_601, 6_600)), vec!["VERDICT_UNKNOWN"]);
    assert!(decide(&mut s, &t, Verdict::Unknown, 1_100, 20_000, 6_600).is_empty(), "once");
    assert!(decide(&mut s, &t, Verdict::OnOurChain, 1_100, 20_001, 6_600).is_empty());
    assert!(decide(&mut s, &t, Verdict::Unknown, 1_100, 20_002, 6_600).is_empty());
    assert_eq!(events(&decide(&mut s, &t, Verdict::Unknown, 1_100, 20_002 + 6_601, 6_600)), vec!["VERDICT_UNKNOWN"], "re-armed");
    // the state survives a restart
    let back: WatchState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back, s);
    // a watch.json of the older watcher loads
    let old: WatchState = serde_json::from_str(r#"{"incident": null, "stale_alerted": "ab"}"#).unwrap();
    assert_eq!((old.above_since, old.unknown_since, old.unknown_alerted), (None, None, false));
}

/// GU-2: a second L1 endpoint (`l1_rpc_second`, optional: without it, as before): the tip must read the same from both
/// (point, locked block, locked height), else no verdict on it (no pause on one source's word).
#[test]
fn gu2_a_second_l1_source_must_agree_before_any_pause() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let c = WatchConfig::parse(&watch_toml(&mainnet, 1, "l1_rpc_second = \"http://127.0.0.1:8546\"\n")).expect("a second L1 endpoint");
    assert_eq!(c.l1_rpc_second.as_deref(), Some("http://127.0.0.1:8546"));
    let testnet = hex::encode(crate::TESTNET_NETWORK);
    assert_eq!(WatchConfig::parse(&watch_toml(&testnet, 84_532, "")).unwrap().l1_rpc_second, None, "none by default off the mainnet: as before");
    let same = WatchConfig::parse(&watch_toml(&mainnet, 1, "l1_rpc_second = \"http://127.0.0.1:8545\"\n")).unwrap_err().to_string();
    assert!(same.contains("l1_rpc_second"), "{same}");
    assert!(WatchConfig::parse(&watch_toml(&mainnet, 1, "l1_rpc_second = \"http://10.1.2.3:8546\"\n")).is_err(), "checked like l1_rpc");
    let t = tip(1, 1_000);
    assert_eq!(merge_tips(Some(t), None), Ok(Some(t)), "no second source");
    assert_eq!(merge_tips(Some(t), Some(Some(t))), Ok(Some(t)));
    assert_eq!(merge_tips(None, Some(None)), Ok(None));
    assert!(merge_tips(Some(t), Some(Some(tip(2, 1_000)))).is_err());
    assert!(merge_tips(Some(t), Some(Some(Tip { locked_block: [0xcd; 32], ..t }))).is_err());
    assert!(merge_tips(Some(t), Some(Some(Tip { locked_height: 101, ..t }))).is_err());
    assert!(merge_tips(Some(t), Some(None)).is_err());
}

// ---------------------------------------------------------------- external audit 2026-11-18 (E-2, E-3, M-9, F-9)

/// E-3: the two L1 endpoints must also agree on the tip's acceptance time and on `pausedUntil`: one endpoint that says
/// "frozen" (no pause run) or "a new point" (no NO_NEW_POINT) on its own word gives no verdict.
#[test]
fn e3_the_two_l1_sources_must_agree_on_the_freeze_and_the_acceptance_time_too() {
    let t = tip(1, 1_000);
    assert_eq!(merge_tips(Some(t), Some(Some(t))), Ok(Some(t)));
    let e = merge_tips(Some(Tip { paused_until: 90_000, ..t }), Some(Some(t))).unwrap_err();
    assert!(e.contains("paused"), "{e}");
    let e = merge_tips(Some(t), Some(Some(Tip { accepted_at: 999, ..t }))).unwrap_err();
    assert!(e.contains("accepted"), "{e}");
}

/// E-3: on the mainnet, a second L1 endpoint is required (one lying endpoint must not blind the watcher); the public
/// test network keeps it optional (its deployed watch.toml has none).
#[test]
fn e3_the_mainnet_watcher_requires_a_second_l1_endpoint() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let without = watch_toml(&mainnet, 1, "").replace("l1_rpc_second = \"http://127.0.0.1:8547\"\n", "");
    let e = WatchConfig::parse(&without).unwrap_err().to_string();
    assert!(e.contains("l1_rpc_second"), "{e}");
    assert!(WatchConfig::parse(&watch_toml(&mainnet, 1, "")).is_ok());
    let testnet = hex::encode(crate::TESTNET_NETWORK);
    assert!(WatchConfig::parse(&watch_toml(&testnet, 84_532, "")).is_ok(), "the public test network: optional");
}

/// M-9: a freeze this watcher saw (its pause, or one it found) that ENDS while the tip is still off the chain is alerted
/// (PAUSE_EXPIRED) and the incident pauses again (its runs counted afresh); a pause whose freeze was never seen is not
/// taken for an expired one.
#[test]
fn m9_a_freeze_that_expires_with_the_tip_still_off_the_chain_is_alerted_and_paused_again() {
    let mut s = WatchState::default();
    let t = tip(1, 1_000);
    assert_eq!(events(&decide(&mut s, &t, Verdict::OffOurChain, 1_100, 10, 6_600)), vec!["pause"]);
    pause_started(&mut s, 10);
    assert_eq!(events(&[pause_ended(&mut s, true, "exit status 0")]), vec!["PAUSED"]);
    // not frozen yet (the pause not mined at this poll): nothing, never an "expiry"
    assert!(decide(&mut s, &t, Verdict::OffOurChain, 1_150, 70, 6_600).iter().all(|a| !matches!(a, Action::Pause)));
    let frozen = Tip { paused_until: 1_100 + 172_800, ..t };
    let quiet = |a: &[Action]| events(a).into_iter().filter(|e| e != "NO_NEW_POINT").collect::<Vec<_>>();
    assert!(quiet(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 130, 6_600)).is_empty(), "frozen: nothing");
    assert!(quiet(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_100 + 172_799, 200_000, 6_600)).is_empty(), "still frozen");
    assert_eq!(quiet(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_100 + 172_800, 200_060, 6_600)), vec!["PAUSE_EXPIRED", "pause"]);
    assert!(s.incident.as_ref().is_some_and(|i| !i.paused && i.attempts == 0), "{:?}", s.incident);
    // an operator's freeze found at the incident's start expires the same way
    let mut s = WatchState::default();
    assert_eq!(quiet(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 10, 6_600)), vec!["ALREADY_PAUSED"]);
    assert_eq!(quiet(&decide(&mut s, &frozen, Verdict::OffOurChain, 1_100 + 172_801, 99_000, 6_600)), vec!["PAUSE_EXPIRED", "pause"]);
    // back on the chain meanwhile: over, nothing
    let mut s = WatchState::default();
    decide(&mut s, &frozen, Verdict::OffOurChain, 1_200, 10, 6_600);
    assert!(quiet(&decide(&mut s, &frozen, Verdict::OnOurChain, 1_100 + 172_801, 99_000, 6_600)).is_empty());
}

/// F-9: an alert whose command failed is kept (in watch.json) and sent again, oldest first, a bounded queue; the late
/// point keeps its own re-arm.
#[test]
fn f9_an_alert_whose_command_failed_is_kept_and_sent_again() {
    let mut s = WatchState::default();
    alert_failed(&mut s, "PAUSED", "d1");
    alert_failed(&mut s, "PAUSE_FAILED", "d2");
    s.stale_alerted = Some("ab".into());
    alert_failed(&mut s, "NO_NEW_POINT", "d3");
    assert_eq!(s.stale_alerted, None, "the late point is raised again by decide");
    let back: WatchState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back.pending_alerts, vec![("PAUSED".to_string(), "d1".to_string()), ("PAUSE_FAILED".to_string(), "d2".to_string())]);
    for i in 0..(2 * MAX_PENDING_ALERTS) {
        alert_failed(&mut s, "VERDICT_UNKNOWN", &format!("{i}"));
    }
    assert_eq!(s.pending_alerts.len(), MAX_PENDING_ALERTS, "bounded");
    assert_eq!(s.pending_alerts.last().unwrap().1, format!("{}", 2 * MAX_PENDING_ALERTS - 1), "the newest kept");
    let old: WatchState = serde_json::from_str(r#"{"incident": null}"#).unwrap();
    assert!(old.pending_alerts.is_empty(), "an older watch.json loads");
}

/// E-2: the watcher's blindness is judged by a thread of its own: a read that hangs (a call held by its server) no
/// longer silences WATCHER_BLIND; said once per blind stretch, again after a read.
#[test]
fn e2_the_blind_alert_goes_out_while_the_watchers_loop_hangs() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    let last_ok = Arc::new(AtomicU64::new(crate::now_unix().saturating_sub(10)));
    let said = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = said.clone();
    let _dog = spawn_blind_watchdog(last_ok.clone(), 5, Duration::from_millis(50), move |event: &str, _detail: &str| {
        sink.lock().unwrap().push(event.to_string());
        true
    });
    // the loop "hangs": last_ok never moves
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(*said.lock().unwrap(), vec!["WATCHER_BLIND".to_string()], "once");
    last_ok.store(crate::now_unix(), Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(200));
    last_ok.store(crate::now_unix().saturating_sub(10), Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(said.lock().unwrap().len(), 2, "again after a read");
}

/// F-6 (external audit 2026-11-18): the watcher's two nodes on the same loopback port are one node.
#[test]
fn f6_the_watchers_same_loopback_port_twice_is_one_node() {
    let testnet = hex::encode(crate::TESTNET_NETWORK);
    let text = watch_toml(&testnet, 84_532, "").replace("http://127.0.0.1:19721", "http://127.0.0.2:19711/");
    assert!(WatchConfig::parse(&text).unwrap_err().to_string().contains("twice"));
}
