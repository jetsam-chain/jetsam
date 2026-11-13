use super::*;

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::rpc::Egress;

/// A fake node on 127.0.0.1: every request answered with the JSON-RPC envelope `answer()` gives at that moment (a
/// test changes it while the client runs); the methods asked, in order.
fn fake_node(answer: Arc<Mutex<Value>>) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let asked: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen = asked.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut req = String::new();
            let mut buf = vec![0u8; 4096];
            while !req.contains("\"params\"") {
                match s.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => req.push_str(&String::from_utf8_lossy(&buf[..n])),
                }
            }
            let method = req.split("\"method\":\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or_default();
            seen.lock().unwrap().push(method.to_string());
            let body = answer.lock().unwrap().to_string();
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
        }
    });
    (port, asked)
}

fn result(v: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": 1, "result": v})
}

fn error(code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": message}})
}

/// A port nothing listens on (bound, then closed).
fn silent_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn client(port: u16) -> NodeClient {
    NodeClient::fill(&format!("http://127.0.0.1:{port}"), Egress::PrivateBridge)
}

/// `this_build()` with `key` set to `value` (or removed: `None`).
fn with(key: &str, value: Option<Value>) -> Value {
    let mut v = this_build();
    let map = v.as_object_mut().unwrap();
    match value {
        Some(value) => {
            map.insert(key.to_string(), value);
        }
        None => {
            map.remove(key);
        }
    }
    v
}

/// Another value than `v`, of the same JSON kind where it has one.
fn other(v: &Value) -> Value {
    match v {
        Value::Null => json!(12_345),
        Value::Number(n) => json!(n.as_u64().unwrap() + 1),
        Value::String(s) => json!(if s.starts_with('0') { s.replacen('0', "1", 1) } else { format!("0{}", &s[1..]) }),
        other => panic!("not a field of the activations: {other}"),
    }
}

#[test]
fn activ_this_build_is_the_record_of_its_rules() {
    let built = this_build();
    assert_eq!(built, serde_json::to_value(jetsam_chain::consensus::activations::Activations::current()).unwrap());
    assert_eq!(built["network"], json!(hex::encode(jetsam_chain::consensus::guardian::network_id())));
    assert_eq!(
        built["deposits_from_height"],
        json!(jetsam_chain::consensus::vault::VaultRules::current().deposits_from_height)
    );
}

#[test]
fn activ_a_node_of_this_build_is_accepted() {
    let built = this_build();
    for scope in [Scope::All, Scope::Network] {
        assert_eq!(judge("n0", &Ok(built.clone()), &built, scope), Ok(()), "{scope:?}");
    }
}

/// Every field, one at a time: another value refuses the node, and the refusal names the node, the field, the node's
/// value and this build's (`mauvais champ comparé`: each field is compared, with itself).
#[test]
fn activ_every_field_that_differs_refuses_the_node_and_is_named() {
    let built = this_build();
    let keys: Vec<String> = built.as_object().unwrap().keys().cloned().collect();
    // 14 since FIXRC (`return_fix_height`): a program of before refuses a node of this build at its start.
    assert_eq!(keys.len(), 14, "{keys:?}");
    for key in &keys {
        let value = other(&built[key]);
        let published = with(key, Some(value.clone()));
        let refusal = judge("n1", &Ok(published), &built, Scope::All).expect_err(key);
        let text = refusal.to_string();
        assert_eq!(refusal.node, "n1");
        assert!(text.starts_with("node n1: "), "{text}");
        assert!(text.contains(&format!("{key} = {value}")) && text.contains(&format!("this build {}", built[key])), "{key}: {text}");
        // no other field is named
        for k in keys.iter().filter(|k| *k != key && !key.contains(k.as_str()) && !k.contains(key.as_str())) {
            assert!(!text.contains(&format!("{k} =")), "{key}: {k} named too: {text}");
        }
    }
}

/// D171's case: the node applies a rule (`F`, `H`, the relay key) the build does not: refused, both values said.
#[test]
fn activ_the_incident_fields_are_compared() {
    let built = this_build();
    for key in ["deposits_from_height", "base_return_height", "fast_release_key", "v1_7_height", "network"] {
        let published = with(key, Some(other(&built[key])));
        assert!(judge("n0", &Ok(published), &built, Scope::All).is_err(), "{key}");
    }
}

#[test]
fn activ_a_node_without_a_field_this_build_has_is_refused_as_older() {
    let built = this_build();
    for key in built.as_object().unwrap().keys() {
        let refusal = judge("n0", &Ok(with(key, None)), &built, Scope::All).expect_err(key).to_string();
        assert!(refusal.contains(&format!("publishes no {key}")) && refusal.contains("update the node"), "{key}: {refusal}");
    }
}

#[test]
fn activ_a_rule_this_build_does_not_know_refuses_the_node_but_not_the_watcher() {
    let built = this_build();
    let published = with("v1_8_height", Some(json!(9_600)));
    let refusal = judge("n2", &Ok(published.clone()), &built, Scope::All).unwrap_err().to_string();
    assert!(refusal.contains("v1_8_height = 9600") && refusal.contains("does not know") && refusal.contains("update this program"), "{refusal}");
    assert_eq!(judge("n2", &Ok(published), &built, Scope::Network), Ok(()));
}

/// The watcher reads block ids, no deposit: the network only, but the network always.
#[test]
fn activ_the_watcher_compares_the_network_only() {
    let built = this_build();
    for key in built.as_object().unwrap().keys().filter(|k| *k != "network") {
        let published = with(key, Some(other(&built[key])));
        assert_eq!(judge("w0", &Ok(published.clone()), &built, Scope::Network), Ok(()), "{key}");
        assert!(judge("w0", &Ok(published), &built, Scope::All).is_err(), "{key}");
        assert_eq!(judge("w0", &Ok(with(key, None)), &built, Scope::Network), Ok(()), "{key} absent");
    }
    let refusal = judge("w0", &Ok(with("network", Some(other(&built["network"])))), &built, Scope::Network).unwrap_err().to_string();
    assert!(refusal.contains("network = ") && refusal.contains("node w0"), "{refusal}");
    let refusal = judge("w0", &Ok(with("network", None)), &built, Scope::Network).unwrap_err().to_string();
    assert!(refusal.contains("publishes no network"), "{refusal}");
}

/// A node older than the method: refused (fail-closed), told to update, under every scope.
#[test]
fn activ_a_node_without_the_method_is_refused_and_told_to_update() {
    let built = this_build();
    for scope in [Scope::All, Scope::Network] {
        let refusal = judge("n0", &Err((METHOD_NOT_FOUND, "Method not found".into())), &built, scope).unwrap_err().to_string();
        assert!(refusal.starts_with("node n0: ") && refusal.contains("jetsam_getActivations") && refusal.contains("update the node"), "{refusal}");
    }
}

#[test]
fn activ_an_error_or_an_answer_that_is_not_a_record_refuses_the_node() {
    let built = this_build();
    let refusal = judge("n0", &Err((-32603, "internal".into())), &built, Scope::All).unwrap_err().to_string();
    assert!(refusal.contains("-32603") && refusal.contains("internal"), "{refusal}");
    for answer in [Value::Null, json!("activations"), json!([1, 2]), json!(7)] {
        for scope in [Scope::All, Scope::Network] {
            let refusal = judge("n0", &Ok(answer.clone()), &built, scope).unwrap_err().to_string();
            assert!(refusal.contains("not a record"), "{answer}: {refusal}");
        }
    }
}

/// Over HTTP: this build's record accepted, no method refused, a silent node an error (neither refused nor accepted).
#[test]
fn activ_check_reads_the_node_and_tells_a_silent_one_from_a_refused_one() {
    let answer = Arc::new(Mutex::new(result(this_build())));
    let (port, asked) = fake_node(answer.clone());
    assert_eq!(check(&client(port), "n0", Scope::All).unwrap(), Ok(()));
    assert_eq!(asked.lock().unwrap().as_slice(), ["jetsam_getActivations"]);
    *answer.lock().unwrap() = error(METHOD_NOT_FOUND, "Method not found");
    let refusal = check(&client(port), "n0", Scope::All).unwrap().unwrap_err();
    assert!(refusal.reason.contains("update the node"), "{refusal}");
    *answer.lock().unwrap() = result(with("deposits_from_height", Some(json!(1))));
    assert!(check(&client(port), "n0", Scope::All).unwrap().is_err());
    assert_eq!(check(&client(port), "n0", Scope::Network).unwrap(), Ok(()));
    assert!(check(&client(silent_port()), "n0", Scope::All).is_err(), "a silent node is not read");
}

/// At a start: a refusal or a silent node is an error naming the node and saying the program does not start.
#[test]
fn activ_require_names_the_node_and_the_program_that_does_not_start() {
    let answer = Arc::new(Mutex::new(result(this_build())));
    let (port, _) = fake_node(answer.clone());
    require(&client(port), "n0", Scope::All, "jg-relay").unwrap();
    *answer.lock().unwrap() = result(with("v1_7_height", Some(json!(8_640))));
    let e = format!("{:#}", require(&client(port), "n0", Scope::All, "jg-relay").unwrap_err());
    assert!(e.contains("node n0") && e.contains("v1_7_height = 8640") && e.ends_with("jg-relay does not start"), "{e}");
    *answer.lock().unwrap() = error(METHOD_NOT_FOUND, "Method not found");
    let e = format!("{:#}", require(&client(port), "n0", Scope::Network, "jg-relay watch").unwrap_err());
    assert!(e.contains("update the node") && e.ends_with("jg-relay watch does not start"), "{e}");
    let e = format!("{:#}", require(&client(silent_port()), "n9", Scope::All, "jg-fill").unwrap_err());
    assert!(e.contains("node n9") && e.contains("not read") && e.contains("jg-fill does not start"), "{e}");
}

/// `fast-run`'s node: silent, it is not used and read again at the next use (no refusal); then read wrong, refused
/// (the cause a `Refusal`); once read right, used without being read again.
#[test]
fn activ_a_checked_node_is_read_before_its_first_use_and_refused_when_wrong() {
    let port = silent_port();
    let mut node = Checked::new(client(port), "http://127.0.0.1:x", Scope::All);
    let e = node.get(|c| c).map(|_| ()).unwrap_err();
    assert!(!e.is::<Refusal>() && !node.verified(), "{e:#}");
    let answer = Arc::new(Mutex::new(result(with("base_return_height", Some(json!(9_999))))));
    let (port, asked) = fake_node(answer.clone());
    let mut node = Checked::new(client(port), "fast", Scope::All);
    let e = node.get(|c| c).map(|_| ()).unwrap_err();
    assert!(e.is::<Refusal>() && !node.verified(), "{e:#}");
    assert!(format!("{e:#}").contains("base_return_height = 9999"), "{e:#}");
    *answer.lock().unwrap() = result(this_build());
    assert!(node.get(|c| c).is_ok() && node.verified());
    let reads = asked.lock().unwrap().len();
    *answer.lock().unwrap() = error(METHOD_NOT_FOUND, "Method not found");
    assert!(node.get(|c| c).is_ok(), "read once, before its first use");
    assert_eq!(asked.lock().unwrap().len(), reads, "not read again");
}

/// `fast-run`'s start: its node found this build's, said; silent, said as an ALERT and not refused (read before its
/// first use); refused, an error naming the node and the value, ending with "fast-run does not start".
#[test]
fn activ_a_checked_node_at_the_start_stops_the_program_only_when_refused() {
    let said = Mutex::new(Vec::new());
    let say = |line: &str| said.lock().unwrap().push(line.to_string());
    let mut silent = Checked::new(client(silent_port()), "s", Scope::All);
    silent.at_start(|c| c, "fast-run", &say).unwrap();
    assert!(!silent.verified());
    let answer = Arc::new(Mutex::new(result(with("fast_release_key", Some(json!("00".repeat(48)))))));
    let (port, _) = fake_node(answer.clone());
    let mut wrong = Checked::new(client(port), "w", Scope::All);
    let e = format!("{:#}", wrong.at_start(|c| c, "fast-run", &say).unwrap_err());
    assert!(e.contains("node w: ") && e.contains("fast_release_key = ") && e.ends_with("fast-run does not start"), "{e}");
    assert!(!wrong.verified());
    *answer.lock().unwrap() = result(this_build());
    let mut right = Checked::new(client(port), "r", Scope::All);
    right.at_start(|c| c, "fast-run", &say).unwrap();
    assert!(right.verified());
    let said = said.into_inner().unwrap();
    assert_eq!(said.len(), 2, "{said:?}");
    assert!(said[0].starts_with("ALERT node s not read at the start") && said[0].contains("read before its first use"), "{said:?}");
    assert_eq!(said[1], "node r: its activations are this build's");
}
