// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! D171, D176 (11 November 2026): a relayer and a robot built before a rule of their nodes read the nodes' blocks
//! otherwise than the nodes accept them (`BadMarker`, then `ReleasesDoNotMatchMarkers`), and nothing said so at their
//! start. Before a program of this crate uses a node, it reads the activations the node applies
//! (`jetsam_getActivations`: [`jetsam_chain::consensus::activations::Activations`], read from the rules the node's
//! validation reads) and compares them with its own build's, read by the same function. A value that differs, a
//! field one of the two does not know, or no such method (a node older than this build): the node is refused and the
//! program does not start. A node that cannot be read is not refused, and not used either before it was read.

use serde_json::Value;

use crate::rpc::NodeClient;

/// The node method (wire name `jetsam_getActivations`).
pub const METHOD: &str = "getActivations";

/// JSON-RPC's "method not found": a node older than [`METHOD`].
pub const METHOD_NOT_FOUND: i64 = -32601;

/// What a program compares of a node's activations with its build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every activation: the programs that read blocks and their deposits, or hand the node objects judged under its
    /// rules (the relayer, `fast-run`, `jg-fill`).
    All,
    /// The network only: the watcher, which reads block ids and no deposit.
    Network,
}

/// A node whose activations are not this build's: the program does not use it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub node: String,
    pub reason: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node {}: {}", self.node, self.reason)
    }
}

impl std::error::Error for Refusal {}

/// This build's activations, as a node of this build publishes them.
pub fn this_build() -> Value {
    serde_json::to_value(jetsam_chain::consensus::activations::Activations::current())
        .expect("the activations are plain JSON")
}

/// The node `node`'s answer to [`METHOD`] (its result, or its JSON-RPC error: code and message) against this build's
/// activations `built`, under `scope`.
pub fn judge(node: &str, answer: &Result<Value, (i64, String)>, built: &Value, scope: Scope) -> Result<(), Refusal> {
    let refuse = |reason: String| Err(Refusal { node: node.to_string(), reason });
    let published = match answer {
        Err((METHOD_NOT_FOUND, message)) => {
            return refuse(format!(
                "it publishes no activations (jetsam_{METHOD}: {message}): a node older than this build, which cannot tell \
                 whether this build reads its blocks as it accepts them; update the node (the nodes first, then the \
                 guardian programs)"
            ))
        }
        Err((code, message)) => {
            return refuse(format!("jetsam_{METHOD} answered the error {code} ({message}): its activations are unknown"))
        }
        Ok(published) => published,
    };
    let (Some(published), Some(built)) = (published.as_object(), built.as_object()) else {
        return refuse(format!("its jetsam_{METHOD} answer is not a record of activations: {published}"));
    };
    let mut gaps = Vec::new();
    for (key, value) in built.iter().filter(|(key, _)| scope == Scope::All || key.as_str() == "network") {
        match published.get(key) {
            None => gaps.push(format!("it publishes no {key} (this build {value}): a node older than this build; update the node")),
            Some(theirs) if theirs != value => gaps.push(format!("it applies {key} = {theirs}, this build {value}")),
            Some(_) => {}
        }
    }
    if scope == Scope::All {
        for (key, theirs) in published.iter().filter(|(key, _)| !built.contains_key(*key)) {
            gaps.push(format!("it applies {key} = {theirs}, a rule this build does not know: update this program"));
        }
    }
    if gaps.is_empty() {
        return Ok(());
    }
    let why = match scope {
        Scope::All => "this build would not read the node's blocks, nor make its objects, as the node accepts them (build it \
                       from the nodes' version, with their profile)",
        Scope::Network => "the node is not of this build's network",
    };
    refuse(format!("{}: {why}", gaps.join("; ")))
}

/// Read the activations of the node `node` and judge them: an error while it cannot be read (not reached, no HTTP
/// 200, no JSON-RPC answer): the node is neither refused nor used then.
pub fn check(client: &NodeClient, node: &str, scope: Scope) -> anyhow::Result<Result<(), Refusal>> {
    let answer = client.answer(METHOD, serde_json::json!([]))?;
    Ok(judge(node, &answer, &this_build(), scope))
}

/// [`check`] at the start of a program that reads each of its nodes there (the relayer, the watcher, `jg-fill`): a
/// node refused, or not read (as a silent node always stopped them there), is an error that ends with
/// "`program` does not start".
pub fn require(client: &NodeClient, node: &str, scope: Scope, program: &str) -> anyhow::Result<()> {
    match check(client, node, scope) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(refusal)) => Err(anyhow::anyhow!("{refusal}: {program} does not start")),
        Err(e) => Err(anyhow::anyhow!("node {node}: its activations (jetsam_{METHOD}) not read ({e:#}): {program} does not start")),
    }
}

/// The node of a program that starts without reading it (`fast-run`: its captures of Base run whatever its node
/// says): its activations read before its first use, then not again.
pub struct Checked<T> {
    node: T,
    name: String,
    scope: Scope,
    verified: bool,
}

impl<T> Checked<T> {
    pub fn new(node: T, name: &str, scope: Scope) -> Self {
        Self { node, name: name.to_string(), scope, verified: false }
    }

    /// The node, once its activations were found this build's (read now while they were not; `client` is its RPC
    /// client): an error while they cannot be read (the node is not used; read again at the next call), whose cause
    /// is a [`Refusal`] when they are not this build's.
    pub fn get(&mut self, client: impl Fn(&T) -> &NodeClient) -> anyhow::Result<&T> {
        if !self.verified {
            let judged = check(client(&self.node), &self.name, self.scope).map_err(|e| {
                e.context(format!("node {}: its activations (jetsam_{METHOD}) not read yet: not used", self.name))
            })?;
            judged?;
            self.verified = true;
        }
        Ok(&self.node)
    }

    /// At the program's start: its activations read now. Found this build's, or the node silent (said as an ALERT: it
    /// is read before its first use): `Ok`. Refused: an error that ends with "`program` does not start".
    pub fn at_start(&mut self, client: impl Fn(&T) -> &NodeClient, program: &str, say: &dyn Fn(&str)) -> anyhow::Result<()> {
        let read = self.get(client).map(|_| ());
        match read {
            Ok(()) => say(&format!("node {}: its activations are this build's", self.name)),
            Err(e) if e.is::<Refusal>() => anyhow::bail!("{e:#}: {program} does not start"),
            Err(e) => say(&format!("ALERT node {} not read at the start ({e:#}): read before its first use", self.name)),
        }
        Ok(())
    }

    /// Whether its activations were read and found this build's.
    pub fn verified(&self) -> bool {
        self.verified
    }
}

#[cfg(test)]
#[path = "activations_tests.rs"]
mod tests;
