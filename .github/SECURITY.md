# Security

## Supported versions

| Release line | Security support |
|---|---|
| `1.0.x` mainnet | Supported |
| Withdrawn pre-v2 releases | Unsupported |

Security fixes are published for the current mainnet release line. Node
operators and wallet users should run its latest patch release.

## Reporting a vulnerability

Use **[Report a vulnerability](https://github.com/jetsam-chain/jetsam/security/advisories/new)**
on this repository's Security tab. The report stays private to the maintainers
until an advisory is published, and it needs nothing beyond a GitHub account.

**Do not open an issue, a discussion, or any other public thread for an
unpatched vulnerability.** Jetsam secures a live network: a public report arms
an attacker before a fix exists.

Include the affected release or commit, component, platform, expected and
observed behavior, security impact, and a minimal reproducer when possible.
Reports concerning consensus, proof verification, wallet authorization or
secret handling, synchronization, peer-to-peer networking, RPC boundaries, or
release artifacts are especially important.

We will confirm receipt, assess the report, and coordinate remediation and
disclosure with the reporter.

Jetsam does not currently operate a formal bug bounty. Receipt of a report
does not imply compensation.

## Everything that is not a vulnerability

- **A bug in the node, the wallet or the miner** —
  [open an issue](https://github.com/jetsam-chain/jetsam/issues/new/choose).
- **A question, an idea, mining or operations talk** —
  [Discussions](https://github.com/jetsam-chain/jetsam/discussions).

## Verifying a release

Every release publishes `SHA256SUMS` next to its binaries. Check what you
downloaded before you run it:

```sh
sha256sum -c --ignore-missing SHA256SUMS
```

A binary whose checksum does not match the release page did not come from this
project. Report that through the vulnerability channel above.
