#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G6 = V4 through the SP1 EXECUTOR (execution only, no proof, CPU): the node's vectors
(jetsam_guardian/tests/fixtures/g6_v4_vectors*.json) run through the claim guest v3 built with the
vectors' clock (build-guest.sh claim test-clock).

For every vector and every user group of its block (and one index past the last): one execution of the
guest on the whole block (input JCLMIN03). Required: an execution succeeds exactly when the node accepts
the block and that group is one of its deposits, and then the 192 public values carry the node's height,
tx_root, recipient, amount and logical txid.

  sp1-exec-vectors.py <sp1-host> <claim-v3-test-clock.elf> <workdir> <fixture.json>...
<sp1-host> is a runner `host exec <elf> <input>` that prints `exit=<code> ... public=<hex>` (the v2
prototype's spv-host). Each execution runs in an empty network namespace (unshare --net)."""
import json
import os
import re
import struct
import subprocess
import sys

CLAIM_DOMAIN_KECCAK = "eef878f12f17c5566eaa3d77b723dc5bfc9372f30c4ed3acc794211526398161"


def user_groups(pages):
    """The number of user groups (START..END), after the coinbase and the optional development payout."""
    b = [bytes.fromhex(p) for p in pages]
    bitmap = lambda x: x[320] | (x[321] << 8)
    i = 1 + (1 if len(b) > 1 and b[1][322] == 1 and bitmap(b[1]) == 0x300 else 0)
    n = 0
    while i < len(b):
        while not bitmap(b[i]) & (1 << 11):
            i += 1
        i += 1
        n += 1
    return n


def claim_input(v, group):
    pages = [bytes.fromhex(p) for p in v["pages"]]
    return b"JCLMIN03" + struct.pack("<Q", v["height"]) + bytes.fromhex(v["tx_root"]) + struct.pack("<II", group, len(pages)) + b"".join(pages)


def word(x):
    return "%064x" % x


def main():
    host, elf, work = sys.argv[1:4]
    os.makedirs(work, exist_ok=True)
    total = agree = 0
    cycles = []
    for fixture in sys.argv[4:]:
        for v in json.load(open(fixture))["vectors"]:
            node = v["node"]
            deposits = {d["group"]: d for d in node.get("deposits", [])} if node["accepted"] else {}
            for g in range(user_groups(v["pages"]) + 1):
                path = os.path.join(work, "%s-%d.bin" % (v["name"], g))
                open(path, "wb").write(claim_input(v, g))
                r = subprocess.run(["unshare", "--net", host, "exec", elf, path], capture_output=True, text=True)
                m = re.search(r"instructions=(\d+) .*exit=(\d+) .*public=([0-9a-f]*)", r.stdout)
                ok = r.returncode == 0 and m is not None and m.group(2) == "0"
                want = deposits.get(g)
                if want is None:
                    same = not ok
                    what = "refused" if not ok else "PROVED " + m.group(3)
                else:
                    pv = m.group(3) if ok else ""
                    expect = CLAIM_DOMAIN_KECCAK + word(v["height"]) + v["tx_root"] + "0" * 24 + want["recipient"][2:] + word(want["amount"]) + want["txid"]
                    same = ok and pv == expect
                    what = ("proved %d" % want["amount"]) if same else ("WRONG " + (pv or r.stdout[-200:] + r.stderr[-300:]))
                    if ok:
                        cycles.append(int(m.group(1)))
                total += 1
                agree += same
                print("%-52s group %d: %s %s" % (v["name"], g, "MATCH" if same else "MISMATCH", what), flush=True)
    print("TOTAL executions %d, agree %d, disagree %d" % (total, agree, total - agree))
    if cycles:
        print("instructions of the proved claims: min %d max %d" % (min(cycles), max(cycles)))
    sys.exit(0 if agree == total else 1)


if __name__ == "__main__":
    main()
