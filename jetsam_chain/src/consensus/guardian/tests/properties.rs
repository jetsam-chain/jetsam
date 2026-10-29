// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.9 — properties under a random simulation (T1.1 §12, §13.2): stake
//! locks and unlocks, fundings (by pairs before the latch), claims, payouts,
//! attestations, one equivocation, reorganisations of 1 to 8 blocks and a
//! state rebuilt from its bytes every few blocks, all through
//! `apply_block_guardian` with synthetic group facts and a test key. Each
//! block checks INV-S, INV-R, INV-R2, INV-C, INV-U and the encoding.

use super::*;
use crate::consensus::params::MICRO_PER_JTM;

const JTM: u64 = MICRO_PER_JTM;
const J_D: u64 = J + 15_360;
const AFTER_J: u64 = 5_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, one_in: u64) -> bool {
        self.below(one_in) == 0
    }
}

/// The pools' balances as the UTXO set would hold them (V_S, V_R).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pools {
    stake_vault: i128,
    reward_pool: i128,
}

fn owner(i: u64) -> Address {
    Address([0x30 + i as u8; 32])
}

fn facts(
    group: usize,
    input_owner: Address,
    outputs: Vec<(Address, u64)>,
    fee: u64,
    ids: Vec<u64>,
) -> GuardianGroupFacts {
    GuardianGroupFacts {
        group,
        input_owner,
        outputs,
        fee,
        input_creation_ids: ids,
    }
}

struct Sim {
    rng: Rng,
    rules: GuardianRules,
    branch: MapBranch,
    height: u64,
    state: GuardianState,
    pools: Pools,
    /// Per applied block: the undo and the pools before it.
    history: Vec<(GuardianUndo, Pools)>,
    equivocated: bool,
    blocks: u64,
    reorgs: u64,
    attestations: u64,
}

fn stake_sum(state: &GuardianState) -> i128 {
    state
        .stakes
        .values()
        .map(|p| i128::from(p.active) + i128::from(p.activating))
        .sum::<i128>()
        + state
            .entries
            .values()
            .filter(|e| e.kind == PayoutEntryKind::Stake)
            .map(|e| i128::from(e.amount))
            .sum::<i128>()
}

fn reward_liabilities(state: &GuardianState) -> i128 {
    i128::from(state.fund_current)
        + i128::from(state.confiscated_current)
        + i128::from(state.carry)
        + state
            .balances
            .values()
            .map(|v| i128::from(*v))
            .sum::<i128>()
        + state
            .guardian_pending
            .iter()
            .map(|(v, _)| i128::from(*v))
            .sum::<i128>()
        + i128::from(state.guardian_balance)
        + state
            .entries
            .values()
            .filter(|e| e.kind != PayoutEntryKind::Stake)
            .map(|e| i128::from(e.amount))
            .sum::<i128>()
}

impl Sim {
    fn new(seed: u64, blocks: u64) -> Self {
        let start = J_D - 20;
        Self {
            rng: Rng(seed),
            rules: rules(),
            branch: MapBranch::new("sim", start - 1_000, start + blocks + 10),
            height: start,
            state: GuardianState::at_v1_7(1_000),
            pools: Pools::default(),
            history: Vec::new(),
            equivocated: false,
            blocks: 0,
            reorgs: 0,
            attestations: 0,
        }
    }

    /// One random block's groups and objects, and the pool flows they make.
    fn block(&mut self) -> (BlockGuardianContext, Pools) {
        let h = self.height;
        let mut groups = Vec::new();
        let mut objects = Vec::new();
        let mut flows = Pools::default();
        let latched = self.state.activated_at.is_some();
        let active = delegation_active(&self.state, h);
        let push = |groups: &mut Vec<GuardianGroupFacts>,
                    objects: &mut Vec<(usize, GuardianObject)>,
                    object: GuardianObject,
                    input: Address,
                    mut outputs: Vec<(Address, u64)>,
                    fee: u64,
                    ids: Vec<u64>| {
            outputs.push((object.marker(), 0));
            let index = groups.len();
            groups.push(facts(index, input, outputs, fee, ids));
            objects.push((index, object));
        };
        let (net, lab) = (self.rules.network_fund, self.rules.lab_fund);
        // Fundings: a pair before the latch, a single one after.
        if h >= J_D && !latched && self.rng.chance(3) {
            for funder in [net, lab] {
                let amount = (1 + self.rng.below(50)) * JTM;
                flows.reward_pool += i128::from(amount);
                push(
                    &mut groups,
                    &mut objects,
                    RewardFund { funder, amount }.object(),
                    funder,
                    vec![(REWARD_POOL, amount)],
                    FEE,
                    vec![1],
                );
            }
        } else if latched && self.rng.chance(4) {
            let funder = if self.rng.chance(2) { net } else { lab };
            let amount = (1 + self.rng.below(80)) * JTM;
            flows.reward_pool += i128::from(amount);
            push(
                &mut groups,
                &mut objects,
                RewardFund { funder, amount }.object(),
                funder,
                vec![(REWARD_POOL, amount)],
                FEE,
                vec![1],
            );
        }
        if active {
            // A stake lock.
            if self.rng.chance(3) {
                let who = owner(self.rng.below(8));
                let amount = (10 + self.rng.below(40)) * JTM;
                flows.stake_vault += i128::from(amount);
                push(
                    &mut groups,
                    &mut objects,
                    StakeLock { owner: who, amount }.object(),
                    who,
                    vec![(STAKE_VAULT, amount)],
                    FEE,
                    vec![1],
                );
            }
            // An unlock of part of a position.
            if self.rng.chance(5) {
                if let Some((key, position)) =
                    self.state.stakes.iter().nth(self.rng.below(8) as usize)
                {
                    let total = position.active + position.activating;
                    if total >= JTM {
                        let amount = JTM + self.rng.below(total - JTM + 1);
                        let who = Address(*key);
                        push(
                            &mut groups,
                            &mut objects,
                            StakeUnlock { owner: who, amount }.object(),
                            who,
                            Vec::new(),
                            FEE,
                            vec![1],
                        );
                    }
                }
            }
            // A claim of part of a balance (delegator or guardian).
            if self.rng.chance(4) {
                if let Some((key, balance)) =
                    self.state.balances.iter().nth(self.rng.below(8) as usize)
                {
                    if *balance >= JTM {
                        let amount = JTM + self.rng.below(*balance - JTM + 1);
                        let who = Address(*key);
                        push(
                            &mut groups,
                            &mut objects,
                            RewardClaim {
                                account: RewardAccount::Delegator,
                                owner: who,
                                amount,
                            }
                            .object(),
                            who,
                            Vec::new(),
                            FEE,
                            vec![1],
                        );
                    }
                }
            }
            if self.state.guardian_balance >= JTM && self.rng.chance(6) {
                let who = self.rules.guardian_reward_owner;
                let amount = self.state.guardian_balance;
                push(
                    &mut groups,
                    &mut objects,
                    RewardClaim {
                        account: RewardAccount::Guardian,
                        owner: who,
                        amount,
                    }
                    .object(),
                    who,
                    Vec::new(),
                    FEE,
                    vec![1],
                );
            }
            // Payouts of due entries (at most two per block).
            let due: Vec<(u64, PayoutEntry)> = self
                .state
                .entries
                .iter()
                .filter(|(_, e)| e.due <= h)
                .take(2)
                .map(|(id, e)| (*id, *e))
                .collect();
            for (id, entry) in due {
                if !self.rng.chance(2) {
                    continue;
                }
                let pool = if entry.kind == PayoutEntryKind::Stake {
                    STAKE_VAULT
                } else {
                    REWARD_POOL
                };
                let fee = (self.rng.below(PAYOUT_FEE_MAX_MICRO + 1)).min(entry.amount);
                let change = self.rng.below(3) * JTM;
                if entry.kind == PayoutEntryKind::Stake {
                    flows.stake_vault -= i128::from(entry.amount);
                } else {
                    flows.reward_pool -= i128::from(entry.amount);
                }
                let mut outputs = vec![(entry.owner, entry.amount - fee)];
                if change > 0 {
                    outputs.push((pool, change));
                }
                push(
                    &mut groups,
                    &mut objects,
                    GuardianPayout { entry_id: id }.object(),
                    pool,
                    outputs,
                    fee,
                    vec![AFTER_J + h],
                );
            }
        }
        // An attestation now and then; one equivocation in the run.
        if self.state.revoked_at.is_none() && self.rng.chance(40) {
            let x = h - 8 - self.rng.below(20);
            if self.state.last_attested_height.is_none_or(|last| x > last) {
                self.attestations += 1;
                push(
                    &mut groups,
                    &mut objects,
                    attestation_of(&statement_at(&self.branch, x)),
                    Address([0x99; 32]),
                    Vec::new(),
                    FEE,
                    vec![1],
                );
            }
        } else if !self.equivocated && h > J_D + 20_000 && self.rng.chance(100) {
            let mut other = self.branch.clone();
            other.extend("sim-fork", h - 50, h - 40);
            let (a, b) = (
                statement_at(&self.branch, h - 45).encode(),
                statement_at(&other, h - 45).encode(),
            );
            let (s1, s2) = if a < b { (a, b) } else { (b, a) };
            let eq = GuardianEquivocation {
                s1,
                sig1: sign(&s1),
                s2,
                sig2: sign(&s2),
                witnesses: None,
            };
            push(
                &mut groups,
                &mut objects,
                eq.object(),
                Address([0x98; 32]),
                Vec::new(),
                FEE,
                vec![1],
            );
        }
        (
            BlockGuardianContext {
                height: h,
                system_outputs: Vec::new(),
                groups,
                objects,
            },
            flows,
        )
    }

    fn step(&mut self) {
        // Now and then, a reorganisation of 1 to 8 blocks.
        if self.history.len() >= 8 && self.rng.chance(150) {
            let depth = 1 + self.rng.below(8) as usize;
            for _ in 0..depth {
                let (undo, pools) = self.history.pop().expect("history");
                self.state = undo.revert(&self.state);
                self.pools = pools;
                self.height -= 1;
            }
            self.reorgs += 1;
            // A reorganised-out equivocation may come again.
            self.equivocated = self.state.revoked_at.is_some();
            self.check_invariants();
            return;
        }
        let (ctx, flows) = self.block();
        let before = self.state.clone();
        let after =
            apply_block_guardian(&before, &ctx, &self.branch, &self.rules).unwrap_or_else(|e| {
                panic!(
                    "block {} refused: {e:?} ({} objects)",
                    ctx.height,
                    ctx.objects.len()
                )
            });
        if ctx
            .objects
            .iter()
            .any(|(_, o)| o.kind == GuardianObjectKind::Equivocation)
        {
            self.equivocated = true;
            // INV-C: nothing of a delegator decreases.
            assert_eq!(after.stakes, before.stakes);
            assert!(before
                .balances
                .iter()
                .all(|(k, v)| after.balances.get(k) >= Some(v)));
            assert!(before
                .entries
                .iter()
                .filter(|(_, e)| e.kind != PayoutEntryKind::GuardianReward)
                .all(|(id, e)| after.entries.get(id) == Some(e)));
        }
        // INV-U: the undo restores the bytes.
        let undo = GuardianUndo::between(&before, &after);
        assert_eq!(
            undo.revert(&after).encode(),
            before.encode(),
            "undo at {}",
            ctx.height
        );
        assert_eq!(GuardianUndo::decode(&undo.encode()).as_ref(), Ok(&undo));
        self.history.push((undo, self.pools));
        if self.history.len() > 16 {
            self.history.remove(0);
        }
        self.pools.stake_vault += flows.stake_vault;
        self.pools.reward_pool += flows.reward_pool;
        self.state = after;
        self.height += 1;
        self.blocks += 1;
        // A node rebuilt from the state's bytes every few blocks.
        if self.blocks.is_multiple_of(97) {
            self.state = GuardianState::decode(&self.state.encode()).expect("canonical state");
        }
        self.check_invariants();
    }

    fn check_invariants(&self) {
        let s = &self.state;
        // INV-S.
        assert_eq!(
            stake_sum(s),
            self.pools.stake_vault,
            "INV-S at {}",
            self.height
        );
        // INV-R and INV-R2.
        assert_eq!(
            reward_liabilities(s),
            self.pools.reward_pool,
            "INV-R at {}",
            self.height
        );
        assert_eq!(
            i128::try_from(s.total_funded).expect("i128")
                - i128::try_from(s.total_paid).expect("i128"),
            self.pools.reward_pool,
            "INV-R2 at {}",
            self.height
        );
        assert!(s.total_paid <= s.total_funded);
        // INV-D.
        if s.activated_at.is_none() {
            assert_eq!(reward_liabilities(s), 0);
            assert!(s.stakes.is_empty() && s.entries.is_empty());
        }
    }
}

fn run(seed: u64, blocks: u64) -> Sim {
    let mut sim = Sim::new(seed, blocks + 2_000);
    while sim.blocks < blocks {
        sim.step();
    }
    sim
}

#[test]
fn prop_sum_stakes_equals_stake_vault_balance() {
    let sim = run(0x5eed_0001, 3_000);
    assert!(sim.state.activated_at.is_some());
    assert!(stake_sum(&sim.state) > 0);
}

#[test]
fn prop_reward_pool_balance_equals_liabilities() {
    let sim = run(0x5eed_0002, 3_000);
    assert!(sim.state.total_funded > 0);
}

#[test]
fn prop_sum_claimed_le_sum_funded() {
    let sim = run(0x5eed_0003, 3_000);
    assert!(sim.state.total_paid <= sim.state.total_funded);
}

#[test]
fn guardian_simulation_100k_steps_without_violation() {
    let sim = run(0x5eed_0100, 100_000);
    // The run did go through every phase.
    assert!(sim.state.activated_at.is_some());
    assert!(sim.equivocated, "no equivocation happened");
    assert!(sim.state.revoked_at.is_some());
    assert!(sim.reorgs > 100, "{} reorgs", sim.reorgs);
    assert!(sim.attestations > 100, "{} attestations", sim.attestations);
    assert!(sim.state.total_paid > 0);
}
