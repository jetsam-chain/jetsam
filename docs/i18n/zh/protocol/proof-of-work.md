# 工作量证明

Jetsam 的[工作量证明](../reference/glossary.md#proof-of-work)只在不含 [nonce](../reference/glossary.md#nonce) 的 `HistoryStep` 完成证明后运行，分为两个阶段：

1. **TowerHash**：在固定语义区块头字段与 nonce 上计算的带[域分离](../reference/glossary.md#domain-separation) [Poseidon2b](../reference/glossary.md#poseidon2b) 海绵摘要，产出一个 32 字节的种子（seed）；
2. **TowerWalk**：在 512 KiB 暂存区（scratchpad）上进行的缓存驻留遍历，把种子变为与目标值比较的摘要。

在公共网络上，遍历自区块 24,846 起成为共识的一部分（于 2026 年 9 月 29 日 18:21:52 UTC 激活）。低于该高度时，摘要仍是 TowerHash 的输出，因此此前的每个区块都保留其原有的工作量证明。TowerWalk 是驻留在每个核心 L2 缓存中的 CPU 工作量证明；区块 24,846 之前使用的 GPU 挖矿程序已不再可用。

## 字段序列

`POWHDR__` 海绵函数精确吸收 16 个 `GF(2^128)` 元素：

| 索引 | 字段 |
|---:|---|
| 0 | 128 位 nonce |
| 1–2 | `prev_block_hash` |
| 3–4 | `state_root` |
| 5–6 | `tx_root` |
| 7 | `timestamp` |
| 8 | `height` |
| 9–10 | `miner_address` |
| 11–12 | `difficulty_target` |
| 13 | `log_slots` |
| 14 | `active_slot_count` |
| 15 | `alloc_counter` |

32 字节值拆成两个小端序 128 位半段，标量整数以零扩展。海绵结构的速率为二，因此该字段序列恰好占八个速率块，不需要变长填充。

Nonce 是字段 0（`POW_NONCE_FIELD_INDEX`），因此它进入第一次置换，任何海绵状态都无法跨 nonce 预先计算。

PoW 域同时区别于含 nonce 的区块身份域和无 nonce 的语义区块头承诺域。

## TowerWalk

自区块 24,846 起，TowerHash 的输出是种子，而不是摘要。TowerWalk 将其读作四个小端序 64 位字，然后：

1. 用由种子派生的“乘法 + xorshift”序列填充 65,536 个 64 位单元（512 KiB）的暂存区，每填充 4,096 个单元就把通道状态经 Poseidon2b 置换折叠（fold）一次；
2. 以四条通道遍历暂存区 131,072 轮，共 524,288 次数据相关读取：每条通道读取由自身状态寻址的单元，混入该值，再把结果写回同一单元；同一轮内通道按顺序执行，因此后一条通道会读到前一条通道刚写入的值；
3. 遍历中每 8,192 轮折叠一次，结束时再折叠一次，共 33 次折叠，最后把四个通道字作为 32 字节摘要输出。

所有算术均按 2^64 取模回绕。以下常量由共识固定：

| 常量 | 值 |
|---|---:|
| 暂存区单元数（`CELLS`） | 65,536（512 KiB） |
| 通道数（`LANES`） | 4 |
| 遍历轮数（`ROUNDS`） | 131,072 |
| 遍历折叠周期（`PERM_PERIOD`） | 8,192 轮 |
| 填充折叠周期（`FILL_PERM_PERIOD`） | 4,096 个单元 |
| 混合乘数（`MULT_C`） | `0x9E3779B97F4A7C15` |
| xorshift 位移（`XORSHIFT`） | 29 |

暂存区大小按单个核心的私有 L2 缓存设定。该函数位于 `jetsam_poseidon2b::towerwalk`，由节点与外部挖矿程序共用，使共识哈希只有一份实现。逐位规范、无依赖的参考实现以及 256 个固定测试向量见[挖矿规范](../../../mining/stratum.md#310-towerwalk-the-digest-from-block-24846)（英文）。

## 目标值比较

摘要与目标值均解释为 256 位小端序整数。Nonce 只有在以下条件成立时有效：

```text
seed       = TowerHash(fields)
pow_digest = TowerWalk(seed)     高度 ≥ 24,846
pow_digest = seed                高度 < 24,846
pow_digest < difficulty_target
```

相等也视为失败。

## ASERT

已接受区块之间的目标间隔为 90 秒。证明准备、nonce 搜索与区块传播共同占用
这一完整间隔。[ASERT](../reference/glossary.md#asert) 使用六区块参考周期和 90
秒半衰期。在每个高度，验证过程根据规范锚点、经过时间与高度差推导精确目标值。

时间戳还必须大于前 11 个区块头的过去时间中位数（median time past），并且最多领先验证节点本地时钟 190 秒。

第一个 TowerWalk 区块无法沿用海绵时代的目标值：一次遍历尝试的成本远高于一次海绵尝试，而 ASERT 以父区块时间戳为锚点，只能吸收算力的下降，无法吸收出块停滞。因此区块 24,846 携带固定的锚定目标值 `2^235`，刻意设在偏容易的一侧；ASERT 从这里向 90 秒间隔收紧。

## 运行时内核

同一个固定置换按 nonce 批次计算。发布版二进制文件会在运行时选择主机支持的最佳实现：

- x86-64 上以 `pclmul` 为基线；
- 可用时使用 `avx2+vpclmul`；
- 支持主机上的 `avx512bw+vpclmul`；
- ARM64 上的 `neon+pmull`。

批量执行只改变吞吐量，不改变摘要。标量实现是用于交叉校验的参考实现，不作为发布版运行时的回退路径。

这些内核加速的是 Poseidon2b 置换。遍历本身是标量 64 位算术，其速度取决于对 512 KiB 暂存区读取的延迟。每个挖矿线程持有自己的暂存区，因此通常每个物理核心一个线程最佳；请按 `jetsam --bench` 输出的 `walked digest` 速率设置 `--cpu-threads`。

## 外部挖矿边界

外部挖矿进程收到精确的 16 字段输入序列、nonce 索引、目标值以及 `pow_walk` 标志。`pow_walk` 为 `true` 时，挖矿进程必须搜索 `TowerWalk(TowerHash(fields))`；缺失或为 `false` 时只比较 TowerHash 摘要。节点依据模板自身的高度设置该标志；挖矿进程必须读取它，绝不能根据高度自行推断。挖矿进程只返回规范的 16 字节小端序 nonce。节点在提交区块前，按该高度要求的摘要，根据不可变的一次性模板验证结果。

`jetsam-miner` 通过 HTTP 头 `X-Jetsam-PoW: walk` 声明自己能执行遍历。节点忽略该头；分发 TowerWalk 任务的矿池据此区分执行遍历的挖矿程序与分叉前的挖矿程序，后者无法找到区块。

外部挖矿进程无法修改交易、State 根、收益地址或证明。使用过期模板求得的 nonce 会被拒绝。
