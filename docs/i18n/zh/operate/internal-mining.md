# 内置挖矿

内置挖矿把交易选择、证明构建、nonce 搜索和区块提交全部保留在同一个 Core
进程中。

## 准备

先运行发布版硬件检查：

```sh
jetsam --check-hardware
```

从节点钱包获取奖励地址：

```sh
jetsam
jetsam-cli address --list
jetsam-cli stop
```

未显式配置奖励地址时，会自动使用钱包活动地址。

## 启动

在前台运行：

```sh
jetsam --mode miner --cpu-threads 12
```

也可更新 systemd unit：

```ini
ExecStart=/usr/local/bin/jetsam \
  --config /etc/jetsam/jetsam.toml \
  --mode miner \
  --cpu-threads 12
```

然后重新加载并重启：

```sh
sudo systemctl daemon-reload
sudo systemctl restart jetsam
```

## 就绪条件

普通挖矿需要一个经过认证的对等节点，以及已同步的链。检查：

```sh
jetsam-cli status
jetsam-cli peers
jetsam-cli mining
```

进程会准备两类内嵌证明矩阵，并选择最合适的 CPU 后端。每个
进程都先用小类别（代码中为 `B25`，自区块 17,750 起为 24 个页面位置）开始生产区块；只有实测的完整准备时间足够快时，才会使用
更大的 B255 模板。

## CPU 规划

`--cpu-threads` 是证明阶段和 PoW 阶段共用的总预算，不应超过 cgroup 或
虚拟机实际分配给服务的逻辑 CPU 数。

自区块 24,846 起，PoW 阶段是 TowerWalk：每个搜索线程遍历自己的 512 KiB 暂存区，其大小按单个核心的私有 L2 缓存设定。搜索时通常每个物理核心一个线程最佳。请按 `jetsam --bench` 输出的 `walked digest` 速率设置预算，而不是其上方的 `sponge digest` 速率。

基础设施节点应为操作系统和公网 P2P 服务留出资源；专用矿机可以使用全部
可见逻辑 CPU。

钱包交易证明在本机优先于正在进行的挖矿，但这不会改变其他节点接受哪些
交易或区块。

## 更改奖励地址

每个新模板都会重新解析钱包活动地址。地址变化会在安全边界使本地工作失效
或刷新；已经不可变的模板不能改写奖励地址。

若要固定独立的进程奖励地址：

```sh
jetsam --mode miner --miner-address j1...
```

请使用完整 bech32m 地址。

## 停止

通过 RPC 或服务管理器停止：

```sh
jetsam-cli stop
```

```sh
sudo systemctl stop jetsam
```

正常关闭会取消挖矿、关闭网络并刷入 MDBX。不要仅因活动证明需要几秒到达
安全取消边界，就反复发送强制终止信号。
