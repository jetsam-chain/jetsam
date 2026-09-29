# 外部矿工

外部挖矿把 PoW nonce 搜索与节点分离。内存池、交易选择、State 转换、
`HistoryStep` 证明、模板以及区块中继仍由节点掌控。

外部挖矿进程不会收到区块体或生成证明所需的见证数据。

## 本地挖矿进程

使用 Bearer 令牌以外部矿工模式启动节点：

```sh
jetsam --mode extminer --mining-key 'LONG-RANDOM-TOKEN'
```

在另一个终端运行：

```sh
jetsam-miner \
  --rpc http://127.0.0.1:9701 \
  --key 'LONG-RANDOM-TOKEN'
```

如果节点使用 `--mining-key` 启动，即使通过回环地址连接也必须提供
token。

需要时可限制挖矿进程的线程数：

```sh
jetsam-miner --key 'LONG-RANDOM-TOKEN' --threads 8
```

每个遍历线程在核心 L2 缓存中保留 512 KiB 暂存区，因此通常每个物理核心一个线程最佳。请先用 `jetsam --bench` 测量，并查看 `walked digest` 一行。

## 远程挖矿进程

切勿把未加密的 Bearer 令牌和通用 RPC 接口直接暴露到互联网。

应把挖矿进程与节点放在经过认证的私有网络中，或由反向代理终止 TLS 并
限制暴露路径。只有安全传输就绪后才绑定公网 RPC：

```sh
jetsam \
  --mode extminer \
  --rpc-listen 0.0.0.0:9701 \
  --mining-key 'LONG-RANDOM-TOKEN'
```

防火墙应只允许指定挖矿进程或代理访问该端口。

## 奖励地址

模板默认使用节点配置的奖励地址，这是更安全的单机挖矿方式。

若允许挖矿进程请求自己的奖励地址，节点运营者必须显式启用：

```sh
jetsam \
  --mode extminer \
  --mining-key 'LONG-RANDOM-TOKEN' \
  --allow-custom-coinbase
```

此后挖矿进程可以使用：

```sh
jetsam-miner \
  --key 'LONG-RANDOM-TOKEN' \
  --coinbase j1...
```

自定义 coinbase 只改变证明构建前嵌入的奖励地址，挖矿进程仍无法修改已经
证明的模板。

## 模板生命周期

`getBlockTemplate` 返回不透明的一次性 ID、16 字段 PoW 输入序列、
nonce 索引、目标值，并自区块 24,846 起返回 `pow_walk: true`。带有该标志时，挖矿进程必须搜索 `TowerWalk(TowerHash(fields))`，而不只是 TowerHash 摘要；`jetsam-miner` 读取的是该标志而不是高度。挖矿进程搜索随机且互不重叠的 nonce 范围，再通过
`submitBlock` 提交恰好 16 个小端序 nonce 字节。

`jetsam-miner` 的每个请求都带两个 HTTP 头：`X-Jetsam-Version`（其版本）和 `X-Jetsam-PoW: walk`（能搜索遍历摘要）。节点忽略这两个头；矿池借助后者区分执行遍历的挖矿程序与分叉前的挖矿程序，后者在 TowerWalk 任务上无法找到区块。

模板在 120 秒后过期；规范链尖变化、成功提交或节点主动取消也会使其失效。
结果过期是正常现象，挖矿进程会在下一次轮询时请求新模板。

## 诊断

运行：

```sh
jetsam-miner --check-hardware
```

请求失败时：

- `401 Unauthorized` 表示 token 缺失或不匹配；
- 自定义 coinbase 错误表示节点未启用该功能；
- 模板不断过期通常表示节点持续收到新链尖，或证明准备超过模板生命周期；
- 每次提交都返回 `-32025`（摘要不低于目标值）表示挖矿进程忽略了 `pow_walk`：v1.4 之前构建的挖矿程序搜索的是错误的摘要；
- 没有模板表示节点尚未同步、对等节点数量不足，或不在 `extminer` 模式。
