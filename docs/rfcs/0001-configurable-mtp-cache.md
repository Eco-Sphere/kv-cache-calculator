# RFC：Qwen3.5/3.6 可配置num speculative tokens


## 背景

当前 Web 计算器只对部分模型提供了`Include draft KV cache`开关，Qwen3.5/3.6 暂不支持`Include draft KV cache`。

Qwen3.5/3.6的MTP层固定为Full Attention结构，开启MTP后额外的KV Cache占用为MTP 草稿模型自己的 full-attention KV cache，加上主模型 GDN 层的递归状态。

和 Full Attention 的区别在于第二项主模型 GDN 层的递归状态。投机长度会增加GDN state 占用的KV Cache空间，且`num_speculative_tokens` 正相关。为了进一步理解，我们需要先介绍投机推理的实现框架。

### 投机推理框架与GDN回退

投机推理将生成过程分为 draft proposal 和 target verification两个阶段。Draft model 先快速提出多个候选 token，target model 再并行验证这些候选，并一次提交连续通过验证的 token。对单个请求而言，可以抽象为以下循环：

```text
已提交前缀
  -> MTP 自回归生成 num_speculative_tokens 个 draft tokens
  -> Target 一次前向验证这些候选位置
  -> Rejection sampler 从左到右决定接受前缀
  -> 提交已接受位置的缓存状态，对于拒绝的位置，回退修改
  -> 进入下一轮
```

对于Qwen3.5这类主模型采用了GDN的模型来说，第四步**提交已接受位置的缓存状态，释放或复用其余候选状态**和普通 Full Attention有所不同：

| 缓存类型 | 验证期间 | 验收完成后 |
| --- | --- | --- |
| 主模型Full Attention KV | 将候选 token 写入预留的 lookahead slots | 保留接受位置对应的 slots，缩短有效 sequence length，复用拒绝位置的 slots可自动自用 |
| MTP模型draft KV | 写入独立的 draft attention cache | 同主模型Full Attention KV |
| 主模型 GDN state | 保留 running state 以及每个 speculative position 的候选状态 | 保留接受位置对应的GDN State，作为下一轮 running state，回收拒绝位置的GDN State |

GDN 回退不能只修改 sequence length。若验证阶段原地覆盖唯一的 GDN state，一旦中途拒绝就无法回退到最后一个接收位置。因此投机请求需要保留当前 GDN state 以及各 speculative position 的候选GDN state，并把这些候选状态保留到拒绝采样。下一轮开始前，系统选择与最终有效前缀对应的状态作为新的 running state，其余候选状态将被回收。

## 目标与方案

### 目标

1. Qwen3.5/3.6 支持配置 `num_speculative_tokens`，取值为大于等于 0 的整数。
2. 开启 `Include draft KV cache` 后，分别展示主模型缓存、linear-attention state 和 MTP draft KV cache，避免把不同语义的缓存合并成一个无法解释的数字。
3. 区分 `num_speculative_tokens` 和 MTP 物理层数：前者是每轮最多生成的 draft token 数，后者来自模型配置（Qwen3.5/3.6 为 `mtp_num_hidden_layers`，DeepSeek V4 为对应的 next-N/MTP 层配置）。
4. 所有分项沿用当前页面的 per-device 口径，只有 `Total cache size` 使用 all-device 口径。
5. 计算结果能够解释 TP、投机长度和运行时 block/page 对齐对缓存占用的影响。


### 方案

开启MTP后，GDN State包含以下状态：

1. Conv state：保留 1 + N 份；单份长度为 K - 1 + N
2. Recurrent state：同样保留 1 + N 份；单份 shape 不随 N 变化

这里 K 表示 GDN 一维因果卷积的卷积核大小，即`K = linear_conv_kernel_dim`；`N = num_speculative_tokens`；两种 state 需要保留的份数相同。

### Conv state 的份数与长度

Conv state 的“份数”和“长度”表示两个不同维度：

```text
Conv state 总量
  = 状态份数 × 每份状态的长度 × 每个位置的 channel 数
```

**长度**表示一份状态内部需要保存多少个位置。卷积核大小为 `K` 时，普通解码需要保留最近 `K - 1` 个位置；投机长度为 `N` 时，还要容纳候选 token 对卷积窗口的连续推进，因此每份 Conv state 的长度扩展为：

```text
K - 1 + N
```

**份数**表示需要保留多少个相互独立、可供提交或回退的状态快照。投机验证结束前，最终可能接受 0 到 `N` 个 draft token，因此需要保留当前 running state 和 `N` 个候选位置对应的状态，共：

```text
1 + N 份
```

例如 `K = 4`、`N = 2` 时，每份 Conv state 的长度为 `4 - 1 + 2 = 5`，共需要保留 `1 + 2 = 3` 份：

```text
状态 0：[5 个位置 × Conv channels]
状态 1：[5 个位置 × Conv channels]
状态 2：[5 个位置 × Conv channels]
```

因此这里保存的是 `3` 份长度为 `5` 的 Conv state。长度保证每份状态能够容纳完整的卷积历史和候选位置，份数保证发生拒绝时可以选择正确的提交位置。

Recurrent state 没有卷积窗口维度，所以它只增加保留份数，单份 shape 不会从 `K - 1` 扩展到 `K - 1 + N`。

