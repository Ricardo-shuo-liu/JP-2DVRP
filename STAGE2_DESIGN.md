# Stage2 设计说明

本文档说明当前仓库中 Stage2 的**实际实现逻辑**，包括数据流、分簇网络、
容量掩码、簇容量惩罚、TSP padding、reward、训练损失、CLI 参数和仍存在的缺口。

本文档描述的是当前代码，而不是最终论文目标。最后一节列出了当前代码与
`docs.md` 中联合训练设计和 TAM 原始 Step2 的差异。

## 1. 相关文件

| 文件 | 作用 |
|---|---|
| `scripts/stage2/run_stage2.py` | Stage2 训练入口、环境构造、模型构造 |
| `scripts/stage2/policy.py` | 分簇 policy、容量 mask、reward 组装 |
| `scripts/stage2/route_evaluator.py` | 冻结 Stage1 AM 的簇 TSP 评价器 |
| `nets/partition.py` | 2D-Ptr 风格分簇网络 |
| `nets/vehicle_encoder.py` | 车辆编码器和 DPN rotary PE |
| `utils/config/stage2.py` | Stage2 超参数 |
| `utils/data.py` | RL4CO CVRP 环境与可选的可行性过滤 |
| `scripts/stage2/generate_data.py` | CVRP 固定数据生成器 |

## 2. 总体流程

当前 Stage2 的完整流程如下：

```text
RL4CO CVRPEnv batch
  ↓
PartitionNetwork
  ↓
logits: [B, M, N]
  ↓
对每个 node 独立做车辆维 softmax
  ↓
容量静态 mask
  ↓
采样/贪心得到 assignments: [B, N]
  ↓
按车辆拆成 M 个 cluster
  ↓
cluster padding 到统一长度
  ↓
批量调用冻结的 Stage1 AM
  ↓
得到每个 cluster 的 TSP 路径长度 cost: [B, M]
  ↓
计算每个车辆容量溢出
  ↓
min-max 或 sum reward
  ↓
REINFORCE 更新 PartitionNetwork
```

当前默认设置：

```text
Stage1 路由评价器：AttentionModel
目标函数：minmax
容量掩码：static
baseline：rollout
```

## 3. 数据与环境

### 3.1 CVRP 环境

Stage2 使用 RL4CO 的 `CVRPEnv`：

```python
env = CVRPEnv(...)
```

每个 batch 的基本字段：

```text
locs:             [B, N+1, 2]   # index 0 是 depot
depot:            [B, 1, 2]
demand:           [B, N]        # RL4CO 已按 capacity 归一化
vehicle_capacity: [B, 1]        # CVRP 中通常为 1.0
```

其中：

```text
节点 i 的原始需求 / 车辆容量 = demand[i]
```

因此后续的容量判断都使用归一化后的 demand：

```text
0 < demand_i <= 1
```

### 3.2 总需求可行性

`FeasibleCVRPGenerator` 默认过滤：

```text
sum(demand) <= num_vehicles
```

这等价于：

```text
sum(original_demand) <= num_vehicles * capacity
```

总需求是否超过全部车辆容量属于**环境可行性检查**，不属于模型训练目标。
当前实现采用过滤而不是奖励惩罚：

- `FeasibleCVRPGenerator` 在线生成时只保留 `sum(demand) <= num_vehicles`
  的样本；
- `scripts/stage2/generate_data.py` 固定数据生成时也做同样过滤；
- 如果固定数据文件中存在违反该条件的样本，应在数据加载阶段拒绝或重新
  生成，而不是让 model reward 去承担这个错误。

## 4. 分簇网络

分簇网络定义在 `nets/partition.py`。

### 4.1 NodeEncoder

NodeEncoder 输入：

```text
depot:         [B, 1, 2]
customer_locs: [B, N, 2]
demand:        [B, N]
```

处理过程：

```text
客户特征 = [x, y, demand]
  ↓
Linear -> 客户 embedding
  ↓
depot embedding
  ↓
concat(depot, customers)
  ↓
多层 self-attention
```

输出：

```text
node_embeddings: [B, N+1, d]
```

后续只取客户部分：

```text
customer_embeddings: [B, N, d]
```

### 4.2 VehicleEncoder

车辆初始状态是相同的：

```text
time = 0
capacity = 1.0
used_capacity = 0
speed = 1
```

经过 MLP 后得到：

```text
vehicle_embeddings: [B, M, d]
```

为了区分同质的 CVRP 车辆，加入了 DPN 风格的 depot-aware rotary PE：

```text
PE(m, q) = Re[
    (x_d W_a + b_a)
    * exp(i m / 1000^(floor(q/2)/d))
]
```

`m` 是车辆编号。这样相同状态的车辆也会得到不同的表示。

当前实现还按 DPN Appendix C.6 加入最后的可学习投影：

```text
PE = rotate(depot_proj(depot), vehicle_index) @ W_PE
```

`W_PE` 初始化为单位矩阵，因此初始行为与旧实现一致，训练过程中再学习
适合当前任务的投影。

之后车辆表示依次经过：

```text
vehicle self-attention
vehicle-to-customer cross-attention
融合层
```

### 4.3 分簇 logits

最终：

```text
logits[b, m, i]
= vehicle_embedding[b, m] · customer_embedding[b, i] / sqrt(d)
```

形状：

```text
[B, M, N]
```

其中：

```text
B = batch size
M = num_vehicles
N = number of customers
```

### 4.4 模型输入输出 tensor 总表

#### 分簇模型输入

| 字段 | shape | dtype | 含义 |
|---|---:|---|---|
| `locs` | `[B, N+1, 2]` | float32 | index 0 是 depot，其余是客户坐标 |
| `demand` | `[B, N]` | float32 | 归一化需求 |
| `vehicle_capacity` | `[B, 1]` | float32 | 归一化车辆容量，CVRP 中通常为 1.0 |
| `num_vehicles` | scalar / `[B]` | int | 运行时车辆数 M；不是网络参数 |

#### 分簇模型内部 tensor

| tensor | shape | 含义 |
|---|---:|---|
| depot embedding | `[B, 1, d]` | depot 坐标投影 |
| customer embedding | `[B, N, d]` | `[x, y, demand]` 投影 |
| node token sequence | `[B, N+1, d]` | 当前实现包含 depot token |
| vehicle state | `[B, M, 4]` | `[time, capacity, used_capacity, speed]` |
| vehicle embedding | `[B, M, d]` | 车辆状态 + rotary PE + cross-attention |
| logits | `[B, M, N]` | `node i` 分配给 `vehicle m` 的分数 |

#### 分簇模型输出

| 字段 | shape | 含义 |
|---|---:|---|
| `assignments` | `[B, N]` | 每个客户所属的车辆编号 |
| `log_likelihood` | `[B]` | 所有客户分配概率的对数和 |
| `cost` | `[B, M]` | 每个车辆对应簇的 TSP 路径长度 |
| `objective_cost` | `[B]` | min-max 或 sum 后的目标值 |
| `overflow_per_vehicle` | `[B, M]` | 每个车辆的容量溢出 |

#### Stage1 AM 路由评价器输入

路由评价器只接收 TSP 数据：

| 字段 | shape | dtype | 含义 |
|---|---:|---|---|
| `locs` | `[K, L, 2]` | float32 | K = B × M 个簇；L = 真实节点 + padding depot |

路由评价器不需要 `demand`。每个 cluster 的第 0 个位置是真实 depot，
后面是簇内真实客户，最后是重复的 depot padding。

#### Stage1 AM 路由评价器输出

| 字段 | shape | 含义 |
|---|---:|---|
| `reward` | `[K]` | 每个 padded TSP 的负路径长度 |
| `cost` | `[K]` | 每个 padded TSP 的路径长度 |
| `per_vehicle_cost` | `[B, M]` | reshape 后的每簇路径长度 |

### 4.5 depot 在模型中的出现位置

当前分簇网络包含两处 depot 信息：

1. `NodeEncoder` 把 depot embedding 作为一个 token 与客户 token 拼接；
2. `VehicleEncoder` 使用 depot 坐标生成 DPN rotary PE。

这是当前代码的实际实现。

如果希望严格遵循“模型只需要 node 和 vehicle，depot 只通过 rotary PE 进入”
的设计，可以移除 `NodeEncoder` 中的 depot token，只保留：

```text
node tokens = customer nodes
vehicle tokens = vehicle state + depot-aware rotary PE
```

这样 depot 仍然会影响分簇，因为 rotary PE 和车辆 query 已经包含 depot 信息。
但 depot 不再作为 self-attention 序列中的额外节点。

这一点需要在最终实验中做消融，因为它会改变模型输入和注意力计算。

## 5. Assignment Policy 与容量掩码

实现位置：`scripts/stage2/policy.py`。

### 5.1 每-node 独立分配

对第 `i` 个 customer，只在车辆维度做 softmax：

```text
p(z_i = m | data) = softmax_m(logits[:, i])
```

因此不同 customer 之间没有依赖：

```text
p(z | data) = Π_i p(z_i | data)
```

训练时：

```python
p(z_i) = Categorical(logits=logits[:, i])
z_i ~ p(z_i)
```

评估时：

```python
z_i = argmax_m logits[m, i]
```

### 5.2 静态容量 mask

对应参数：

```bash
--capacity-mask-mode static
```

对每个 `(vehicle m, node i)`：

```text
如果 capacity[m] < demand[i]
    该位置 logits[m, i] = -inf
```

公式：

```text
allowed[m, i] = 1[capacity[m] >= demand[i]]
masked_logits[m, i] = logits[m, i] if allowed else -inf
```

然后仍然在每个 node 的车辆维度独立做 softmax：

```text
p(z_i = m) = softmax_m(masked_logits[:, i])
```

这个设计不会引入节点间的前后依赖。

### 5.3 无可行车辆时的处理

如果某个 node 对所有车辆都属于：

```text
capacity[m] < demand[i]
```

那么所有位置都会被 mask 掉。

为了避免 softmax 出现全 `-inf`：

1. 暂时取消该 node 的 mask；
2. 仍然采样/选择一个车辆；
3. 由后续的簇容量溢出惩罚处理这个非法分配。

也就是说，静态 mask 只负责直接禁止单个节点与车辆的明显不兼容；
多个节点累计导致的容量溢出由后验惩罚处理。

### 5.4 `none` 模式

对应：

```bash
--capacity-mask-mode none
```

不进行任何静态 mask，用于消融实验。

## 6. 后验收容量惩罚

分簇完成后，对每个车辆计算：

```text
load_m = Σ_{i: z_i = m} demand_i
overflow_m = max(0, load_m - capacity_m)
```

注意这里是在整个簇上求和，因此可以发现：

```text
多个节点单独都合法
但分配到同一个车辆后总需求超容量
```

这部分不会再改变 assignment，而是通过 reward 惩罚。

### 6.1 单车溢出惩罚

每个车辆的惩罚后成本是：

```text
penalized_cost_m
= TSP_cost_m
 + penalty * overflow_m
```

其中：

```text
penalty = --penalty
```

### 6.2 总需求可行性

总需求与总车辆容量的比较只在数据/环境层面检查：

```text
sum(demand) <= num_vehicles * capacity
```

这个条件不进入 reward。因为如果它不成立，问题本身对于给定车队就是不可行的，
不应该由分簇模型来学习补偿。

## 7. Stage1 AM 路由评价器

实现位置：`scripts/stage2/route_evaluator.py`。

当前主要验证目标是：

```bash
--route-model AttentionModel
```

### 7.1 冻结模型

Stage1 checkpoint 只加载 `policy.*` 参数：

```text
冻结 Stage1 AM
requires_grad = False
只在 eval 模式推理
```

Stage2 只优化 `PartitionNetwork`。

### 7.2 cluster 收集

对每个 batch 和每个车辆：

```text
member_indices = {i | assignment_i = m}
```

每个 cluster 的输入是：

```text
[depot, member_1, member_2, ..., member_n]
```

当前 `member_i` 的顺序是原始客户索引从小到大：

```python
member_indices = (assignments[b] == m).nonzero()
```

这个顺序不是 TSP 的访问顺序。原因有三个：

1. AM encoder 没有节点位置编码，理论上对输入节点排列不敏感；
2. TSP 本身是循环路径，循环的起始点不改变路径长度；
3. 访问顺序由 AM decoder 在推理时逐个选择，而不是由输入顺序决定。

因此“簇内节点在 tensor 中的排列顺序”和“最终 TSP 的访问顺序”是两件事。

每个 cluster 的实际 depot 处理是：

```text
位置 0：真实 depot
位置 1..n：真实客户
位置 n+1..L-1：重复 depot padding
```

所以真实 depot 始终在 cluster 中，并且 AM 的回路会包含这个 depot。
padding 的重复 depot 只是为了让不同 cluster 能组成一个 batch。

### 7.3 AM 如何选择下一个 action

Stage1 AM 的输入虽然只是：

```text
locs: [K, L, 2]
```

但 decoder 并不是只依赖这组坐标。AM 的内部状态还包括：

```text
first_node:   当前 TSP 回路的第一个节点
current_node: 当前已经到达的节点
i:            已经解码的步数
action_mask:  [K, L]，哪些节点还没有访问
```

Decoder 每一步会构造一个 context query：

```text
q_t = project([
    embedding(first_node),
    embedding(current_node)
])
```

第一步还没有 first/current 的真实上下文，因此使用一个可学习的 placeholder。

然后使用 pointer attention：

```text
score_i = q_t · key_i / sqrt(d)
```

再通过 `action_mask` 屏蔽已经访问过的节点：

```text
score_i = -inf    if node i already visited
```

最后对未访问节点做 softmax，选择：

```text
sampling:  a_t ~ Categorical(softmax(score))
greedy:    a_t = argmax(score)
```

因此“输入只有 `[node, 2]`”并不代表 decoder 不知道当前位置。
坐标只负责产生节点 embedding，当前位置、已访问节点和回路起点由 decoder
的 `current_node`、`first_node` 和 `action_mask` 维护。

### 7.3 簇 padding

不同 cluster 的大小不同，无法直接组成一个 tensor，因此做 padding：

```text
n_max = max(max_cluster_size, route_pad_to)
cluster_len = n_max + 1
```

每个 cluster 的一行是：

```text
[depot, 真实客户..., depot 重复节点...]
```

例如：

```text
cluster A 有 3 个客户
cluster B 有 1 个客户
n_max = 5

A: [depot, c1, c2, c3, depot, depot]
B: [depot, c1, depot, depot, depot, depot]
```

重复的 depot 节点与真实 depot 坐标相同，因此不会增加真实的 depot
回路距离，但会让 AM 的输入长度统一。

原始 TAM 使用同样的 padding 思想：

```text
不同 sub-TSP -> padding 到同一长度 -> GPU 并行求解
```

### 7.4 批量 TSP 推理

所有 cluster 被组成：

```text
[B * M, cluster_len, 2]
```

一次送入冻结 AM：

```text
reward: [B * M]
cost:   [-reward][B * M]
```

然后 reshape：

```text
per_vehicle_cost: [B, M]
```

### 7.5 TSP 评价 batch 大小

Stage2 batch 和 TSP forward batch 是分离的。

Stage2 batch：

```bash
--batch-size
```

TSP 评价 batch：

```bash
--route-eval-batch-size
```

例如：

```text
Stage2 batch = 32
num_vehicles = 8
总簇数 = 256

--route-eval-batch-size 64
→ 分 4 次 AM forward
```

如果：

```bash
--route-eval-batch-size 0
```

则一次评价所有簇。

## 8. Reward 设计

### 8.1 min-max

对应：

```bash
--objective minmax
```

定义：

```text
objective_cost
= max_m penalized_cost_m
```

奖励：

```text
reward = -objective_cost
```

即最小化最长车辆路径，而不是最小化所有路径长度之和。

### 8.2 sum

对应：

```bash
--objective sum
```

定义：

```text
objective_cost
= Σ_m penalized_cost_m
```

奖励公式相同。这是原始 TAM 的 min-sum CVRP 目标。

## 9. 训练循环

当前使用 RL4CO 的 `REINFORCE`。

### 9.1 sampling

训练阶段：

```text
对每个 node 采样一个车辆
得到 assignment z
```

### 9.2 log likelihood

因为节点独立：

```text
log p(z | data)
= Σ_i log p(z_i | data)
```

所以 REINFORCE 的 log-likelihood 也是逐节点求和。

### 9.3 baseline

默认：

```bash
--baseline rollout
```

这更接近 TAM Step2 中使用的 greedy rollout baseline。

也支持：

```bash
--baseline exponential
--baseline no
```

`shared` baseline 不适用于当前 Stage2 reward。Stage2 的 reward 形状是
`[B]`，而 RL4CO 的 `SharedBaseline` 期望在 `dim=1` 上求均值；直接使用会报
`IndexError: Dimension out of range`。

### 9.4 损失

基本形式：

```text
loss
= -(reward - baseline) * log p(z | data)
```

只有 `PartitionNetwork` 的参数收到梯度。

## 10. 推荐命令

### 10.1 当前 AM + min-max 设计

```bash
python -m scripts.stage2.run_stage2 \
    --num-loc 50 \
    --num-vehicles 8 \
    --capacity 40 \
    --objective minmax \
    --capacity-mask-mode static \
    --penalty 10.0 \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --route-pad-to 20 \
    --route-eval-batch-size 64 \
    --baseline rollout \
    --epochs 100 \
    --batch-size 32 \
    --train-data-size 1280000 \
    --val-data-size 10000 \
    --accelerator gpu \
    --devices 1
```

### 10.2 原始 TAM 的 min-sum 版本

```bash
python -m scripts.stage2.run_stage2 \
    --num-loc 50 \
    --num-vehicles 8 \
    --capacity 40 \
    --objective sum \
    --capacity-mask-mode static \
    --penalty 10.0 \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --route-pad-to 20 \
    --baseline rollout \
    --epochs 100 \
    --batch-size 32
```

## 11. 当前实现与 `docs.md` 的差异

### 11.1 联合训练尚未实现

`docs.md` 中的联合损失是：

```text
L_joint
= -(R - b)
   * (log P_assign(z)
      + α Σ_k log P_route(τ_k))
```

当前代码只训练：

```text
PartitionNetwork
```

Stage1 AM：

```text
冻结，不计算梯度，不更新参数
```

因此当前版本是：

```text
TAM Step2 的“冻结 TSP 评价器”版本
```

不是完整的联合训练版本。

### 11.2 没有全局 sequence mask

原始 TAM 通过 autoregressive decoder 和 depot mask 控制最大车辆数。

当前分簇网络是一次性输出 `[B, M, N]`，所以：

- 车辆数通过运行时输出维度 `M` 决定；同一个权重可以用于不同的 M；
- 没有“下一次是否访问 depot”的顺序决策；
- 没有 TAM 式 global mask；
- 空簇允许存在。

如果问题需要固定使用所有车辆，或者需要严格限制实际使用车辆数，还需要增加
额外的车辆使用约束。

注意：结构上可接收不同的 M，不等于训练过的模型可以在不同 M 上零样本泛化。
如果训练只见过 M=5，推断时直接使用 M=7，输出维度虽然可以正常计算，但
vehicle 竞争、PE 的 m 分布和 partition 行为都处于训练分布之外。要真正训练
一个跨 M 复用的模型，需要在训练时随机化车辆数量并对无效 vehicle 做 padding
和 mask。

### 11.3 静态容量 mask 不等于累计容量可行

当前默认 `static` 只检查：

```text
capacity[m] < demand[i]
```

它不检查：

```text
load_m + demand_i > capacity_m
```

累计容量不可行时，通过后验 `overflow_m` 惩罚处理。

这保留了节点独立分配，但不能从数学上保证最终每个簇都满足容量约束。

### 11.4 目标函数不同

原始 TAM 是：

```text
sum of sub-route lengths
```

当前默认是：

```text
max of sub-route lengths
```

如果要做 TAM 复现，应使用：

```bash
--objective sum
```

## 12. 仍然欠缺的工程部分

### 12.1 联合路由训练

需要：

1. AM 返回每个 cluster 的 sampled actions；
2. 计算每个 cluster 的 route log-prob；
3. 把 `Σ log P_route` 加入 Stage2 loss；
4. 将 AM 参数加入优化器；
5. 确定上界 gradient clip 和学习率。

### 12.2 更严格的容量可行分配

如果需要严格容量可行，需要考虑：

- 精确 bin-packing；
- 容量修复；
- feasibility projection；
- 或者 sequential mask。

但这些方案都会破坏“节点完全独立分配”的简单性。

### 12.3 训练指标记录

目前 policy 返回：

```text
cost
objective_cost
overflow
overflow_per_vehicle
```

但还没有分别记录成 CSV 曲线：

```text
train/max_route
train/total_route
train/overflow
val/max_route
```

### 12.4 Padding 构造优化

TSP forward 已经批量化，但构造 `[B * M, cluster_len, 2]` 仍使用 Python
循环。后续可以用 scatter/index 操作进一步加速。

### 12.5 POMO/MDAM 不是当前主要验证路径

当前主要保证：

```text
AM route evaluator
```

可以正常工作。

POMO 和 MDAM 的接口保留，但还需要针对以下问题单独验证：

- POMO 的 padded starts 是否影响 multi-start 分布；
- `--route-num-starts` 是否与 Stage1 POMO checkpoint 的训练设置一致；
- MDAM 是否使用 Stage1 中修复后的 decoder。

## 13. 结论

当前 Stage2 已经实现：

```text
2D-Ptr 风格的一次性分簇
+ 每-node 独立车辆 softmax
+ static 容量 mask
+ 簇累计容量溢出惩罚
+ depot padding
+ 批量冻结 AM 评价
+ min-max / sum 可切换目标
+ rollout / exponential / no baseline
+ REINFORCE 训练
```

它已经不是原来的“逐簇循环调用 Stage1 模型”版本，也已经拆开了：

```text
Stage2 batch size
和
TSP evaluation batch size
```

但距离完整设计仍然缺：

```text
AM 联合训练
严格累计容量可行性
完整的 Stage2 指标曲线
```
