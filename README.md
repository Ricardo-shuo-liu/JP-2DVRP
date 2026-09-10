# JP‑2DVRP

两阶段车辆路径求解代码：

- Stage1：训练单簇/单车 **TSP** 路由模型，模型来自 RL4CO。
- Stage2：使用 RL4CO `CVRPEnv` 生成完整 CVRP 实例，训练 2D-Ptr 风格的分簇网络；Stage1 路由模型作为**冻结的 reward 评价器**，为每个簇计算 TSP 路线成本。

依赖环境需要 Python 3.10+ 以及 `rl4co>=0.6`、`torch`、`lightning`、`tensordict`。

## 目录结构

```text
utils/
  config/                 Stage1/Stage2 配置
  data.py                 环境构造、npz 规范读写、可行 CVRP 生成器
  solvers/tsp.py          OR-Tools/LKH3 TSP baseline 适配与计时
nets/
  blocks.py               Transformer 基础块
  rotary_pe.py            DPN 旋转位置编码
  vehicle_encoder.py      迁移自 2D-Ptr 的车辆编码/交叉注意力
  partition.py            一次分簇网络（节点维独立 softmax）
scripts/
  stage1/                 Stage1 训练、TSP 数据生成与固定数据集评估
  stage2/                 Stage2 数据生成、训练、分簇策略、路由评价器
```

## Stage1：训练 TSP 路由模型

Stage1 永远是“一辆车、一个簇”的子问题，因此环境固定为 `TSPEnv`，不接受 `CVRPEnv`。

```bash
# Attention Model（默认）
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m AttentionModel \
    --num-loc 20 --epochs 10 --batch-size 256

# POMO
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m POMO \
    --num-loc 20 --epochs 10 --batch-size 64

# POMO TSP20 论文配置（510 epochs，每 epoch 100k 实例）
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m POMO --num-loc 20 \
    --epochs 510 --batch-size 64 --train-data-size 100000 \
    --lr 1e-4 --weight-decay 1e-6 \
    --num-encoder-layers 6 --num-heads 8 --normalization instance \
    --baseline shared --lr-scheduler multi_step \
    --lr-milestones 501 --lr-gamma 0.1 \
    --logit-clipping 10 --feedforward-hidden 512 \
    --precision 32 --accelerator gpu --devices 1

# MDAM TSP20 论文配置（100 epochs，每 epoch 1.28M 实例）
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m MDAM --num-loc 20 \
    --epochs 100 --batch-size 512 --train-data-size 1280000 \
    --val-data-size 10000 --lr 1e-4 --gradient-clip-val 1.0 \
    --num-encoder-layers 3 --num-heads 8 --normalization batch \
    --baseline rollout --mdam-num-paths 5 \
    --mdam-eg-step-gap 2 --mdam-kl-loss 0.01 \
    --logit-clipping 10 --precision 32 \
    --accelerator gpu --devices 1
```

可选模型：`AttentionModel`、`POMO`、`MDAM`、`PointerNetwork`。
查看全部参数：

```bash
python -m scripts.stage1.run_stage1 --help
```

生成固定 TSP 数据，用于固定测试集比较不同路由模型：

```bash
python -m scripts.stage1.generate_data \
    --num-loc 50 --num-samples 10000 \
    --split val --out-dir data/tsp50 --seed 1234

python -m scripts.stage1.run_stage1 \
    -m AttentionModel --num-loc 50 \
    --data-dir data/tsp50 --val-file val.npz --test-file test.npz
```

Stage1 checkpoint 默认写入 `logs/stage1/TSPEnv-<model>/version_*/checkpoints/`，
Stage2 会通过 `--route-checkpoint` 加载它。

## Stage1：在同一数据集上评估模型

训练数据每个 epoch 都重新随机生成，因此把单独生成的 `test.npz` 交给所有模型
评估，就等于测它们对“完全没见过的实例”的泛化效果。

```bash
# 先生成一个固定的测试集
python -m scripts.stage1.generate_data \
    --num-loc 20 --num-samples 10000 --split test \
    --out-dir data/tsp20 --seed 4321

# 自动扫描 logs/stage1 下四种模型，取每个模型最新 version 的 last.ckpt
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --checkpoint-root logs/stage1 \
    --device cuda

# 也可以只评估某个 checkpoint
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --device cuda
```

默认使用 greedy 解码；要按论文式 sampling 评估（多个候选取最好）：

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --decode-type sampling \
    --width 100 \
    --batch-size 128 \
    --device cuda
```

`--width` 是每个实例采样的候选路线数。显存有限时把 `--batch-size` 调小即可。
输出会打印每个模型的 `mean_cost`、与最优模型的 gap、std/min/max 和吞吐，
也可以用 `--output results.json` 保存结构化结果。

### Stage1：神经模型与传统求解器对比

同一个评估入口可以同时加入 OR-Tools 和 LKH3。传统求解器不需要 checkpoint，
但必须先把数据固定到 `--data-file`，这样所有方法看到的是同一批实例：

```bash
# 同时评估已有 checkpoint，以及 OR-Tools / LKH3
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --solver-num-procs 4 \
    --lkh-path /path/to/LKH \
    --output assets/results/stage1_tsp20_compare.json

# 一次评估多个神经模型；按默认目录结构自动找每个模型最新 version
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --checkpoint-root logs/stage1 \
    --models AttentionModel POMO MDAM \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --solver-num-procs 4 \
    --output assets/results/stage1_tsp20_multi_compare.json

# 只跑传统求解器
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --skip-neural \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --output assets/results/stage1_tsp20_classical.json
```

OR-Tools 需要安装 `pip install ortools`；LKH3 不需要 Python 包装器，
脚本会自动查找 `assets/solver/LKH-*/LKH`，也可以用 `--lkh-path` 显式指定。
`--solver-time-limit` 是每个实例的限制，不是整批时间。输出 JSON 会记录数据集
SHA256、每个实例的解、状态、耗时，以及成功/失败统计，便于复查。
默认每 10 秒输出一次进度和 ETA，可用 `--progress-interval 0` 关闭。
结果会在每个模型/求解器完成后增量写入一次，未全部完成时 JSON 中的
`partial=true`；全部完成后写为 `partial=false`。

## Stage2：生成 CVRP 数据

Stage2 数据来自 RL4CO `CVRPEnv`。生成器会保证每个样本
`sum(demand) <= num_vehicles * capacity`：

```bash
python -m scripts.stage2.generate_data \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --num-samples 10000 --split train \
    --out-dir data/cvrp50 --seed 1234

# val / test 使用不同 seed 再生成一次
python -m scripts.stage2.generate_data \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --num-samples 2048 --split val \
    --out-dir data/cvrp50 --seed 4321
```

生成文件的字段为：

- `locs`：客户坐标 `[N, num_loc, 2]`
- `depot`：仓库坐标 `[N, 2]`
- `demand`：未归一化整数需求 `[N, num_loc]`
- `capacity`：原始容量 `[N]`
- `num_vehicles`：车辆数 `[N, 1]`

读取时由 RL4CO `CVRPEnv.load_data` 按 capacity 归一化一次。

## Stage2：训练分簇网络

不指定固定文件时，训练/验证数据由 RL4CO CVRPEnv 在线生成；也可以读取上面的 npz：

```bash
# 在线生成，随机初始化 Stage1 路由模型（先做 smoke test）
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --route-model AttentionModel \
    --epochs 5 --batch-size 32

# 使用 Stage1 checkpoint 作为冻结 reward 评价器
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --epochs 5 --batch-size 32

# 使用固定 CVRP 数据
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --data-dir data/cvrp50 --train-file train.npz \
    --val-file val.npz \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt
```

Stage2 每次实验通过 `--num-vehicles` 固定车辆数；`--penalty` 控制容量惩罚系数；
`--baseline` 可选 `exponential`（默认）、`shared`、`no`。

Stage2 分簇网络输出 `[batch, num_vehicles, num_nodes]` logits，对每个 node 维做
独立 softmax 后采样/贪心得到簇分配；每个簇再交给冻结的 Stage1 TSP 模型求解。
训练损失只更新分簇网络，不更新 Stage1 路由参数。

## 查看参数与常见错误

```bash
python -m scripts.stage1.run_stage1 --help
python -m scripts.stage2.run_stage2 --help
python -m scripts.stage2.generate_data --help
```

- Stage1 传入 `-e CVRPEnv` 会被拒绝：CVRP 的多车/容量问题属于 Stage2。
- Stage2 的 `--route-model` 必须与生成 checkpoint 的 Stage1 模型一致，
  否则加载时报告 `policy.*` 参数缺失。
- `--route-checkpoint` 省略时使用随机初始化的路由模型，适合先验证整条流水线。
