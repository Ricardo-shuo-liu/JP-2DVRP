# JP-2DVRP

English | [中文](README.zh-CN.md)

A two-stage vehicle-routing codebase:

- **Stage 1**: train a single-vehicle / single-cluster **TSP** routing model from RL4CO.
- **Stage 2**: generate complete CVRP instances with RL4CO's `CVRPEnv` and train a
  2D-Ptr-style partition network. The frozen Stage 1 model evaluates each cluster
  and returns its TSP route cost.

Required environment: Python 3.10+, `rl4co>=0.6`, `torch`, `lightning`, and
`tensordict`.

## Repository layout

```text
utils/
  config/                 Stage1/Stage2 configuration
  data.py                 environment builders, npz helpers, feasible CVRP generator
  solvers/tsp.py          OR-Tools/LKH3 TSP baselines with timing
nets/
  blocks.py               shared Transformer blocks
  rotary_pe.py            DPN rotation-based positional encoding
  vehicle_encoder.py      2D-Ptr-style vehicle encoder / cross-attention
  partition.py            one-shot partition network (node-wise softmax)
scripts/
  stage1/                 Stage1 training, TSP data generation, fixed-set evaluation
  stage2/                 Stage2 CVRP data generation, training, policy, route evaluator
  plot_metrics.py         plot training curves from metrics.csv
  plot_eval_results.py    plot evaluation JSON files
```

## Stage 1: train a TSP routing model

Stage 1 is always a one-vehicle, one-cluster subproblem, so the environment is
fixed to `TSPEnv`; `CVRPEnv` is rejected here.

```bash
# Attention Model (default)
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m AttentionModel \
    --num-loc 20 --epochs 10 --batch-size 256

# POMO
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m POMO \
    --num-loc 20 --epochs 10 --batch-size 64

# POMO TSP20 paper recipe (510 epochs, 100k instances per epoch)
python -m scripts.stage1.run_stage1 \
    -e TSPEnv -m POMO --num-loc 20 \
    --epochs 510 --batch-size 64 --train-data-size 100000 \
    --lr 1e-4 --weight-decay 1e-6 \
    --num-encoder-layers 6 --num-heads 8 --normalization instance \
    --baseline shared --lr-scheduler multi_step \
    --lr-milestones 501 --lr-gamma 0.1 \
    --logit-clipping 10 --feedforward-hidden 512 \
    --num-starts 20 --precision 32 \
    --accelerator gpu --devices 1

# MDAM TSP20 paper recipe (100 epochs, 1.28M instances per epoch)
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

Supported models: `AttentionModel`, `POMO`, `MDAM`, `PointerNetwork`.

```bash
python -m scripts.stage1.run_stage1 --help
```

Generate a fixed TSP dataset for fair comparisons:

```bash
python -m scripts.stage1.generate_data \
    --num-loc 50 --num-samples 10000 \
    --split val --out-dir data/tsp50 --seed 1234

python -m scripts.stage1.run_stage1 \
    -m AttentionModel --num-loc 50 \
    --data-dir data/tsp50 --val-file val.npz --test-file test.npz
```

Stage 1 checkpoints are written to
`logs/stage1/TSPEnv-<model>/version_*/checkpoints/`. Stage 2 loads a checkpoint
through `--route-checkpoint`.

## Stage 1: evaluate models on one fixed dataset

Training regenerates random data every epoch, so a separately generated
`test.npz` measures generalization to unseen instances.

```bash
# Create a fixed test set
python -m scripts.stage1.generate_data \
    --num-loc 20 --num-samples 10000 --split test \
    --out-dir data/tsp20 --seed 4321

# Auto-discover the latest checkpoint of each supported model
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --checkpoint-root logs/stage1 \
    --device cuda

# Evaluate one checkpoint explicitly
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --device cuda
```

Greedy decoding is the default. For paper-style sampling, use:

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

`--width` is the number of sampled candidates per instance. Lower
`--batch-size` if GPU memory is tight. The terminal prints `mean_cost`, gap to
the best mean cost, std/min/max, and throughput. Use `--output results.json` to
save a structured result.

### Stage 1: compare neural models with classical solvers

The same evaluator can add OR-Tools and LKH3. Classical solvers need no
checkpoint, but all methods must use the same `--data-file`:

```bash
# Neural model plus OR-Tools / LKH3
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --solver-num-procs 4 \
    --lkh-path /path/to/LKH \
    --output assets/results/stage1_tsp20_compare.json

# Evaluate several neural models at once
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --checkpoint-root logs/stage1 \
    --models AttentionModel POMO MDAM \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --solver-num-procs 4 \
    --output assets/results/stage1_tsp20_multi_compare.json

# Classical solvers only
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --skip-neural \
    --solvers ortools lkh \
    --solver-time-limit 1.0 \
    --output assets/results/stage1_tsp20_classical.json
```

Install OR-Tools with `pip install ortools`. LKH3 does not require a Python
wrapper; the evaluator automatically searches `assets/solver/LKH-*/LKH`, and
`--lkh-path` can override it.

`--solver-time-limit` is a per-instance limit, not a whole-batch limit. The
output JSON records the dataset SHA256, per-instance costs, statuses, timings,
and success/failure statistics. Progress and ETA are printed every 10 seconds
by default; use `--progress-interval 0` to disable this.

Results are written incrementally after each model or solver finishes. An
incomplete file has `partial=true`; a completed file has `partial=false`.

Plot evaluation results:

```bash
python scripts/plot_eval_results.py \
    assets/results/stage1_tsp20_compare.json \
    -o assets/images/stage1_tsp20_eval.png
```

Add `--distributions` to save a per-instance cost-distribution box plot:

```bash
python scripts/plot_eval_results.py \
    assets/results/stage1_tsp20_compare.json \
    --distributions \
    -o assets/images/stage1_tsp20_eval.png
```

The distribution plot labels `mean/p95` above each box and `p5/p50` below it.
Use `--no-distribution-labels` to disable the labels.

## Stage 2: generate CVRP data

Stage 2 data comes from RL4CO's `CVRPEnv`. The generator ensures that every
sample satisfies `sum(demand) <= num_vehicles * capacity`:

```bash
python -m scripts.stage2.generate_data \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --num-samples 10000 --split train \
    --out-dir data/cvrp50 --seed 1234

# Generate validation data with a different seed
python -m scripts.stage2.generate_data \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --num-samples 2048 --split val \
    --out-dir data/cvrp50 --seed 4321
```

Total-demand feasibility is treated as an environment/data check, not as a
model objective. The generator rejects samples with
`sum(demand) > num_vehicles * capacity`.

Generated fields:

- `locs`: customer coordinates `[N, num_loc, 2]`
- `depot`: depot coordinates `[N, 2]`
- `demand`: unnormalized integer demand `[N, num_loc]`
- `capacity`: raw capacity `[N]`
- `num_vehicles`: number of vehicles `[N, 1]`

When loading, RL4CO's `CVRPEnv.load_data` normalizes demand by capacity once.

## Stage 2: train the partition network

When no fixed data file is given, RL4CO `CVRPEnv` generates training and
validation data online. Fixed npz files are also supported:

```bash
# Online data and a randomly initialized Stage 1 route model (smoke test)
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --route-model AttentionModel \
    --epochs 5 --batch-size 32

# Use a Stage 1 checkpoint as the frozen reward evaluator
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --epochs 5 --batch-size 32

# Use fixed CVRP data
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --data-dir data/cvrp50 --train-file train.npz \
    --val-file val.npz \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt
```

Each Stage 2 experiment fixes the vehicle count through `--num-vehicles`.
`--penalty` controls the capacity-overflow penalty, and `--baseline` accepts
`rollout` (paper-style default), `exponential`, `shared`, or `no`.

The Stage 2 objective can be either total route length or the maximum route
length:

```bash
# Min-max objective with capacity-aware assignment
python -m scripts.stage2.run_stage2 \
    --num-loc 50 --num-vehicles 8 --capacity 40 \
    --objective minmax \
    --capacity-mask-mode static \
    --penalty 10.0 \
    --route-model AttentionModel \
    --route-checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
    --route-pad-to 20
```

- `--objective minmax` uses `max_k TSP(cluster_k)`; `sum` uses the
  conventional total CVRP length.
- `--capacity-mask-mode static` disables `(vehicle, node)` entries whose
  vehicle capacity is smaller than that node's demand. It keeps node
  assignments independent; cumulative cluster overflow is handled by the
  post-assignment penalty.
- `--penalty` penalizes per-vehicle capacity overflow.
- `--route-pad-to` pads every cluster to at least this many customer slots
  before calling the Stage 1 TSP evaluator; `0` uses the largest cluster in
  the batch.
- `--route-eval-batch-size` limits how many clusters are sent to the TSP model
  in one forward pass; `0` uses all clusters. Stage 2 batch size and TSP
  evaluation batch size are therefore independent.

The partition network outputs `[batch, num_vehicles, num_nodes]` logits and
selects one vehicle per customer. With `--capacity-mask-mode none` or `static`
the assignments are independent over nodes; cumulative overflow is handled by
the post-assignment penalty.
Every cluster is then solved by the frozen Stage 1 TSP model. Training updates
the partition network only; Stage 1 routing parameters remain frozen.

## Help and common errors

```bash
python -m scripts.stage1.run_stage1 --help
python -m scripts.stage2.run_stage2 --help
python -m scripts.stage2.generate_data --help
```

- Passing `-e CVRPEnv` to Stage 1 is rejected because multi-vehicle capacity
  constraints belong to Stage 2.
- Stage 2's `--route-model` must match the model that produced the Stage 1
  checkpoint, otherwise loading reports missing `policy.*` parameters.
- Omitting `--route-checkpoint` uses a randomly initialized route model, which
  is useful only for end-to-end smoke tests.
