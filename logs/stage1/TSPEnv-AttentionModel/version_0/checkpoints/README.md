# AttentionModel TSP20 checkpoint

English | [中文](README.zh-CN.md)

Download the files into this directory:

```text
logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/
```

## File mapping

| Hugging Face file | Local filename |
|---|---|
| `AM_tsp_20_last.ckpt` | `last.ckpt` |
| `AM_tsp_20_epoch_094.ckpt` | `epoch_094.ckpt` |

In the current release, `last.ckpt` and `epoch_094.ckpt` have identical
contents. The best `val/reward` occurred at epoch 94.

## Download

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    AM_tsp_20_last.ckpt AM_tsp_20_epoch_094.ckpt \
    --repo-type dataset \
    --local-dir .

mv AM_tsp_20_last.ckpt last.ckpt
mv AM_tsp_20_epoch_094.ckpt epoch_094.ckpt
```

## Checksum

```bash
sha256sum last.ckpt epoch_094.ckpt
```

Expected checksums for the current version:

```text
8cc216396ae47c3016c3618034e155128338748bf671a05b2a7699f2933692d1  last.ckpt
8cc216396ae47c3016c3618034e155128338748bf671a05b2a7699f2933692d1  epoch_094.ckpt
```

## Model configuration

```text
model: AttentionModel
env: TSPEnv
num_loc: 20
embed_dim: 128
encoder_layers: 3
heads: 8
normalization: batch
baseline: rollout
```

## Evaluation

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt
```
