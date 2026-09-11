# POMO TSP20 checkpoint

English | [中文](README.zh-CN.md)

Download the files into this directory:

```text
logs/stage1/TSPEnv-POMO/version_0/checkpoints/
```

## File mapping

| Hugging Face file | Local filename |
|---|---|
| `POMO_tsp_20_last.ckpt` | `last.ckpt` |
| `POMO_tsp_20_epoch_509.ckpt` | `epoch_509.ckpt` |

In the current release, `last.ckpt` and `epoch_509.ckpt` have identical
contents. The best `val/reward` occurred at epoch 509.

## Download

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    POMO_tsp_20_last.ckpt POMO_tsp_20_epoch_509.ckpt \
    --repo-type dataset \
    --local-dir .

mv POMO_tsp_20_last.ckpt last.ckpt
mv POMO_tsp_20_epoch_509.ckpt epoch_509.ckpt
```

## Checksum

```bash
sha256sum last.ckpt epoch_509.ckpt
```

Expected checksums for the current version:

```text
98603e9a9efead73b1025b1cc35eb5d422c8622de4eec83d687a383d9ce0be11  last.ckpt
98603e9a9efead73b1025b1cc35eb5d422c8622de4eec83d687a383d9ce0be11  epoch_509.ckpt
```

## Model configuration

```text
model: POMO
env: TSPEnv
num_loc: 20
embed_dim: 128
encoder_layers: 6
heads: 8
normalization: instance
baseline: shared
num_starts: 20
logit_clipping: 10
feedforward_hidden: 512
```

## Evaluation

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model POMO \
    --checkpoint logs/stage1/TSPEnv-POMO/version_0/checkpoints/last.ckpt
```
