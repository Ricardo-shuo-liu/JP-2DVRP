# POMO TSP20 Checkpoint

[English](README.md) | 中文

下载后的文件必须放在当前目录：

```text
logs/stage1/TSPEnv-POMO/version_0/checkpoints/
```

## 文件对应关系

| Hugging Face 文件 | 本地文件名 |
|---|---|
| `POMO_tsp_20_last.ckpt` | `last.ckpt` |
| `POMO_tsp_20_epoch_509.ckpt` | `epoch_509.ckpt` |

当前版本中 `last.ckpt` 和 `epoch_509.ckpt` 内容相同；训练时最好的
`val/reward` 出现在 epoch 509。

## 下载

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    POMO_tsp_20_last.ckpt POMO_tsp_20_epoch_509.ckpt \
    --repo-type dataset \
    --local-dir .

mv POMO_tsp_20_last.ckpt last.ckpt
mv POMO_tsp_20_epoch_509.ckpt epoch_509.ckpt
```

## 校验

```bash
sha256sum last.ckpt epoch_509.ckpt
```

当前版本期望值：

```text
98603e9a9efead73b1025b1cc35eb5d422c8622de4eec83d687a383d9ce0be11  last.ckpt
98603e9a9efead73b1025b1cc35eb5d422c8622de4eec83d687a383d9ce0be11  epoch_509.ckpt
```

## 模型配置

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

## 评估

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model POMO \
    --checkpoint logs/stage1/TSPEnv-POMO/version_0/checkpoints/last.ckpt
```
