# AttentionModel TSP20 Checkpoint

[English](README.md) | 中文

下载后的文件必须放在当前目录：

```text
logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/
```

## 文件对应关系

| Hugging Face 文件 | 本地文件名 |
|---|---|
| `AM_tsp_20_last.ckpt` | `last.ckpt` |
| `AM_tsp_20_epoch_094.ckpt` | `epoch_094.ckpt` |

当前版本中 `last.ckpt` 和 `epoch_094.ckpt` 内容相同；训练时最好的
`val/reward` 出现在 epoch 94。

## 下载

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    AM_tsp_20_last.ckpt AM_tsp_20_epoch_094.ckpt \
    --repo-type dataset \
    --local-dir .

mv AM_tsp_20_last.ckpt last.ckpt
mv AM_tsp_20_epoch_094.ckpt epoch_094.ckpt
```

## 校验

```bash
sha256sum last.ckpt epoch_094.ckpt
```

当前版本期望值：

```text
8cc216396ae47c3016c3618034e155128338748bf671a05b2a7699f2933692d1  last.ckpt
8cc216396ae47c3016c3618034e155128338748bf671a05b2a7699f2933692d1  epoch_094.ckpt
```

## 模型配置

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

## 评估

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --model AttentionModel \
    --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt
```
