# TSP20 固定测试集

[English](README.md) | 中文

下载后的文件必须放在当前目录，并命名为：

```text
data/tsp20/test.npz
```

## 从 Hugging Face 下载

当前目录下执行：

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    tsp_20_test.npz \
    --repo-type dataset \
    --local-dir .
mv tsp_20_test.npz test.npz
```

如果没有 `huggingface-cli`：

```bash
pip install huggingface_hub
```

也可以直接下载：

```bash
curl -L \
    https://huggingface.co/datasets/flyRicardo/JP-2DVRP/resolve/main/tsp_20_test.npz \
    -o test.npz
```

## 数据格式

`test.npz` 必须满足：

```text
字段：locs
shape：[100000, 20, 2]
dtype：float32
坐标范围：[0, 1]
```

检查：

```bash
python - <<'PY'
import numpy as np

data = np.load("test.npz")
assert data["locs"].shape == (100000, 20, 2), data["locs"].shape
assert data["locs"].dtype == np.float32, data["locs"].dtype
assert data["locs"].min() >= 0.0
assert data["locs"].max() <= 1.0
print("TSP20 test data OK")
PY
```

## 校验

```bash
sha256sum test.npz
```

当前版本期望值：

```text
93890219c6a30092a3151747e490d17798bdcb35ab489ecbe5770fc90a2f0db8
```

如果校验值不同，不要直接和其他评测结果比较，先确认下载的文件版本。

## 使用位置

Stage1 评估时使用：

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    ...
```
