# TSP20 fixed test set

English | [中文](README.zh-CN.md)

Download the file into this directory and rename it to:

```text
data/tsp20/test.npz
```

## Download from Hugging Face

Run these commands in this directory:

```bash
huggingface-cli download flyRicardo/JP-2DVRP \
    tsp_20_test.npz \
    --repo-type dataset \
    --local-dir .
mv tsp_20_test.npz test.npz
```

If `huggingface-cli` is unavailable:

```bash
pip install huggingface_hub
```

The file can also be downloaded directly:

```bash
curl -L \
    https://huggingface.co/datasets/flyRicardo/JP-2DVRP/resolve/main/tsp_20_test.npz \
    -o test.npz
```

## Data format

`test.npz` must contain:

```text
field: locs
shape: [100000, 20, 2]
dtype: float32
coordinate range: [0, 1]
```

Validate the file:

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

## Checksum

```bash
sha256sum test.npz
```

Expected checksum for the current version:

```text
93890219c6a30092a3151747e490d17798bdcb35ab489ecbe5770fc90a2f0db8
```

If the checksum differs, do not compare the file with existing evaluation
results until the version is confirmed.

## Usage

The Stage 1 evaluator expects:

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    ...
```
