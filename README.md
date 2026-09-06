

```bash
python scripts/stage1/run_stage1.py
```

指定环境、模型和超参数：

```bash
python scripts/stage1/run_stage1.py \
    -e TSPEnv -m AttentionModel \
    --num-loc 50 --lr 1e-4 --epochs 3 --batch-size 512
```

生成固定数据（便于不同模型在同一份数据上比较）：

```bash
python scripts/stage1/generate_data.py \
    -e TSPEnv --num-loc 50 --num-samples 10000 \
    --split val --out-dir data/tsp50 --seed 1234
```

用固定数据训练并比较不同模型：

```bash
python scripts/stage1/run_stage1.py \
    -e TSPEnv -m AttentionModel \
    --data-dir data/tsp50 --val-file val.npz --test-file test.npz

python scripts/stage1/run_stage1.py \
    -e TSPEnv -m POMO \
    --data-dir data/tsp50 --val-file val.npz --test-file test.npz
```

查看全部参数：

```bash
python scripts/stage1/run_stage1.py --help
python scripts/stage1/generate_data.py --help
```
