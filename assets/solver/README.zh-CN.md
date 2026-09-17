# LKH-3 求解器

[English](README.md) | 中文

该目录保存 Stage1 TSP 评测所需的 LKH-3 外部程序。OR-Tools 是 Python 包，
不放在这里，需要单独执行 `pip install ortools`。

## 目录内容

```text
assets/solver/
  LKH-3.0.14.tgz          LKH-3 上游源码压缩包
  LKH-3.0.14/
    LKH                   项目使用的可执行文件
    SRC/                  LKH-3 C 源码和 Makefile
    DOC/                  上游文档文件
    *.tsp, *.par          上游示例实例和参数文件
    LKH.dSYM/             调试符号，运行时不使用
    README.txt            上游 LKH 文档，保持原文
```

上游 LKH-3 项目地址：

```text
http://webhotel4.ruc.dk/~keld/research/LKH-3/
```

LKH 由 Keld Helsgaun 开发，仅供研究使用。LKH 源码和二进制适用上游
`README.txt` 以及作者的使用条款；本仓库不会对其重新授权。

上游 `README.txt` 保持原版不翻译；本仓库的中英文文档只说明如何编译和调用
LKH，不是上游手册的翻译版本。

## 编译

依赖 `gcc` 和 `make`。

```bash
cd assets/solver/LKH-3.0.14
make
```

强制重新编译：

```bash
cd assets/solver/LKH-3.0.14
make clean
make
```

生成的可执行文件：

```text
assets/solver/LKH-3.0.14/LKH
```

根目录 Makefile 会转调用 `SRC/Makefile`。上游 Makefile 使用
`-march=native` 编译，因此迁移到不同 CPU 机器时需要重新编译。

## 在项目中运行

Stage1 评估脚本会直接调用该可执行文件：

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --skip-neural \
    --solvers lkh \
    --solver-time-limit 1.0 \
    --lkh-path assets/solver/LKH-3.0.14/LKH \
    --output assets/results/lkh_tsp20.json
```

不需要执行 `pip install lkh`。PyPI 上的 `lkh` 只是 Python 包装器，
本仓库的评测代码直接调用 LKH 可执行文件。
如果可执行文件已经放在 `assets/solver/LKH-*/LKH`，通常可以不传
`--lkh-path`，评估脚本会自动查找该位置。

相关参数：

- `--solver-time-limit`：每个 TSP 实例的时间上限，单位秒。
- `--lkh-runs`：每个实例运行 LKH 的次数。
- `--solver-num-procs`：评估时使用的工作进程数。
- `--progress-interval`：进度和 ETA 输出间隔；设为 `0` 可关闭。

## CVRP 基线（Stage2 验证）

`utils/solvers/cvrp.py` 用同一个可执行文件求解带容量 VRP，
`scripts/stage2/eval_cvrp.py` 则在 Stage2 测试集上批量评估：

```bash
python -m scripts.stage2.eval_cvrp \
    --data-file data/cvrp100/test.npz \
    --num-instances 100 \
    --solvers lkh \
    --objectives minsum minmax \
    --lkh-path assets/solver/LKH-3.0.14/LKH \
    --output assets/results/cvrp100_baseline.json
```

- `--objectives minsum`：LKH 原生 CVRP 目标，最小化总路径长度。
- `--objectives minmax`：与 Stage2 奖励一致，最小化最长单条路径。LKH-3
  **没有**原生 min-max CVRP 目标：`TYPE: CVRP` 下 `MTSP_OBJECTIVE = MINMAX`
  会被忽略（见 `SRC/MTSP2TSP.c`），而 `TYPE: CCVRP` + `MINMAX` 优化的是另一套
  内部量，并不对应实际路径长度。因此这里通过二分单车路径长度上限
  （`DISTANCE = λ`）逼近 min-max 解。得到的是**强启发式上界，不是可证明最优**。
- 路径长度、载重、容量超出量都在 Python 侧用坐标重新计算；LKH 打印的
  `Cost` 使用的是内部变换量纲，不作为结果。

正确性对照（6 个客户、2-3 辆车，穷举全部划分与路径排列）：min-sum 与精确最优
在 17 个实例中命中 16 个；min-max 二分搜索命中 14 个，最差相对偏差约 3.6%。

## 可移植性

当前二进制是针对编译机器的 CPU 生成的。如果迁移到其他 CPU 或操作系统环境
后出现 `Illegal instruction` 或动态库错误，请在目标机器上重新编译：

```bash
cd assets/solver/LKH-3.0.14
make clean
make
```

## 检查

```bash
test -x assets/solver/LKH-3.0.14/LKH && echo "LKH executable OK"
file assets/solver/LKH-3.0.14/LKH
```

小规模 smoke test：

```bash
python -m scripts.stage1.generate_data \
    --num-loc 20 --num-samples 100 --split test \
    --out-dir /tmp/tsp20_smoke --seed 1

python -m scripts.stage1.eval_stage1 \
    --data-file /tmp/tsp20_smoke/test.npz \
    --skip-neural \
    --solvers lkh \
    --solver-time-limit 0.1 \
    --solver-num-procs 1 \
    --output /tmp/lkh_smoke.json
```

不要第一次就用 10 万实例做完整评估。
