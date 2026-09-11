# LKH-3 solver

English | [中文](README.zh-CN.md)

This directory contains the external LKH-3 backend used by the Stage 1 TSP
evaluation interface. OR-Tools is a Python package and is not stored here;
install it separately with `pip install ortools`.

## Contents

```text
assets/solver/
  LKH-3.0.14.tgz          upstream LKH-3 source archive
  LKH-3.0.14/
    LKH                   compiled executable used by the project
    SRC/                  LKH-3 C source and Makefile
    DOC/                  upstream documentation files
    *.tsp, *.par          upstream sample instances and parameter files
    LKH.dSYM/             debug symbols, not required at runtime
    README.txt            upstream LKH documentation (not translated)
```

The upstream LKH-3 project is available at:

```text
http://webhotel4.ruc.dk/~keld/research/LKH-3/
```

LKH is developed by Keld Helsgaun and is distributed for research use. The
upstream `README.txt` and the author's terms apply to the LKH source and
binary. This project does not relicense LKH.

The upstream `README.txt` is intentionally kept in its original form. The
English/Chinese bilingual documentation in this repository describes how to
build and call LKH; it is not a translation of the upstream manual.

## Build

Requirements: `gcc` and `make`.

```bash
cd assets/solver/LKH-3.0.14
make
```

To force a clean rebuild:

```bash
cd assets/solver/LKH-3.0.14
make clean
make
```

The resulting executable is:

```text
assets/solver/LKH-3.0.14/LKH
```

The root Makefile delegates to `SRC/Makefile`. The upstream Makefile compiles
with `-march=native`, so rebuild the binary when moving to a different CPU.

## Run with this project

The Stage 1 evaluator calls the executable directly:

```bash
python -m scripts.stage1.eval_stage1 \
    --data-file data/tsp20/test.npz \
    --skip-neural \
    --solvers lkh \
    --solver-time-limit 1.0 \
    --lkh-path assets/solver/LKH-3.0.14/LKH \
    --output assets/results/lkh_tsp20.json
```

`pip install lkh` is not required. That PyPI package is only a Python wrapper;
the evaluator in this repository invokes the LKH executable directly.
If the executable is already at `assets/solver/LKH-*/LKH`, `--lkh-path` can
normally be omitted because the evaluator searches this location
automatically.

Useful related options:

- `--solver-time-limit`: time limit per TSP instance, in seconds.
- `--lkh-runs`: number of LKH runs per instance.
- `--solver-num-procs`: number of worker processes used by the evaluator.
- `--progress-interval`: progress/ETA reporting interval; `0` disables it.

## Portability

The bundled binary was built for the build machine's CPU. If it is moved to a
different CPU or operating-system environment and reports `Illegal instruction`
or a loader error, rebuild it on the target machine:

```bash
cd assets/solver/LKH-3.0.14
make clean
make
```

## Verify

```bash
test -x assets/solver/LKH-3.0.14/LKH && echo "LKH executable OK"
file assets/solver/LKH-3.0.14/LKH
```

Run a small solver-only smoke test:

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

Do not use the full 100k-instance evaluation set for a first smoke test.
