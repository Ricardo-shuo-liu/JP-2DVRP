"""Stage 1 evaluation: compare trained TSP route models on one fixed dataset.

Training uses freshly generated random data every epoch, so evaluating on a
separately generated ``*.npz`` dataset measures the model on instances it has
never seen.  The same file can be passed to every model checkpoint, which makes
the comparison fair.

Usage (project root):

    # Evaluate one checkpoint
    python -m scripts.stage1.eval_stage1 \
        --data-file data/tsp20/test.npz \
        --model AttentionModel \
        --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt

    # Auto-discover one checkpoint per model under logs/stage1
    python -m scripts.stage1.eval_stage1 \
        --data-file data/tsp20/test.npz \
        --checkpoint-root logs/stage1

The default decoding is greedy.  For AM-style sampling evaluation add
``--decode-type sampling --width 100`` (lower the batch size if GPU memory is
tight).

Classical baselines can be added to the same comparison:

    python -m scripts.stage1.eval_stage1 \
        --data-file data/tsp20/test.npz \
        --model AttentionModel \
        --checkpoint logs/stage1/TSPEnv-AttentionModel/version_0/checkpoints/last.ckpt \
        --solvers ortools lkh \
        --solver-time-limit 1.0 \
        --output assets/results/stage1_tsp20_compare.json

Use ``--skip-neural --solvers ortools lkh`` to evaluate only classical solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl4co.data.utils import load_npz_to_tensordict

from scripts.stage1.run_stage1 import get_model
from utils.config.stage1 import Config
from utils.data import make_env
from utils.solvers import (
    find_lkh_executable,
    solve_tsp_lkh,
    solve_tsp_ortools,
)


SUPPORTED_MODELS = ("AttentionModel", "POMO", "MDAM", "PointerNetwork")
SUPPORTED_SOLVERS = ("ortools", "lkh")
PREFERRED_CKPT_NAMES = ("last.ckpt", "best.ckpt")


@dataclass
class Run:
    model: str
    checkpoint: Path
    config_path: Path | None = None


def _version_key(path: Path) -> int:
    match = re.search(r"version[_-]?(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _epoch_key(path: Path) -> int:
    match = re.search(r"epoch[_-]?(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _pick_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    for name in PREFERRED_CKPT_NAMES:
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    files = list(ckpt_dir.glob("*.ckpt"))
    if not files:
        return None
    return max(files, key=_epoch_key)


def _model_from_path(path: Path) -> str | None:
    text = str(path)
    for model in SUPPORTED_MODELS:
        if f"TSPEnv-{model}" in text:
            return model
    return None


def _find_config_dir(checkpoint: Path) -> Path | None:
    """Look for the version/run directory that contains ``config.json``."""
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        if (parent / "config.json").is_file():
            return parent
        if (parent.parent / "config.json").is_file():
            return parent.parent
    return None


def _read_run_config(config_path: Path) -> dict:
    if config_path is None or not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_meta(checkpoint: Path) -> dict:
    """Read hyperparameters saved by Lightning inside the checkpoint."""
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report the real checkpoint error
        raise RuntimeError(f"无法读取 checkpoint {checkpoint}: {exc}") from exc
    hparams = raw.get("hyper_parameters", {})
    if not isinstance(hparams, dict):
        return {}
    policy_kwargs = hparams.get("policy_kwargs", {})
    return {
        "policy_kwargs": policy_kwargs if isinstance(policy_kwargs, dict) else {},
        "baseline": hparams.get("baseline"),
    }


def discover_runs(args) -> list[Run]:
    runs: list[Run] = []
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint 不存在: {checkpoint}")
        model = args.model or _model_from_path(checkpoint)
        if model is None:
            raise SystemExit(
                "无法从 checkpoint 路径推断模型；请用 --model 指定，"
                f"可用: {', '.join(SUPPORTED_MODELS)}"
            )
        runs.append(
            Run(
                model=model,
                checkpoint=checkpoint,
                config_path=_find_config_dir(checkpoint),
            )
        )
        return runs

    root = Path(args.checkpoint_root).resolve()
    models = args.models or list(SUPPORTED_MODELS)
    missing = []
    for model in models:
        base = root / f"TSPEnv-{model}"
        if not base.is_dir():
            missing.append(model)
            continue

        # Versioned run layout: TSPEnv-<model>/version_<i>/checkpoints
        version_dirs = sorted(
            (d for d in base.glob("version_*") if d.is_dir()),
            key=_version_key,
            reverse=True,
        )
        candidates: list[tuple[Path, Path]] = []
        for version_dir in version_dirs if args.all_versions else version_dirs[:1]:
            ckpt_dir = version_dir / "checkpoints"
            ckpt = _pick_checkpoint(ckpt_dir)
            if ckpt is not None:
                candidates.append((ckpt, version_dir))

        # Fallback for logger=none layout: TSPEnv-<model>/checkpoints
        if not candidates:
            ckpt = _pick_checkpoint(base / "checkpoints")
            if ckpt is not None:
                candidates.append((ckpt, base))

        if candidates:
            for ckpt, run_dir in candidates:
                runs.append(
                    Run(
                        model=model,
                        checkpoint=ckpt,
                        config_path=run_dir / "config.json",
                    )
                )
        else:
            missing.append(model)

    if missing:
        print(
            "以下模型未找到可用 checkpoint（已跳过）: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    if not runs:
        raise SystemExit(
            f"{root} 下没有找到任何 Stage1 checkpoint。"
            "请先运行训练，或改用 --checkpoint/--model 显式指定。"
        )
    return runs


def load_policy_weights(model, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint} 中没有可加载的 state_dict")

    policy_state = {
        key[len("policy.") :]: value
        for key, value in state.items()
        if key.startswith("policy.")
    }
    if not policy_state:
        raise RuntimeError(
            f"{checkpoint} 中没有 policy.* 参数；"
            "请确认这是 Stage1 路由模型 checkpoint"
        )
    model.policy.load_state_dict(policy_state, strict=True)


def build_model_for_eval(
    model_name: str,
    num_loc: int,
    checkpoint: Path,
    device: str,
    check_solution: bool = False,
):
    meta = _checkpoint_meta(checkpoint)
    policy_kwargs = dict(meta.get("policy_kwargs") or {})

    config = Config(
        env="TSPEnv",
        model=model_name,
        num_loc=num_loc,
        embed_dim=int(policy_kwargs.get("embed_dim", 128)),
        num_encoder_layers=int(policy_kwargs.get("num_encoder_layers", 3)),
        num_heads=int(policy_kwargs.get("num_heads", 8)),
        normalization=str(policy_kwargs.get("normalization", "batch")),
        baseline="auto",
        batch_size=8,
        train_data_size=8,
        val_data_size=8,
    )
    env = make_env("TSPEnv", num_loc=num_loc, check_solution=check_solution)
    model = get_model(env, config)
    load_policy_weights(model, checkpoint)

    device = torch.device(device)
    model = model.to(device)
    model.policy.eval()
    return model, env


def _expand_width(td: TensorDict, width: int) -> TensorDict:
    if width <= 1:
        return td
    locs = td["locs"].repeat_interleave(width, dim=0)
    return TensorDict({"locs": locs}, batch_size=[locs.shape[0]], device=locs.device)


@torch.no_grad()
def _decode_chunk(
    model,
    env,
    td: TensorDict,
    model_name: str,
    decode_type: str,
    width: int,
    max_parallel: int | None = None,
) -> torch.Tensor:
    """Return best reward per original instance in ``td``."""
    num_instances = td.batch_size[0]
    best_reward = torch.full(
        (num_instances,), float("-inf"), device=td.device
    )

    # Split a large width into smaller parallel chunks to avoid OOM.
    sub_width = width
    if max_parallel and num_instances * width > max_parallel:
        sub_width = max(1, max_parallel // num_instances)

    for start_w in range(0, width, sub_width):
        current_width = min(sub_width, width - start_w)
        td_in = _expand_width(td, current_width)
        td_reset = env.reset(td_in)

        if model_name == "POMO":
            # POMO policy already stores a multistart decode type; only sampling /
            # greedy changes.  num_starts=None means: start from every feasible node.
            policy_decode = (
                "multistart_sampling"
                if decode_type == "sampling"
                else "multistart_greedy"
            )
            model.policy.test_decode_type = policy_decode
            num_starts = env.get_num_starts(td_reset)
            out = model.policy(
                td_reset,
                env,
                phase="test",
                num_starts=num_starts,
            )
            reward = out["reward"]
            # POMO returns rewards in start-major order:
            # [num_starts, expanded_batch], not [batch, width, num_starts].
            reward = reward.reshape(
                num_starts, num_instances, current_width
            ).amax(dim=0)
            reward = reward.amax(dim=1)

        elif model_name == "MDAM":
            out = model.policy(
                td_reset,
                env,
                phase="test",
                decode_type=decode_type,
            )
            reward = out["reward"]
            reward = reward.reshape(
                num_instances, current_width, -1
            ).amax(dim=(1, 2))

        else:
            # AttentionModel / PointerNetwork: one reward per expanded instance.
            out = model.policy(
                td_reset,
                env,
                phase="test",
                decode_type=decode_type,
            )
            reward = out["reward"]
            reward = reward.reshape(num_instances, current_width).amax(dim=1)

        best_reward = torch.maximum(best_reward, reward)
    return best_reward


def evaluate_run(run: Run, td: TensorDict, args) -> dict:
    num_instances = td.batch_size[0]
    num_loc = int(td["locs"].shape[-2])
    device = args.device
    print(
        f"[{run.model}] 开始评估 {num_instances} 个实例，device={device}",
        flush=True,
    )
    model, env = build_model_for_eval(
        run.model,
        num_loc,
        run.checkpoint,
        device,
        check_solution=args.check_solution,
    )
    torch.manual_seed(args.seed)

    rewards = []
    started = time.perf_counter()
    progress = ProgressReporter(
        run.model, num_instances, args.progress_interval
    )
    with torch.no_grad():
        for start in range(0, num_instances, args.batch_size):
            end = min(start + args.batch_size, num_instances)
            batch_td = td[start:end].to(device)
            reward = _decode_chunk(
                model,
                env,
                batch_td,
                model_name=run.model,
                decode_type=args.decode_type,
                width=args.width,
                max_parallel=args.max_parallel,
            )
            rewards.append(reward.detach().cpu())
            progress.update(end)
    progress.update(num_instances, force=True)

    rewards = torch.cat(rewards)
    costs = -rewards  # TSPEnv reward = -tour length
    elapsed = time.perf_counter() - started
    return {
        "kind": "neural",
        "model": run.model,
        "checkpoint": str(run.checkpoint),
        "num_samples": int(num_instances),
        "num_loc": num_loc,
        "decode_type": args.decode_type,
        "width": args.width,
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(unbiased=True)),
        "min_cost": float(costs.min()),
        "max_cost": float(costs.max()),
        "seconds": elapsed,
        "instances_per_second": float(num_instances / elapsed),
        "instance_costs": [float(cost) for cost in costs.tolist()],
    }


_WORKER_LOCS = None


def _init_solver_worker(locs):
    """Initialize one ProcessPool worker with the shared instance array."""
    global _WORKER_LOCS
    _WORKER_LOCS = locs


def _solver_kwargs(name: str, args) -> dict:
    if name == "ortools":
        return {
            "time_limit": args.solver_time_limit,
            "seed": args.seed,
            "first_solution_strategy": args.ortools_first_solution,
            "local_search_metaheuristic": args.ortools_local_search,
        }
    if name == "lkh":
        return {
            "lkh_path": args.lkh_path,
            "time_limit": args.solver_time_limit,
            "runs": args.lkh_runs,
            "seed": args.seed,
        }
    raise ValueError(f"未知传统求解器: {name}")


def _solve_one(index: int, locs, name: str, kwargs: dict) -> dict:
    solver = solve_tsp_ortools if name == "ortools" else solve_tsp_lkh
    started = time.perf_counter()
    try:
        result = solver(locs, **kwargs)
    except Exception as exc:  # noqa: BLE001 - preserve solver failures in the report
        return {
            "index": int(index),
            "cost": None,
            "seconds": float(time.perf_counter() - started),
            "status": "exception",
            "error": str(exc),
        }

    cost = float(result.cost)
    if not math.isfinite(cost):
        cost = None
    return {
        "index": int(index),
        "cost": cost,
        "seconds": float(result.seconds),
        "status": result.status,
        "error": result.error,
    }


def _solve_worker(payload) -> dict:
    index, name, kwargs = payload
    return _solve_one(index, _WORKER_LOCS[index], name, kwargs)


def _validate_solver(name: str, args) -> None:
    """Fail before launching thousands of per-instance solver calls."""
    if name == "ortools":
        try:
            from ortools.constraint_solver import (  # noqa: F401
                pywrapcp,
                routing_enums_pb2,
            )
        except ImportError as exc:
            raise SystemExit(
                "未安装 OR-Tools；请先执行：pip install ortools"
            ) from exc
        return

    if name == "lkh":
        try:
            find_lkh_executable(args.lkh_path)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        return

    raise SystemExit(f"未知传统求解器: {name}")


def _fmt_optional(value, precision: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def _fmt_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class ProgressReporter:
    """Small time-based progress printer shared by neural/classical runs."""

    def __init__(self, name: str, total: int, interval: float):
        self.name = name
        self.total = int(total)
        self.interval = float(interval)
        self.started = time.perf_counter()
        self.last_print = self.started
        self.last_done = 0

    def update(self, done: int, force: bool = False) -> None:
        if force and done == self.last_done and done > 0:
            return
        now = time.perf_counter()
        if not force and (
            self.interval <= 0 or now - self.last_print < self.interval
        ):
            return
        elapsed = max(now - self.started, 1e-9)
        rate = done / elapsed
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        percent = 100.0 * done / self.total if self.total else 100.0
        print(
            f"[{self.name}] {done}/{self.total} ({percent:6.2f}%) "
            f"elapsed={_fmt_duration(elapsed)} "
            f"eta={_fmt_duration(eta)}",
            flush=True,
        )
        self.last_print = now
        self.last_done = done


def evaluate_classical_solver(name: str, td: TensorDict, args) -> dict:
    """Evaluate one classical TSP solver on the same fixed dataset."""
    _validate_solver(name, args)
    locs = td["locs"].detach().cpu().numpy()
    num_instances, num_loc, _ = locs.shape
    kwargs = _solver_kwargs(name, args)
    label = "OR-Tools" if name == "ortools" else "LKH3"
    print(
        f"[{label}] 开始求解 {num_instances} 个实例，"
        f"time_limit={args.solver_time_limit}s，"
        f"num_procs={args.solver_num_procs}",
        flush=True,
    )
    progress = ProgressReporter(label, num_instances, args.progress_interval)

    started = time.perf_counter()
    if args.solver_num_procs > 1:
        with ProcessPoolExecutor(
            max_workers=args.solver_num_procs,
            mp_context=mp.get_context("spawn"),
            initializer=_init_solver_worker,
            initargs=(locs,),
        ) as pool:
            records = []
            for record in pool.map(
                _solve_worker,
                [(index, name, kwargs) for index in range(num_instances)],
            ):
                records.append(record)
                progress.update(len(records))
    else:
        records = []
        for index in range(num_instances):
            records.append(_solve_one(index, locs[index], name, kwargs))
            progress.update(len(records))
    progress.update(num_instances, force=True)
    total_elapsed = time.perf_counter() - started

    successful = [row for row in records if row["cost"] is not None]
    costs = [row["cost"] for row in successful]
    status_counts: dict[str, int] = {}
    for row in records:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    mean_cost = statistics.fmean(costs) if costs else None
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0 if costs else None
    return {
        "kind": "classical",
        "model": label,
        "solver": name,
        "checkpoint": None,
        "num_samples": int(num_instances),
        "num_loc": int(num_loc),
        "successes": len(successful),
        "failures": int(num_instances - len(successful)),
        "status_counts": status_counts,
        "decode_type": "classical",
        "width": None,
        "mean_cost": float(mean_cost) if mean_cost is not None else None,
        "std_cost": float(std_cost) if std_cost is not None else None,
        "min_cost": float(min(costs)) if costs else None,
        "max_cost": float(max(costs)) if costs else None,
        "seconds": float(total_elapsed),
        "mean_solve_seconds": (
            float(statistics.fmean([row["seconds"] for row in successful]))
            if successful
            else None
        ),
        "instances_per_second": (
            float(num_instances / total_elapsed) if total_elapsed > 0 else None
        ),
        "solver_time_limit": args.solver_time_limit,
        "lkh_runs": args.lkh_runs if name == "lkh" else None,
        "instance_results": records,
    }


def _finalize_results(results: list[dict]) -> None:
    valid = [
        row for row in results if row.get("mean_cost") is not None and row["mean_cost"] > 0
    ]
    if not valid:
        return
    best_cost = min(row["mean_cost"] for row in valid)
    for row in results:
        if row.get("mean_cost") is None:
            row["gap_to_best_pct"] = None
        else:
            row["gap_to_best_pct"] = (
                (row["mean_cost"] - best_cost) / best_cost * 100.0
            )


def print_results(results: list[dict]) -> None:
    valid_costs = [
        row["mean_cost"] for row in results if row.get("mean_cost") is not None
    ]
    best_cost = min(valid_costs) if valid_costs else None
    header = (
        f"{'model':<16}{'mean_cost':>12}{'gap_to_best':>14}"
        f"{'std':>10}{'min':>10}{'max':>10}{'samples/s':>12}{'status':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        mean_cost = row.get("mean_cost")
        gap = (
            (mean_cost - best_cost) / best_cost * 100.0
            if mean_cost is not None and best_cost is not None
            else float("nan")
        )
        status = (
            "neural"
            if row.get("kind") == "neural"
            else (
                f"{row.get('successes', 0)}ok/"
                f"{row.get('failures', 0)}fail"
            )
        )
        print(
            f"{row['model']:<16}"
            f"{_fmt_optional(mean_cost):>12}"
            f"{gap:>13.3f}%"
            f"{_fmt_optional(row.get('std_cost')):>10}"
            f"{_fmt_optional(row.get('min_cost')):>10}"
            f"{_fmt_optional(row.get('max_cost')):>10}"
            f"{(row.get('instances_per_second') or 0.0):>11.1f}"
            f"{status:>14}"
        )
    print()
    for row in results:
        if row.get("kind") == "classical":
            print(
                f"{row['model']}: time_limit={row['solver_time_limit']}s "
                f"status={row['status_counts']} "
                f"mean_cost={_fmt_optional(row.get('mean_cost'))} "
                f"total={row['seconds']:.1f}s "
                f"per_instance={_fmt_optional(row.get('mean_solve_seconds'), 6)}s"
            )
            continue
        print(
            f"{row['model']}: {row['checkpoint']}  "
            f"mean={row['mean_cost']:.6f}  "
            f"decode={row['decode_type']} width={row['width']}  "
            f"took={row['seconds']:.1f}s"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage1：在同一固定 TSP npz 数据集上评估各路由模型"
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Stage1 generate_data.py 生成的 npz（字段: locs）",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="logs/stage1",
        help="自动扫描目录，默认取每个模型最新的 version 的 last.ckpt",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="扫描 checkpoint-root 时评估哪些模型；默认全部",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="只评估传统求解器，不扫描神经模型 checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="显式指定单个 checkpoint；与 --model 一起使用",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=None,
        help="显式评估的模型；路径中含 TSPEnv-<Model> 时可省略",
    )
    parser.add_argument("--all-versions", action="store_true")
    parser.add_argument(
        "--decode-type",
        choices=("greedy", "sampling"),
        default="greedy",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1,
        help="sampling 时的候选数；greedy 请保持 1",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2048,
        help="sampling 时单次并行展开的最大轨迹数（不含 POMO 内部多起点）",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto")
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径")
    parser.add_argument(
        "--check-solution",
        action="store_true",
        help="对每个生成路线做完整性校验，失败即报错",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=SUPPORTED_SOLVERS,
        default=[],
        help="额外的传统求解器 baseline：ortools、lkh",
    )
    parser.add_argument(
        "--solver-time-limit",
        type=float,
        default=0.1,
        help="每个 TSP 实例的求解时间上限（秒）",
    )
    parser.add_argument(
        "--solver-num-procs",
        type=int,
        default=1,
        help="传统求解器并行进程数；1 表示顺序执行",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="进度输出间隔（秒）；设为 0 可关闭周期性进度日志",
    )
    parser.add_argument(
        "--lkh-path",
        default=None,
        help="LKH-3 可执行文件路径；不指定时从 PATH 查找 LKH/LKH-3/lkh",
    )
    parser.add_argument(
        "--lkh-runs",
        type=int,
        default=1,
        help="LKH 对每个实例运行的次数",
    )
    parser.add_argument(
        "--ortools-first-solution",
        default="PATH_CHEAPEST_ARC",
        help="OR-Tools FirstSolutionStrategy",
    )
    parser.add_argument(
        "--ortools-local-search",
        default="GUIDED_LOCAL_SEARCH",
        help="OR-Tools LocalSearchMetaheuristic",
    )
    return parser


def _write_results(
    output: str | None,
    data_file: Path,
    td: TensorDict,
    args,
    results: list[dict],
    *,
    partial: bool,
    data_sha256: str,
) -> None:
    if not output:
        return
    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "partial": partial,
                "data_file": str(data_file),
                "data_sha256": data_sha256,
                "num_samples": int(td.batch_size[0]),
                "num_loc": int(td["locs"].shape[-2]),
                "seed": args.seed,
                "device": args.device,
                "solvers": args.solvers,
                "solver_time_limit": args.solver_time_limit,
                "lkh_runs": args.lkh_runs,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.width < 1:
        raise SystemExit("--width 必须 >= 1")
    if args.solver_time_limit < 0:
        raise SystemExit("--solver-time-limit 必须 >= 0")
    if args.solver_num_procs < 1:
        raise SystemExit("--solver-num-procs 必须 >= 1")
    if args.progress_interval < 0:
        raise SystemExit("--progress-interval 必须 >= 0")
    if args.lkh_runs < 1:
        raise SystemExit("--lkh-runs 必须 >= 1")
    if args.skip_neural and not args.solvers:
        raise SystemExit("--skip-neural 时至少需要指定一个 --solvers")
    if args.decode_type == "greedy" and args.width != 1:
        print("提示：greedy 解码下 --width 无意义，已按 1 处理", file=sys.stderr)
        args.width = 1

    data_file = Path(args.data_file).resolve()
    if not data_file.is_file():
        raise SystemExit(f"数据集不存在: {data_file}")
    td = load_npz_to_tensordict(data_file)
    if "locs" not in td.keys():
        raise SystemExit(f"{data_file} 中缺少 locs 字段，不是 Stage1 TSP 数据")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[eval] 数据={data_file}，实例数={td.batch_size[0]}，"
        f"节点数={td['locs'].shape[-2]}，device={args.device}",
        flush=True,
    )

    runs = [] if args.skip_neural else discover_runs(args)
    data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
    results = []

    for run in runs:
        results.append(evaluate_run(run, td, args))
        _finalize_results(results)
        _write_results(
            args.output,
            data_file,
            td,
            args,
            results,
            partial=True,
            data_sha256=data_sha256,
        )

    for solver_name in args.solvers:
        results.append(evaluate_classical_solver(solver_name, td, args))
        _finalize_results(results)
        _write_results(
            args.output,
            data_file,
            td,
            args,
            results,
            partial=True,
            data_sha256=data_sha256,
        )
    if not results:
        raise SystemExit("没有可评估的模型或传统求解器")

    _finalize_results(results)
    print_results(results)

    if args.output:
        _write_results(
            args.output,
            data_file,
            td,
            args,
            results,
            partial=False,
            data_sha256=data_sha256,
        )
        out_path = Path(args.output).resolve()
        print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
