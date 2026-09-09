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
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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


SUPPORTED_MODELS = ("AttentionModel", "POMO", "MDAM", "PointerNetwork")
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
            reward = reward.reshape(
                num_instances, current_width, num_starts
            ).amax(dim=(1, 2))

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

    rewards = torch.cat(rewards)
    costs = -rewards  # TSPEnv reward = -tour length
    elapsed = time.perf_counter() - started
    return {
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
    }


def print_results(results: list[dict]) -> None:
    best_cost = min(row["mean_cost"] for row in results)
    header = (
        f"{'model':<16}{'mean_cost':>12}{'gap_to_best':>14}"
        f"{'std':>10}{'min':>10}{'max':>10}{'samples/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        gap = (row["mean_cost"] - best_cost) / best_cost * 100.0
        print(
            f"{row['model']:<16}"
            f"{row['mean_cost']:>12.6f}"
            f"{gap:>13.3f}%"
            f"{row['std_cost']:>10.6f}"
            f"{row['min_cost']:>10.6f}"
            f"{row['max_cost']:>10.6f}"
            f"{row['instances_per_second']:>11.1f}"
        )
    print()
    for row in results:
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.width < 1:
        raise SystemExit("--width 必须 >= 1")
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

    runs = discover_runs(args)
    results = [evaluate_run(run, td, args) for run in runs]
    print_results(results)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "data_file": str(data_file),
                    "device": args.device,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
