"""Plot RL4CO ``metrics.csv`` training curves from one or more runs.

Usage examples (project root):

    # Plot a single model
    python scripts/plot_metrics.py \\
        logs/stage1/TSPEnv-AttentionModel/version_0/metrics.csv \\
        -o assets/images/attention_model.png

    # Compare several models on one figure
    python scripts/plot_metrics.py \\
        logs/stage1/TSPEnv-AttentionModel/version_0/metrics.csv \\
        logs/stage1/TSPEnv-POMO/version_0/metrics.csv \\
        --labels AM POMO \\
        -o assets/images/stage1_compare.png

    # Pass a directory instead of a file; all metrics.csv under it are collected
    python scripts/plot_metrics.py logs/stage1/TSPEnv-AttentionModel \\
        -o assets/images/am.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT = Path("assets/images/training_curves.png")


def _num(value):
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_column(fieldnames, preferred):
    """Match a CSV column flexibly, e.g. ``train/reward`` or ``train_reward``."""
    cleaned = [str(name).strip().replace(" ", "_") for name in fieldnames]
    for candidate in preferred:
        exact = candidate.replace(" ", "_")
        if exact in cleaned:
            return fieldnames[cleaned.index(exact)]
    for name in fieldnames:
        norm = str(name).strip().lower().replace(" ", "_")
        for candidate in preferred:
            cand = candidate.lower().replace(" ", "_")
            if norm == cand or norm.endswith("/" + cand) or norm.endswith("_" + cand):
                return name
    return None


def _discover(path: Path):
    """Return a list of metrics.csv paths under ``path`` (file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("metrics.csv") if p.is_file())
    return []


def _default_label(
    metrics_path: Path, input_path: Path, multiple_in_input: bool = False
) -> str:
    parent = metrics_path.parent
    # logs/stage1/<Model>/version_0/metrics.csv -> <Model>
    if (
        parent.name.lower().startswith("version")
        and parent.parent.name
        and not (multiple_in_input and input_path.is_dir())
    ):
        return parent.parent.name
    if multiple_in_input and input_path.is_dir():
        try:
            rel = metrics_path.relative_to(input_path)
            parts = list(rel.parts[:-1])
            return "/".join(parts) if parts else input_path.name
        except ValueError:
            pass
    if metrics_path.parent == input_path:
        return input_path.name
    try:
        rel = metrics_path.relative_to(input_path)
        parts = list(rel.parts[:-1])
        if parts and parts[-1].lower().startswith("version"):
            parts = parts[:-1]
        return "/".join(parts) if parts else input_path.name
    except ValueError:
        return parent.name


def _read_metrics(path: Path):
    """Read an RL4CO-style metrics.csv and return train/val row lists."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        epoch_col = _find_column(fieldnames, ["epoch"])
        step_col = _find_column(fieldnames, ["step"])
        loss_col = _find_column(fieldnames, ["train/loss", "loss"])
        train_reward_col = _find_column(
            fieldnames, ["train/reward", "reward", "train_reward"]
        )
        val_reward_col = _find_column(
            fieldnames, ["val/reward", "validation/reward", "val_reward"]
        )

        if epoch_col is None and step_col is None:
            raise ValueError(f"{path}: metrics.csv 缺少 epoch/step 列")

        train_rows, val_rows = [], []
        for raw in reader:
            epoch = _num(raw.get(epoch_col)) if epoch_col else None
            step = _num(raw.get(step_col)) if step_col else None
            loss = _num(raw.get(loss_col)) if loss_col else None
            reward = _num(raw.get(train_reward_col)) if train_reward_col else None
            val_reward = _num(raw.get(val_reward_col)) if val_reward_col else None

            if loss is not None or reward is not None:
                train_rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss,
                        "reward": reward,
                    }
                )
            if val_reward is not None:
                val_rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "reward": val_reward,
                    }
                )

    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "has_loss": loss_col is not None,
        "has_val": val_reward_col is not None,
        "columns": fieldnames,
    }


def _aggregate_by_epoch(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        epoch = row.get("epoch")
        value = row.get(key)
        if epoch is None or value is None:
            continue
        grouped[round(epoch, 6)].append(value)
    epochs = sorted(grouped)
    return epochs, [sum(grouped[e]) / len(grouped[e]) for e in epochs]


def _x_and_y(train_rows, val_rows, x_axis):
    """Return train/val curves ready for plotting."""
    if x_axis == "epoch":
        train_epochs, train_loss = _aggregate_by_epoch(train_rows, "loss")
        _, train_reward = _aggregate_by_epoch(train_rows, "reward")
        val_epochs, val_reward = _aggregate_by_epoch(val_rows, "reward")
        train_x = train_epochs
        val_x = val_epochs
    else:
        train_x = [r["step"] for r in train_rows]
        train_loss = [r["loss"] for r in train_rows]
        train_reward = [r["reward"] for r in train_rows]
        val_x = [r["step"] for r in val_rows]
        val_reward = [r["reward"] for r in val_rows]
    return train_x, train_loss, train_reward, val_x, val_reward


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot RL4CO training curves from metrics.csv files and save the "
            "figure to assets/images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help=(
            "metrics.csv 文件路径或目录；目录会递归收集其中的 metrics.csv。"
            "传入多个即可同图比较"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="每条曲线的图例名称，数量需与收集到的 metrics.csv 数量一致",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "输出图片路径（默认 assets/images/training_curves.png）；"
            "只给目录时会自动写入该目录"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="整张图标题；默认只显示图例，不写标题",
    )
    parser.add_argument(
        "--x-axis",
        choices=("epoch", "step"),
        default="epoch",
        help="横轴：epoch 会对每个 epoch 内的多条记录取均值；step 保留原始记录",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="输出图片分辨率",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(10.0, 7.0),
        metavar=("W", "H"),
        help="图片尺寸，单位 inch",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)

    metrics_paths = []
    for raw in args.paths:
        path = Path(raw)
        found = _discover(path)
        if not found:
            raise SystemExit(f"未找到 metrics.csv: {raw}")
        # Record whether this input expands to more than one metrics.csv.
        metrics_paths.extend(
            [(item, path, len(found) > 1) for item in found]
        )

    if args.labels is not None:
        if len(args.labels) != len(metrics_paths):
            raise SystemExit(
                f"--labels 数量 ({len(args.labels)}) 与 metrics.csv 数量 "
                f"({len(metrics_paths)}) 不一致"
            )
        labels = list(args.labels)
    else:
        labels = [
            _default_label(item, root, multiple)
            for item, root, multiple in metrics_paths
        ]

    runs = []
    for (path, _, _), label in zip(metrics_paths, labels):
        data = _read_metrics(path)
        if not data["train_rows"]:
            print(f"警告: {path} 中没有 train 记录，已跳过", file=sys.stderr)
            continue
        train_x, train_loss, train_reward, val_x, val_reward = _x_and_y(
            data["train_rows"], data["val_rows"], args.x_axis
        )
        runs.append(
            {
                "label": label,
                "path": path,
                "train_x": train_x,
                "train_loss": train_loss,
                "train_reward": train_reward,
                "val_x": val_x,
                "val_reward": val_reward,
                "has_loss": data["has_loss"],
            }
        )

    if not runs:
        raise SystemExit("没有可绘制的 metrics.csv 数据")

    has_any_loss = any(run["has_loss"] for run in runs)
    n_rows = 2 if has_any_loss else 1
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=args.figsize,
        sharex=False,
        squeeze=False,
    )
    color_cycle = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    if has_any_loss:
        ax_loss = axes[0][0]
        ax_reward = axes[1][0]
    else:
        ax_loss = None
        ax_reward = axes[0][0]

    for run in runs:
        color = next(color_cycle)
        if ax_loss is not None and run["has_loss"]:
            ax_loss.plot(
                run["train_x"],
                run["train_loss"],
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=run["label"],
            )

        ax_reward.plot(
            run["train_x"],
            run["train_reward"],
            color=color,
            linewidth=1.4,
            alpha=0.65,
            label=f"{run['label']} (train reward)",
        )
        if run["val_reward"]:
            ax_reward.plot(
                run["val_x"],
                run["val_reward"],
                color=color,
                linewidth=1.2,
                marker="o",
                markersize=3,
                alpha=0.9,
                label=f"{run['label']} (val reward)",
            )

    if ax_loss is not None:
        ax_loss.set_ylabel("train/loss")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(fontsize=9, ncol=2)
    ax_reward.set_ylabel("reward (higher is better)")
    ax_reward.set_xlabel("epoch" if args.x_axis == "epoch" else "global step")
    ax_reward.grid(alpha=0.3)
    ax_reward.legend(fontsize=9, ncol=2)

    if args.title:
        fig.suptitle(args.title, fontsize=13)

    out_path = Path(args.output) if args.output else DEFAULT_OUT
    if out_path.suffix.lower() not in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
        out_path = out_path / DEFAULT_OUT.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
