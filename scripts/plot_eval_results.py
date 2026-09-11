"""Plot Stage1 evaluation JSON files produced by ``eval_stage1.py``.

Examples:

    python scripts/plot_eval_results.py \
        assets/results/stage1_tsp20_compare.json \
        -o assets/images/stage1_tsp20_eval.png

    # Compare multiple JSON files, and also save the per-instance distribution
    python scripts/plot_eval_results.py \
        assets/results/am.json assets/results/pomo.json \
        --labels AM POMO \
        --distributions \
        -o assets/images/am_pomo_eval.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("assets/images/stage1_eval.png")


def _fmt(value, digits=4):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _extract_instance_costs(row):
    if row.get("instance_costs"):
        return [float(x) for x in row["instance_costs"] if x is not None]
    instance_results = row.get("instance_results") or []
    return [
        float(item["cost"])
        for item in instance_results
        if item.get("cost") is not None
    ]


def _load_eval_files(paths, labels=None):
    if labels is None:
        labels = [None] * len(paths)
    if len(labels) != len(paths):
        raise SystemExit(
            f"--labels 数量 ({len(labels)}) 与 JSON 文件数量 ({len(paths)}) 不一致"
        )

    runs = []
    for path, file_label in zip(paths, labels):
        path = Path(path)
        data = json.loads(path.read_text())
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise SystemExit(f"{path} 中没有 results 数据")
        prefix = file_label or (path.stem if len(paths) > 1 else "")
        if data.get("partial"):
            print(f"警告: {path} 是 partial 结果，图中的数据可能不完整")

        for row in results:
            method = row.get("model") or row.get("solver") or "unknown"
            label = f"{prefix}:{method}" if prefix else method
            runs.append(
                {
                    "label": label,
                    "method": method,
                    "kind": row.get("kind", "unknown"),
                    "mean_cost": row.get("mean_cost"),
                    "std_cost": row.get("std_cost"),
                    "min_cost": row.get("min_cost"),
                    "max_cost": row.get("max_cost"),
                    "gap": row.get("gap_to_best_pct"),
                    "throughput": row.get("instances_per_second"),
                    "total_seconds": row.get("seconds"),
                    "mean_solve_seconds": row.get("mean_solve_seconds"),
                    "instance_costs": _extract_instance_costs(row),
                }
            )
    return runs


def _add_value_labels(ax, bars, values, *, digits=4):
    for bar, value in zip(bars, values):
        if value is None or not isinstance(value, (int, float)):
            text = "n/a"
        elif not math.isfinite(value):
            text = "n/a"
        else:
            text = _fmt(value, digits)
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _bar_panel(ax, runs, key, title, ylabel, *, log=False, digits=4):
    labels = [run["label"] for run in runs]
    values = [run[key] for run in runs]
    colors = plt.cm.tab10(range(len(runs)))
    bars = ax.bar(labels, [0 if value is None else value for value in values],
                  color=colors, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    if log:
        ax.set_yscale("log")
    _add_value_labels(ax, bars, values, digits=digits)
    ax.grid(axis="y", alpha=0.25)


def _plot_box_distributions(runs, output, title, dpi, annotate=True):
    rows = [run for run in runs if run["instance_costs"]]
    if not rows:
        print("没有 instance_costs/instance_results，跳过分布图")
        return None
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(rows)), 5))
    try:
        box_result = ax.boxplot(
            [run["instance_costs"] for run in rows],
            tick_labels=[run["label"] for run in rows],
            showfliers=False,
            whis=(5, 95),
        )
    except TypeError:  # Matplotlib < 3.9
        box_result = ax.boxplot(
            [run["instance_costs"] for run in rows],
            labels=[run["label"] for run in rows],
            showfliers=False,
            whis=(5, 95),
        )
    if annotate:
        p5s, p95s = [], []
        for idx, run in enumerate(rows, start=1):
            values = np.asarray(run["instance_costs"], dtype=float)
            mean = float(values.mean())
            median = float(np.median(values))
            p5 = float(np.percentile(values, 5))
            p95 = float(np.percentile(values, 95))
            p5s.append(p5)
            p95s.append(p95)
            ax.annotate(
                f"mean={mean:.3f}\np95={p95:.3f}",
                xy=(idx, p95),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75),
            )
            ax.annotate(
                f"p5={p5:.3f}\np50={median:.3f}",
                xy=(idx, p5),
                xytext=(0, -6),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75),
            )
        if p5s and p95s:
            margin = max(0.05, 0.12 * (max(p95s) - min(p5s)))
            ax.set_ylim(min(p5s) - margin, max(p95s) + margin)
    ax.set_title(f"{title} — per-instance cost distribution")
    ax.set_ylabel("TSP tour length")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.25)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot evaluation JSON produced by scripts.stage1.eval_stage1"
    )
    parser.add_argument("paths", nargs="+", metavar="JSON")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="每个 JSON 文件的标签前缀；数量需与 JSON 文件数一致",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"主图输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument("--title", default="Stage1 evaluation comparison")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--distributions",
        action="store_true",
        help="额外保存逐实例成本分布箱线图",
    )
    parser.add_argument(
        "--distribution-output",
        default=None,
        help="逐实例分布图输出路径；默认在主图文件名后加 _distribution",
    )
    parser.add_argument(
        "--no-distribution-labels",
        action="store_true",
        help="不在分布图上标注 mean/p5/p50/p95",
    )
    parser.add_argument(
        "--no-log-throughput",
        action="store_true",
        help="吞吐量面板使用线性坐标而不是对数坐标",
    )
    return parser


def main():
    args = build_parser().parse_args()
    runs = _load_eval_files(args.paths, args.labels)
    if not runs:
        raise SystemExit("没有可绘制的数据")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Mean cost with standard-deviation error bars.
    labels = [run["label"] for run in runs]
    means = [0 if run["mean_cost"] is None else run["mean_cost"] for run in runs]
    stds = [0 if run["std_cost"] is None else run["std_cost"] for run in runs]
    colors = plt.cm.tab10(range(len(runs)))
    bars = axes[0, 0].bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                          capsize=4)
    axes[0, 0].set_title("Mean TSP cost (lower is better)")
    axes[0, 0].set_ylabel("tour length")
    _add_value_labels(axes[0, 0], bars, [run["mean_cost"] for run in runs])
    axes[0, 0].grid(axis="y", alpha=0.25)

    valid_means = [run["mean_cost"] for run in runs if run["mean_cost"] is not None]
    best = min(valid_means) if valid_means else None
    for run in runs:
        gap = run["gap"]
        if gap is None and run["mean_cost"] is not None and best:
            gap = (run["mean_cost"] - best) / best * 100.0
        run["gap"] = gap
    _bar_panel(
        axes[0, 1],
        runs,
        "gap",
        "Gap to best mean cost (lower is better)",
        "%",
        digits=3,
    )

    _bar_panel(
        axes[1, 0],
        runs,
        "throughput",
        "Throughput (higher is better)",
        "instances / second",
        log=not args.no_log_throughput,
        digits=2,
    )
    _bar_panel(
        axes[1, 1],
        runs,
        "total_seconds",
        "Total wall time (log scale)",
        "seconds",
        log=True,
        digits=2,
    )

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output = Path(args.output) if args.output else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output.resolve()}")

    if args.distributions:
        dist_output = (
            Path(args.distribution_output)
            if args.distribution_output
            else output.with_name(output.stem + "_distribution" + output.suffix)
        )
        path = _plot_box_distributions(
            runs,
            dist_output,
            args.title,
            args.dpi,
            annotate=not args.no_distribution_labels,
        )
        if path is not None:
            print(f"已保存: {path.resolve()}")


if __name__ == "__main__":
    main()
