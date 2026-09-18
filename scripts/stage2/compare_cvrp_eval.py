"""把多份 ``eval_cvrp`` 结果 JSON 合并成对比表 / 对比图。

用法（项目根目录）：

    python -m scripts.stage2.compare_cvrp_eval \
        assets/results/cvrp100_model_vs_lkh.json \
        assets/results/cvrp100_lkh_minmax.json \
        --reference lkh-minmax \
        --table assets/results/cvrp100_compare_table.csv \
        -o assets/images/cvrp100_model_vs_lkh.png

指标优先级：``lkh_*``（精确重算）> ``max_route/total``（求解器原生）>
``am_*``（冻结 Stage1 模型估计）。这样模型分簇和传统求解器可以在同一口径上比较。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _metric(record: dict, names: tuple[str, ...]) -> float:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    return float("nan")


def load_methods(paths: list[str]) -> dict[str, dict]:
    methods: dict[str, dict] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, method in payload.get("methods", {}).items():
            records = [
                r for r in method.get("instances", []) if r.get("status") == "success"
            ]
            if not records:
                continue
            if name in methods:
                name = f"{name}@{Path(path).stem}"
            methods[name] = {
                "records": records,
                "num_instances": len(method.get("instances", [])),
                "source": str(path),
                "partial": bool(payload.get("partial")),
            }
    return methods


def summarize(name: str, entry: dict) -> dict:
    records = entry["records"]
    max_route = [_metric(r, ("lkh_max_route", "max_route", "am_max_route")) for r in records]
    total = [_metric(r, ("lkh_total", "total", "am_total")) for r in records]
    seconds = [float(r.get("seconds", float("nan"))) for r in records]
    overflow = [float(r.get("overflow", 0.0)) for r in records]
    return {
        "method": name,
        "n_ok": len(records),
        "n_total": entry["num_instances"],
        "max_route_mean": st.fmean(max_route),
        "max_route_median": st.median(max_route),
        "max_route_p95": float(np.percentile(max_route, 95)),
        "total_mean": st.fmean(total),
        "seconds_mean": st.fmean(seconds),
        "overflow_mean": st.fmean(overflow),
        "overflow_instances": sum(1 for value in overflow if value > 1e-9),
    }


def gap_to_reference(entry: dict, reference: dict) -> dict[str, float]:
    ref = {
        int(r["index"]): _metric(r, ("lkh_max_route", "max_route", "am_max_route"))
        for r in reference["records"]
    }
    gaps: list[float] = []
    for record in entry["records"]:
        index = int(record["index"])
        if index not in ref:
            continue
        value = _metric(record, ("lkh_max_route", "max_route", "am_max_route"))
        if ref[index] > 0 and not math.isnan(value):
            gaps.append(100.0 * (value - ref[index]) / ref[index])
    if not gaps:
        return {}
    return {
        "gap_mean_pct": float(st.fmean(gaps)),
        "gap_median_pct": float(st.median(gaps)),
        "gap_worse_pct": float(100.0 * sum(1 for g in gaps if g > 0) / len(gaps)),
        "gap_n": len(gaps),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 eval_cvrp 结果 JSON，输出对比表与对比图"
    )
    parser.add_argument("paths", nargs="+", help="一个或多个 eval_cvrp 结果 JSON")
    parser.add_argument(
        "--reference",
        default=None,
        help="作为 100%% 基准的方法名（例如 lkh-minmax），用于计算相对 gap",
    )
    parser.add_argument("--table", default=None, help="输出 CSV 表格路径")
    parser.add_argument("-o", "--output", default=None, help="输出图片路径")
    parser.add_argument(
        "--plot-methods",
        nargs="*",
        default=None,
        help="只画这些方法（默认全部，最多 8 个）",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.0, 5.5))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = load_methods(args.paths)
    if not methods:
        raise SystemExit("没有可用结果（检查 JSON 是否包含 methods）")

    rows = []
    for name, entry in methods.items():
        row = summarize(name, entry)
        if args.reference and args.reference in methods and args.reference != name:
            row.update(gap_to_reference(entry, methods[args.reference]))
        rows.append(row)
    rows.sort(key=lambda row: row["max_route_mean"])

    header = (
        f"{'method':<22}{'n_ok':>8}{'max_route':>11}{'p95':>8}"
        f"{'total':>9}{'sec/inst':>10}{'overflow':>10}"
    )
    if args.reference:
        header += f"{'gap%':>9}{'worse%':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (
            f"{row['method']:<22}{row['n_ok']:>4}/{row['n_total']:<3}"
            f"{row['max_route_mean']:>11.4f}{row['max_route_p95']:>8.3f}"
            f"{row['total_mean']:>9.3f}{row['seconds_mean']:>10.4f}"
            f"{row['overflow_mean']:>10.3f}"
        )
        if args.reference:
            gap = row.get("gap_mean_pct")
            worse = row.get("gap_worse_pct")
            line += f"{gap:>9.2f}" if gap is not None else f"{'-':>9}"
            line += f"{worse:>8.1f}" if worse is not None else f"{'-':>8}"
        print(line)

    if args.table:
        table_path = Path(args.table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n表格已写入 {table_path}")

    if args.output:
        names = args.plot_methods or [row["method"] for row in rows][:8]
        names = [name for name in names if name in methods]
        fig, (ax_bar, ax_cdf) = plt.subplots(
            1, 2, figsize=tuple(args.figsize), gridspec_kw={"width_ratios": [1.0, 1.15]}
        )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for position, name in enumerate(names):
            records = methods[name]["records"]
            values = [
                _metric(r, ("lkh_max_route", "max_route", "am_max_route"))
                for r in records
            ]
            color = colors[position % len(colors)]
            ax_bar.bar(
                position,
                st.fmean(values),
                yerr=float(np.percentile(values, 95)) - st.fmean(values),
                color=color,
                alpha=0.85,
                capsize=4,
            )
            ax_bar.text(
                position,
                st.fmean(values),
                f"{st.fmean(values):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ordered = np.sort(np.asarray(values))
            cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
            ax_cdf.plot(ordered, cumulative, color=color, linewidth=1.6, label=name)

        ax_bar.set_xticks(range(len(names)))
        ax_bar.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax_bar.set_ylabel("max route length (mean, p95 whisker)")
        ax_bar.grid(alpha=0.3, axis="y")
        ax_bar.set_title("Mean longest route (lower is better)")

        ax_cdf.set_xlabel("max route length")
        ax_cdf.set_ylabel("fraction of instances")
        ax_cdf.grid(alpha=0.3)
        ax_cdf.legend(fontsize=9)
        ax_cdf.set_title("Per-instance distribution")

        figure_path = Path(args.output)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=args.dpi)
        print(f"图片已保存: {figure_path}")


if __name__ == "__main__":
    main()
