import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_learning_rate_from_run_key(run_key: str) -> str:
    """
    Extract the learning rate substring from a run_key like:
    "experiment_runs/learning_rate_0.0001/network_depth_2/network_width_8"

    Returns the string form (e.g., "0.0001" or "1e-05") so that we preserve
    the original representation for labeling, while also enabling numeric
    sorting by converting to float when needed.
    """
    try:
        prefix = "learning_rate_"
        start = run_key.index(prefix) + len(prefix)
        end = run_key.index("/network_depth_", start)
        return run_key[start:end]
    except ValueError:
        return "unknown"


def _sort_unique(values: List[str]) -> List[str]:
    """Sort a list of numeric strings by their float value."""
    return sorted(
        values,
        key=lambda s: float(s) if s not in {"unknown", ""} else float("inf"),
    )


def load_results(json_path: str) -> pd.DataFrame:
    """
    Load results JSON into a DataFrame with columns:
    [learning_rate, depth, width, avg_success_rate,
    avg_average_episodic_reward]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    records: List[Dict] = []
    for row in data:
        run_key = row.get("run_key", "")
        lr_str = _parse_learning_rate_from_run_key(run_key)
        records.append(
            {
                "learning_rate_str": lr_str,
                "learning_rate": (
                    float(lr_str) if lr_str not in {"unknown", ""} else np.nan
                ),
                "depth": int(row["depth"]),
                "width": int(row["width"]),
                "avg_success_rate": float(row["avg_success_rate"]),
                "avg_average_episodic_reward": float(
                    row["avg_average_episodic_reward"]
                ),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


def _plot_heatmaps_for_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: str,
    cmap: str = "viridis",
    annotate_as_percent: bool = False,
) -> None:
    """
    Create a row of heatmaps (one per learning rate) with depth on Y and
    width on X.
    """
    learning_rates = _sort_unique(df["learning_rate_str"].unique().tolist())
    num_lrs = len(learning_rates)

    # Establish global vmin/vmax for consistent color scaling across subplots
    vmin = df[metric].min()
    vmax = df[metric].max()

    fig_width = max(4 * num_lrs, 6)
    fig, axes = plt.subplots(
        1,
        num_lrs,
        figsize=(fig_width, 5),
        squeeze=False,
        layout="constrained",
    )

    for idx, lr_str in enumerate(learning_rates):
        ax = axes[0, idx]
        sub = df[df["learning_rate_str"] == lr_str]

        # Pivot depth (rows) x width (cols)
        pivot = sub.pivot_table(
            index="depth", columns="width", values=metric, aggfunc="mean"
        )

        # Sort axes numerically
        pivot = pivot.sort_index(axis=0)  # depth
        pivot = pivot.sort_index(axis=1)  # width

        # Prepare data for imshow; handle NaNs with masked array
        data = pivot.values.astype(float)
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # Axis ticks and labels
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(int))
        ax.set_xlabel("width")

        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(int))
        ax.set_ylabel("depth")

        ax.set_title(f"lr={lr_str}")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iat[i, j]
                if pd.isna(val):
                    continue
                if annotate_as_percent:
                    text = f"{val * 100:.0f}%"
                else:
                    text = f"{val:.3f}"
                # import pdb

                # pdb.set_trace()
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=(
                        "black"
                        if (val - vmin) / (vmax - vmin + 1e-12) > 0.5
                        else "white"
                    ),
                    fontsize=9,
                )

    # One shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(metric)
    fig.suptitle(title)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_lines_vs_width(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: str,
) -> None:
    """
    For each learning rate, plot lines of metric vs width, one line per depth.
    Useful to see how capacity scaling behaves at fixed lr.
    """
    learning_rates = _sort_unique(df["learning_rate_str"].unique().tolist())
    num_lrs = len(learning_rates)

    fig_width = max(4 * num_lrs, 6)
    fig, axes = plt.subplots(
        1, num_lrs, figsize=(fig_width, 4), squeeze=False, layout="constrained"
    )

    for idx, lr_str in enumerate(learning_rates):
        ax = axes[0, idx]
        sub = df[df["learning_rate_str"] == lr_str]
        depths = sorted(sub["depth"].unique().tolist())

        for depth in depths:
            sub_depth = (
                sub[sub["depth"] == depth].sort_values("width").dropna(subset=[metric])
            )
            ax.plot(
                sub_depth["width"],
                sub_depth[metric],
                marker="o",
                label=f"depth={depth}",
            )

        ax.set_title(f"lr={lr_str}")
        ax.set_xlabel("width")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _compute_best_lr_success(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (depth, width), select the learning rate with the highest
    avg_success_rate from aggregated JSON results, and capture both
    success rate and average reward at that best LR.
    Returns columns:
      [depth, width, best_lr_str, best_success_rate, best_avg_reward]
    """
    idx = df.groupby(["depth", "width"])["avg_success_rate"].idxmax()
    best = (
        df.loc[
            idx,
            [
                "depth",
                "width",
                "learning_rate_str",
                "avg_success_rate",
                "avg_average_episodic_reward",
            ],
        ]
        .rename(
            columns={
                "learning_rate_str": "best_lr_str",
                "avg_success_rate": "best_success_rate",
                "avg_average_episodic_reward": "best_avg_reward",
            }
        )
        .sort_values(["depth", "width"])
    )  # type: ignore[call-arg]
    best = best.reset_index(drop=True)
    return best


# Removed seed-based computation; we select the best LR from JSON only.


def _plot_bars_capacity(
    df_vals: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    """
    Bar chart with x=capacity label "depthxwidth" and y=value_col.
    """
    if df_vals.empty:
        return
    df_vals = df_vals.copy().sort_values(["depth", "width"])
    labels = [f"{d}x{w}" for d, w in zip(df_vals["depth"], df_vals["width"])]
    values = df_vals[value_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 4))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_combined_lines_by_lr(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    """
    Combined line chart where each line is a (depth, width) capacity and the
    x-axis enumerates learning rates (sorted). y-axis is the given metric.
    """
    if df.empty:
        return

    # All learning rates sorted numerically, keep string labels for ticks
    lr_strs = _sort_unique(df["learning_rate_str"].unique().tolist())
    x = np.arange(len(lr_strs))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(lr_strs)), 5))

    capacities = (
        df[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )

    for depth_val, width_val in capacities:
        sub = df[(df["depth"] == depth_val) & (df["width"] == width_val)]
        sub_map = {
            lr: v
            for lr, v in zip(sub["learning_rate_str"].tolist(), sub[metric].tolist())
        }
        y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs], dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            alpha=0.9,
            label=f"{depth_val}x{width_val}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_strs, rotation=45, ha="right")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="depth x width", fontsize=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter results heatmaps and line plots"
    )
    parser.add_argument(
        "--json",
        default="hyperparam_results.json",
        help="Path to hyperparam_results.json",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("plots"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    df = load_results(args.json)

    # Heatmaps per learning rate
    _plot_heatmaps_for_metric(
        df=df,
        metric="avg_success_rate",
        title="Avg Success Rate (last-10) by depth x width per learning rate",
        output_path=os.path.join(args.outdir, "heatmaps_success_rate.png"),
        cmap="viridis",
        annotate_as_percent=True,
    )

    _plot_heatmaps_for_metric(
        df=df,
        metric="avg_average_episodic_reward",
        title=(
            "Avg Average Episodic Reward (last-10) by depth x width per "
            "learning rate"
        ),
        output_path=os.path.join(args.outdir, "heatmaps_avg_reward.png"),
        cmap="magma",
        annotate_as_percent=False,
    )

    # Lines vs width per depth for each LR
    _plot_lines_vs_width(
        df=df,
        metric="avg_success_rate",
        title="Success Rate vs Width (one line per depth) per learning rate",
        output_path=os.path.join(
            args.outdir,
            "lines_success_rate_vs_width.png",
        ),
    )

    _plot_lines_vs_width(
        df=df,
        metric="avg_average_episodic_reward",
        title="Avg Reward vs Width (one line per depth) per learning rate",
        output_path=os.path.join(
            args.outdir,
            "lines_avg_reward_vs_width.png",
        ),
    )

    # New: bar charts by capacity (best LR chosen by success rate)
    best_lr_success = _compute_best_lr_success(df)
    _plot_bars_capacity(
        df_vals=best_lr_success[
            [
                "depth",
                "width",
                "best_success_rate",
            ]
        ],
        value_col="best_success_rate",
        title="Best LR Success Rate by Capacity (depth x width)",
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "best_lr_success_by_capacity.png",
        ),
    )
    _plot_bars_capacity(
        df_vals=best_lr_success[
            [
                "depth",
                "width",
                "best_avg_reward",
            ]
        ],
        value_col="best_avg_reward",
        title="Avg Reward at Best LR by Capacity (depth x width)",
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "best_lr_avg_reward_by_capacity.png",
        ),
    )

    # New: combined lines by learning rate (one line per capacity)
    _plot_combined_lines_by_lr(
        df=df,
        metric="avg_success_rate",
        title=("Success Rate across Learning Rates (one line per depth x width)"),
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_success_rate_by_lr.png",
        ),
    )
    _plot_combined_lines_by_lr(
        df=df,
        metric="avg_average_episodic_reward",
        title=("Avg Reward across Learning Rates (one line per depth x width)"),
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_avg_reward_by_lr.png",
        ),
    )


if __name__ == "__main__":
    main()
