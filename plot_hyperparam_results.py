import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

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


def _parse_optimal_path_from_run_key(run_key: str) -> bool:
    """
    Determine whether the run was performed with optimal path generation.

    We infer this from substrings in the run_key, e.g.:
    ".../generate_optimal_path_True/..." or
    ".../generate_optimal_path_False/..."
    If neither is present, default to False.
    """
    if "generate_optimal_path_True" in run_key:
        return True
    if "generate_optimal_path_False" in run_key:
        return False
    return False


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
        optimal_path = _parse_optimal_path_from_run_key(run_key)
        records.append(
            {
                "learning_rate_str": lr_str,
                "learning_rate": (
                    float(lr_str) if lr_str not in {"unknown", ""} else np.nan
                ),
                "optimal_path": optimal_path,
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


def _plot_grouped_bars_path_vs_pathless_by_capacity(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
    agg: str = "max",  # y-value per bar: "max" or "mean" across LRs
    show_error_bars: bool = True,  # toggle error bars
    capsize: float = 3.0,  # error bar cap size
    path_label: str = "Path",
    pathless_label: str = "No Path",
    path_color: str = "#4C78A8",
    pathless_color: str = "#F58518",
) -> None:
    if df.empty:
        return
    if agg not in {"max", "mean"}:
        raise ValueError("agg must be 'max' or 'mean'")

    # Per (depth,width,optimal_path) stats across learning rates
    stats = (
        df.groupby(["depth", "width", "optimal_path"])[metric]
        .agg(mean=np.mean, std=np.std, count="count", max=np.max)
        .reset_index()
        .sort_values(["depth", "width", "optimal_path"])  # type: ignore[call-arg]
    )

    # Consistent capacity ordering
    capacities = (
        stats[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )
    capacities = list(capacities)
    labels = [f"{d}x{w}" for d, w in capacities]

    value_col = "max" if agg == "max" else "mean"
    value_pivot = stats.pivot_table(
        index=["depth", "width"],
        columns="optimal_path",
        values=value_col,
        aggfunc="first",
    )
    std_pivot = (
        stats.pivot_table(
            index=["depth", "width"],
            columns="optimal_path",
            values="std",
            aggfunc="first",
        )
        if show_error_bars
        else None
    )

    def _get(pvt: pd.DataFrame | None, d: int, w: int, opt: bool) -> float:
        if pvt is None or (d, w) not in pvt.index or opt not in pvt.columns:
            return float("nan")
        val = pvt.loc[(d, w), opt]
        return float(val) if pd.notna(val) else float("nan")

    # Bar heights
    y_pathless = [_get(value_pivot, d, w, False) for d, w in capacities]
    y_path = [_get(value_pivot, d, w, True) for d, w in capacities]

    # Error bars (std across LRs)
    yerr_pathless = (
        [_get(std_pivot, d, w, False) for d, w in capacities]
        if show_error_bars
        else None
    )
    yerr_path = (
        [_get(std_pivot, d, w, True) for d, w in capacities]
        if show_error_bars
        else None
    )

    # Plot
    fig_width = max(8, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    x = np.arange(len(labels))
    bar_width = 0.38

    # No Path (left), Path (right)
    ax.bar(
        x - bar_width / 2,
        y_pathless,
        width=bar_width,
        color=pathless_color,
        label=pathless_label,
        yerr=yerr_pathless,
        capsize=capsize if show_error_bars else 0.0,
        error_kw=(
            dict(ecolor="black", elinewidth=1, alpha=0.7) if show_error_bars else None
        ),
    )
    ax.bar(
        x + bar_width / 2,
        y_path,
        width=bar_width,
        color=path_color,
        label=path_label,
        yerr=yerr_path,
        capsize=capsize if show_error_bars else 0.0,
        error_kw=(
            dict(ecolor="black", elinewidth=1, alpha=0.7) if show_error_bars else None
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

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

    base_width = max(8, 0.6 * len(lr_strs))
    fig_width = base_width + 3.0  # reserve space on the right for legend
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Build a stable color mapping per (depth, width) capacity
    pairs = (
        df[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )
    pairs = list(pairs)
    num_pairs = max(1, len(pairs))
    cmap = plt.get_cmap("tab20")
    pair_to_color: Dict[tuple, tuple] = {
        pair: cmap(idx / max(1, num_pairs - 1)) for idx, pair in enumerate(pairs)
    }

    # Each line corresponds to a unique (depth, width, optimal_path) triple
    capacities = (
        df[["depth", "width", "optimal_path"]]
        .drop_duplicates()
        .sort_values(["optimal_path", "depth", "width"])  # type: ignore
        .itertuples(index=False, name=None)
    )

    for depth_val, width_val, optimal_path_val in capacities:
        sub = df[
            (df["depth"] == depth_val)
            & (df["width"] == width_val)
            & (df["optimal_path"] == optimal_path_val)
        ]
        sub_map = {
            lr: v
            for lr, v in zip(
                sub["learning_rate_str"].tolist(),
                sub[metric].tolist(),
            )
        }
        y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs], dtype=float)
        # Style by optimal_path for clear differentiation
        linestyle = "-" if optimal_path_val else "--"
        marker = "o" if optimal_path_val else "x"
        color = pair_to_color[(depth_val, width_val)]

        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1.8,
            alpha=0.9,
            linestyle=linestyle,
            color=color,
            label=(
                f"{depth_val}x{width_val}"
                + (" | optimal_path" if optimal_path_val else "")
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_strs, rotation=45, ha="right")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="depth x width | optimal_path",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_pair_color_map(
    df_all: pd.DataFrame,
    cmap_name: str = "tab20",
) -> Dict[Tuple[int, int], tuple]:
    """Create a stable color map for (depth,width) pairs across plots."""
    pairs_iter = (
        df_all[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )
    pairs = list(pairs_iter)
    num_pairs = max(1, len(pairs))
    cmap = plt.get_cmap(cmap_name)
    pair_to_color: Dict[Tuple[int, int], tuple] = {
        pair: cmap(idx / max(1, num_pairs - 1)) for idx, pair in enumerate(pairs)
    }
    return pair_to_color


def _plot_combined_lines_by_lr_filtered(
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
    pair_to_color: Optional[Dict[Tuple[int, int], tuple]] = None,
) -> None:
    """Plot combined lines for a filtered subset (e.g., only optimal_path).

    Colors are assigned by (depth,width) using a shared mapping so that
    colors match across related plots.
    """
    if df_filtered.empty:
        return

    lr_strs = _sort_unique(df_filtered["learning_rate_str"].unique().tolist())
    x = np.arange(len(lr_strs))

    base_width = max(8, 0.6 * len(lr_strs))
    fig_width = base_width + 3.0
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    if pair_to_color is None:
        pair_to_color = _build_pair_color_map(df_all)

    capacities = (
        df_filtered[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )

    for depth_val, width_val in capacities:
        sub = df_filtered[
            (df_filtered["depth"] == depth_val) & (df_filtered["width"] == width_val)
        ]
        sub_map = {
            lr: v
            for lr, v in zip(
                sub["learning_rate_str"].tolist(),
                sub[metric].tolist(),
            )
        }
        y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs], dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            alpha=0.9,
            linestyle="-",
            color=pair_to_color[(depth_val, width_val)],
            label=f"{depth_val}x{width_val}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_strs, rotation=45, ha="right")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="depth x width",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_combined_lines_on_ax(
    ax,
    df_filtered: pd.DataFrame,
    metric: str,
    lr_strs: List[str],
    pair_to_color: Dict[Tuple[int, int], tuple],
    ylabel: str,
    title: str,
) -> None:
    """Draw the combined-lines chart on a provided Axes.

    - Colors are assigned by (depth,width) using pair_to_color
    - Solid lines with circle markers are used
    """
    x = np.arange(len(lr_strs))

    capacities = (
        df_filtered[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore[call-arg]
        .itertuples(index=False, name=None)
    )

    for depth_val, width_val in capacities:
        sub = df_filtered[
            (df_filtered["depth"] == depth_val) & (df_filtered["width"] == width_val)
        ]
        sub_map = {
            lr: v
            for lr, v in zip(
                sub["learning_rate_str"].tolist(),
                sub[metric].tolist(),
            )
        }
        y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs], dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            alpha=0.9,
            linestyle="-",
            color=pair_to_color[(depth_val, width_val)],
            label=f"{depth_val}x{width_val}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_strs, rotation=45, ha="right")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_side_by_side_path_vs_pathless(
    df_all: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    """
    Side-by-side plots: left=pathless, right=optimal_path, shared colors and
    legend.
    """
    if df_all.empty:
        return

    df_path = df_all[df_all["optimal_path"] == True]  # noqa: E712
    df_pathless = df_all[df_all["optimal_path"] == False]  # noqa: E712

    if df_path.empty and df_pathless.empty:
        return

    # Shared x-axis across both
    lr_strs = _sort_unique(df_all["learning_rate_str"].unique().tolist())
    base_width = max(8, 0.6 * len(lr_strs))
    fig_width = base_width * 2 + 4.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5), squeeze=False)
    ax_left = axes[0, 0]
    ax_right = axes[0, 1]

    # Shared colors
    pair_to_color = _build_pair_color_map(df_all)

    # Draw both panels
    if not df_pathless.empty:
        _draw_combined_lines_on_ax(
            ax=ax_left,
            df_filtered=df_pathless,
            metric=metric,
            lr_strs=lr_strs,
            pair_to_color=pair_to_color,
            ylabel=ylabel,
            title=f"{title} (pathless)",
        )
    else:
        ax_left.set_visible(False)

    if not df_path.empty:
        _draw_combined_lines_on_ax(
            ax=ax_right,
            df_filtered=df_path,
            metric=metric,
            lr_strs=lr_strs,
            pair_to_color=pair_to_color,
            ylabel=ylabel,
            title=f"{title} (optimal_path)",
        )
    else:
        ax_right.set_visible(False)

    # Match y-limits for comparability
    ymins: List[float] = []
    ymaxs: List[float] = []
    for ax in [ax_left, ax_right]:
        if ax.get_visible():
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
    if ymins and ymaxs:
        common_ylim = (min(ymins), max(ymaxs))
        for ax in [ax_left, ax_right]:
            if ax.get_visible():
                ax.set_ylim(*common_ylim)

    # Single legend on the right using handles from left (fallback to right)
    handles, labels = [], []
    if ax_left.get_visible():
        handles, labels = ax_left.get_legend_handles_labels()
    elif ax_right.get_visible():
        handles, labels = ax_right.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="depth x width",
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
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

    # # Heatmaps per learning rate
    # _plot_heatmaps_for_metric(
    #     df=df,
    #     metric="avg_success_rate",
    #     title="Avg Success Rate (last-10) by depth x width per learning rate",
    #     output_path=os.path.join(args.outdir, "heatmaps_success_rate.png"),
    #     cmap="viridis",
    #     annotate_as_percent=True,
    # )

    # _plot_heatmaps_for_metric(
    #     df=df,
    #     metric="avg_average_episodic_reward",
    #     title=(
    #         "Avg Average Episodic Reward (last-10) by depth x width per "
    #         "learning rate"
    #     ),
    #     output_path=os.path.join(args.outdir, "heatmaps_avg_reward.png"),
    #     cmap="magma",
    #     annotate_as_percent=False,
    # )

    # # Lines vs width per depth for each LR
    # _plot_lines_vs_width(
    #     df=df,
    #     metric="avg_success_rate",
    #     title="Success Rate vs Width (one line per depth) per learning rate",
    #     output_path=os.path.join(
    #         args.outdir,
    #         "lines_success_rate_vs_width.png",
    #     ),
    # )

    # _plot_lines_vs_width(
    #     df=df,
    #     metric="avg_average_episodic_reward",
    #     title="Avg Reward vs Width (one line per depth) per learning rate",
    #     output_path=os.path.join(
    #         args.outdir,
    #         "lines_avg_reward_vs_width.png",
    #     ),
    # )

    # # New: bar charts by capacity (best LR chosen by success rate)
    # best_lr_success = _compute_best_lr_success(df)
    # _plot_bars_capacity(
    #     df_vals=best_lr_success[
    #         [
    #             "depth",
    #             "width",
    #             "best_success_rate",
    #         ]
    #     ],
    #     value_col="best_success_rate",
    #     title="Best LR Success Rate by Capacity (depth x width)",
    #     ylabel="success rate",
    #     output_path=os.path.join(
    #         args.outdir,
    #         "best_lr_success_by_capacity.png",
    #     ),
    # )
    # _plot_bars_capacity(
    #     df_vals=best_lr_success[
    #         [
    #             "depth",
    #             "width",
    #             "best_avg_reward",
    #         ]
    #     ],
    #     value_col="best_avg_reward",
    #     title="Avg Reward at Best LR by Capacity (depth x width)",
    #     ylabel="avg episodic reward",
    #     output_path=os.path.join(
    #         args.outdir,
    #         "best_lr_avg_reward_by_capacity.png",
    #     ),
    # )

    # New: combined lines by learning rate (one line per capacity)
    _plot_combined_lines_by_lr(
        df=df,
        metric="avg_success_rate",
        title=(
            "Success Rate across Learning Rates "
            "(one line per depth x width x optimal_path)"
        ),
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_success_rate_by_lr.png",
        ),
    )
    _plot_combined_lines_by_lr(
        df=df,
        metric="avg_average_episodic_reward",
        title=(
            "Avg Reward across Learning Rates "
            "(one line per depth x width x optimal_path)"
        ),
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_avg_reward_by_lr.png",
        ),
    )

    # New: separate plots for optimal_path True/False using shared colors
    shared_colors = _build_pair_color_map(df)

    _plot_combined_lines_by_lr_filtered(
        df_all=df,
        df_filtered=df[df["optimal_path"] == True],  # noqa: E712
        metric="avg_success_rate",
        title=("Success Rate across Learning Rates (optimal_path only)"),
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_success_rate_by_lr_optimal_only.png",
        ),
        pair_to_color=shared_colors,
    )
    _plot_combined_lines_by_lr_filtered(
        df_all=df,
        df_filtered=df[df["optimal_path"] == False],  # noqa: E712
        metric="avg_success_rate",
        title=("Success Rate across Learning Rates (pathless only)"),
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_success_rate_by_lr_pathless_only.png",
        ),
        pair_to_color=shared_colors,
    )

    _plot_combined_lines_by_lr_filtered(
        df_all=df,
        df_filtered=df[df["optimal_path"] == True],  # noqa: E712
        metric="avg_average_episodic_reward",
        title=("Avg Reward across Learning Rates (optimal_path only)"),
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_avg_reward_by_lr_optimal_only.png",
        ),
        pair_to_color=shared_colors,
    )
    _plot_combined_lines_by_lr_filtered(
        df_all=df,
        df_filtered=df[df["optimal_path"] == False],  # noqa: E712
        metric="avg_average_episodic_reward",
        title=("Avg Reward across Learning Rates (pathless only)"),
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "combined_lines_avg_reward_by_lr_pathless_only.png",
        ),
        pair_to_color=shared_colors,
    )

    # Side-by-side pathless vs optimal_path
    _plot_side_by_side_path_vs_pathless(
        df_all=df,
        metric="avg_success_rate",
        title="Success Rate across Learning Rates",
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "side_by_side_success_rate_pathless_vs_optimal.png",
        ),
    )
    _plot_side_by_side_path_vs_pathless(
        df_all=df,
        metric="avg_average_episodic_reward",
        title="Avg Reward across Learning Rates",
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "side_by_side_avg_reward_pathless_vs_optimal.png",
        ),
    )

    # Grouped bars: Path vs No Path by capacity
    _plot_grouped_bars_path_vs_pathless_by_capacity(
        df=df,
        metric="avg_success_rate",
        title="Success Rate by Architecture: Path vs No Path",
        ylabel="success rate",
        output_path=os.path.join(
            args.outdir,
            "grouped_bars_success_rate_path_vs_pathless_by_capacity.png",
        ),
        agg="max",
    )
    _plot_grouped_bars_path_vs_pathless_by_capacity(
        df=df,
        metric="avg_average_episodic_reward",
        title="Avg Reward by Architecture: Path vs No Path",
        ylabel="avg episodic reward",
        output_path=os.path.join(
            args.outdir,
            "grouped_bars_avg_reward_path_vs_pathless_by_capacity.png",
        ),
        agg="max",
    )


if __name__ == "__main__":
    main()
