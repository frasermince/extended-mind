import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Prefer Helvetica-like sans fonts (Nature-style)
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica",
            "Arial",
            "Nimbus Sans",
            "DejaVu Sans",
        ],
    }
)


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
    # Fallback: some aggregators may append a plain "optimal_path" marker
    if (
        "/optimal_path" in run_key
        or run_key.endswith("optimal_path")
        or "optimal_path" in run_key
    ):
        return True
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
        # Optional new metrics
        auc_val = row.get("average_reward_area_under_curve")
        try:
            auc_float = float(auc_val) if auc_val is not None else np.nan
        except (TypeError, ValueError):
            auc_float = np.nan
        reward_curve_val = row.get("average_reward_curve", None)
        reward_curve_standard_error_val = row.get(
            "average_reward_curve_standard_error", None
        )
        # Optional per-seed curves at each timestep (list-of-lists [T][S])
        all_avg_episodic_reward_val = row.get("all_avg_episodic_reward", None)
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
                # New columns (may be NaN/None if not present)
                "average_reward_area_under_curve": auc_float,
                "average_reward_curve": reward_curve_val,
                "average_reward_curve_standard_error": reward_curve_standard_error_val,
                "all_avg_episodic_reward": all_avg_episodic_reward_val,
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


def _export_best_auc_per_seed_individual_plots(
    df_all: pd.DataFrame,
    output_dir: str,
    ylabel: str = "Average Reward",
) -> None:
    """Export one PNG per seed for the best-AUC LR per (depth,width),
    separated into subfolders for path visibility.

    Expects columns:
      - "average_reward_area_under_curve"
      - "all_avg_episodic_reward" (list-of-lists [T][S])
      - "learning_rate_str"
    """
    if df_all.empty:
        return
    if (
        "average_reward_area_under_curve" not in df_all.columns
        or "all_avg_episodic_reward" not in df_all.columns
    ):
        return

    for opt_flag, subfolder in [(False, "pathless"), (True, "path")]:
        df_panel = df_all[df_all["optimal_path"] == opt_flag]
        if df_panel.empty:
            continue
        panel_dir = os.path.join(output_dir, subfolder)
        os.makedirs(panel_dir, exist_ok=True)

        groups = df_panel.groupby(["depth", "width"], dropna=False)
        for (depth_val, width_val), sub in groups:
            sub_valid = sub.dropna(
                subset=[
                    "average_reward_area_under_curve",
                    "all_avg_episodic_reward",
                ]
            )
            if sub_valid.empty:
                continue
            idx = sub_valid["average_reward_area_under_curve"].idxmax()
            best_row = df_panel.loc[idx]

            all_vals = best_row.get("all_avg_episodic_reward")
            if not isinstance(all_vals, list) or len(all_vals) == 0:
                continue
            arr_ts = np.asarray(all_vals, dtype=float)
            if arr_ts.ndim != 2:
                continue
            T, S = arr_ts.shape
            x = np.arange(T, dtype=float)
            lr_str = str(best_row.get("learning_rate_str", "unknown"))

            for s in range(S):
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.plot(x, arr_ts[:, s], linewidth=1.2, color="#4C78A8")
                ax.set_xlabel("Time Step (x 10^3)")
                ax.set_ylabel(ylabel)
                title = (
                    f"{depth_val}x{width_val} | LR={lr_str} | "
                    f"{'Path' if opt_flag else 'No Path'} | Seed {s+1}"
                )
                ax.set_title(title)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                desired_ticks = [0, 10000, 20000, 30000, 40000, 50000]
                ticks_in_range = [t for t in desired_ticks if t <= int(x[-1])]
                if ticks_in_range:
                    ax.set_xticks(ticks_in_range)
                    ax.set_xlim(0, ticks_in_range[-1])

                fname = (
                    f"best_auc_seed_{s+1}_depth_{int(depth_val)}_"
                    f"width_{int(width_val)}_lr_{lr_str}.png"
                )
                outpath = os.path.join(panel_dir, fname)
                fig.tight_layout()
                fig.savefig(outpath, dpi=150, bbox_inches="tight")
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
            sub_depth = sub[sub["depth"] == depth]
            sub_depth = sub_depth.sort_values("width").dropna(subset=[metric])
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
    stats = df.groupby(["depth", "width", "optimal_path"])[metric].agg(
        mean=np.mean,
        sem=lambda x: (
            float(np.std(x, ddof=1)) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        ),
        count="count",
        max=np.max,
    )
    stats = stats.reset_index()
    stats = stats.sort_values(["depth", "width", "optimal_path"])  # type: ignore[call-arg]

    # Consistent capacity ordering
    capacities = (
        stats[["depth", "width"]]
        .drop_duplicates()
        .sort_values(["depth", "width"])  # type: ignore
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
    sem_pivot = (
        stats.pivot_table(
            index=["depth", "width"],
            columns="optimal_path",
            values="sem",
            aggfunc="first",
        )
        if show_error_bars
        else None
    )

    def _get(pvt: pd.DataFrame | None, d: int, w: int, opt: bool) -> float:
        if pvt is None or (d, w) not in pvt.index or opt not in pvt.columns:
            return float("nan")
        # Ensure we index with a tuple index correctly
        idx_key = (d, w)
        val = pvt.loc[idx_key, opt]
        return float(val) if pd.notna(val) else float("nan")

    # Bar heights
    y_pathless = [_get(value_pivot, d, w, False) for d, w in capacities]
    y_path = [_get(value_pivot, d, w, True) for d, w in capacities]

    # Error bars (std across LRs)
    yerr_pathless = (
        [_get(sem_pivot, d, w, False) for d, w in capacities]
        if show_error_bars
        else None
    )
    yerr_path = (
        [_get(sem_pivot, d, w, True) for d, w in capacities]
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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
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
    base_pairs = df[["depth", "width"]].drop_duplicates()
    base_pairs = base_pairs.sort_values(["depth", "width"])  # type: ignore
    pairs = base_pairs.itertuples(index=False, name=None)
    pairs = list(pairs)
    num_pairs = max(1, len(pairs))
    cmap = plt.get_cmap("tab20")
    pair_to_color: Dict[tuple, tuple] = {
        pair: cmap(idx / max(1, num_pairs - 1)) for idx, pair in enumerate(pairs)
    }

    # Each line corresponds to a unique (depth, width, optimal_path) triple
    caps_df = df[["depth", "width", "optimal_path"]].drop_duplicates()
    caps_df = caps_df.sort_values(["optimal_path", "depth", "width"])  # type: ignore
    capacities = caps_df.itertuples(index=False, name=None)

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
        .sort_values(["depth", "width"])  # type: ignore
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

    caps_df = df_filtered[["depth", "width"]].drop_duplicates()
    caps_df = caps_df.sort_values(["depth", "width"])  # type: ignore
    capacities = caps_df.itertuples(index=False, name=None)

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

    caps_df2 = df_filtered[["depth", "width"]].drop_duplicates()
    caps_df2 = caps_df2.sort_values(["depth", "width"])  # type: ignore
    capacities = caps_df2.itertuples(index=False, name=None)

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


# def _plot_side_by_side_reward_curves_by_lr(
#     df_all: pd.DataFrame,
#     title_base: str,
#     ylabel: str,
#     output_dir: str,
# ) -> None:
#     """For each learning rate, draw side-by-side (pathless vs optimal_path)
#     reward curves averaged over seeds for each (depth,width).

#     Expects column "average_reward_curve" to contain a list of floats per row.
#     """
#     if df_all.empty:
#         return
#     if "average_reward_curve" not in df_all.columns:
#         return

#     lr_strs = _sort_unique(df_all["learning_rate_str"].unique().tolist())
#     pair_to_color = _build_pair_color_map(df_all)

#     def _mean_sem_curve(
#         rows: pd.DataFrame,
#     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         """Compute per-timestep mean and SEM from list-of-lists.

#         - Expected shape per row: [num_timesteps][num_seeds]
#         - If row stores a 1D list, treat it as [num_timesteps]
#         - If multiple rows exist (should not normally), prefer the first
#         """
#         # Collect candidate arrays
#         arrays: List[np.ndarray] = []
#         for v in rows["average_reward_curve"]:
#             if isinstance(v, list) and len(v) > 0:
#                 arr = np.asarray(v, dtype=float)
#                 arrays.append(arr)
#         if not arrays:
#             return None, None

#         # Use the first non-empty array (per (lr, depth, width, path) we expect one row)
#         arr0 = arrays[0]
#         # If 1D, no SEM available; return mean as the array itself
#         if arr0.ndim == 1:
#             mean_vals = arr0.astype(float)
#             sem_vals = np.full_like(mean_vals, np.nan, dtype=float)
#             return mean_vals, sem_vals

#         # If 2D: shape [T, S]. Compute mean and SEM across seeds axis=1 per timestep
#         if arr0.ndim == 2:
#             # Handle NaNs per timestep
#             with np.errstate(all="ignore"):
#                 # counts per timestep ignoring NaNs
#                 valid_counts = np.sum(~np.isnan(arr0), axis=1).astype(float)
#                 means = np.nanmean(arr0, axis=1)
#                 stds = np.nanstd(arr0, axis=1, ddof=1)
#                 sems = np.divide(
#                     stds,
#                     np.sqrt(np.where(valid_counts > 0, valid_counts, np.nan)),
#                 )

#             return means, sems

#         # Unexpected shape
#         return None, None

#     for lr in lr_strs:
#         sub_lr = df_all[df_all["learning_rate_str"] == lr]
#         if sub_lr.empty:
#             continue

#         df_path = sub_lr[sub_lr["optimal_path"] == True]  # noqa: E712
#         df_pathless = sub_lr[sub_lr["optimal_path"] == False]  # noqa: E712

#         base_width = 10
#         fig, axes = plt.subplots(1, 2, figsize=(base_width, 4.5), squeeze=False)
#         ax_left = axes[0, 0]
#         ax_right = axes[0, 1]

#         # Plot pathless panel
#         if not df_pathless.empty:
#             caps_df3 = df_pathless[["depth", "width"]].drop_duplicates()
#             caps_df3 = caps_df3.sort_values(["depth", "width"])  # type: ignore[call-arg]
#             caps = caps_df3.itertuples(index=False, name=None)
#             for depth_val, width_val in caps:
#                 rows = df_pathless[
#                     (df_pathless["depth"] == depth_val)
#                     & (df_pathless["width"] == width_val)
#                 ]

#                 mean_curve, sem_curve = _mean_sem_curve(rows)
#                 if mean_curve is None:
#                     continue
#                 x = np.arange(mean_curve.shape[0])
#                 ax_left.plot(
#                     x,
#                     mean_curve,
#                     color=pair_to_color[(depth_val, width_val)],
#                     linewidth=1.8,
#                     alpha=0.9,
#                     label=f"{depth_val}x{width_val}",
#                 )
#                 if sem_curve is not None and np.all(np.isfinite(sem_curve)):
#                     ax_left.fill_between(
#                         x,
#                         mean_curve - sem_curve,
#                         mean_curve + sem_curve,
#                         color=pair_to_color[(depth_val, width_val)],
#                         alpha=0.15,
#                         linewidth=0,
#                     )
#             ax_left.set_title(f"{title_base} (pathless) | lr={lr}")
#             ax_left.set_xlabel("timestep")
#             ax_left.set_ylabel(ylabel)
#             ax_left.grid(True, alpha=0.3)
#         else:
#             ax_left.set_visible(False)

#         # Plot optimal_path panel
#         if not df_path.empty:
#             caps_df4 = df_path[["depth", "width"]].drop_duplicates()
#             caps_df4 = caps_df4.sort_values(["depth", "width"])  # type: ignore[call-arg]
#             caps = caps_df4.itertuples(index=False, name=None)
#             for depth_val, width_val in caps:
#                 rows = df_path[
#                     (df_path["depth"] == depth_val) & (df_path["width"] == width_val)
#                 ]
#                 mean_curve, sem_curve = _mean_sem_curve(rows)
#                 if mean_curve is None:
#                     continue
#                 x = np.arange(mean_curve.shape[0])
#                 ax_right.plot(
#                     x,
#                     mean_curve,
#                     color=pair_to_color[(depth_val, width_val)],
#                     linewidth=1.8,
#                     alpha=0.9,
#                     label=f"{depth_val}x{width_val}",
#                 )
#                 if sem_curve is not None and np.all(np.isfinite(sem_curve)):
#                     ax_right.fill_between(
#                         x,
#                         mean_curve - sem_curve,
#                         mean_curve + sem_curve,
#                         color=pair_to_color[(depth_val, width_val)],
#                         alpha=0.15,
#                         linewidth=0,
#                     )
#             ax_right.set_title(f"{title_base} (optimal_path) | lr={lr}")
#             ax_right.set_xlabel("timestep")
#             ax_right.set_ylabel(ylabel)
#             ax_right.grid(True, alpha=0.3)
#         else:
#             ax_right.set_visible(False)

#         # Match y-limits
#         ymins: List[float] = []
#         ymaxs: List[float] = []
#         for ax in [ax_left, ax_right]:
#             if ax.get_visible():
#                 ymin, ymax = ax.get_ylim()
#                 ymins.append(ymin)
#                 ymaxs.append(ymax)
#         if ymins and ymaxs:
#             common_ylim = (min(ymins), max(ymaxs))
#             for ax in [ax_left, ax_right]:
#                 if ax.get_visible():
#                     ax.set_ylim(*common_ylim)

#         # Legend
#         handles, labels = [], []
#         if ax_left.get_visible():
#             handles, labels = ax_left.get_legend_handles_labels()
#         elif ax_right.get_visible():
#             handles, labels = ax_right.get_legend_handles_labels()
#         if handles and labels:
#             fig.legend(
#                 handles,
#                 labels,
#                 title="depth x width",
#                 fontsize=8,
#                 loc="center left",
#                 bbox_to_anchor=(1.02, 0.5),
#                 borderaxespad=0.0,
#             )

#         os.makedirs(output_dir, exist_ok=True)
#         fig.tight_layout()
#         outfile = os.path.join(
#             output_dir,
#             f"side_by_side_reward_curves_lr_{lr}.png",
#         )
#         fig.savefig(outfile, dpi=150, bbox_inches="tight")
#         plt.close(fig)


def _plot_best_auc_reward_curves_side_by_side(
    df_all: pd.DataFrame,
    title_base: str,
    ylabel: str,
    output_path: str,
) -> None:
    """Side-by-side panels (Path Not Visible vs Path Visible) of reward curves
    where, for each (depth, width), we select the learning rate that maximizes
    reward AUC within that panel.

    Requires columns:
      - "average_reward_area_under_curve"
      - "average_reward_curve"
    """
    if df_all.empty:
        return
    if (
        "average_reward_area_under_curve" not in df_all.columns
        or "average_reward_curve" not in df_all.columns
    ):
        return

    df_path = df_all[df_all["optimal_path"] == True]  # noqa: E712
    df_pathless = df_all[df_all["optimal_path"] == False]  # noqa: E712

    if df_path.empty and df_pathless.empty:
        return

    pair_to_color = _build_pair_color_map(df_all)

    def _curve_mean_sem_from_row_val(
        val: object,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not isinstance(val, list) or len(val) == 0:
            return None, None
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            mean_vals = arr.astype(float)
            sem_vals = np.full_like(mean_vals, np.nan, dtype=float)
            return mean_vals, sem_vals
        if arr.ndim == 2:
            with np.errstate(all="ignore"):
                valid_counts = np.sum(~np.isnan(arr), axis=1).astype(float)
                means = np.nanmean(arr, axis=1)
                stds = np.nanstd(arr, axis=1, ddof=1)
                sems = np.divide(
                    stds, np.sqrt(np.where(valid_counts > 0, valid_counts, np.nan))
                )
            return means, sems
        return None, None

    def _draw_panel(ax, df_panel: pd.DataFrame, title: str) -> None:
        if df_panel.empty:
            ax.set_visible(False)
            return
        groups = df_panel.groupby(["depth", "width"], dropna=False)
        panel_max_x: Optional[int] = None
        for (depth_val, width_val), sub in groups:
            sub_valid = sub.dropna(
                subset=[
                    "average_reward_area_under_curve",
                    "average_reward_curve",
                    "average_reward_curve_standard_error",
                ]
            )
            if sub_valid.empty:
                continue
            idx = sub_valid["average_reward_area_under_curve"].idxmax()
            best_row = df_panel.loc[idx]

            # mean_curve, sem_curve = _curve_mean_sem_from_row_val(
            #     best_row.get("average_reward_curve")
            # )
            mean_curve = np.asarray(best_row["average_reward_curve"])
            sem_curve = np.asarray(best_row["average_reward_curve_standard_error"])
            if mean_curve is None:
                continue
            # Plot x in thousands of steps: raw step index divided by 1000
            x = np.arange(mean_curve.shape[0], dtype=float) / 1000.0
            panel_max_x = max(panel_max_x or 0, int(x[-1]))
            color = pair_to_color[(int(depth_val), int(width_val))]
            label = f"{int(depth_val)}x{int(width_val)}"  # capacity only; no LR
            ax.plot(
                x,
                mean_curve,
                linewidth=1.0,
                alpha=0.9,
                linestyle="-",
                color=color,
                label=label,
            )
            if sem_curve is not None and np.all(np.isfinite(sem_curve)):
                ax.fill_between(
                    x,
                    mean_curve - sem_curve,
                    mean_curve + sem_curve,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
        ax.set_title(title)
        ax.set_xlabel("Time Step (x 10^3)")
        ax.set_ylabel("Average Reward Per Step")
        # Specific x-ticks and limits (0..50), representing thousands of steps
        max_x = panel_max_x if panel_max_x is not None else 50
        desired_ticks = [0, 10, 20, 30, 40, 50]
        ticks_in_range = [t for t in desired_ticks if t <= max_x]
        if ticks_in_range:
            ax.set_xticks(ticks_in_range)
            ax.set_xlim(0, ticks_in_range[-1])
        # Minimal frame: hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Create figure and draw panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    ax_left = axes[0, 0]
    ax_right = axes[0, 1]

    _draw_panel(ax_left, df_pathless, "Path Not Visible")
    _draw_panel(ax_right, df_path, "Path Visible")

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

    # Legend inside the right plot with a descriptive title
    if ax_right.get_visible():
        handles, labels = ax_right.get_legend_handles_labels()
        if handles and labels:
            ax_right.legend(
                title="Network Size",
                fontsize=8,
                title_fontsize=9,
                loc="upper left",
                frameon=False,
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_best_auc_per_seed_curves_side_by_side(
    df_all: pd.DataFrame,
    title_base: str,
    ylabel: str,
    output_path: str,
) -> None:
    """Side-by-side panels where, for each (depth,width), we pick the LR with
    max AUC and plot all per-seed curves (thin), plus the mean curve (bold).

    Requires columns in the input DataFrame:
      - "average_reward_area_under_curve"
      - "all_avg_episodic_reward" (shape [T][S])
      - "average_reward_curve" (for the overlaid mean)
    """
    if df_all.empty:
        return
    if (
        "average_reward_area_under_curve" not in df_all.columns
        or "all_avg_episodic_reward" not in df_all.columns
        or "average_reward_curve" not in df_all.columns
    ):
        return

    df_path = df_all[df_all["optimal_path"] == True]  # noqa: E712
    df_pathless = df_all[df_all["optimal_path"] == False]  # noqa: E712

    if df_path.empty and df_pathless.empty:
        return

    pair_to_color = _build_pair_color_map(df_all)

    def _draw_panel(ax, df_panel: pd.DataFrame, title: str) -> None:
        if df_panel.empty:
            ax.set_visible(False)
            return
        groups = df_panel.groupby(["depth", "width"], dropna=False)
        panel_max_x: Optional[int] = None
        for (depth_val, width_val), sub in groups:
            sub_valid = sub.dropna(
                subset=[
                    "average_reward_area_under_curve",
                    "all_avg_episodic_reward",
                    "average_reward_curve",
                ]
            )
            if sub_valid.empty:
                continue
            idx = sub_valid["average_reward_area_under_curve"].idxmax()
            best_row = df_panel.loc[idx]

            # Per-seed curves: list-of-lists [T][S]
            all_vals = best_row.get("all_avg_episodic_reward")
            if not isinstance(all_vals, list) or len(all_vals) == 0:
                continue
            arr_ts = np.asarray(all_vals, dtype=float)  # shape [T, S]
            if arr_ts.ndim != 2:
                continue
            T = arr_ts.shape[0]
            x = np.arange(T, dtype=float) / 1000.0
            panel_max_x = max(panel_max_x or 0, int(x[-1]))

            color = pair_to_color[(int(depth_val), int(width_val))]

            # Plot each seed (thin, translucent)
            for s in range(arr_ts.shape[1]):
                ax.plot(
                    x,
                    arr_ts[:, s],
                    linewidth=0.8,
                    alpha=0.25,
                    linestyle="-",
                    color=color,
                )

            # Overlay mean curve (bold)
            mean_curve = np.asarray(best_row["average_reward_curve"], dtype=float)
            ax.plot(
                x,
                mean_curve,
                linewidth=1.4,
                alpha=0.95,
                linestyle="-",
                color=color,
                label=f"{int(depth_val)}x{int(width_val)}",
            )

        ax.set_title(title)
        ax.set_xlabel("Time Step (x 10^3)")
        ax.set_ylabel(ylabel)
        max_x = panel_max_x if panel_max_x is not None else 50
        desired_ticks = [0, 10, 20, 30, 40, 50]
        ticks_in_range = [t for t in desired_ticks if t <= max_x]
        if ticks_in_range:
            ax.set_xticks(ticks_in_range)
            ax.set_xlim(0, ticks_in_range[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), squeeze=False)
    ax_left = axes[0, 0]
    ax_right = axes[0, 1]

    _draw_panel(ax_left, df_pathless, "Path Not Visible (per-seed)")
    _draw_panel(ax_right, df_path, "Path Visible (per-seed)")

    # Match y-limits
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

    if ax_right.get_visible():
        handles, labels = ax_right.get_legend_handles_labels()
        if handles and labels:
            ax_right.legend(
                title="Network Size",
                fontsize=8,
                title_fontsize=9,
                loc="upper left",
                frameon=False,
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

    # Side-by-side AUC across learning rates (one line per capacity)
    if "average_reward_area_under_curve" in df.columns:
        _plot_side_by_side_path_vs_pathless(
            df_all=df,
            metric="average_reward_area_under_curve",
            title="Avg Reward AUC across Learning Rates",
            ylabel="reward AUC",
            output_path=os.path.join(
                args.outdir,
                "side_by_side_avg_reward_auc_pathless_vs_optimal.png",
            ),
        )

    # Best-AUC reward curves per (depth,width), side-by-side pathless vs path
    _plot_best_auc_reward_curves_side_by_side(
        df_all=df,
        title_base="Avg Reward Curve at Best AUC LR",
        ylabel="Average Reward",
        output_path=os.path.join(
            args.outdir,
            "best_auc_reward_curves_side_by_side.png",
        ),
    )

    # New: per-seed curves for the best-AUC LR per capacity
    _plot_best_auc_per_seed_curves_side_by_side(
        df_all=df,
        title_base="Per-seed Avg Reward Curves at Best AUC LR",
        ylabel="Average Reward",
        output_path=os.path.join(
            args.outdir,
            "best_auc_per_seed_reward_curves_side_by_side.png",
        ),
    )

    # Export individual per-seed plots into a folder
    _export_best_auc_per_seed_individual_plots(
        df_all=df,
        output_dir=os.path.join(args.outdir, "best_auc_per_seed_individual"),
        ylabel="Average Reward",
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
    _plot_grouped_bars_path_vs_pathless_by_capacity(
        df=df,
        metric="average_reward_area_under_curve",
        title="Avg Reward AUC by Architecture: Path vs No Path",
        ylabel="Average Reward Per Step AUC",
        output_path=os.path.join(
            args.outdir,
            "grouped_bars_avg_reward_auc_path_vs_pathless_by_capacity.png",
        ),
        agg="max",
    )


if __name__ == "__main__":
    main()
