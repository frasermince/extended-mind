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

# Improved color palette (colorblind-friendly)
COLORS = [
    "#4c72b0",  # blue
    "#dd8452",  # orange
    "#55a868",  # green
    "#c44e52",  # red
    "#8172b3",  # purple
    "#937860",  # brown
    "#da8bc3",  # pink
    "#8c8c8c",  # gray
    "#ccb974",  # olive
    "#64b5cd",  # cyan
]


# @title More Utility Code
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
    max_learning_rate_per_capacity = {}
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
        per_seed_auc_val = row.get("per_seed_aucs", None)
        per_seed_auc_standard_error_val = row.get("per_seed_auc_standard_error", None)
        # Optional per-seed curves at each timestep (list-of-lists [T][S])
        all_avg_episodic_reward_val = row.get("all_avg_episodic_reward", None)
        record = {
            "learning_rate_str": lr_str,
            "learning_rate": (
                float(lr_str) if lr_str not in {"unknown", ""} else np.nan
            ),
            "optimal_path": optimal_path,
            "depth": int(row["depth"]),
            "width": int(row["width"]),
            # "avg_success_rate": float(row["avg_success_rate"]),
            # "avg_average_episodic_reward": float(
            #    row["avg_average_episodic_reward"]
            # ),
            # New columns (may be NaN/None if not present)
            "average_reward_area_under_curve": auc_float,
            "average_reward_curve": reward_curve_val,
            "average_reward_curve_standard_error": reward_curve_standard_error_val,
            "per_seed_aucs": per_seed_auc_val,
            "per_seed_auc_standard_error": per_seed_auc_standard_error_val,
            # "all_avg_episodic_reward": all_avg_episodic_reward_val,
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)
    return df


# @title Plotting Utilities
def _to_pdf_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def _snap_axes_to_ticks(ax: plt.Axes) -> None:
    """Make both axes end exactly on tick positions."""
    # X: if ticks exist, clamp limits to first/last tick
    xt = ax.get_xticks()
    if xt is not None and len(xt) > 1:
        ax.set_xlim(xt[0], xt[-1])
    # Y: use auto ticks (force draw first), then clamp
    ax.figure.canvas.draw_idle()
    yt = ax.get_yticks()
    if yt is not None and len(yt) > 1:
        ax.set_ylim(yt[0], yt[-1])


def _strip_axes(ax: plt.Axes) -> None:
    """Apply visual cleanup: no grid, hide top/right spines."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mark_curve_max(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color) -> None:
    """Place a star marker at the maximum y of a curve (ignores NaNs)."""
    if x.size == 0 or y.size == 0:
        return
    with np.errstate(invalid="ignore"):
        idx = np.nanargmax(y) if np.isfinite(y).any() else None
    if idx is None:
        return
    ax.scatter(x[idx], y[idx], marker="*", s=240, edgecolor="none", c=[color], zorder=5)


def _add_legend(ax: plt.Axes, title: str = "Network Size") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title=title,
            fontsize=12,
            title_fontsize=12,
            loc="upper right",
            frameon=False,
        )


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


def _plot_step_size_sweep(
    paths: List[Tuple[str, pd.DataFrame]],
    shared_colors: Dict[Tuple[int, int], tuple],
    ylabel: str,
    output_path: str,
) -> None:
    """
    Plot total reward (AUC) across step sizes, one panel per group.
    """
    metric = "average_reward_area_under_curve"
    for path_label, df in paths:
        if df.empty:
            return

    lr_strs = _sort_unique(paths[0][1]["learning_rate_str"].unique().tolist())
    base_width = max(8, 0.6 * len(lr_strs))
    fig_width = base_width * 2 + 4.0
    fig, axes = plt.subplots(1, len(paths), figsize=(fig_width, 5), squeeze=False)

    pair_to_color = shared_colors

    def _draw_panel(ax, df_filtered: pd.DataFrame, title: str) -> None:
        if df_filtered.empty:
            ax.set_visible(False)
            return

        # Build per-(depth,width) curves over learning rates
        lr_strs_local = _sort_unique(df_filtered["learning_rate_str"].unique().tolist())
        x = np.arange(len(lr_strs_local))
        caps_df = df_filtered[["depth", "width"]].drop_duplicates().sort_values(["depth", "width"])  # type: ignore

        for depth_val, width_val in caps_df.itertuples(index=False, name=None):
            sub = df_filtered[
                (df_filtered["depth"] == depth_val)
                & (df_filtered["width"] == width_val)
            ]
            sub_map = {
                lr: v
                for lr, v in zip(
                    sub["learning_rate_str"].tolist(), sub[metric].tolist()
                )
            }
            y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs_local], dtype=float)

            color = pair_to_color[(depth_val, width_val)]
            (line,) = ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                alpha=0.9,
                linestyle="-",
                color=color,
                label=f"{depth_val}x{width_val}",
            )
            _mark_curve_max(ax, x, y, color)

        # X ticks are LR strings - styling with larger fonts
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(lr_strs_local, rotation=45, ha="right")
        ax.set_xlabel("Step-size", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        # Visual cleanup & snap limits to ticks
        _strip_axes(ax)
        _snap_axes_to_ticks(ax)

    # Draw panels
    for i, (path_label, df) in enumerate(paths):
        _draw_panel(axes[0, i], df, path_label)

    # Match y-limits for comparability
    ymins, ymaxs = [], []
    for ax in axes[0, :]:
        if ax.get_visible():
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
    if ymins and ymaxs:
        common_ylim = (min(ymins), max(ymaxs))
        for ax in axes[0, :]:
            if ax.get_visible():
                ax.set_ylim(*common_ylim)
                _snap_axes_to_ticks(ax)

    # After plotting both panels
    for ax in axes[0, :]:
        if ax.get_visible():
            _add_legend(ax, "Network Size")
            ax.legend(fontsize=12, title_fontsize=12)

    fig.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(_to_pdf_path(output_path), bbox_inches="tight")
    # plt.show()


def _plot_best_avg_reward_curves(
    label: str,
    df: pd.DataFrame,
    shared_colors: Dict[Tuple[int, int], tuple],
    output_path: str,
) -> None:
    """Plot reward curves by capacity, selecting the learning rate that maximizes
    reward AUC for each (depth, width) group.

    Requires columns:
      - "average_reward_area_under_curve"
      - "average_reward_curve"
      - "average_reward_curve_standard_error"
    """
    if df.empty:
        return
    if (
        "average_reward_area_under_curve" not in df.columns
        or "average_reward_curve" not in df.columns
    ):
        return

    pair_to_color = shared_colors

    def _draw_panel(ax, df_panel: pd.DataFrame, title: str) -> None:
        if df_panel.empty:
            ax.set_visible(False)
            return
        groups = df_panel.groupby(["depth", "width"], dropna=False)
        panel_max_x: Optional[int] = None
        ymax = 0
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
            mean_curve = np.asarray(best_row["average_reward_curve"]).mean(axis=0)
            sem_curve = np.asarray(best_row["average_reward_curve_standard_error"])
            if mean_curve is None:
                continue
            # Plot x in thousands of steps: raw step index divided by 1000
            x = np.arange(mean_curve.shape[0], dtype=float) / 100.0
            panel_max_x = max(panel_max_x or 0, int(x[-1]))
            color = pair_to_color[(int(depth_val), int(width_val))]
            label = f"{int(depth_val)}x{int(width_val)}"  # capacity only; no LR
            ax.plot(
                x,
                mean_curve,
                linewidth=3.0,
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
            ymax = np.max(np.append([ymax], mean_curve + sem_curve))

        # Optimal policy
        # reward, total_reward = 0., 0.
        # optimal_average_rewards = []
        # for t in range(1, 100000+1):
        #   reward = 1. if t % 15 == 0 else 0.
        #   total_reward += reward
        #   optimal_average_rewards += [total_reward / t]

        # ax.axhline(y=np.max(optimal_average_rewards),
        #         linewidth=1.0,
        #         alpha=0.9,
        #         linestyle="dashed",
        #         color='gray',
        #         label='Max',)

        # Styling with larger fonts
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Time Step (x $10^3$)", fontsize=16, labelpad=10)
        ax.set_ylabel("Average Reward", fontsize=16, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=14)
        # Specific x-ticks and limits (0..50), representing thousands of steps

        max_x = panel_max_x + 1 if panel_max_x is not None else 100
        desired_ticks = [i for i in range(0, 210, 10)]
        ticks_in_range = [t for t in desired_ticks if t <= max_x]
        if ticks_in_range:
            ax.set_xticks(ticks_in_range)
            ax.set_xlim(0, ticks_in_range[-1])
        # Minimal frame: hide top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, ax = plt.subplots(figsize=(18, 6))
    _draw_panel(ax, df, label)

    ax.set_ylim(ax.get_ylim()[0], 0.05)

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            title="Network Size",
            fontsize=12,
            title_fontsize=14,
            loc="upper left",
            frameon=False,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_total_reward_by_capacity(
    paths: List[Tuple[str, pd.DataFrame]],
    metric: str,
    dot_metric: Optional[str],
    error_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    agg: str = "max",  # y-value per bar: "max" or "mean" across LRs
    show_error_bars: bool = True,  # toggle error bars
    capsize: float = 3.0,  # error bar cap size
    path_label: str = "Path",
    pathless_label: str = "No Path",
    pathless_color: str = "#EC8380",  # red
    path_color: str = "#89BEDF",  # blue
    show_sample_dots: bool = True,
    sample_dot_size: float = 18.0,
    sample_dot_alpha: float = 0.75,
    sample_jitter: float = 0,
) -> None:
    # Use the global color palette
    colors = COLORS
    for path_label, df in paths:
        if df.empty:
            return
    if agg not in {"max", "mean"}:
        raise ValueError("agg must be 'max' or 'mean'")

    def _get(pvt: pd.DataFrame | None, d: int, w: int, value_col: str) -> float:
        if pvt is None or (d, w) not in pvt.index:
            return float("nan")
        # Ensure we index with a tuple index correctly
        idx_key = (d, w)
        try:
            raw_val = pvt.loc[idx_key][value_col]
        except (KeyError, IndexError, TypeError, ValueError):
            return float("nan")
        coerced = pd.to_numeric(pd.Series([raw_val]), errors="coerce").iloc[0]
        return float(coerced) if pd.notna(coerced) else float("nan")

    fig = None
    ax = None

    # Per (depth,width,optimal_path) stats across learning rates
    for i, (path_label, df) in enumerate(paths):
        idx = df.groupby(["depth", "width"])[metric].idxmax()
        best_rows = df.loc[idx]
        stats = best_rows.groupby(["depth", "width"]).agg(
            mean=(metric, np.mean),
            sem=((f"{error_metric}", "first")),
            count=(metric, "count"),
            max=(metric, np.max),
        )
        stats = stats.reset_index()
        stats = stats.sort_values(["depth", "width"])  # type: ignore[call-arg]

        # Consistent capacity ordering
        capacities = (
            stats[["depth", "width"]]
            .drop_duplicates()
            .sort_values(["depth", "width"])  # type: ignore
            .itertuples(index=False, name=None)
        )
        capacities = list(capacities)
        labels = [f"{d}x{w}" for d, w in capacities]

        value_col = "max"
        value_pivot = stats.pivot_table(
            index=["depth", "width"],
            values=value_col,
            aggfunc="first",
        )
        sem_pivot = (
            stats.pivot_table(
                index=["depth", "width"],
                values="sem",
                aggfunc="first",
            )
            if show_error_bars
            else None
        )

        y = [_get(value_pivot, d, w, value_col) for d, w in capacities]

        # Error bars (std across LRs)
        yerr = (
            [
                _get(
                    sem_pivot,
                    d,
                    w,
                    "sem",
                )
                for d, w in capacities
            ]
            if show_error_bars
            else None
        )

        fig_width = max(8, 0.7 * len(labels))
        if i == 0:
            fig, ax = plt.subplots(figsize=(fig_width, 4.5))

        x = np.arange(len(labels))
        bar_width = 0.38
        group_gap = 1.8  # extra horizontal gap between label groups
        x = np.arange(len(labels)) * (1.0 + group_gap)
        # fig_width = max(8, 0.7 * len(labels) * (1.0 + group_gap))

        offset = (i - (len(paths) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            y,
            width=bar_width,
            color=colors[i],
            label=path_label,
            yerr=yerr,
            capsize=capsize if show_error_bars else 0.0,
            error_kw=(
                dict(ecolor="black", elinewidth=1, alpha=0.7)
                if show_error_bars
                else None
            ),
        )
    # Bar heights
    # y_pathless = [_get(value_pivot, d, w, False) for d, w in capacities]
    # y_path = [_get(value_pivot, d, w, True) for d, w in capacities]

    # Error bars (std across LRs)
    # yerr_pathless = (
    #     [_get(sem_pivot, d, w, False) for d, w in capacities]
    #     if show_error_bars
    #     else None
    # )
    # yerr_path = (
    #     [_get(sem_pivot, d, w, True) for d, w in capacities]
    #     if show_error_bars
    #     else None
    # )

    # No Path (left), Path (right)
    # ax.bar(
    #     x - bar_width / 2,
    #     y_pathless,
    #     width=bar_width,
    #     color=pathless_color,
    #     label=pathless_label,
    #     yerr=yerr_pathless,
    #     capsize=capsize if show_error_bars else 0.0,
    #     error_kw=(
    #         dict(ecolor="black", elinewidth=1, alpha=0.7) if show_error_bars else None
    #     ),
    # )
    # ax.bar(
    #     x + bar_width / 2,
    #     y_path,
    #     width=bar_width,
    #     color=path_color,
    #     label=path_label,
    #     yerr=yerr_path,
    #     capsize=capsize if show_error_bars else 0.0,
    #     error_kw=(
    #         dict(ecolor="black", elinewidth=1, alpha=0.7) if show_error_bars else None
    #     ),
    # )

    # Overlay per-seed dots at the best LR per group, if a per-seed dot_metric is provided.
    # Fallback: if dot_metric is None or missing/invalid, draw one dot per LR (previous behavior).
    zorder = 1
    max_vals = [0] * len(paths)
    if show_sample_dots:
        rng = np.random.default_rng(0)

        def _best_lr_seed_values(sub_df: pd.DataFrame) -> Optional[np.ndarray]:
            if sub_df.empty:
                return None
            # Choose best LR by argmax of the plotting metric
            tmp = sub_df.copy()
            best_idx = tmp[metric].idxmax()
            candidate = sub_df.loc[best_idx, dot_metric]
            return np.array(candidate)

        x = np.arange(len(labels)) * (1.0 + group_gap)
        for idx_cap, (d_val, w_val) in enumerate(capacities):
            for i, (path_label, df) in enumerate(paths):
                # Pathless (left bar)
                sub = df[(df["depth"] == d_val) & (df["width"] == w_val)]
                vals = _best_lr_seed_values(sub)

                # print(f'Pathless {d_val}x{w_val}')
                # df_describe = pd.DataFrame(left_vals)
                # print(df_describe.describe())
                if not sub.empty:
                    max_vals[i] = np.max(np.append(vals, [max_vals[i]]))
                    if vals is None:
                        # If no per-seed and no best value, fallback to all LR values as dots
                        vals_fallback = pd.to_numeric(
                            sub[metric], errors="coerce"
                        ).to_numpy()
                        left_vals = vals_fallback if vals_fallback.size > 0 else None
                    if vals is not None and vals.size > 0:
                        offset = (i - (len(paths) - 1) / 2) * bar_width
                        xs = [x[idx_cap] + offset] * vals.size
                        ax.scatter(
                            xs,
                            vals,
                            s=sample_dot_size,
                            color=colors[i],
                            edgecolors="white",
                            linewidths=0.4,
                            alpha=sample_dot_alpha,
                            zorder=zorder,
                        )

            # # Path (right bar)
            # sub_right = df[
            #     (df["depth"] == d_val)
            #     & (df["width"] == w_val)
            #     & (df["optimal_path"] == True)  # noqa: E712
            # ]
            # right_vals = _best_lr_seed_values(sub_right)

            # # print(f'Path {d_val}x{w_val}')
            # # df_describe = pd.DataFrame(right_vals)
            # # print(df_describe.describe())

            # max_path = np.max(np.append(right_vals, [max_path]))
            # if right_vals is None:
            #     vals_fallback = (
            #         pd.to_numeric(sub_right[metric], errors="coerce")
            #         .dropna()
            #         .to_numpy()
            #     )
            #     right_vals = vals_fallback if vals_fallback.size > 0 else None
            # if right_vals is not None and right_vals.size > 0:
            #     xs_right = (
            #         x[idx_cap]
            #         + bar_width / 2
            #         + rng.uniform(-sample_jitter, sample_jitter, size=right_vals.size)
            #     )
            #     ax.scatter(
            #         xs_right,
            #         right_vals,
            #         s=sample_dot_size,
            #         color=path_color,
            #         edgecolors="white",
            #         linewidths=0.4,
            #         alpha=sample_dot_alpha,
            #         zorder=zorder,
            #     )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False, fontsize=12)

    # Ensure the y-axis upper limit ends at the next major tick above data max
    try:

        def _series_upper_max(y_vals, y_errs):
            y_arr = np.asarray(y_vals, dtype=float)
            if y_errs is None:
                err_arr = np.zeros_like(y_arr)
            else:
                err_arr = np.asarray(y_errs, dtype=float)
                # Treat non-finite errors as zero to avoid shrinking the max
                err_arr[~np.isfinite(err_arr)] = 0.0
            # Ignore non-finite y values
            y_arr[~np.isfinite(y_arr)] = np.nan
            upper = y_arr + np.where(np.isfinite(y_arr), err_arr, 0.0)
            return np.nanmax(upper) if np.any(np.isfinite(upper)) else np.nan

        ymax_data = np.max(max_vals)

        # ymax_data = np.nanmax(
        #     [
        #         _series_upper_max(y_pathless, yerr_pathless),
        #         _series_upper_max(y_path, yerr_path),
        #     ]
        # )

        ticks = ax.get_yticks()
        if len(ticks) >= 2 and np.isfinite(ymax_data):
            # Find the first tick strictly greater than the data max
            higher_ticks = [t for t in ticks if t > ymax_data + 1e-12]
            if higher_ticks:
                new_top = higher_ticks[0]
            else:
                # Extend by one tick step if no higher tick is present
                steps = np.diff(ticks)
                # Use median positive step to be robust to irregular spacing
                pos_steps = steps[steps > 0]
                step = np.median(pos_steps) if pos_steps.size > 0 else steps[-1]
                new_top = ticks[-1] + step
            bottom, _ = ax.get_ylim()
            # ax.set_ylim(bottom, new_top)
            ax.set_ylim(bottom, 7000)
    except (ValueError, TypeError, RuntimeError, FloatingPointError):
        # If anything goes wrong, fall back silently to default autoscaling
        pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    outdir = "plots"
    misleading_path = "misleading_path_results.json"
    path_and_no_path = "path_and_no_path_results.json"
    visited_cells = "visited_paths.json"
    suboptimal_path = "suboptimal_results.json"
    random_path = "random_path_results.json"
    landmarks = "landmarks_path_results.json"
    landmarks_grey = "landmarks_grey_results.json"
    landmarks_black = "landmarks_black_results.json"
    landmarks_extra = "landmarks_extra_results.json"
    misleading_path_df = load_results(misleading_path)
    path_and_no_path_df = load_results(path_and_no_path)
    visited_cells_df = load_results(visited_cells)
    suboptimal_path_df = load_results(suboptimal_path)
    random_path_df = load_results(random_path)
    # landmarks_df = load_results(landmarks)
    landmarks_grey_df = load_results(landmarks_grey)
    landmarks_black_df = load_results(landmarks_black)
    landmarks_extra_df = load_results(landmarks_extra)

    parser = argparse.ArgumentParser()
    shared_colors = _build_pair_color_map(path_and_no_path_df)
    just_misleading_path = [("Misleading Path", misleading_path_df)]
    just_visited_cells = [("Non Stationary Path", visited_cells_df)]
    just_suboptimal_path = [("Suboptimal Path", suboptimal_path_df)]
    just_random_path = [("Random Path", random_path_df)]
    # just_landmarks = [("Landmarks", landmarks_df)]
    for paths in [
        # just_misleading_path,
        # just_visited_cells,
        # just_suboptimal_path,
        # just_random_path,
        # just_landmarks,
    ]:
        _plot_best_avg_reward_curves(
            label=paths[0][0],
            df=paths[0][1],
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"average_reward_{paths[0][0]}.pdf",
            ),
        )

        _plot_step_size_sweep(
            paths=paths,
            ylabel="Total Reward",
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"sweeps_{paths[0][0]}.pdf",
            ),
        )

    paths = [
        # ("No Path", path_and_no_path_df[path_and_no_path_df["optimal_path"] == False]),
        # ("Random Path", random_path_df),
        # ("Landmarks", landmarks_df),
        # ("Non Stationary Path", visited_cells_df),
        # ("Misleading Path", misleading_path_df),
        # ("Suboptimal Path", suboptimal_path_df),
        # (
        #     "Optimal Path",
        #     path_and_no_path_df[path_and_no_path_df["optimal_path"] == True],
        # ),
        ("Landmarks Grey", landmarks_grey_df),
        ("Landmarks Black", landmarks_black_df),
        ("Landmarks Extra", landmarks_extra_df),
    ]
    _plot_total_reward_by_capacity(
        paths=paths,
        metric="average_reward_area_under_curve",
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        title="",
        xlabel="Network Size",
        ylabel="Total Reward",
        output_path=os.path.join(
            outdir,
            "total_reward.pdf",
        ),
        agg="max",
        show_sample_dots=True,
    )
