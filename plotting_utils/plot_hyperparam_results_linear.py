import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from matplotlib.patches import Patch

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


# Grouping column for capacity (edge_dim)
# We'll find the winning config first, then show bars by capacity for that config
GROUP_COLS = ["edge_dim"]

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


def _make_label(vals) -> str:
    """Create a label from the grouping column values (tuple or dict-like)."""
    if isinstance(vals, tuple):
        # vals = (edge_dim,) - single element tuple
        return f"{vals[0] * vals[0]}"
    # dict-like (row from DataFrame) or single value
    if hasattr(vals, "__getitem__") and not isinstance(vals, (int, float)):
        return f"{vals['edge_dim'] * vals['edge_dim']}"
    # single value (e.g., just the edge_dim int)
    return f"{vals}"


def _make_config_label(row) -> str:
    """Create a label for a config (decay_pixels, decay_chance, inclusion_pixels)."""
    return (
        f"decay={row['decay_pixels']}, "
        f"chance={row['decay_chance']:.2f}, "
        f"incl={row['inclusion_pixels']}"
    )


def _sort_unique(values: List[str]) -> List[str]:
    """Sort a list of numeric strings by their float value."""
    return sorted(
        values,
        key=lambda s: float(s) if s not in {"unknown", ""} else float("inf"),
    )


def _process_row(row: Dict) -> Dict:
    """Process a single row into the DataFrame record format."""
    lr_str = row.get("learning_rate_str", "unknown")
    optimal_path = row.get("optimal_path", False)
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
    path_mode = row.get("path_mode", None)
    return {
        "learning_rate_str": lr_str,
        "learning_rate": (float(lr_str) if lr_str not in {"unknown", ""} else np.nan),
        "optimal_path": optimal_path,
        # Linear agent hyperparameters (replaces depth/width)
        "edge_dim": int(row["edge_dim"]),
        "decay_pixels": (
            int(row["decay_pixels"])
            if "decay_pixels" in row and row["decay_pixels"] is not None
            else None
        ),
        "decay_chance": (
            float(row["decay_chance"])
            if "decay_chance" in row and row["decay_chance"] is not None
            else None
        ),
        "inclusion_pixels": (
            int(row["inclusion_pixels"])
            if "inclusion_pixels" in row and row["inclusion_pixels"] is not None
            else None
        ),
        # New columns (may be NaN/None if not present)
        "average_reward_area_under_curve": auc_float,
        "average_reward_curve": reward_curve_val,
        "average_reward_curve_standard_error": reward_curve_standard_error_val,
        "per_seed_aucs": per_seed_auc_val,
        "per_seed_auc_standard_error": per_seed_auc_standard_error_val,
        "path_mode": path_mode,
    }


def load_results(json_path: str) -> pd.DataFrame:
    """
    Load results JSON or JSONL into a DataFrame with columns:
    [learning_rate, edge_dim (capacity), decay_pixels, decay_chance,
    inclusion_pixels (hyperparams), ...]

    Supports both JSON array format (legacy) and JSONL format (one JSON object per line).
    """
    records: List[Dict] = []

    with open(json_path, "r", encoding="utf-8") as f:
        # Try to detect format: JSON array starts with '[', JSONL doesn't
        first_char = f.read(1)
        f.seek(0)  # Reset to beginning

        if first_char == "[":
            # JSON array format (legacy)
            data: List[Dict] = json.load(f)
            for row in data:
                records.append(_process_row(row))
        else:
            # JSONL format (one JSON object per line)
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    row = json.loads(line)
                    records.append(_process_row(row))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {json_path}")
                    continue

    df = pd.DataFrame.from_records(records)
    return df


def _to_pdf_path(path: str) -> str:
    root, _ = os.path.splitext(path)
    return root + ".pdf"


def _snap_axes_to_ticks(ax: plt.Axes) -> None:
    """Make both axes end exactly on tick positions within data range."""
    # Get current data limits before modifying
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xt = ax.get_xticks()
    if xt is not None and len(xt) > 1:
        # Only use ticks that are within or near the current data range
        valid_xt = xt[(xt >= xlim[0] - 1e-9) & (xt <= xlim[1] + 1e-9)]
        if len(valid_xt) >= 2:
            ax.set_xlim(valid_xt[0], valid_xt[-1])

    ax.figure.canvas.draw_idle()

    yt = ax.get_yticks()
    if yt is not None and len(yt) > 1:
        # Only use ticks that are within or near the current data range
        # Include one tick above the data max so results aren't cut off
        valid_yt_lower = yt[yt >= ylim[0] - 1e-9]
        valid_yt = valid_yt_lower[valid_yt_lower <= ylim[1] + 1e-9]
        # Find first tick above the data range for the upper bound
        ticks_above = yt[yt > ylim[1] + 1e-9]
        if len(ticks_above) > 0 and len(valid_yt) >= 1:
            ax.set_ylim(valid_yt[0], ticks_above[0])
        elif len(valid_yt) >= 2:
            ax.set_ylim(valid_yt[0], valid_yt[-1])


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


def _add_legend(ax: plt.Axes, title: str = "Edge Dim") -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            title=title,
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            frameon=False,
        )


def _build_pair_color_map(
    df_all: pd.DataFrame,
    cmap_name: str = "tab20",
) -> Dict[tuple, str]:
    """Create a stable color map for GROUP_COLS values across plots.

    Works with both single-column (linear: edge_dim) and multi-column
    (DQN: depth, width) groupings.
    """
    pairs_iter = (
        df_all[GROUP_COLS]
        .drop_duplicates()
        .sort_values(GROUP_COLS)  # type: ignore
        .itertuples(index=False, name=None)
    )
    colors = [
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
        "#937860",
        "#da8bc3",
        "#8c8c8c",
        "#ccb974",
        "#64b5cd",
    ]
    pairs = list(pairs_iter)
    pair_to_color: Dict[tuple, str] = {
        pair: colors[idx % len(colors)] for idx, pair in enumerate(pairs)
    }
    return pair_to_color


def _plot_side_by_side_path_vs_pathless(
    paths: List[Tuple[str, pd.DataFrame]],
    metric: str,
    shared_colors: Dict[tuple, str],
    title: str,  # kept for signature compatibility; ignored
    ylabel: str,
    output_path: str,
) -> None:
    """
    Side-by-side plots: shared colors and legend.
    """
    for path_label, df in paths:
        if df.empty:
            return

    lr_strs = _sort_unique(paths[0][1]["learning_rate_str"].unique().tolist())
    base_width = max(8, 0.6 * len(lr_strs))
    fig_width = base_width * 2 + 4.0
    fig, axes = plt.subplots(1, len(paths), figsize=(8, 6), squeeze=False)

    pair_to_color = shared_colors

    def _draw_panel(ax, df_filtered: pd.DataFrame, panel_title: str) -> None:
        if df_filtered.empty:
            ax.set_visible(False)
            return

        lr_strs_local = _sort_unique(df_filtered["learning_rate_str"].unique().tolist())
        x = np.arange(len(lr_strs_local))
        caps_df = (
            df_filtered[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS)
        )  # type: ignore

        for config_vals in caps_df.itertuples(index=False, name=None):
            mask = pd.Series(True, index=df_filtered.index)
            for col, val in zip(GROUP_COLS, config_vals):
                mask &= df_filtered[col] == val
            sub = df_filtered[mask]

            sub_map = {
                lr: v
                for lr, v in zip(
                    sub["learning_rate_str"].tolist(), sub[metric].tolist()
                )
            }
            y = np.array([sub_map.get(lr, np.nan) for lr in lr_strs_local], dtype=float)

            color = pair_to_color[config_vals]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                alpha=0.9,
                linestyle="-",
                color=color,
                label=_make_label(config_vals),
            )
            _mark_curve_max(ax, x, y, color)

        # Styling
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.tick_params(axis="both", which="minor", labelsize=24)
        ax.set_title(panel_title, fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(lr_strs_local, rotation=45, ha="right")
        ax.set_xlabel("Step-size", fontsize=26)
        yticks = np.linspace(0, 10000, 5)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, fontsize=26)
        ax.set_ylim(0, 10000)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        _snap_axes_to_ticks(ax)

    for i, (path_label, df) in enumerate(paths):
        _draw_panel(axes[0, i], df, path_label)

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

    for ax in axes[0, :]:
        if ax.get_visible():
            _add_legend(ax, "Capacity")
            ax.legend(title="Capacity", fontsize=22, title_fontsize=20)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(_to_pdf_path(output_path), bbox_inches="tight")
    fig.savefig(f"{output_path}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_best_auc_reward_curves_side_by_side(
    paths: List[Tuple[str, pd.DataFrame]],
    title_base: str,
    ylabel: str,
    shared_colors: Dict[tuple, str],
    output_path: str,
) -> None:
    """Side-by-side panels of reward curves where, for each hyperparameter config,
    we select the learning rate that maximizes reward AUC within that panel.

    Requires columns:
      - "average_reward_area_under_curve"
      - "average_reward_curve"
    """
    for path_label, df in paths:
        if df.empty:
            return
        if (
            "average_reward_area_under_curve" not in df.columns
            or "average_reward_curve" not in df.columns
        ):
            return

    def _draw_panel(ax, df_panel: pd.DataFrame, title: str) -> None:
        if df_panel.empty:
            ax.set_visible(False)
            return
        groups = df_panel.groupby(GROUP_COLS, dropna=False)
        ymax = 0
        x = None
        for i, (group_vals, sub) in enumerate(groups):
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

            mean_curve = np.asarray(best_row["average_reward_curve"]).mean(axis=0)
            sem_curve = np.asarray(best_row["average_reward_curve_standard_error"])
            if mean_curve is None:
                continue
            # Plot x in thousands of steps: raw step index divided by 100
            x = np.arange(mean_curve.shape[0], dtype=float) / 100.0
            color = COLORS[i % len(COLORS)]
            label = _make_label(group_vals)
            ax.plot(
                x,
                mean_curve,
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

        ax.set_title(title, fontsize=26)
        ax.set_xlabel("Time Step (x $10^3$)", fontsize=26)
        ax.set_ylabel("Average Reward", fontsize=26)

        if x is not None:
            x_tick_locations = np.linspace(start=x[0], stop=x[-1], num=5, dtype=int)
            ax.set_xticks(x_tick_locations)
            x_tick_labels = ["0", "50", "100", "150", "200"]
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlim(0, x_tick_locations[-1])

        yticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
        ax.set_yticks(yticks)
        ax.set_ylim(0, 0.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.tick_params(axis="both", which="minor", labelsize=24)
        ax.grid(False)

    # Create figure and draw panels
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (path_label, df) in enumerate(paths):
        _draw_panel(ax, df, path_label)

    ax.legend(
        title="Capacity",
        fontsize=22,
        title_fontsize=20,
        loc="upper left",
        frameon=False,
        # bbox_to_anchor=(0, 1.05),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # fig.tight_layout()
    fig.savefig(_to_pdf_path(output_path), dpi=150, bbox_inches="tight")
    fig.savefig(f"{output_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_config_stats(
    config_stats: pd.DataFrame,
    path_label: str,
    output_path: str,
    color: str,
) -> None:
    """Plot reward and diff metrics per config as separate bar charts.

    Args:
        config_stats: DataFrame with columns ['decay_pixels', 'decay_chance',
            'inclusion_pixels', 'max_reward', 'mean_reward', 'max_diff', 'mean_diff']
        path_label: Label for the path (used in titles)
        output_path: Base output path for saving plots
        color: Color to use for bars
    """
    # Sort by max_reward for max plot (descending)
    config_stats_max = config_stats.sort_values("max_reward", ascending=False)
    config_labels_max = [
        _make_config_label(row) for _, row in config_stats_max.iterrows()
    ]

    # Plot 1: max_reward per config
    fig_max, ax_max = plt.subplots(
        figsize=(max(10, 0.5 * len(config_labels_max)), 5), layout="constrained"
    )
    x_pos = np.arange(len(config_labels_max))
    bars_max = ax_max.bar(
        x_pos,
        config_stats_max["max_reward"],
        color=color,
        alpha=0.7,
    )
    ax_max.set_xlabel("Config", fontsize=12)
    ax_max.set_ylabel("Max Reward", fontsize=12)
    ax_max.set_title(f"{path_label}: Max Reward per Config", fontsize=14)
    ax_max.set_xticks(x_pos)
    ax_max.set_xticklabels(config_labels_max, rotation=45, ha="right", fontsize=9)
    ax_max.grid(axis="y", alpha=0.3)
    ax_max.tick_params(labelsize=10)

    # Add value labels on bars
    for bar in bars_max:
        height = bar.get_height()
        ax_max.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Save max_reward plot (PDF and PNG)
    base_path = f"{output_path}_max_reward_per_config"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig_max.savefig(f"{base_path}.pdf", dpi=150, bbox_inches="tight")
    fig_max.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_max)

    # Sort by mean_reward for mean plot (descending)
    config_stats_mean = config_stats.sort_values("mean_reward", ascending=False)
    config_labels_mean = [
        _make_config_label(row) for _, row in config_stats_mean.iterrows()
    ]

    # Plot 2: mean_reward per config
    fig_mean, ax_mean = plt.subplots(
        figsize=(max(10, 0.5 * len(config_labels_mean)), 5), layout="constrained"
    )
    x_pos_mean = np.arange(len(config_labels_mean))
    bars_mean = ax_mean.bar(
        x_pos_mean,
        config_stats_mean["mean_reward"],
        color=color,
        alpha=0.7,
    )
    ax_mean.set_xlabel("Config", fontsize=12)
    ax_mean.set_ylabel("Mean Reward", fontsize=12)
    ax_mean.set_title(f"{path_label}: Mean Reward per Config", fontsize=14)
    ax_mean.set_xticks(x_pos_mean)
    ax_mean.set_xticklabels(config_labels_mean, rotation=45, ha="right", fontsize=9)
    ax_mean.grid(axis="y", alpha=0.3)
    ax_mean.tick_params(labelsize=10)

    # Add value labels on bars
    for bar in bars_mean:
        height = bar.get_height()
        ax_mean.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Save mean_reward plot (PDF and PNG)
    base_path = f"{output_path}_mean_reward_per_config"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig_mean.savefig(f"{base_path}.pdf", dpi=150, bbox_inches="tight")
    fig_mean.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mean)

    # Only plot diff metrics if they exist
    if "max_diff" in config_stats.columns:
        # Sort by max_diff for max diff plot (descending)
        config_stats_max_diff = config_stats.sort_values("max_diff", ascending=False)
        config_labels_max_diff = [
            _make_config_label(row) for _, row in config_stats_max_diff.iterrows()
        ]

        # Plot 3: max_diff per config
        fig_max_diff, ax_max_diff = plt.subplots(
            figsize=(max(10, 0.5 * len(config_labels_max_diff)), 5),
            layout="constrained",
        )
        x_pos_max_diff = np.arange(len(config_labels_max_diff))
        bars_max_diff = ax_max_diff.bar(
            x_pos_max_diff,
            config_stats_max_diff["max_diff"],
            color=color,
            alpha=0.7,
        )
        ax_max_diff.set_xlabel("Config", fontsize=12)
        ax_max_diff.set_ylabel("Max Diff from Pathless", fontsize=12)
        ax_max_diff.set_title(
            f"{path_label}: Max Diff from Pathless per Config", fontsize=14
        )
        ax_max_diff.set_xticks(x_pos_max_diff)
        ax_max_diff.set_xticklabels(
            config_labels_max_diff, rotation=45, ha="right", fontsize=9
        )
        ax_max_diff.grid(axis="y", alpha=0.3)
        ax_max_diff.tick_params(labelsize=10)
        ax_max_diff.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        for bar in bars_max_diff:
            height = bar.get_height()
            ax_max_diff.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

        base_path = f"{output_path}_max_diff_per_config"
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        fig_max_diff.savefig(f"{base_path}.pdf", dpi=150, bbox_inches="tight")
        fig_max_diff.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_max_diff)

    if "mean_diff" in config_stats.columns:
        # Sort by mean_diff for mean diff plot (descending)
        config_stats_mean_diff = config_stats.sort_values("mean_diff", ascending=False)
        config_labels_mean_diff = [
            _make_config_label(row) for _, row in config_stats_mean_diff.iterrows()
        ]

        # Plot 4: mean_diff per config
        fig_mean_diff, ax_mean_diff = plt.subplots(
            figsize=(max(10, 0.5 * len(config_labels_mean_diff)), 5),
            layout="constrained",
        )
        x_pos_mean_diff = np.arange(len(config_labels_mean_diff))
        bars_mean_diff = ax_mean_diff.bar(
            x_pos_mean_diff,
            config_stats_mean_diff["mean_diff"],
            color=color,
            alpha=0.7,
        )
        ax_mean_diff.set_xlabel("Config", fontsize=12)
        ax_mean_diff.set_ylabel("Mean Diff from Pathless", fontsize=12)
        ax_mean_diff.set_title(
            f"{path_label}: Mean Diff from Pathless per Config", fontsize=14
        )
        ax_mean_diff.set_xticks(x_pos_mean_diff)
        ax_mean_diff.set_xticklabels(
            config_labels_mean_diff, rotation=45, ha="right", fontsize=9
        )
        ax_mean_diff.grid(axis="y", alpha=0.3)
        ax_mean_diff.tick_params(labelsize=10)
        ax_mean_diff.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        for bar in bars_mean_diff:
            height = bar.get_height()
            ax_mean_diff.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

        base_path = f"{output_path}_mean_diff_per_config"
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        fig_mean_diff.savefig(f"{base_path}.pdf", dpi=150, bbox_inches="tight")
        fig_mean_diff.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_mean_diff)


def _prepare_data_for_plotting(
    df: pd.DataFrame,
    path_label: str,
    pathless_df: pd.DataFrame,
    metric: str,
    output_path: str,
    agg: str = "max_reward",
    color: str = COLORS[1],
    is_nonstationary: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Prepare DataFrame for plotting by selecting best config (if nonstationary) or using all data (if pathless).

    Args:
        df: Input DataFrame with results
        path_label: Label for the path (used in titles and config stats plots)
        pathless_df: DataFrame with pathless results (for diff calculations)
        metric: Metric column name
        output_path: Output path for config stats plots (only used for nonstationary)
        agg: Aggregation method - 'max_reward', 'mean_reward', 'max_diff', or 'mean_diff'
        color: Color for config stats plots
        is_nonstationary: If True, perform config selection. If False, use all data.

    Returns:
        Tuple of (filtered DataFrame ready for plotting, title suffix string or None)
    """
    # Step 1: For each (config, edge_dim), find best step size
    config_cols = ["decay_pixels", "decay_chance", "inclusion_pixels"]
    config_and_capacity = config_cols + ["edge_dim"]
    idx = df.groupby(config_and_capacity)[metric].idxmax()
    best_rows = df.loc[idx].copy()

    # Calculate pathless baseline per edge_dim (best learning rate for each capacity)
    pathless_idx = pathless_df.groupby(GROUP_COLS)[metric].idxmax()
    pathless_best = pathless_df.loc[pathless_idx].set_index("edge_dim")[metric]

    # Add diff column: nonstationary - pathless for each row
    best_rows["pathless_baseline"] = best_rows["edge_dim"].map(pathless_best)
    best_rows["diff_from_pathless"] = best_rows[metric] - best_rows["pathless_baseline"]

    # Step 2: For each config, aggregate across capacities
    config_stats = best_rows.groupby(config_cols).agg(
        max_reward=(metric, "max"),
        mean_reward=(metric, "mean"),
        sum_reward=(metric, "sum"),
        max_diff=("diff_from_pathless", "max"),
        mean_diff=("diff_from_pathless", "mean"),
        sum_diff=("diff_from_pathless", "sum"),
    )
    config_stats = config_stats.reset_index()

    # Create plots for max_reward and mean_reward per config
    _plot_config_stats(config_stats, path_label, output_path, color)

    # Step 3: Find winning config (argmax_M) using mean_N
    winning_config_idx = config_stats[agg].idxmax()
    winning_config = config_stats.loc[winning_config_idx]
    winning_config_vals = (
        winning_config["decay_pixels"],
        winning_config["decay_chance"],
        winning_config["inclusion_pixels"],
    )

    # Step 4: Filter to winning config only
    mask = pd.Series(True, index=df.index)
    for col, val in zip(config_cols, winning_config_vals):
        mask &= df[col] == val
    df_filtered = df[mask]

    # Create title suffix for this path
    decay, chance, incl = winning_config_vals
    title_suffix = f"{path_label}: decay={decay}, chance={chance:.3f}, incl={incl}"

    return df_filtered, title_suffix


def _plot_grouped_bars_path_vs_pathless_by_capacity(
    paths: List[Tuple[str, pd.DataFrame]],
    metric: str,
    dot_metric: Optional[str],
    error_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    title_suffix: Optional[str] = None,
    agg: str = "max",
    show_error_bars: bool = True,
    capsize: float = 5.0,
    show_sample_dots: bool = True,
    sample_dot_size: float = 18.0,
    sample_dot_alpha: float = 0.75,
    is_nonstationary: bool = True,
) -> None:
    # Use the global color palette
    colors = COLORS
    for path_label, df in paths:
        if df.empty:
            return
    if agg not in {"max", "mean"}:
        raise ValueError("agg must be 'max' or 'mean'")

    def _get(pvt: pd.DataFrame | None, config_key: tuple, value_col: str) -> float:
        if pvt is None or config_key not in pvt.index:
            return float("nan")
        try:
            raw_val = pvt.loc[config_key][value_col]
        except (KeyError, IndexError, TypeError, ValueError):
            return float("nan")
        coerced = pd.to_numeric(pd.Series([raw_val]), errors="coerce").iloc[0]
        return float(coerced) if pd.notna(coerced) else float("nan")

    fig = None
    ax = None
    capacities_per_path = []  # Store capacities for each path

    # learning_rate 0.02, edge_dim 20, decay_pixels 600, decay_chance: 0.01, inclusion_pixels: 22, seed: 14

    for i, (path_label, df) in enumerate(paths):
        # df is already pre-filtered/prepared, just need to aggregate stats
        # Group by capacity (edge_dim) and find best step size for each
        idx_capacity = df.groupby(GROUP_COLS)[metric].idxmax()
        best_rows_capacity = df.loc[idx_capacity]
        stats = best_rows_capacity.groupby(GROUP_COLS).agg(
            mean=(metric, np.mean),
            sem=((f"{error_metric}", "first")),
            count=(metric, "count"),
            max=(metric, np.max),
            argmax=(metric, np.argmax),
        )
        stats = stats.reset_index()
        stats = stats.sort_values(GROUP_COLS)  # type: ignore[call-arg]

        capacities = (
            stats[GROUP_COLS]
            .drop_duplicates()
            .sort_values(GROUP_COLS)  # type: ignore
            .itertuples(index=False, name=None)
        )

        capacities = list(capacities)
        labels = [_make_label(c) for c in capacities]

        capacities_per_path.append(capacities)

        value_col = "max"
        value_pivot = stats.pivot_table(
            index=GROUP_COLS,
            values=value_col,
            aggfunc="first",
        )
        sem_pivot = (
            stats.pivot_table(
                index=GROUP_COLS,
                values="sem",
                aggfunc="first",
            )
            if show_error_bars
            else None
        )

        y = [_get(value_pivot, c[0], value_col) for c in capacities]
        yerr = (
            [_get(sem_pivot, c[0], "sem") for c in capacities]
            if show_error_bars
            else None
        )

        fig_width = max(8, 0.7 * len(labels))
        if len(paths) <= 2:
            figsize = (8, 6)
        else:
            figsize = (16, 6)
        if i == 0:
            fig, ax = plt.subplots(figsize=figsize)

        num_bars = len(paths)
        bar_width = 0.38

        if num_bars > 2:
            group_gap = 0.3 * num_bars  # extra horizontal gap between label groups
        else:
            group_gap = 0
        x = np.arange(len(labels)) * (1.0 + group_gap)

        offset = (i - (len(paths) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            y,
            width=bar_width,
            color=colors[i],
            label=path_label,
            yerr=yerr,
            capsize=capsize if show_error_bars else 0.0,
        )

    zorder = 1
    max_vals = [0] * len(paths)
    if show_sample_dots:

        def _best_lr_seed_values(sub_df: pd.DataFrame) -> Optional[np.ndarray]:
            if sub_df.empty:
                return None
            tmp = sub_df.copy()
            best_idx = tmp[metric].idxmax()
            candidate = sub_df.loc[best_idx, dot_metric]
            return np.array(candidate)

        # Use capacities from first path (they should be the same across paths)
        capacities_for_scatter = capacities_per_path[0] if capacities_per_path else []
        labels = [_make_label(c) for c in capacities_for_scatter]
        x = np.arange(len(capacities_for_scatter)) * (1.0 + group_gap)
        for idx_cap, capacity_val in enumerate(capacities_for_scatter):
            for i, (path_label, df) in enumerate(paths):
                # Filter to this capacity
                mask = pd.Series(True, index=df.index)
                cap_tuple = (
                    capacity_val if isinstance(capacity_val, tuple) else (capacity_val,)
                )
                for col, val in zip(GROUP_COLS, cap_tuple):
                    mask &= df[col] == val

                # df is already filtered to winning config by _prepare_data_for_plotting
                sub = df[mask]
                vals = _best_lr_seed_values(sub)
                if not sub.empty and vals is not None:
                    max_vals[i] = np.max(np.append(vals, [max_vals[i]]))
                    if vals.size > 0:
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

    # Build title with config info from title_suffixes
    if title_suffix:
        title_with_config = f"{title}\n(Winning config: {title_suffix})"
    else:
        title_with_config = title

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks = np.arange(len(labels)) * (1 + group_gap)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.tick_params(axis="both", which="minor", labelsize=24)
    ax.set_xlabel(xlabel, fontsize=26)

    yticks = np.linspace(0, 9000, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([int(t) for t in yticks])
    ax.set_ylim(0, 9000)
    ax.set_ylabel(ylabel, fontsize=26)
    if is_nonstationary:
        # Move the title up by using the Figure instead of Axes
        fig.suptitle(title_with_config, fontsize=26, y=1.05)  # Move title further up
    else:
        ax.set_title(
            title_with_config,
            fontsize=26,
        )
    ax.grid(False)

    # Split legend into two halves
    handles, labels_leg = ax.get_legend_handles_labels()
    if len(paths) > 2:
        midpoint = len(labels_leg) // 2
    else:
        midpoint = len(labels_leg)
    handles1, labels1 = handles[:midpoint], labels_leg[:midpoint]
    if len(paths) > 2:
        handles2, labels2 = handles[midpoint:], labels_leg[midpoint:]

    legend1 = ax.legend(
        handles1,
        labels1,
        loc="upper left",
        fontsize=22,
        title_fontsize=20,
        ncols=1,
        bbox_to_anchor=(0, 1.40) if len(paths) > 2 else (0, 1.05),
        frameon=False,
    )
    if len(labels_leg) > 2:
        legend2 = ax.legend(
            handles2,
            labels2,
            loc="upper right",
            fontsize=22,
            title_fontsize=20,
            ncols=1,
            bbox_to_anchor=(0.75, 1.40),
            frameon=False,
        )
    ax.add_artist(legend1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{output_path}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(f"{output_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# @title P-vals Plotting Code
def _plot_pvals(
    pathless_df: pd.DataFrame,
    path_df: pd.DataFrame,
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
    path_label: str = "Path Visible",
    pathless_label: str = "No Path",
    pathless_color: str = "#d62728",  # red
    path_color: str = "#1f77b4",  # blue
    show_sample_dots: bool = True,
    sample_dot_size: float = 18.0,
    sample_dot_alpha: float = 0.75,
) -> None:
    from scipy import stats

    def series_to_list(series):
        return series.tolist()

    # idx = df.groupby(["depth", "width", "optimal_path"])[metric].idxmax()
    # best_rows = df.loc[idx]
    # info = best_rows.groupby(["depth", "width", "optimal_path"]).agg(
    #    mean=(metric, np.mean),
    #    sem=(
    #        (f"{error_metric}", "first")
    #    ),
    #    count=(metric, "count"),
    #    max=(metric, np.max),
    #    data=(metric, series_to_list), # only the mean?
    # )
    # info = info.reset_index()
    # info = info.sort_values(["depth", "width", "optimal_path"])  # type: ignore[call-arg]

    # Consistent capacity ordering
    # capacities = (
    #    info[["depth", "width"]]
    #    .drop_duplicates()
    #    .sort_values(["depth", "width"])  # type: ignore
    #    .itertuples(index=False, name=None)
    # )
    # capacities = list(capacities)
    pathless_df = pathless_df.copy()
    pathless_df["edge_dim"] = pathless_df["edge_dim"] ** 2
    path_df = path_df.copy()
    path_df["edge_dim"] = path_df["edge_dim"] ** 2

    pathless_idx = pathless_df.groupby(["edge_dim", "optimal_path"])[metric].idxmax()
    pathless_best_rows = pathless_df.loc[pathless_idx]
    pathless_info = pathless_best_rows.groupby(["edge_dim", "optimal_path"]).agg(
        mean=(metric, np.mean),
        sem=((f"{error_metric}", "first")),
        count=(metric, "count"),
        max=(metric, np.max),
        data=(metric, series_to_list),  # only the mean?
    )
    pathless_info = pathless_info.reset_index()
    pathless_info = pathless_info.sort_values(["edge_dim", "optimal_path"])  # type: ignore[call-arg]

    # Consistent capacity ordering
    capacities = (
        pathless_info[["edge_dim"]]
        .drop_duplicates()
        .sort_values(["edge_dim"])  # type: ignore
        .itertuples(index=False, name=None)
    )
    capacities = list(capacities)

    # path_idx = path_df.groupby(["depth", "width"])[metric].idxmax()
    # path_best_rows = path_df.loc[path_idx]
    # path_info = path_best_rows.groupby(["depth", "width", "optimal_path"]).agg(
    #    mean=(metric, np.mean),
    #    sem=(
    #        (f"{error_metric}", "first")
    #    ),
    #    count=(metric, "count"),
    #    max=(metric, np.max),
    #    data=(metric, series_to_list), # only the mean?
    # )
    # print("CAPACITIES")
    labels = [f"{edge_dim}" for edge_dim in capacities]
    # print("labels", labels)
    # print([f"{d}x{w}" for d, w in capacities])

    # Hyper param selection
    def _best_lr_seed_values(sub_df: pd.DataFrame) -> Optional[np.ndarray]:
        if sub_df.empty:
            return None
        # Choose best LR by argmax of the plotting metric
        tmp = sub_df.copy()
        best_idx = tmp[metric].idxmax()
        candidate = sub_df.loc[best_idx, dot_metric]
        return np.array(candidate)

    p_values = {}
    for edge_dim in capacities:
        # print("capacity", d_i, w_i)
        size_i = str(edge_dim[0])
        df_sub_i = pathless_df[(pathless_df["edge_dim"] == edge_dim[0])]
        data_i = _best_lr_seed_values(df_sub_i)

        p_values[size_i] = {}
        for edge_dim_j in capacities:
            size_j = str(edge_dim_j[0])
            df_sub_j = path_df[(path_df["edge_dim"] == edge_dim_j[0])]
            data_j = _best_lr_seed_values(df_sub_j)

            _, p_value = stats.ttest_ind(
                data_i, data_j, equal_var=True, alternative="less"
            )
            p_values[size_i][size_j] = p_value

    # Convert to DataFrame for easier manipulation
    p_values_df = pd.DataFrame(p_values)
    print(p_values_df)

    # Create a mask for the lower triangle (size_i < size_j)
    # We'll mask out the lower triangle so only upper triangle shows
    mask = np.zeros_like(p_values_df.values, dtype=bool)
    size_labels = list(p_values_df.index)

    for i, size_i in enumerate(size_labels):
        for j, size_j in enumerate(size_labels):
            # Parse dimensions to compare sizes
            edge_dim_i = int(size_i)
            edge_dim_j = int(size_j)

            # Mask if edge_dim_i > edge_dim_j (hide lower triangle)
            if edge_dim_i > edge_dim_j:
                mask[i, j] = True

    # Create the plot
    fig = plt.figure(figsize=(6, 8))
    plt.rcParams["font.family"] = "sans-serif"  # Use default sans-serif font
    plt.rcParams["font.sans-serif"] = [
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]

    # Create custom colors based on significance
    from matplotlib.colors import ListedColormap

    # Create a custom colormap: gray for non-significant, green for significant
    colors = ["gray", "green"]
    custom_cmap = ListedColormap(colors)

    # Create a binary matrix for coloring (0 = non-significant, 1 = significant)
    color_matrix = (p_values_df <= 0.05).astype(int)
    ax = sns.heatmap(
        color_matrix,
        mask=mask,
        annot=p_values_df,  # Show actual p-values as annotations
        fmt=".2f",
        cmap=custom_cmap,
        square=True,
        linewidths=0.1,
        cbar=False,
        annot_kws={"size": 20},  # Font size for p-value annotations
    )  # Remove colorbar since it's just binary
    ax.tick_params(axis="both", labelsize=20)  # Capacity tick labels

    # Remove all tick marks and grid lines
    plt.gca().grid(False)

    # Add a custom legend

    legend_elements = [
        Patch(facecolor="green", label="$p \\leq 0.05$ (significant)"),
        Patch(facecolor="gray", label="$p > 0.05$ (not significant)"),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=1,
        fontsize=22,
    )

    ax.set_title(
        f"P-values for the Null Hypothesis\nNo Path $\\geq$ {path_label}",
        fontsize=26,
        pad=100,  # Padding to make room for legend
    )
    plt.xlabel(f"Capacity ({pathless_label})", fontsize=26)
    plt.ylabel(f"Capacity ({path_label})", fontsize=26)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(f"{output_path}.pdf", dpi=150)
    fig.savefig(f"{output_path}.png", dpi=150)
    plt.close(fig)

    # plt.show()


def _generate_latex_pvals_table(
    pathless_df: pd.DataFrame,
    path_experiments: list,  # List of (label, df) tuples
    metric: str,
    dot_metric: Optional[str],
    error_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    decimal_places: int = 2,
) -> None:
    """
    Generates a complete LaTeX document using the 'article' class,
    containing a table of performance results (Mean \\pm SEM) by edge_dim.
    Includes No Path and all path variants in a single table.
    """

    # --- Data Aggregation for Mean and Error ---

    # 1. Identify the best row (best hyperparameter) for each unique group
    pathless_idx = pathless_df.groupby(["edge_dim", "optimal_path"])[metric].idxmax()
    pathless_best_rows = pathless_df.loc[pathless_idx]

    # 2. Aggregate the mean of the metric and the error_metric (SEM/STD)
    pathless_agg_results = (
        pathless_best_rows.groupby(["edge_dim", "optimal_path"])
        .agg(
            mean=(metric, np.mean),
            sem=(error_metric, "first"),
        )
        .reset_index()
        .sort_values(["edge_dim"])
    )

    # 3. Consistent edge_dim ordering
    edge_dims = sorted(pathless_agg_results["edge_dim"].unique())

    # 4. Format the data for the table structure: Mean \pm SEM
    formatted_data = {}
    row_labels = ["No Path"]

    # Format No Path data
    for edge_dim in edge_dims:
        no_path = pathless_agg_results[pathless_agg_results["edge_dim"] == edge_dim]
        if not no_path.empty:
            mean_np = no_path["mean"].iloc[0]
            sem_np = no_path["sem"].iloc[0]
            formatted_data[edge_dim, "No Path"] = (
                f"${mean_np:.{decimal_places}f} \\pm {sem_np:.{decimal_places}f}$"
            )
        else:
            formatted_data[edge_dim, "No Path"] = "N/A"

    # Process each path experiment
    for path_label, path_df in path_experiments:
        row_labels.append(path_label)

        # Identify the best row for each unique group
        path_idx = path_df.groupby(["edge_dim", "optimal_path"])[metric].idxmax()
        path_best_rows = path_df.loc[path_idx]

        # Aggregate the mean of the metric and the error_metric
        path_agg_results = (
            path_best_rows.groupby(["edge_dim", "optimal_path"])
            .agg(
                mean=(metric, np.mean),
                sem=(error_metric, "first"),
            )
            .reset_index()
            .sort_values(["edge_dim"])
        )

        # Format path data
        for edge_dim in edge_dims:
            path = path_agg_results[path_agg_results["edge_dim"] == edge_dim]
            if not path.empty:
                mean_p = path["mean"].iloc[0]
                sem_p = path["sem"].iloc[0]
                formatted_data[edge_dim, path_label] = (
                    f"${mean_p:.{decimal_places}f} \\pm {sem_p:.{decimal_places}f}$"
                )
            else:
                formatted_data[edge_dim, path_label] = "N/A"

    def make_tabular(edge_dims, formatted_data, row_labels):
        """Generate a single tabular with edge_dim as columns and all settings as rows."""
        if not edge_dims:
            return ""

        num_cols = len(edge_dims)
        col_format = "l" + "c" * num_cols
        header_row = " & ".join([f"\\textbf{{{ed**2}}}" for ed in edge_dims])

        tabular = f"""\\begin{{center}}
\\begin{{tabular}}{{{col_format}}}
\\toprule
\\textbf{{Setting}} & \\multicolumn{{{num_cols}}}{{c}}{{\\textbf{{Capacity}}}} \\\\
\\cmidrule(lr){{2-{num_cols + 1}}}
 & {header_row} \\\\
\\midrule
        """

        for i, status in enumerate(row_labels):
            row_values = [status]
            for edge_dim in edge_dims:
                row_values.append(formatted_data.get((edge_dim, status), "N/A"))
            tabular += " & ".join(row_values) + " \\\\\n"

        tabular += "\\bottomrule\n\\end{tabular}\n\\end{center}"
        return tabular

    # Build the tabular
    tabular_content = make_tabular(edge_dims, formatted_data, row_labels)

    # Calculate paper height based on number of rows
    paper_height = 3 + len(row_labels)

    # --- Full LaTeX Document Structure (article class) ---
    latex_document = f"""\\documentclass{{article}}
\\usepackage[paperwidth=30cm,paperheight={paper_height}cm,margin=3mm]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage[T1]{{fontenc}}
\\pagestyle{{empty}}

\\begin{{document}}
\\large

{tabular_content}

\\end{{document}}"""

    # --- File Writing ---
    try:
        with open(output_path, "w") as f:
            f.write(latex_document)
        print(f"Successfully wrote LaTeX table to: {output_path}")
        print("To compile to PDF, use: pdflatex " + output_path)
        print("Then crop with: pdfcrop " + output_path.replace(".tex", ".pdf"))
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")


def nonstationary_local_test():
    outputs_name = "seasonal_grid_runs_test"
    outdir = f"nonstationary_local_test_plots_{outputs_name}"
    # misleading_path = "misleading_path_results.json"
    # path_and_no_path = "path_and_no_path_results.json"
    # visited_cells = "visited_paths.json"
    # suboptimal_path = "suboptimal_results.json"
    # random_path = "random_path_results.json"
    # landmarks = "landmarks_path_results.json"
    # landmarks_grey = "landmarks_grey_results.json"
    # landmarks_black = "landmarks_black_results.json"
    # landmarks_extra = "landmarks_extra_results.json"
    # pathless_linear = f"/Users/frasermince/Programming/hidden_llava/{outputs_name}/{outputs_name}_PATH_MODE_NONE_aggregation_progress.jsonl"
    pathless_linear = "/Users/frasermince/Programming/hidden_llava/current_processed/linear_pathless_aggregation_progress.jsonl"
    nonstationary_linear = f"/Users/frasermince/Programming/hidden_llava/{outputs_name}/{outputs_name}_PATH_MODE_VISITED_CELLS_aggregation_progress.jsonl"

    nonstationary_linear_df = load_results(nonstationary_linear)
    pathless_linear_df = load_results(pathless_linear)

    shared_colors = _build_pair_color_map(pathless_linear_df)

    # Raw paths before preparation
    raw_paths = [
        (
            "Non Stationary",
            nonstationary_linear_df,
            True,
        ),  # (label, df, is_nonstationary)
        # ("Pathless Linear", pathless_linear_df, False),
    ]

    # Prepare data for plotting
    output_path = os.path.join(outdir, "total_reward")
    metric = "average_reward_area_under_curve"

    prepared_paths = [
        ("Pathless", pathless_linear_df),
        ("Non Stationary", nonstationary_linear_df),
    ]

    _plot_grouped_bars_path_vs_pathless_by_capacity(
        paths=prepared_paths,
        metric=metric,
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        title="",
        xlabel="Capacity",
        ylabel="Total Reward",
        output_path=f"{output_path}",
        # title_suffix=title_suffix,
        agg="max",
        show_sample_dots=True,
        is_nonstationary=True,
    )

    # paths = [(f"{agg} {title_suffix}", df_prepared)]
    for path_label, path_df in prepared_paths:
        _plot_side_by_side_path_vs_pathless(
            paths=[(path_label, path_df)],
            metric="average_reward_area_under_curve",
            title=f"Avg Reward AUC across Learning Rates",
            ylabel="Total Reward",
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"sweeps/sweeps",
            ),
        )


def nonstationary_main():
    outdir = "nonstationary_plots"
    # misleading_path = "misleading_path_results.json"
    # path_and_no_path = "path_and_no_path_results.json"
    # visited_cells = "visited_paths.json"
    # suboptimal_path = "suboptimal_results.json"
    # random_path = "random_path_results.json"
    # landmarks = "landmarks_path_results.json"
    # landmarks_grey = "landmarks_grey_results.json"
    # landmarks_black = "landmarks_black_results.json"
    # landmarks_extra = "landmarks_extra_results.json"
    nonstationary_linear = "/Users/frasermince/Programming/hidden_llava/current_processed/linear_env_sweep_v2_aggregation_progress.jsonl"
    pathless_linear = "/Users/frasermince/Programming/hidden_llava/current_processed/linear_pathless_aggregation_progress.jsonl"

    nonstationary_linear_df = load_results(nonstationary_linear)
    pathless_linear_df = load_results(pathless_linear)

    shared_colors = _build_pair_color_map(pathless_linear_df)

    # Raw paths before preparation
    raw_paths = [
        (
            "Non Stationary",
            nonstationary_linear_df,
            True,
        ),  # (label, df, is_nonstationary)
        # ("Pathless Linear", pathless_linear_df, False),
    ]

    # Prepare data for plotting
    output_path = os.path.join(outdir, "total_reward")
    metric = "average_reward_area_under_curve"
    filtered_nonstationary_df_01 = nonstationary_linear_df[
        nonstationary_linear_df["decay_chance"] == 0.01
    ]
    filtered_nonstationary_df_03 = nonstationary_linear_df[
        nonstationary_linear_df["decay_chance"] == 0.03
    ]
    path_tuples = [
        ("Pathless", pathless_linear_df),
        ("Non Stationary 0.01", filtered_nonstationary_df_01),
        ("Non Stationary 0.03", filtered_nonstationary_df_03),
    ]

    import pdb

    pdb.set_trace()
    _plot_grouped_bars_path_vs_pathless_by_capacity(
        paths=path_tuples,
        metric="average_reward_area_under_curve",
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        sample_dot_size=50,
        title="",
        xlabel="Capacity",
        ylabel="Total Reward",
        output_path=os.path.join(
            outdir,
            "capacity_total_reward",
        ),
        agg="max",
        show_sample_dots=True,
    )
    print("AFTER")

    title_suffixes = []
    aggs = [
        "max_reward",
        "mean_reward",
        "sum_reward",
        "max_diff",
        "mean_diff",
        "sum_diff",
    ]
    for agg in aggs:
        prepared_paths = [("Pathless", pathless_linear_df)]
        df_prepared, title_suffix = _prepare_data_for_plotting(
            df=nonstationary_linear_df,
            path_label="Non Stationary",
            metric=metric,
            pathless_df=pathless_linear_df,
            output_path=output_path,
            agg=agg,
            color=COLORS[1],
            is_nonstationary=True,
        )
        prepared_paths.append(("Non Stationary " + agg, df_prepared))

        _plot_grouped_bars_path_vs_pathless_by_capacity(
            paths=prepared_paths,
            metric=metric,
            dot_metric="per_seed_aucs",
            error_metric="per_seed_auc_standard_error",
            title="",
            xlabel="Capacity",
            ylabel="Total Reward",
            output_path=f"{output_path}_{agg}",
            title_suffix=title_suffix,
            agg="max",
            show_sample_dots=True,
            is_nonstationary=True,
        )

        paths = [(f"{agg} {title_suffix}", df_prepared)]
        _plot_side_by_side_path_vs_pathless(
            paths=paths,
            metric="average_reward_area_under_curve",
            title=f"{title_suffix} Avg Reward AUC across Learning Rates",
            ylabel="Total Reward",
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"sweeps/sweeps_{agg}",
            ),
        )


def all_paths_main():
    outdir = "linear_plots"

    # Use explicit variable names for each path mode according to file_context_1
    no_path = "/Users/frasermince/Programming/hidden_llava/current_processed/linear_pathless_aggregation_progress.jsonl"
    misleading_path = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_MISLEADING_PATH_aggregation_progress.jsonl"
    random_path = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_RANDOM_PATH_aggregation_progress.jsonl"
    suboptimal_path = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_SUBOPTIMAL_PATH_aggregation_progress.jsonl"
    optimal_path = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_SHORTEST_PATH_aggregation_progress.jsonl"
    landmarks = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_LANDMARKS_aggregation_progress.jsonl"

    misleading_path_df = load_results(misleading_path)
    random_path_df = load_results(random_path)
    suboptimal_path_df = load_results(suboptimal_path)
    optimal_path_df = load_results(optimal_path)
    no_path_df = load_results(no_path)
    landmarks_df = load_results(landmarks)

    just_optimal_path = [("Optimal Path", optimal_path_df)]
    just_no_path = [("No Path", no_path_df)]
    just_misleading_path = [("Misleading Path", misleading_path_df)]
    # just_visited_cells = [("Non Stationary Path", visited_cells_df)]
    just_suboptimal_path = [("Suboptimal Path", suboptimal_path_df)]
    just_random_path = [("Random Path", random_path_df)]
    just_landmarks = [("Landmarks", landmarks_df)]

    path_tuples = [
        ("No Path", no_path_df),
        ("Random Path", random_path_df),
        ("Landmarks", landmarks_df),
        ("Misleading Path", misleading_path_df),
        ("Suboptimal Path", suboptimal_path_df),
        ("Optimal Path", optimal_path_df),
    ]

    path_tuples_no_path_optimal = [
        ("No Path", no_path_df),
        ("Optimal Path", optimal_path_df),
    ]

    path_variants = [
        just_optimal_path,
        just_no_path,
        just_misleading_path,
        # just_visited_cells,
        just_suboptimal_path,
        just_random_path,
        just_landmarks,
    ]

    shared_colors = _build_pair_color_map(no_path_df)
    for paths in path_variants:
        _plot_best_auc_reward_curves_side_by_side(
            paths=paths,
            title_base="Avg Reward Curve at Best AUC LR",
            ylabel="Average Reward",
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"average_reward_{paths[0][0]}",
            ),
        )
        print(os.path.join(outdir, f"average_reward_{paths[0][0]}"))

    _plot_grouped_bars_path_vs_pathless_by_capacity(
        paths=path_tuples_no_path_optimal,
        metric="average_reward_area_under_curve",
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        sample_dot_size=50,
        title="",
        xlabel="Capacity",
        ylabel="Total Reward",
        output_path=os.path.join(
            outdir,
            "total_reward_pathless_path",
        ),
        agg="max",
        show_sample_dots=True,
    )
    _plot_grouped_bars_path_vs_pathless_by_capacity(
        paths=path_tuples,
        metric="average_reward_area_under_curve",
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        sample_dot_size=50,
        title="",
        xlabel="Capacity",
        ylabel="Total Reward",
        output_path=os.path.join(
            outdir,
            "total_reward_all_paths",
        ),
        agg="max",
        show_sample_dots=True,
    )
    for paths in path_variants:
        _plot_side_by_side_path_vs_pathless(
            paths=paths,
            metric="average_reward_area_under_curve",
            title="Avg Reward AUC across Learning Rates",
            ylabel="Total Reward",
            shared_colors=shared_colors,
            output_path=os.path.join(
                outdir,
                f"sweeps_{paths[0][0]}",
            ),
        )

    p_val_experiments = [
        ("Optimal Path", optimal_path_df),
        ("Suboptimal Path", suboptimal_path_df),
        ("Misleading Path", misleading_path_df),
        ("Random Path", random_path_df),
        ("Landmarks", landmarks_df),
    ]
    for path_label, path_variant_df in p_val_experiments:
        _plot_pvals(
            pathless_df=no_path_df,
            path_df=path_variant_df,
            metric="average_reward_area_under_curve",
            dot_metric="per_seed_aucs",
            error_metric="per_seed_auc_standard_error",
            title="",
            xlabel="Capacity",
            ylabel="Total Reward",
            path_label=path_label,
            output_path=os.path.join(
                outdir,
                f"{path_label}_pvals",
            ),
            agg="max",
            show_sample_dots=True,
        )
    p_val_experiments = [
        ("Optimal", optimal_path_df),
        ("Suboptimal", suboptimal_path_df),
        ("Misleading", misleading_path_df),
        ("Random", random_path_df),
        ("Landmarks", landmarks_df),
    ]
    _generate_latex_pvals_table(
        pathless_df=no_path_df,
        path_experiments=p_val_experiments,
        metric="average_reward_area_under_curve",
        dot_metric="per_seed_aucs",
        error_metric="per_seed_auc_standard_error",
        title="",
        xlabel="Capacity",
        ylabel="Total Reward",
        output_path=os.path.join(
            outdir + "/tables",
            "all_paths_table.tex",
        ),
    )


if __name__ == "__main__":

    # nonstationary_main()
    # all_paths_main()
    nonstationary_local_test()
