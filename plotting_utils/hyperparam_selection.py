import os
import json
import numpy as np


try:
    import pyarrow.dataset as ds  # type: ignore[import]
except Exception:
    ds = None


def load_runs_from_parquet(
    outputs_root,
    exp_group_id=None,
    date=None,
):
    """Load run-level summaries from nested Parquet dataset.

    Returns a pandas DataFrame with parsed depth/width and path_variant.
    """
    if ds is None:
        raise RuntimeError("pyarrow is required: pip install pyarrow")
    # roots = _list_results_parquet_roots(outputs_root)
    print("READING ROOTS")
    roots = outputs_root
    dataset = ds.dataset(roots, format="parquet", partitioning="hive")
    flt = None
    if exp_group_id is not None:
        expr = ds.field("exp_group_id") == exp_group_id
        flt = expr if flt is None else flt & expr
    # 'date' is a column (not a hive partition), but we can still filter
    if date is not None:
        expr = ds.field("date") == date
        flt = expr if flt is None else flt & expr
    scanner = dataset.scanner(filter=flt) if flt is not None else dataset.scanner()
    table = scanner.to_table()
    print("CONVERTING TO PANDAS")
    df = table.to_pandas()

    # Derive depth/width from dense_features string
    # depths = []
    # widths = []
    # for s in df.get("dense_features", []).astype(str).tolist():
    #     depth, width = _parse_dense_features_to_depth_width(s)
    #     depths.append(depth)
    #     widths.append(width)
    # if len(depths) == len(df):
    #     df["network_depth"] = depths
    #     df["network_width"] = widths

    # Normalize lr string for grouping consistency
    def _lr_str(x):
        try:
            # Keep common textual forms if present
            s = f"{float(x):.6g}"
            return s
        except Exception:
            return str(x)

    df["learning_rate_str"] = df["learning_rate"].apply(_lr_str)
    df["seed_num"] = df["seed"].astype(int)
    df["path_variant"] = np.where(
        df.get("generate_optimal_path", False), "optimal_path", "standard"
    )

    return df


def aggregate_runs_duckdb(
    outputs_root: str,
    *,
    recursive_glob: str = "**/*.parquet",
    step_subsample: int = 10,
    extra_where: str | None = None,
):
    """Aggregate Parquet logs with DuckDB using pushdown and vectorized ops.

    Returns a pandas DataFrame with one row per
    (depth,width,learning_rate,optimal_path) containing:
      - average_rewards_mean_subsampled: list[float] (subsampled mean per step)
      - average_rewards_standard_error_subsampled: list[float] (SEM per step)
      - total_rewards_mean: float (mean total reward across seeds)
      - total_rewards_standard_error: float (SEM of total reward)
      - total_rewards_individual_seeds: list[float]
      - learning_rate_str: str (stable string for labeling)
    """
    try:
        duckdb = __import__("duckdb")
    except Exception as exc:
        raise RuntimeError("duckdb is required: pip install duckdb") from exc

    # Build a safe parquet glob for DuckDB
    base = outputs_root.rstrip("/")
    # Escape single quotes for SQL literal safety
    base_escaped = base.replace("'", "''")
    glob_path = f"{base_escaped}/{recursive_glob}"

    # Optional extra predicates (e.g., date or exp_group filters)
    where_extra_sql = f" AND ({extra_where})" if extra_where else ""

    # Use window functions and aggregate-to-arrays; rely on Parquet predicate &
    # projection pushdown so we only read required columns/rows.
    query = f"""
        WITH base AS (
            SELECT
                CAST(network_depth AS INTEGER) AS network_depth,
                CAST(network_width AS INTEGER) AS network_width,
                CAST(learning_rate AS DOUBLE) AS learning_rate,
                COALESCE(CAST(optimal_path_available AS BOOLEAN), FALSE) AS optimal_path,
                CAST(seed AS INTEGER) AS seed,
                CAST(step AS BIGINT) AS step,
                CAST(value AS DOUBLE) AS value
            FROM read_parquet('{glob_path}')
            WHERE metric = 'reward_per_timestep'{where_extra_sql}
        ),
        per_seed_auc AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path, seed,
                SUM(value) AS auc
            FROM base
            GROUP BY 1,2,3,4,5
        ),
        per_seed_cum AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path, seed, step,
                AVG(value) OVER (
                    PARTITION BY network_depth, network_width, learning_rate, optimal_path, seed
                    ORDER BY step
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_avg
            FROM base
        ),
        -- Subsampled per-seed cumulative averages for per-seed curves
        per_seed_cum_sub AS (
            SELECT * FROM per_seed_cum
            WHERE MOD(step, {int(step_subsample)}) = 0
        ),
        curves AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path, step,
                AVG(cum_avg) AS mean_curve,
                STDDEV_SAMP(cum_avg) / NULLIF(SQRT(COUNT(*)), 0) AS sem_curve
            FROM per_seed_cum
            GROUP BY 1,2,3,4,5
        ),
        curves_sub AS (
            SELECT * FROM curves
            WHERE MOD(step, {int(step_subsample)}) = 0
        ),
        -- Build per-seed curve arrays (one array per seed)
        per_seed_curve_arrays AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path, seed,
                array_agg(cum_avg ORDER BY step) AS per_seed_curve
            FROM per_seed_cum_sub
            GROUP BY 1,2,3,4,5
        ),
        -- Aggregate per-seed arrays into a list-of-lists per group
        per_seed_curve_matrix AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path,
                array_agg(per_seed_curve ORDER BY seed) AS average_reward_curve
            FROM per_seed_curve_arrays
            GROUP BY 1,2,3,4
        ),
        curve_arrays AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path,
                array_agg(mean_curve ORDER BY step) AS average_rewards_mean_subsampled,
                array_agg(sem_curve ORDER BY step) AS average_rewards_standard_error_subsampled,
                -- Back-compat arrays (mean and SEM across seeds)
                array_agg(mean_curve ORDER BY step) AS average_reward_curve_standard_error_base_mean,
                array_agg(sem_curve ORDER BY step) AS average_reward_curve_standard_error
            FROM curves_sub
            GROUP BY 1,2,3,4
        ),
        auc_group AS (
            SELECT
                network_depth, network_width, learning_rate, optimal_path,
                AVG(auc) AS total_rewards_mean,
                STDDEV_SAMP(auc) / NULLIF(SQRT(COUNT(*)), 0) AS total_rewards_standard_error,
                array_agg(auc ORDER BY seed) AS total_rewards_individual_seeds
            FROM per_seed_auc
            GROUP BY 1,2,3,4
        )
        SELECT
            ca.network_depth,
            ca.network_width,
            ca.learning_rate,
            printf('%.6g', ca.learning_rate) AS learning_rate_str,
            ca.optimal_path,
            ca.average_rewards_mean_subsampled,
            ca.average_rewards_standard_error_subsampled,
            a.total_rewards_mean,
            a.total_rewards_standard_error,
            a.total_rewards_individual_seeds,
            pcm.average_reward_curve,
            ca.average_reward_curve_standard_error
        FROM curve_arrays ca
        JOIN auc_group a USING (network_depth, network_width, learning_rate, optimal_path)
        JOIN per_seed_curve_matrix pcm USING (network_depth, network_width, learning_rate, optimal_path)
        ORDER BY 1,2,3,5
    """

    con = duckdb.connect()
    try:
        df = con.execute(query).df()
    finally:
        con.close()

    # Coerce array columns to native Python lists if needed
    # Back-compat mappings for output column names
    if "total_rewards_mean" in df.columns:
        df["average_reward_area_under_curve"] = df["total_rewards_mean"]
    if "total_rewards_individual_seeds" in df.columns:
        df["per_seed_aucs"] = df["total_rewards_individual_seeds"]
    if "total_rewards_standard_error" in df.columns:
        df["per_seed_auc_standard_error"] = df["total_rewards_standard_error"]
    if "average_rewards_mean_subsampled" in df.columns:
        df["average_reward_mean"] = df["average_rewards_mean_subsampled"]

    # Ensure list-like columns are Python lists
    for col in (
        "average_reward_curve",
        "average_reward_curve_standard_error",
        "average_rewards_mean_subsampled",
        "average_rewards_standard_error_subsampled",
        "total_rewards_individual_seeds",
        "per_seed_aucs",
        "average_reward_mean",
    ):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if not isinstance(x, list) else x)

    return df


if __name__ == "__main__":
    # New path: read from nested Parquet dataset
    redirect_level = 1
    testing_on = True
    accum_results = {}
    depths = (2, 3)
    widths = (4, 8, 16, 32)
    missing_seeds = {}
    learning_rates = (
        "1e-05",
        "5e-05",
        "0.0001",
        "0.0005",
        "0.001",
        "0.005",
        "0.01",
    )
    for optimal_path in (True, False):
        for depth in depths:
            for width in widths:
                for lr in learning_rates:
                    missing_seeds[(depth, width, lr, optimal_path)] = set(range(31))

    use_parquet = True
    if use_parquet:
        outputs_root = "/Users/frasermince/Programming/hidden_llava/parquet_test_reordered/path_mode_MISLEADING_PATH"
        # Aggregate entire directory in one DuckDB call
        try:
            duck_df = aggregate_runs_duckdb(
                outputs_root=outputs_root,
                recursive_glob="**/*.parquet",
                step_subsample=10,
            )
        except Exception as e:
            raise SystemExit(f"DuckDB aggregation failed: {e}") from e

        results_list = []

        # Build records using legacy keys expected by downstream plotting code
        for row in duck_df.itertuples(index=False):
            depth = int(getattr(row, "network_depth"))
            width = int(getattr(row, "network_width"))
            lr_str = str(getattr(row, "learning_rate_str"))
            optimal_path = bool(getattr(row, "optimal_path"))

            # Compose run_key compatible with downstream parsers
            run_key = (
                f"{outputs_root}/learning_rate_{lr_str}/"
                f"network_depth_{depth}/network_width_{width}"
            )
            if optimal_path:
                run_key += "/optimal_path"

            # Normalize arrays for JSON (ensure pure Python lists)
            def _to_float_list(seq):
                return [
                    float(x) for x in (list(seq) if not isinstance(seq, list) else seq)
                ]

            def _to_2d_float_list(list_of_seq):
                outer = (
                    list_of_seq if isinstance(list_of_seq, list) else list(list_of_seq)
                )
                return [
                    [
                        float(v)
                        for v in (list(inner) if not isinstance(inner, list) else inner)
                    ]
                    for inner in outer
                ]

            avg_curve_2d = _to_2d_float_list(getattr(row, "average_reward_curve"))
            avg_curve_sem = _to_float_list(
                getattr(row, "average_reward_curve_standard_error")
            )
            per_seed_totals = _to_float_list(
                getattr(row, "total_rewards_individual_seeds")
            )
            avg_mean_curve = _to_float_list(getattr(row, "average_reward_mean"))

            record = {
                "run_key": run_key,
                "depth": depth,
                "width": width,
                # Back-compat keys sourced from new names
                "average_reward_area_under_curve": float(
                    getattr(row, "total_rewards_mean")
                ),
                "average_reward_curve": avg_curve_2d,
                "average_reward_curve_standard_error": avg_curve_sem,
                "per_seed_aucs": per_seed_totals,
                "per_seed_auc_standard_error": float(
                    getattr(row, "total_rewards_standard_error")
                ),
                "average_reward_mean": avg_mean_curve,
            }
            results_list.append(record)

        # Write consolidated results to repo-root JSON
        out_json = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, "hyperparam_results.json"
            )
        )
        with open(out_json, "w", encoding="utf-8") as json_file:
            json.dump(results_list, json_file, indent=2)
        print(f"Wrote {len(results_list)} records to {out_json}")
    else:
        # Legacy path disabled in favor of DuckDB aggregation demo.
        # Keeping historical code commented for reference.
        pass
