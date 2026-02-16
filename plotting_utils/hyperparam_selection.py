import os
import duckdb
import json
import numpy as np
import glob
from tqdm import tqdm

try:
    import pyarrow.dataset as ds  # type: ignore[import]
except Exception:
    ds = None


def delete_path_metric_records(
    outputs_root: str,
    output_dir: str,
    recursive_glob: str = "**/*.parquet",
):
    """Delete records with metric='path' and add path-derived columns.

    Reads all parquet files, filters out rows where metric='path',
    extracts parameters from the file path, and writes to the same
    relative directory structure in output_dir.

    Args:
        outputs_root: Root directory containing parquet files.
        output_dir: Directory to write the filtered parquet files.
        recursive_glob: Glob pattern for finding parquet files.
    """
    import re

    # Find all input parquet files
    input_files = glob.glob(os.path.join(outputs_root, recursive_glob), recursive=True)
    print(f"Found {len(input_files)} parquet files to process")

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    try:
        for i, input_file in enumerate(input_files):
            # Get relative path and create output path
            rel_path = os.path.relpath(input_file, outputs_root)
            output_file = os.path.join(output_dir, rel_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Extract values from the path
            def extract_int(pattern, path):
                m = re.search(pattern, path)
                return int(m.group(1)) if m else None

            def extract_float(pattern, path):
                m = re.search(pattern, path)
                return float(m.group(1)) if m else None

            edge_dim = extract_int(r"agent_pixel_view_edge_dim_(\d+)", input_file)
            decay_pixels = extract_int(
                r"nonstationary_path_decay_pixels_(\d+)", input_file
            )
            decay_chance = extract_float(
                r"nonstationary_path_decay_chance_([0-9.]+)", input_file
            )
            inclusion_pixels = extract_int(
                r"nonstationary_path_inclusion_pixels_(\d+)", input_file
            )

            input_escaped = input_file.replace("'", "''")
            output_escaped = output_file.replace("'", "''")

            query = f"""
                COPY (
                    SELECT 
                        *,
                        {edge_dim} AS agent_pixel_view_edge_dim,
                        {decay_pixels} AS nonstationary_path_decay_pixels,
                        {decay_chance} AS nonstationary_path_decay_chance,
                        {inclusion_pixels} AS nonstationary_path_inclusion_pixels
                    FROM read_parquet('{input_escaped}')
                    WHERE metric != 'path'
                ) TO '{output_escaped}'
                (FORMAT PARQUET)
            """

            con.execute(query)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(input_files)} files")

        print(f"Filtered parquet files written to {output_dir}")
    finally:
        con.close()


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


def aggregate_runs_linear_duckdb(
    outputs_root: str,
    jsonl_path: str,
    *,
    recursive_glob: str = "**/*.parquet",
    step_subsample: int = 10,
    extra_where: str | None = None,
    start_from_combo: int = 0,  # Skip to this combo index (0-based) for debugging
):
    """Aggregate Parquet logs with DuckDB using pushdown and vectorized ops.

    Processes one learning rate at a time to reduce memory usage.

    Returns a pandas DataFrame with one row per
    (agent_pixel_view_edge_dim, nonstationary_path_decay_pixels,
     nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels,
     learning_rate, optimal_path) containing:
      - average_rewards_mean_subsampled: list[float] (subsampled mean per step)
      - average_rewards_standard_error_subsampled: list[float] (subsampled SEM per step)
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
    base_escaped = base.replace("'", "''")
    glob_path = f"{base_escaped}/{recursive_glob}"

    where_extra_sql = f" AND ({extra_where})" if extra_where else ""

    # First, get unique (learning_rate, agent_pixel_view_edge_dim) combinations
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    print(
        "Discovering unique (learning_rate, agent_pixel_view_edge_dim) combinations..."
    )
    combo_query = f"""
        SELECT DISTINCT 
            CAST(step_size AS DOUBLE) AS learning_rate,
            CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim
        FROM read_parquet('{glob_path}')
        WHERE metric = 'reward_per_timestep'{where_extra_sql}
        ORDER BY learning_rate, agent_pixel_view_edge_dim
    """
    combos = con.execute(combo_query).fetchall()
    print(f"Found {len(combos)} unique (lr, dim) combinations")
    con.close()

    import json
    import gc
    import time

    def print_memory():
        """Print current memory usage (if psutil available)."""
        try:
            import psutil

            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  [Memory: {mem:.2f} GB]", flush=True)
        except ImportError:
            print("  [Memory: psutil not installed]", flush=True)

    # Use a permanent JSONL file in the project directory for resumability
    if not jsonl_path:
        jsonl_path = os.path.join(
            os.path.dirname(__file__),
            "linear_beyond_basic_paths_aggregation_progress.jsonl",
        )
    print(f"Using results file: {jsonl_path}")

    # Load already-processed combinations from existing file
    # Use regex to extract keys WITHOUT parsing full JSON (saves massive memory)
    import re

    lr_pattern = re.compile(r'"learning_rate":\s*([0-9.eE+-]+)')
    dim_pattern = re.compile(r'"agent_pixel_view_edge_dim":\s*(\d+)')

    processed_combos = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    lr_match = lr_pattern.search(line)
                    dim_match = dim_pattern.search(line)
                    if lr_match and dim_match:
                        lr = float(lr_match.group(1))
                        dim = int(dim_match.group(1))
                        processed_combos.add((lr, dim))
                except (ValueError, AttributeError):
                    pass  # Skip malformed lines
        print(f"Found {len(processed_combos)} already-processed combinations")
        gc.collect()  # Free any memory from file reading
        print_memory()

    # Open in append mode so we don't lose previous progress
    total_rows = len(processed_combos) * 125  # Approximate rows from prior runs

    # Force cleanup before starting the main loop
    print("Cleaning up before main loop...")
    gc.collect()
    time.sleep(1)
    print_memory()

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for combo_idx, (lr, dim) in enumerate(combos):
            # Debug: skip to specific combo if requested
            if combo_idx < start_from_combo:
                print(f"\nDebug skip {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim}")
                continue

            # Skip if already processed
            if (lr, dim) in processed_combos:
                print(
                    f"\nSkipping {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim} (already done)"
                )
                continue

            # Extra cleanup before processing - important after skipping many combos
            print(
                f"\n--- Starting combo {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim} ---",
                flush=True,
            )
            gc.collect()
            print_memory()
            print(f"Processing...", flush=True)

            # Query for this (lr, dim, decay_pixels) combination
            query = f"""
        WITH base AS (
            SELECT
                CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim,
                CAST(step_size AS DOUBLE) AS learning_rate,
                COALESCE(CAST(optimal_path_available AS BOOLEAN), FALSE) AS optimal_path,
                CAST(path_mode AS TEXT) AS path_mode,
                CAST(seed AS INTEGER) AS seed,
                CAST(step AS BIGINT) AS step,
                CAST(value AS DOUBLE) AS value
            FROM read_parquet('{glob_path}')
                WHERE metric = 'reward_per_timestep'
                  AND CAST(step_size AS DOUBLE) = {lr}
                  AND CAST(agent_pixel_view_edge_dim AS INTEGER) = {dim}
                  {where_extra_sql}
        ),
        per_seed_auc AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed,
                SUM(value) AS auc
            FROM base
                GROUP BY 1,2,3,4,5
        ),
        per_seed_cum AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed, step,
                AVG(value) OVER (
                    PARTITION BY agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed
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
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, step,
                AVG(cum_avg) AS mean_curve,
                STDDEV_SAMP(cum_avg) / NULLIF(SQRT(COUNT(*)), 0) AS sem_curve
            FROM per_seed_cum
            GROUP BY 1,2,3,4,5
        ),
        curves_sub AS (
            SELECT * FROM curves
            WHERE MOD(step, {int(step_subsample)}) = 0
        ),
        per_seed_curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed,
                array_agg(cum_avg ORDER BY step) AS per_seed_curve
            FROM per_seed_cum_sub
            GROUP BY 1,2,3,4,5
        ),
        per_seed_curve_matrix AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode,
                array_agg(per_seed_curve ORDER BY seed) AS average_reward_curve
            FROM per_seed_curve_arrays
            GROUP BY 1,2,3,4
        ),
        curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode,
                array_agg(mean_curve ORDER BY step) AS average_rewards_mean_subsampled,
                array_agg(sem_curve ORDER BY step) AS average_rewards_standard_error_subsampled,
                array_agg(mean_curve ORDER BY step) AS average_reward_curve_standard_error_base_mean,
                array_agg(sem_curve ORDER BY step) AS average_reward_curve_standard_error
            FROM curves_sub
            GROUP BY 1,2,3,4
        ),
        auc_group AS (
            SELECT
                    agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode,
                AVG(auc) AS total_rewards_mean,
                STDDEV_SAMP(auc) / NULLIF(SQRT(COUNT(*)), 0) AS total_rewards_standard_error,
                array_agg(auc ORDER BY seed) AS total_rewards_individual_seeds
            FROM per_seed_auc
                GROUP BY 1,2,3,4
        )
        SELECT
            ca.agent_pixel_view_edge_dim,
            ca.learning_rate,
            printf('%.6g', ca.learning_rate) AS learning_rate_str,
            ca.optimal_path,
            ca.path_mode,
            ca.average_rewards_mean_subsampled,
            ca.average_rewards_standard_error_subsampled,
            a.total_rewards_mean,
            a.total_rewards_standard_error,
            a.total_rewards_individual_seeds,
            pcm.average_reward_curve,
            ca.average_reward_curve_standard_error
        FROM curve_arrays ca
            JOIN auc_group a USING (agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode)
            JOIN per_seed_curve_matrix pcm USING (agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode)
                ORDER BY 1,2,4,5
            """

            # Force GC and cleanup before starting a new query
            gc.collect()
            time.sleep(2)  # Give OS time to reclaim memory

            # Clean up DuckDB temp files from previous iterations
            duckdb_temp = "/tmp/duckdb_temp"
            if os.path.exists(duckdb_temp):
                import shutil

                shutil.rmtree(duckdb_temp, ignore_errors=True)
            os.makedirs(duckdb_temp, exist_ok=True)

            con = duckdb.connect()
            con.execute("SET preserve_insertion_order = false")
            con.execute("SET threads = 1")  # Single thread to minimize memory
            con.execute(
                "SET memory_limit = '24GB'"
            )  # Conservative - force early disk spill
            con.execute("SET temp_directory = '/tmp/duckdb_temp'")
            con.execute("SET max_temp_directory_size = '150GB'")
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA enable_progress_bar_print = true")

            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, "tolist"):  # handles ndarray and matrix
                        return obj.tolist()
                    if hasattr(obj, "item"):  # handles numpy scalars
                        return obj.item()
                    return super().default(obj)

            def _transform_to_plotting_format(row_dict, outputs_root):
                """Transform DuckDB output to plotting format."""

                # Convert arrays to Python lists
                def _to_float_list(seq):
                    if seq is None:
                        return None
                    if isinstance(seq, list):
                        return [float(x) for x in seq]
                    return [float(x) for x in list(seq)]

                def _to_2d_float_list(list_of_seq):
                    if list_of_seq is None:
                        return None
                    outer = (
                        list_of_seq
                        if isinstance(list_of_seq, list)
                        else list(list_of_seq)
                    )
                    return [
                        [
                            float(v)
                            for v in (
                                list(inner) if isinstance(inner, list) else list(inner)
                            )
                        ]
                        for inner in outer
                    ]

                edge_dim = int(row_dict["agent_pixel_view_edge_dim"])
                lr_str = str(row_dict["learning_rate_str"])
                optimal_path = bool(row_dict["optimal_path"])

                # Compose run_key compatible with downstream parsers
                run_key = (
                    f"{outputs_root}/learning_rate_{lr_str}/"
                    f"agent_pixel_view_edge_dim_{edge_dim}"
                )
                if optimal_path:
                    run_key += "/optimal_path"

                return {
                    "run_key": run_key,
                    "edge_dim": edge_dim,
                    "learning_rate": float(row_dict["learning_rate"]),
                    "learning_rate_str": lr_str,
                    "optimal_path": optimal_path,
                    # Back-compat keys sourced from new names
                    "average_reward_area_under_curve": float(
                        row_dict["total_rewards_mean"]
                    ),
                    "average_reward_curve": _to_2d_float_list(
                        row_dict["average_reward_curve"]
                    ),
                    "average_reward_curve_standard_error": _to_float_list(
                        row_dict["average_reward_curve_standard_error"]
                    ),
                    "per_seed_aucs": _to_float_list(
                        row_dict["total_rewards_individual_seeds"]
                    ),
                    "per_seed_auc_standard_error": float(
                        row_dict["total_rewards_standard_error"]
                    ),
                    "average_reward_mean": _to_float_list(
                        row_dict["average_rewards_mean_subsampled"]
                    ),
                }

            try:
                batch_df = con.execute(query).df()
                # Write each row to JSONL immediately in plotting format (saves memory)
                for _, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    # Transform to plotting format
                    plotting_dict = _transform_to_plotting_format(
                        row_dict, outputs_root
                    )
                    jsonl_file.write(json.dumps(plotting_dict, cls=NumpyEncoder) + "\n")
                jsonl_file.flush()
                total_rows += len(batch_df)
                print(f"  Got {len(batch_df)} rows (total: {total_rows})")
                del batch_df  # Free memory immediately
            except Exception as e:
                print(f"  Error processing lr={lr}, dim={dim}: {e}")
            finally:
                con.close()
                del con  # Explicitly delete connection
                gc.collect()  # Force garbage collection after each batch


def aggregate_runs_linear_exploration_duckdb(
    outputs_root: str,
    jsonl_path: str,
    *,
    recursive_glob: str = "**/*.parquet",
    step_subsample: int = 10,
    extra_where: str | None = None,
    start_from_combo: int = 0,  # Skip to this combo index (0-based) for debugging
):
    """Aggregate Parquet logs with DuckDB, grouping by exploration parameters.

    Similar to aggregate_runs_linear_duckdb but also groups by exploration params
    (start_epsilon, end_epsilon, exploration_fraction) without grouping by
    nonstationary parameters.

    Processes one learning rate at a time to reduce memory usage.

    Returns a pandas DataFrame with one row per
    (agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode,
     start_epsilon, end_epsilon, exploration_fraction) containing:
      - average_rewards_mean_subsampled: list[float] (subsampled mean per step)
      - average_rewards_standard_error_subsampled: list[float] (subsampled SEM per step)
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
    base_escaped = base.replace("'", "''")
    glob_path = f"{base_escaped}/{recursive_glob}"

    where_extra_sql = f" AND ({extra_where})" if extra_where else ""

    # First, get unique (learning_rate, agent_pixel_view_edge_dim) combinations
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    print(
        "Discovering unique (learning_rate, agent_pixel_view_edge_dim) combinations..."
    )
    combo_query = f"""
        SELECT DISTINCT 
            CAST(step_size AS DOUBLE) AS learning_rate,
            CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim
        FROM read_parquet('{glob_path}')
        WHERE metric = 'reward_per_timestep'{where_extra_sql}
        ORDER BY learning_rate, agent_pixel_view_edge_dim
    """
    combos = con.execute(combo_query).fetchall()
    print(f"Found {len(combos)} unique (lr, dim) combinations")
    con.close()

    import json
    import gc
    import time

    def print_memory():
        """Print current memory usage (if psutil available)."""
        try:
            import psutil

            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  [Memory: {mem:.2f} GB]", flush=True)
        except ImportError:
            print("  [Memory: psutil not installed]", flush=True)

    # Use a permanent JSONL file in the project directory for resumability
    if not jsonl_path:
        jsonl_path = os.path.join(
            os.path.dirname(__file__),
            "linear_exploration_aggregation_progress.jsonl",
        )
    print(f"Using results file: {jsonl_path}")

    # Load already-processed combinations from existing file
    # Use regex to extract keys WITHOUT parsing full JSON (saves massive memory)
    import re

    lr_pattern = re.compile(r'"learning_rate":\s*([0-9.eE+-]+)')
    dim_pattern = re.compile(r'"agent_pixel_view_edge_dim":\s*(\d+)')

    processed_combos = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    lr_match = lr_pattern.search(line)
                    dim_match = dim_pattern.search(line)
                    if lr_match and dim_match:
                        lr = float(lr_match.group(1))
                        dim = int(dim_match.group(1))
                        processed_combos.add((lr, dim))
                except (ValueError, AttributeError):
                    pass  # Skip malformed lines
        print(f"Found {len(processed_combos)} already-processed combinations")
        gc.collect()  # Free any memory from file reading
        print_memory()

    # Open in append mode so we don't lose previous progress
    total_rows = len(processed_combos) * 125  # Approximate rows from prior runs

    # Force cleanup before starting the main loop
    print("Cleaning up before main loop...")
    gc.collect()
    time.sleep(1)
    print_memory()

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for combo_idx, (lr, dim) in enumerate(combos):
            # Debug: skip to specific combo if requested
            if combo_idx < start_from_combo:
                print(f"\nDebug skip {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim}")
                continue

            # Skip if already processed
            if (lr, dim) in processed_combos:
                print(
                    f"\nSkipping {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim} (already done)"
                )
                continue

            # Extra cleanup before processing - important after skipping many combos
            print(
                f"\n--- Starting combo {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim} ---",
                flush=True,
            )
            gc.collect()
            print_memory()
            print(f"Processing...", flush=True)

            # Query for this (lr, dim) combination with exploration parameters
            query = f"""
        WITH base AS (
            SELECT
                CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim,
                CAST(step_size AS DOUBLE) AS learning_rate,
                COALESCE(CAST(optimal_path_available AS BOOLEAN), FALSE) AS optimal_path,
                CAST(path_mode AS TEXT) AS path_mode,
                CAST(start_epsilon AS DOUBLE) AS start_epsilon,
                CAST(end_epsilon AS DOUBLE) AS end_epsilon,
                CAST(exploration_fraction AS DOUBLE) AS exploration_fraction,
                CAST(seed AS INTEGER) AS seed,
                CAST(step AS BIGINT) AS step,
                CAST(value AS DOUBLE) AS value
            FROM read_parquet('{glob_path}')
                WHERE metric = 'reward_per_timestep'
                  AND CAST(step_size AS DOUBLE) = {lr}
                  AND CAST(agent_pixel_view_edge_dim AS INTEGER) = {dim}
                  {where_extra_sql}
        ),
        per_seed_auc AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed, start_epsilon, end_epsilon, exploration_fraction,
                SUM(value) AS auc
            FROM base
                GROUP BY 1,2,3,4,5,6,7,8
        ),
        per_seed_cum AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed, step, start_epsilon, end_epsilon, exploration_fraction,
                AVG(value) OVER (
                    PARTITION BY agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed, start_epsilon, end_epsilon, exploration_fraction
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
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, step, start_epsilon, end_epsilon, exploration_fraction,
                AVG(cum_avg) AS mean_curve,
                STDDEV_SAMP(cum_avg) / NULLIF(SQRT(COUNT(*)), 0) AS sem_curve
            FROM per_seed_cum
            GROUP BY 1,2,3,4,5,6,7,8
        ),
        curves_sub AS (
            SELECT * FROM curves
            WHERE MOD(step, {int(step_subsample)}) = 0
        ),
        per_seed_curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, seed, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(cum_avg ORDER BY step) AS per_seed_curve
            FROM per_seed_cum_sub
            GROUP BY 1,2,3,4,5,6,7,8
        ),
        per_seed_curve_matrix AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(per_seed_curve ORDER BY seed) AS average_reward_curve
            FROM per_seed_curve_arrays
            GROUP BY 1,2,3,4,5,6,7
        ),
        curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(mean_curve ORDER BY step) AS average_rewards_mean_subsampled,
                array_agg(sem_curve ORDER BY step) AS average_rewards_standard_error_subsampled,
                array_agg(mean_curve ORDER BY step) AS average_reward_curve_standard_error_base_mean,
                array_agg(sem_curve ORDER BY step) AS average_reward_curve_standard_error
            FROM curves_sub
            GROUP BY 1,2,3,4,5,6,7
        ),
        auc_group AS (
            SELECT
                    agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, start_epsilon, end_epsilon, exploration_fraction,
                AVG(auc) AS total_rewards_mean,
                STDDEV_SAMP(auc) / NULLIF(SQRT(COUNT(*)), 0) AS total_rewards_standard_error,
                array_agg(auc ORDER BY seed) AS total_rewards_individual_seeds
            FROM per_seed_auc
                GROUP BY 1,2,3,4,5,6,7
        )
        SELECT
            ca.agent_pixel_view_edge_dim,
            ca.learning_rate,
            printf('%.6g', ca.learning_rate) AS learning_rate_str,
            ca.optimal_path,
            ca.path_mode,
            ca.start_epsilon,
            ca.end_epsilon,
            ca.exploration_fraction,
            ca.average_rewards_mean_subsampled,
            ca.average_rewards_standard_error_subsampled,
            a.total_rewards_mean,
            a.total_rewards_standard_error,
            a.total_rewards_individual_seeds,
            pcm.average_reward_curve,
            ca.average_reward_curve_standard_error
        FROM curve_arrays ca
            JOIN auc_group a USING (agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, start_epsilon, end_epsilon, exploration_fraction)
            JOIN per_seed_curve_matrix pcm USING (agent_pixel_view_edge_dim, learning_rate, optimal_path, path_mode, start_epsilon, end_epsilon, exploration_fraction)
                ORDER BY 1,2,4,5,6,7,8
            """

            # Force GC and cleanup before starting a new query
            gc.collect()
            time.sleep(2)  # Give OS time to reclaim memory

            # Clean up DuckDB temp files from previous iterations
            duckdb_temp = "/tmp/duckdb_temp"
            if os.path.exists(duckdb_temp):
                import shutil

                shutil.rmtree(duckdb_temp, ignore_errors=True)
            os.makedirs(duckdb_temp, exist_ok=True)

            con = duckdb.connect()
            con.execute("SET preserve_insertion_order = false")
            con.execute("SET threads = 1")  # Single thread to minimize memory
            con.execute(
                "SET memory_limit = '24GB'"
            )  # Conservative - force early disk spill
            con.execute("SET temp_directory = '/tmp/duckdb_temp'")
            con.execute("SET max_temp_directory_size = '150GB'")
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA enable_progress_bar_print = true")

            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, "tolist"):  # handles ndarray and matrix
                        return obj.tolist()
                    if hasattr(obj, "item"):  # handles numpy scalars
                        return obj.item()
                    return super().default(obj)

            def _transform_to_plotting_format(row_dict, outputs_root):
                """Transform DuckDB output to plotting format."""

                # Convert arrays to Python lists
                def _to_float_list(seq):
                    if seq is None:
                        return None
                    if isinstance(seq, list):
                        return [float(x) for x in seq]
                    return [float(x) for x in list(seq)]

                def _to_2d_float_list(list_of_seq):
                    if list_of_seq is None:
                        return None
                    outer = (
                        list_of_seq
                        if isinstance(list_of_seq, list)
                        else list(list_of_seq)
                    )
                    return [
                        [
                            float(v)
                            for v in (
                                list(inner) if isinstance(inner, list) else list(inner)
                            )
                        ]
                        for inner in outer
                    ]

                edge_dim = int(row_dict["agent_pixel_view_edge_dim"])
                lr_str = str(row_dict["learning_rate_str"])
                optimal_path = bool(row_dict["optimal_path"])
                start_epsilon = float(row_dict["start_epsilon"])
                end_epsilon = float(row_dict["end_epsilon"])
                exploration_fraction = float(row_dict["exploration_fraction"])

                # Compose run_key compatible with downstream parsers
                run_key = (
                    f"{outputs_root}/learning_rate_{lr_str}/"
                    f"agent_pixel_view_edge_dim_{edge_dim}/"
                    f"start_epsilon_{start_epsilon}/end_epsilon_{end_epsilon}/exploration_fraction_{exploration_fraction}"
                )
                if optimal_path:
                    run_key += "/optimal_path"

                return {
                    "run_key": run_key,
                    "edge_dim": edge_dim,
                    "agent_pixel_view_edge_dim": edge_dim,
                    "learning_rate": float(row_dict["learning_rate"]),
                    "learning_rate_str": lr_str,
                    "optimal_path": optimal_path,
                    "start_epsilon": start_epsilon,
                    "end_epsilon": end_epsilon,
                    "exploration_fraction": exploration_fraction,
                    # Back-compat keys sourced from new names
                    "average_reward_area_under_curve": float(
                        row_dict["total_rewards_mean"]
                    ),
                    "average_reward_curve": _to_2d_float_list(
                        row_dict["average_reward_curve"]
                    ),
                    "average_reward_curve_standard_error": _to_float_list(
                        row_dict["average_reward_curve_standard_error"]
                    ),
                    "per_seed_aucs": _to_float_list(
                        row_dict["total_rewards_individual_seeds"]
                    ),
                    "per_seed_auc_standard_error": float(
                        row_dict["total_rewards_standard_error"]
                    ),
                    "average_reward_mean": _to_float_list(
                        row_dict["average_rewards_mean_subsampled"]
                    ),
                }

            try:
                batch_df = con.execute(query).df()
                # Write each row to JSONL immediately in plotting format (saves memory)
                for _, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    # Transform to plotting format
                    plotting_dict = _transform_to_plotting_format(
                        row_dict, outputs_root
                    )
                    jsonl_file.write(json.dumps(plotting_dict, cls=NumpyEncoder) + "\n")
                jsonl_file.flush()
                total_rows += len(batch_df)
                print(f"  Got {len(batch_df)} rows (total: {total_rows})")
                del batch_df  # Free memory immediately
            except Exception as e:
                print(f"  Error processing lr={lr}, dim={dim}: {e}")
            finally:
                con.close()
                del con  # Explicitly delete connection
                gc.collect()  # Force garbage collection after each batch


def aggregate_runs_linear_nonstationary_duckdb(
    outputs_root: str,
    *,
    recursive_glob: str = "**/*.parquet",
    step_subsample: int = 10,
    jsonl_path: str,
    extra_where: str | None = None,
    start_from_combo: int = 0,  # Skip to this combo index (0-based) for debugging
):
    """Aggregate Parquet logs with DuckDB using pushdown and vectorized ops.

    Processes one learning rate at a time to reduce memory usage.

    Returns a pandas DataFrame with one row per
    (agent_pixel_view_edge_dim, nonstationary_path_decay_pixels,
     nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels,
     learning_rate, optimal_path) containing:
      - average_rewards_mean_subsampled: list[float] (subsampled mean per step)
      - average_rewards_standard_error_subsampled: list[float] (subsampled SEM per step)
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
    base_escaped = base.replace("'", "''")
    glob_path = f"{base_escaped}/{recursive_glob}"

    where_extra_sql = f" AND ({extra_where})" if extra_where else ""

    # First, get unique (learning_rate, agent_pixel_view_edge_dim) combinations
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")

    print(
        "Discovering unique (learning_rate, agent_pixel_view_edge_dim, decay_chance) combinations..."
    )
    combo_query = f"""
        SELECT DISTINCT 
            CAST(step_size AS DOUBLE) AS learning_rate,
            CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim,
            regexp_extract(filename, 'nonstationary_path_decay_chance_([0-9.]+)', 1) AS decay_chance_str
        FROM read_parquet('{glob_path}', filename=true)
        WHERE metric = 'reward_per_timestep'{where_extra_sql}
        ORDER BY learning_rate, agent_pixel_view_edge_dim, decay_chance_str
    """
    combos = con.execute(combo_query).fetchall()
    print(f"Found {len(combos)} unique (lr, dim, decay_chance) combinations")
    con.close()

    import json
    import gc
    import time

    def print_memory():
        """Print current memory usage (if psutil available)."""
        try:
            import psutil

            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            print(f"  [Memory: {mem:.2f} GB]", flush=True)
        except ImportError:
            print("  [Memory: psutil not installed]", flush=True)

    # Use a permanent JSONL file in the project directory for resumability
    print(f"Using results file: {jsonl_path}")

    # Load already-processed combinations from existing file
    # Use regex to extract keys WITHOUT parsing full JSON (saves massive memory)
    # We track (lr, dim, decay_chance_str) tuples to match our batching granularity
    # decay_chance is kept as string to avoid floating point comparison issues
    import re

    lr_pattern = re.compile(r'"learning_rate":\s*([0-9.eE+-]+)')
    dim_pattern = re.compile(r'"agent_pixel_view_edge_dim":\s*(\d+)')
    # Extract decay_chance as string from run_key path (more reliable than the float value)
    decay_chance_pattern = re.compile(r"nonstationary_path_decay_chance_([0-9.]+)")

    processed_combos = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    lr_match = lr_pattern.search(line)
                    dim_match = dim_pattern.search(line)
                    decay_chance_match = decay_chance_pattern.search(line)
                    if lr_match and dim_match and decay_chance_match:
                        lr = float(lr_match.group(1))
                        dim = int(dim_match.group(1))
                        decay_chance_str = decay_chance_match.group(1)  # Keep as string
                        processed_combos.add((lr, dim, decay_chance_str))
                except (ValueError, AttributeError):
                    pass  # Skip malformed lines
        print(f"Found {len(processed_combos)} already-processed combinations")
        gc.collect()  # Free any memory from file reading
        print_memory()

    # Open in append mode so we don't lose previous progress
    total_rows = len(processed_combos) * 125  # Approximate rows from prior runs

    # Force cleanup before starting the main loop
    print("Cleaning up before main loop...")
    gc.collect()
    time.sleep(1)
    print_memory()

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for combo_idx, (lr, dim, decay_chance_str) in enumerate(combos):
            # Debug: skip to specific combo if requested
            if combo_idx < start_from_combo:
                print(
                    f"\nDebug skip {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim}, decay_chance={decay_chance_str}"
                )
                continue

            # Skip if already processed
            if (lr, dim, decay_chance_str) in processed_combos:
                print(
                    f"\nSkipping {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim}, decay_chance={decay_chance_str} (already done)"
                )
                continue

            # Extra cleanup before processing - important after skipping many combos
            print(
                f"\n--- Starting combo {combo_idx + 1}/{len(combos)}: lr={lr}, dim={dim}, decay_chance={decay_chance_str} ---",
                flush=True,
            )
            gc.collect()
            print_memory()
            print(f"Processing...", flush=True)

            # Query for this (lr, dim, decay_chance_str) combination
            # Use string comparison for decay_chance to avoid floating point issues
            query = f"""
        WITH base AS (
            SELECT
                CAST(agent_pixel_view_edge_dim AS INTEGER) AS agent_pixel_view_edge_dim,
                CAST(regexp_extract(filename, 'nonstationary_path_decay_pixels_(\\d+)', 1) AS INTEGER) AS nonstationary_path_decay_pixels,
                CAST(regexp_extract(filename, 'nonstationary_path_decay_chance_([0-9.]+)', 1) AS DOUBLE) AS nonstationary_path_decay_chance,
                CAST(regexp_extract(filename, 'nonstationary_path_inclusion_pixels_(\\d+)', 1) AS INTEGER) AS nonstationary_path_inclusion_pixels,
                CAST(start_epsilon AS DOUBLE) AS start_epsilon,
                CAST(end_epsilon AS DOUBLE) AS end_epsilon,
                CAST(exploration_fraction AS DOUBLE) AS exploration_fraction,
                CAST(step_size AS DOUBLE) AS learning_rate,
                COALESCE(CAST(optimal_path_available AS BOOLEAN), FALSE) AS optimal_path,
                CAST(seed AS INTEGER) AS seed,
                CAST(step AS BIGINT) AS step,
                CAST(value AS DOUBLE) AS value
            FROM read_parquet('{glob_path}', filename=true)
                WHERE metric = 'reward_per_timestep'
                  AND CAST(step_size AS DOUBLE) = {lr}
                  AND CAST(agent_pixel_view_edge_dim AS INTEGER) = {dim}
                  AND regexp_extract(filename, 'nonstationary_path_decay_chance_([0-9.]+)', 1) = '{decay_chance_str}'
                  {where_extra_sql}
        ),
        per_seed_auc AS (
            SELECT
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, seed, start_epsilon, end_epsilon, exploration_fraction,
                SUM(value) AS auc
            FROM base
                GROUP BY 1,2,3,4,5,6,7,8,9,10
        ),
        per_seed_cum AS (
            SELECT
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, seed, step, start_epsilon, end_epsilon, exploration_fraction,
                AVG(value) OVER (
                    PARTITION BY agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, seed, start_epsilon, end_epsilon, exploration_fraction
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
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, step, start_epsilon, end_epsilon, exploration_fraction,
                AVG(cum_avg) AS mean_curve,
                STDDEV_SAMP(cum_avg) / NULLIF(SQRT(COUNT(*)), 0) AS sem_curve
            FROM per_seed_cum
            GROUP BY 1,2,3,4,5,6,7,8,9,10
        ),
        curves_sub AS (
            SELECT * FROM curves
            WHERE MOD(step, {int(step_subsample)}) = 0
        ),
        per_seed_curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, seed, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(cum_avg ORDER BY step) AS per_seed_curve
            FROM per_seed_cum_sub
            GROUP BY 1,2,3,4,5,6,7,8,9,10
        ),
        per_seed_curve_matrix AS (
            SELECT
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(per_seed_curve ORDER BY seed) AS average_reward_curve
            FROM per_seed_curve_arrays
            GROUP BY 1,2,3,4,5,6,7,8,9
        ),
        curve_arrays AS (
            SELECT
                agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, start_epsilon, end_epsilon, exploration_fraction,
                array_agg(mean_curve ORDER BY step) AS average_rewards_mean_subsampled,
                array_agg(sem_curve ORDER BY step) AS average_rewards_standard_error_subsampled,
                array_agg(mean_curve ORDER BY step) AS average_reward_curve_standard_error_base_mean,
                array_agg(sem_curve ORDER BY step) AS average_reward_curve_standard_error
            FROM curves_sub
            GROUP BY 1,2,3,4,5,6,7,8,9
        ),
        auc_group AS (
            SELECT
                    agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path, start_epsilon, end_epsilon, exploration_fraction,
                AVG(auc) AS total_rewards_mean,
                STDDEV_SAMP(auc) / NULLIF(SQRT(COUNT(*)), 0) AS total_rewards_standard_error,
                array_agg(auc ORDER BY seed) AS total_rewards_individual_seeds
            FROM per_seed_auc
                GROUP BY 1,2,3,4,5,6,7,8,9
        )
        SELECT
                ca.agent_pixel_view_edge_dim,
                ca.nonstationary_path_decay_pixels,
                ca.nonstationary_path_decay_chance,
                ca.nonstationary_path_inclusion_pixels,
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
            JOIN auc_group a USING (agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path)
            JOIN per_seed_curve_matrix pcm USING (agent_pixel_view_edge_dim, nonstationary_path_decay_pixels, nonstationary_path_decay_chance, nonstationary_path_inclusion_pixels, learning_rate, optimal_path)
                ORDER BY 1,2,3,4,5,7
            """

            # Force GC and cleanup before starting a new query
            gc.collect()
            time.sleep(2)  # Give OS time to reclaim memory

            # Clean up DuckDB temp files from previous iterations
            duckdb_temp = "/tmp/duckdb_temp"
            if os.path.exists(duckdb_temp):
                import shutil

                shutil.rmtree(duckdb_temp, ignore_errors=True)
            os.makedirs(duckdb_temp, exist_ok=True)

            con = duckdb.connect()
            con.execute("SET preserve_insertion_order = false")
            con.execute("SET threads = 1")  # Single thread to minimize memory
            con.execute(
                "SET memory_limit = '24GB'"
            )  # Conservative - force early disk spill
            con.execute("SET temp_directory = '/tmp/duckdb_temp'")
            con.execute("SET max_temp_directory_size = '150GB'")
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA enable_progress_bar_print = true")

            # Custom JSON encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, "tolist"):  # handles ndarray and matrix
                        return obj.tolist()
                    if hasattr(obj, "item"):  # handles numpy scalars
                        return obj.item()
                    return super().default(obj)

            def _transform_to_plotting_format(row_dict, outputs_root):
                """Transform DuckDB output to plotting format."""

                # Convert arrays to Python lists
                def _to_float_list(seq):
                    if seq is None:
                        return None
                    if isinstance(seq, list):
                        return [float(x) for x in seq]
                    return [float(x) for x in list(seq)]

                def _to_2d_float_list(list_of_seq):
                    if list_of_seq is None:
                        return None
                    outer = (
                        list_of_seq
                        if isinstance(list_of_seq, list)
                        else list(list_of_seq)
                    )
                    return [
                        [
                            float(v)
                            for v in (
                                list(inner) if isinstance(inner, list) else list(inner)
                            )
                        ]
                        for inner in outer
                    ]

                edge_dim = int(row_dict["agent_pixel_view_edge_dim"])
                decay_pixels = int(row_dict["nonstationary_path_decay_pixels"])
                decay_chance = float(row_dict["nonstationary_path_decay_chance"])
                inclusion_pixels = int(row_dict["nonstationary_path_inclusion_pixels"])
                lr_str = str(row_dict["learning_rate_str"])
                optimal_path = bool(row_dict["optimal_path"])

                # Compose run_key compatible with downstream parsers
                run_key = (
                    f"{outputs_root}/learning_rate_{lr_str}/"
                    f"agent_pixel_view_edge_dim_{edge_dim}/nonstationary_path_decay_pixels_{decay_pixels}/nonstationary_path_decay_chance_{decay_chance}/nonstationary_path_inclusion_pixels_{inclusion_pixels}"
                )
                if optimal_path:
                    run_key += "/optimal_path"

                return {
                    "run_key": run_key,
                    "edge_dim": edge_dim,
                    "agent_pixel_view_edge_dim": edge_dim,
                    "decay_pixels": decay_pixels,
                    "nonstationary_path_decay_pixels": decay_pixels,
                    "decay_chance": decay_chance,
                    "nonstationary_path_decay_chance": decay_chance,
                    "inclusion_pixels": inclusion_pixels,
                    "nonstationary_path_inclusion_pixels": inclusion_pixels,
                    "learning_rate": float(row_dict["learning_rate"]),
                    "learning_rate_str": lr_str,
                    "optimal_path": optimal_path,
                    # Back-compat keys sourced from new names
                    "average_reward_area_under_curve": float(
                        row_dict["total_rewards_mean"]
                    ),
                    "average_reward_curve": _to_2d_float_list(
                        row_dict["average_reward_curve"]
                    ),
                    "average_reward_curve_standard_error": _to_float_list(
                        row_dict["average_reward_curve_standard_error"]
                    ),
                    "per_seed_aucs": _to_float_list(
                        row_dict["total_rewards_individual_seeds"]
                    ),
                    "per_seed_auc_standard_error": float(
                        row_dict["total_rewards_standard_error"]
                    ),
                    "average_reward_mean": _to_float_list(
                        row_dict["average_rewards_mean_subsampled"]
                    ),
                }

            try:
                batch_df = con.execute(query).df()
                # Write each row to JSONL immediately in plotting format (saves memory)
                for _, row in batch_df.iterrows():
                    row_dict = row.to_dict()
                    # Transform to plotting format
                    plotting_dict = _transform_to_plotting_format(
                        row_dict, outputs_root
                    )
                    jsonl_file.write(json.dumps(plotting_dict, cls=NumpyEncoder) + "\n")
                jsonl_file.flush()
                total_rows += len(batch_df)
                print(f"  Got {len(batch_df)} rows (total: {total_rows})")
                del batch_df  # Free memory immediately
            except Exception as e:
                print(
                    f"  Error processing lr={lr}, dim={dim}, decay_chance={decay_chance_str}: {e}"
                )
            finally:
                con.close()
                del con  # Explicitly delete connection
                gc.collect()  # Force garbage collection after each batch


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
    con.execute("SET preserve_insertion_order = false")
    con.execute("PRAGMA enable_progress_bar")
    con.execute("PRAGMA enable_progress_bar_print = true")
    print("Running DuckDB aggregation query...")
    try:
        df = con.execute(query).df()
    finally:
        con.close()
    print("Aggregation complete.")

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


def process_linear_hyperparams(jsonl_path: str):
    """Convert JSONL file to JSON array format for plotting.

    The JSONL file is already in the plotting format, so we just need to
    read it and write it as a JSON array.
    """
    results_list = []

    print(f"Converting JSONL to JSON array from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(
            tqdm(f, desc="Reading lines from JSONL", unit="lines"), 1
        ):
            try:
                record = json.loads(line)
                results_list.append(record)
            except json.JSONDecodeError:
                print(f"  Warning: Skipping malformed line {line_num}")
                continue

    # Write consolidated results to repo-root JSON
    out_json = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.pardir, "nonstationary_results_linear.json"
        )
    )
    with open(out_json, "w", encoding="utf-8") as json_file:
        json.dump(results_list, json_file, indent=2)
    print(f"Wrote {len(results_list)} records to {out_json}")


def process_deep_hyperparams(duck_df):
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
            return [float(x) for x in (list(seq) if not isinstance(seq, list) else seq)]

        def _to_2d_float_list(list_of_seq):
            outer = list_of_seq if isinstance(list_of_seq, list) else list(list_of_seq)
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
        per_seed_totals = _to_float_list(getattr(row, "total_rewards_individual_seeds"))
        avg_mean_curve = _to_float_list(getattr(row, "average_reward_mean"))

        record = {
            "run_key": run_key,
            "depth": depth,
            "width": width,
            "learning_rate": float(getattr(row, "learning_rate")),
            "learning_rate_str": lr_str,
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
        os.path.join(os.path.dirname(__file__), os.pardir, "hyperparam_results.json")
    )
    with open(out_json, "w", encoding="utf-8") as json_file:
        json.dump(results_list, json_file, indent=2)
    print(f"Wrote {len(results_list)} records to {out_json}")


if __name__ == "__main__":
    if False:
        outputs_root = "/Users/frasermince/Programming/parquet_runs_linear_env_sweep"
        delete_path_metric_records(
            outputs_root, "nonstationary_dec_1_same_dir_structure"
        )
    else:
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
            outputs_root = (
                "/Users/frasermince/Programming/hidden_llava/larger_grid_runs_test/"
            )

            # Debug: Read first row from each parquet file (memory efficient)

            # Aggregate entire directory in one DuckDB call
            if False:

                try:
                    duck_df = aggregate_runs_duckdb(
                        outputs_root=outputs_root,
                        recursive_glob="**/*.parquet",
                        step_subsample=10,
                    )

                except Exception as e:
                    raise SystemExit(f"DuckDB aggregation failed: {e}") from e

                process_deep_hyperparams(duck_df)
            elif False:
                try:
                    aggregate_runs_linear_nonstationary_duckdb(
                        outputs_root=outputs_root,
                        recursive_glob="**/*.parquet",
                        start_from_combo=0,  # Start from beginning with new finer batching
                    )
                except Exception as e:
                    raise SystemExit(f"DuckDB aggregation failed: {e}") from e

                # Optional: Convert JSONL to JSON array format (for legacy compatibility)
                # The plotting code can now read JSONL directly, so this is optional
                # Uncomment the line below if you need the JSON file format
                # process_linear_hyperparams(jsonl_path)
                # print(f"\nJSONL file ready at: {jsonl_path}")
                # print("Plotting code can read this JSONL file directly.")
            else:
                try:
                    # outputs_root = "/Users/frasermince/Programming/hidden_llava/current_processed/parquet_runs_linear_beyond_basic_paths/agent_name_linear_qlearning/path_mode_MISLEADING_PATH"
                    # aggregate_runs_linear_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path="/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_MISLEADING_PATH_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_root = "/Users/frasermince/Programming/hidden_llava/current_processed/parquet_runs_linear_beyond_basic_paths/agent_name_linear_qlearning/path_mode_RANDOM_PATH"
                    # aggregate_runs_linear_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path="/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_RANDOM_PATH_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_root = "/Users/frasermince/Programming/hidden_llava/current_processed/parquet_runs_linear_beyond_basic_paths/agent_name_linear_qlearning/path_mode_SUBOPTIMAL_PATH"
                    # aggregate_runs_linear_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path="/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_SUBOPTIMAL_PATH_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_root = "/Users/frasermince/Programming/hidden_llava/current_processed/linear_landmarks_2/agent_name_linear_qlearning/path_mode_LANDMARKS"
                    output_path = "path_mode_LANDMARKS"
                    outputs_name = (
                        # "parquet_runs_linear_visited_cells_eps_schedule_v3_jan_15"
                        "landmarks_sweep_updated_exploration_feb_13"
                    )
                    outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/{outputs_name}/agent_name_linear_qlearning/{output_path}"
                    aggregate_runs_linear_duckdb(
                        outputs_root=outputs_root,
                        jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/{outputs_name}/{outputs_name}_{output_path}_aggregation_progress.jsonl",
                        recursive_glob="**/*.parquet",
                        step_subsample=10,
                    )
                    path_modes = [
                        "SHORTEST_PATH",
                        "NONE",
                        "SUBOPTIMAL_PATH",
                        "MISLEADING_PATH",
                        "RANDOM_PATH",
                    ]
                    # for path_mode in path_modes:
                    #     output_path = f"path_mode_{path_mode}"
                    #     outputs_name = (
                    #         # "parquet_runs_linear_visited_cells_eps_schedule_v3_jan_15"
                    #         "linear_paths_feb_9"
                    #     )
                    #     outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/{outputs_name}/agent_name_linear_qlearning/{output_path}"
                    #     aggregate_runs_linear_duckdb(
                    #         outputs_root=outputs_root,
                    #         jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/{outputs_name}/{outputs_name}_{output_path}_aggregation_progress.jsonl",
                    #         recursive_glob="**/*.parquet",
                    #         step_subsample=10,
                    #     )
                    # output_path = "path_mode_NONE"
                    # outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}/agent_name_linear_qlearning/{output_path}"
                    # aggregate_runs_linear_exploration_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}/{outputs_name}_{output_path}_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_name = (
                    #     "parquet_runs_linear_visited_cells_skinny_paths_counter"
                    # )
                    # outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}"
                    # aggregate_runs_linear_nonstationary_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}/{outputs_name}_PATH_MODE_VISITED_CELLS_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_name = "parquet_runs_linear_visited_cells_skinny_paths"
                    # outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}"
                    # aggregate_runs_linear_nonstationary_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}/{outputs_name}_PATH_MODE_VISITED_CELLS_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_name = "parquet_runs_linear_visited_cells_seasonal"
                    # outputs_root = f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}"
                    # aggregate_runs_linear_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path=f"/Users/frasermince/Programming/hidden_llava/current_processed/nonstationary_runs/{outputs_name}/{outputs_name}_PATH_MODE_VISITED_CELLS_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                    # outputs_root = "/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_SHORTEST_PATH_nov19_linear/path_mode_SHORTEST_PATH"
                    # aggregate_runs_linear_duckdb(
                    #     outputs_root=outputs_root,
                    #     jsonl_path="/Users/frasermince/Programming/hidden_llava/current_processed/path_mode_SHORTEST_PATH_aggregation_progress.jsonl",
                    #     recursive_glob="**/*.parquet",
                    #     step_subsample=10,
                    # )
                except Exception as e:
                    raise SystemExit(f"DuckDB aggregation failed: {e}") from e

                # process_linear_hyperparams(jsonl_path)
        else:
            # Legacy path disabled in favor of DuckDB aggregation demo.
            # Keeping historical code commented for reference.
            pass
