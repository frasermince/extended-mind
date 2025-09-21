import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import ast

try:
    import pyarrow.dataset as ds  # type: ignore[import]
except Exception:
    ds = None


# # @title Utility Code


# def walk_experiment_runs(base_dirs=None):
#     """
#     Recursively walks through the `experiment_runs` directory and yields
#     the path to each `metrics.pkl` file, along with the associated
#     hyperparameters extracted from the folder names.

#     This version uses explicit nested for-loops for each hyperparameter
#     layer. The traversal at each level is sorted by the value at that
#     level (ascending).
#     """
#     if base_dirs is None:
#         base_prefix = (
#             "/Users/frasermince/Programming/hidden_llava/more_exploration_runs/"
#         )
#         base_dirs = [
#             base_prefix + "generate_optimal_path_True",
#             base_prefix + "generate_optimal_path_False",
#         ]
#     for base_dir in base_dirs:
#         print(base_dir)
#         if not os.path.isdir(base_dir):
#             continue

#         # Sort learning rates by value (ascending)
#         lr_dirs = [
#             d
#             for d in os.listdir(base_dir)
#             if (
#                 d.startswith("learning_rate_")
#                 and os.path.isdir(os.path.join(base_dir, d))
#             )
#         ]

#         def lr_key(d):
#             try:
#                 return float(d.split("learning_rate_")[1])
#             except (IndexError, ValueError):
#                 return float("inf")

#         for lr_dir in sorted(lr_dirs, key=lr_key):
#             lr_path = os.path.join(base_dir, lr_dir)
#             learning_rate_string = lr_dir.split("learning_rate_")[1]
#             try:
#                 learning_rate = float(learning_rate_string)
#             except (IndexError, ValueError):
#                 continue

#             # Sort network depths by value (ascending)
#             depth_dirs = [
#                 d
#                 for d in os.listdir(lr_path)
#                 if d.startswith("network_depth_")
#                 and os.path.isdir(os.path.join(lr_path, d))
#             ]

#             def depth_key(d):
#                 try:
#                     return int(d.split("network_depth_")[1])
#                 except (IndexError, ValueError):
#                     return float("inf")

#             for depth_dir in sorted(depth_dirs, key=depth_key):
#                 depth_path = os.path.join(lr_path, depth_dir)
#                 try:
#                     network_depth = int(depth_dir.split("network_depth_")[1])
#                 except (IndexError, ValueError):
#                     continue

#                 # Sort network widths by value (ascending)
#                 width_dirs = [
#                     d
#                     for d in os.listdir(depth_path)
#                     if d.startswith("network_width_")
#                     and os.path.isdir(os.path.join(depth_path, d))
#                 ]

#                 def width_key(d):
#                     try:
#                         return int(d.split("network_width_")[1])
#                     except (IndexError, ValueError):
#                         return float("inf")

#                 for width_dir in sorted(width_dirs, key=width_key):
#                     width_path = os.path.join(depth_path, width_dir)
#                     try:
#                         network_width = int(width_dir.split("network_width_")[1])
#                     except (IndexError, ValueError):
#                         continue

#                     # Sort seeds by value (ascending)
#                     seed_dirs = [
#                         d
#                         for d in os.listdir(width_path)
#                         if d.startswith("seed_")
#                         and os.path.isdir(os.path.join(width_path, d))
#                     ]

#                     def seed_key(d):
#                         try:
#                             return int(d.split("seed_")[1])
#                         except (IndexError, ValueError):
#                             return float("inf")

#                     for seed_dir in sorted(seed_dirs, key=seed_key):
#                         seed_path = os.path.join(width_path, seed_dir)
#                         try:
#                             seed_num = int(seed_dir.split("seed_")[1])
#                         except (IndexError, ValueError):
#                             continue

#                         metrics_path = os.path.join(seed_path, "metrics.pkl")
#                         optimal_metrics_path = os.path.join(
#                             seed_path, "metrics_optimal_path.pkl"
#                         )

#                         if os.path.isfile(metrics_path):
#                             # Yield standard metrics
#                             yield {
#                                 "metrics_path": metrics_path,
#                                 "learning_rate": learning_rate,
#                                 "learning_rate_str": learning_rate_string,
#                                 "network_depth": network_depth,
#                                 "network_width": network_width,
#                                 "seed_num": seed_num,
#                                 "run_key": width_path,
#                                 "path_variant": "standard",
#                             }

#                         # Yield optimal-path metrics as a separate item
#                         if os.path.isfile(optimal_metrics_path):
#                             yield {
#                                 "metrics_path": optimal_metrics_path,
#                                 "learning_rate": learning_rate,
#                                 "learning_rate_str": learning_rate_string,
#                                 "network_depth": network_depth,
#                                 "network_width": network_width,
#                                 "seed_num": seed_num,
#                                 "run_key": os.path.join(width_path, "optimal_path"),
#                                 "path_variant": "optimal_path",
#                             }


# # Can skip if want to use existing results.json
# if __name__ == "__main__":
#     testing_on = True
#     accum_results = {}
#     depths = (2, 3)
#     widths = (4, 8, 16, 32)
#     missing_seeds = {}
#     learning_rates = ("1e-05", "5e-05", "0.0001", "0.0005", "0.001", "0.005", "0.01")
#     for optimal_path in (True, False):
#         for depth in depths:
#             for width in widths:
#                 for lr in learning_rates:
#                     missing_seeds[(depth, width, lr, optimal_path)] = set(range(31))

#     # Example usage: print all found metrics paths and hyperparams
#     for result in tqdm(walk_experiment_runs(), desc="Processing experiments"):
#         if testing_on and (
#             result["network_depth"] != 3
#             or result["network_width"] != 16
#             or result["path_variant"] != "standard"
#         ):
#             continue
#         try:
#             with open(result["metrics_path"], "rb") as f:
#                 metrics = pickle.load(f)
#         except Exception as e:
#             print(f"Error loading {result['metrics_path']}: {e}")
#             continue
#         df = pd.DataFrame(metrics["data"])
#         last_10_success_rate = (
#             df[df["metric"] == "charts/success_rate"].iloc[-10:]
#         ).value
#         last_10_success_rate = last_10_success_rate.values.tolist()
#         last_10_success_percentage = sum(last_10_success_rate) / len(
#             last_10_success_rate
#         )
#         last_10_average_episodic_reward = (
#             df[df["metric"] == "charts/average_episodic_reward"].iloc[-10:].value
#         )
#         last_10_average_episodic_reward = (
#             last_10_average_episodic_reward.values.tolist()
#         )
#         last_10_average_episodic_reward_percentage = sum(
#             last_10_average_episodic_reward
#         ) / len(last_10_average_episodic_reward)

#         reward_sum = 0
#         average_reward_per_timestep = []
#         for i, reward_per_timestep in enumerate(
#             df[df["metric"] == "reward_per_timestep"]["value"].values.tolist()
#         ):
#             reward_sum += reward_per_timestep

#             average_reward_per_timestep.append(reward_sum / (i + 1))
#         print(
#             result["network_depth"],
#             result["network_width"],
#             result["learning_rate_str"],
#             result["path_variant"] == "optimal_path",
#         )
#         missing_seeds[
#             (
#                 result["network_depth"],
#                 result["network_width"],
#                 result["learning_rate_str"],
#                 result["path_variant"] == "optimal_path",
#             )
#         ].remove(result["seed_num"])
#         if result["run_key"] not in accum_results:
#             accum_results[result["run_key"]] = {
#                 "depth": result["network_depth"],
#                 "width": result["network_width"],
#                 "items": [
#                     {
#                         "seed_num": result["seed_num"],
#                         "success_rate": last_10_success_percentage,
#                         "average_episodic_reward": (
#                             last_10_average_episodic_reward_percentage
#                         ),
#                         "average_reward_per_timestep": average_reward_per_timestep,
#                     }
#                 ],
#             }
#         else:
#             accum_results[result["run_key"]]["items"].append(
#                 {
#                     "seed_num": result["seed_num"],
#                     "success_rate": last_10_success_percentage,
#                     "average_episodic_reward": (
#                         last_10_average_episodic_reward_percentage
#                     ),
#                     "average_reward_per_timestep": average_reward_per_timestep,
#                 }
#             )
#     if not testing_on:
#         print("Map of missing seeds:")
#         for key, value in missing_seeds.items():
#             print(f"{key}: {value}")

#     def capacity_key(item):
#         return (item[1]["depth"], item[1]["width"])

#     sorted_results = sorted(accum_results.items(), key=capacity_key)
#     import json

#     results_list = []
#     max_learning_rate_auc = {}
#     for run_key, results in sorted_results:
#         # Sort results by capacity (assuming format "depthxwidth", e.g., "3x64")
#         items = results["items"]

#         avg_success_rate = sum(item["success_rate"] for item in items) / len(items)
#         avg_average_episodic_reward = sum(
#             item["average_episodic_reward"] for item in items
#         ) / len(items)

#         average_rewards_per_seed = np.array(
#             [item["average_reward_per_timestep"] for item in items]
#         )
#         import pdb

#         pdb.set_trace()

#         reward_averaged_over_seeds = np.mean(average_rewards_per_seed, axis=0)
#         total_reward_per_seed = np.sum(average_rewards_per_seed, axis=1)

#         subsampled_average_rewards_per_seed = average_rewards_per_seed[:, ::10]

#         standard_error_reward_curve = np.std(
#             subsampled_average_rewards_per_seed, ddof=1, axis=1
#         ) / np.sqrt(len(subsampled_average_rewards_per_seed[0]))

#         averaged_over_seeds_total_reward = np.mean(total_reward_per_seed)

#         total_reward_per_seed_standard_error = np.std(
#             total_reward_per_seed, ddof=1, axis=0
#         ) / np.sqrt(len(total_reward_per_seed))

#         print()
#         # Collect results for JSON output
#         record = {
#             "run_key": run_key,
#             "depth": results["depth"],
#             "width": results["width"],
#             "avg_success_rate": avg_success_rate,
#             "avg_average_episodic_reward": avg_average_episodic_reward,
#             "average_reward_area_under_curve": averaged_over_seeds_total_reward,
#             "average_reward_curve": list(subsampled_average_rewards_per_seed),
#             "average_reward_curve_standard_error": list(standard_error_reward_curve),
#             "per_seed_aucs": total_reward_per_seed,
#             "per_seed_auc_standard_error": total_reward_per_seed_standard_error,
#         }
#         # if run_key not in max_learning_rate_auc:
#         #     max_learning_rate_auc[run_key] = {
#         #         "record": record,
#         #         "max_auc": average_reward_area_under_curve,
#         #     }
#         # elif (
#         #     average_reward_area_under_curve > max_learning_rate_auc[run_key]["max_auc"]
#         # ):
#         #     max_learning_rate_auc[run_key] = {
#         #         "record": record,
#         #         "max_auc": average_reward_area_under_curve,
#         #     }
#     for key, value in max_learning_rate_auc.items():
#         results_list.append(value["record"])

#     # Write results to JSON file in the same order
#     if not testing_on:
#         with open(
#             "results_more_exploration.json",
#             "w",
#             encoding="utf-8",
#         ) as json_file:
#             json.dump(results_list, json_file, indent=2)


# def _parse_dense_features_to_depth_width(dense_features_str):
#     """Parse a string like "[8, 8]" to (depth=2, width=8).

#     If parsing fails, returns (None, None).
#     """
#     try:
#         features = ast.literal_eval(dense_features_str)
#         if isinstance(features, (list, tuple)) and len(features) > 0:
#             return int(len(features)), int(features[0])
#     except Exception:
#         pass
#     return None, None


def _list_results_parquet_roots(outputs_root):
    roots = []
    if not os.path.isdir(outputs_root):
        return roots
    for file in sorted(os.listdir(outputs_root)):
        if file.endswith(".parquet"):
            roots.append(file)
    return roots


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


if __name__ == "__main__":
    # New path: read from nested Parquet dataset
    use_parquet = True
    if use_parquet:
        outputs_root = (
            "/Users/frasermince/Programming/hidden_llava/parquet_metrics/metrics"
        )
        try:
            runs_df = load_runs_from_parquet(
                outputs_root=outputs_root,
                exp_group_id=None,  # set to a specific id for pruning
                date=None,  # set to YYYY-MM-DD to focus on a day
            )
        except Exception as e:
            raise SystemExit(f"Failed loading Parquet results: {e}")

        # Example aggregation similar to previous flow
        # Group by capacity, lr, and path variant, aggregate across seeds
        group_cols = [
            "network_depth",
            "network_width",
            "learning_rate",
            "optimal_path_available",
            "seed",
        ]

        grouped = runs_df.sort_values(by="step")[
            runs_df["metric"] == "reward_per_timestep"
        ].groupby(group_cols)

        def average_per_timestep(x):
            reward_sum = 0
            average_reward_per_timestep = []
            for i, reward_per_timestep in enumerate(x.value):
                reward_sum += reward_per_timestep

                average_reward_per_timestep.append(reward_sum / (i + 1))
            return np.array(average_reward_per_timestep)

        average_rewards_per_timestep = grouped.apply(average_per_timestep).reset_index()

        # Print a small preview for sanity
        print("Aggregated (by depth,width,lr,path):")
        print(agg.head(20).to_string(index=False))
    else:
        # Legacy path: read pickled metrics by walking folders
        # Can skip if want to use existing results.json
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

        # Example usage: print all found metrics paths and hyperparams
        for result in tqdm(walk_experiment_runs(), desc="Processing experiments"):
            if testing_on and (
                result["network_depth"] != 3
                or result["network_width"] != 16
                or result["path_variant"] != "standard"
            ):
                continue
            try:
                with open(result["metrics_path"], "rb") as f:
                    metrics = pickle.load(f)
            except Exception as e:
                print(f"Error loading {result['metrics_path']}: {e}")
                continue
            df = pd.DataFrame(metrics["data"])
            last_10_success_rate = (
                df[df["metric"] == "charts/success_rate"].iloc[-10:]
            ).value
            last_10_success_rate = last_10_success_rate.values.tolist()
            last_10_success_percentage = sum(last_10_success_rate) / len(
                last_10_success_rate
            )
            last_10_average_episodic_reward = (
                df[df["metric"] == "charts/average_episodic_reward"].iloc[-10:].value
            )
            last_10_average_episodic_reward = (
                last_10_average_episodic_reward.values.tolist()
            )
            last_10_average_episodic_reward_percentage = sum(
                last_10_average_episodic_reward
            ) / len(last_10_average_episodic_reward)

            reward_sum = 0
            average_reward_per_timestep = []
            for i, reward_per_timestep in enumerate(
                df[df["metric"] == "reward_per_timestep"]["value"].values.tolist()
            ):
                reward_sum += reward_per_timestep

                average_reward_per_timestep.append(reward_sum / (i + 1))
            print(
                result["network_depth"],
                result["network_width"],
                result["learning_rate_str"],
                result["path_variant"] == "optimal_path",
            )
            missing_seeds[
                (
                    result["network_depth"],
                    result["network_width"],
                    result["learning_rate_str"],
                    result["path_variant"] == "optimal_path",
                )
            ].remove(result["seed_num"])
            if result["run_key"] not in accum_results:
                accum_results[result["run_key"]] = {
                    "depth": result["network_depth"],
                    "width": result["network_width"],
                    "items": [
                        {
                            "seed_num": result["seed_num"],
                            "success_rate": last_10_success_percentage,
                            "average_episodic_reward": (
                                last_10_average_episodic_reward_percentage
                            ),
                            "average_reward_per_timestep": average_reward_per_timestep,
                        }
                    ],
                }
            else:
                accum_results[result["run_key"]]["items"].append(
                    {
                        "seed_num": result["seed_num"],
                        "success_rate": last_10_success_percentage,
                        "average_episodic_reward": (
                            last_10_average_episodic_reward_percentage
                        ),
                        "average_reward_per_timestep": average_reward_per_timestep,
                    }
                )
        if not testing_on:
            print("Map of missing seeds:")
            for key, value in missing_seeds.items():
                print(f"{key}: {value}")

        def capacity_key(item):
            return (item[1]["depth"], item[1]["width"])

        sorted_results = sorted(accum_results.items(), key=capacity_key)
        import json as _json

        results_list = []
        max_learning_rate_auc = {}
        for run_key, results in sorted_results:
            # Sort results by capacity (assuming format "depthxwidth")
            items = results["items"]

            avg_success_rate = sum(item["success_rate"] for item in items) / len(items)
            avg_average_episodic_reward = sum(
                item["average_episodic_reward"] for item in items
            ) / len(items)

            average_rewards_per_seed = np.array(
                [item["average_reward_per_timestep"] for item in items]
            )
            import pdb

            pdb.set_trace()

            reward_averaged_over_seeds = np.mean(average_rewards_per_seed, axis=0)
            total_reward_per_seed = np.sum(average_rewards_per_seed, axis=1)

            subsampled_average_rewards_per_seed = average_rewards_per_seed[:, ::10]

            standard_error_reward_curve = np.std(
                subsampled_average_rewards_per_seed, ddof=1, axis=1
            ) / np.sqrt(len(subsampled_average_rewards_per_seed[0]))

            averaged_over_seeds_total_reward = np.mean(total_reward_per_seed)

            total_reward_per_seed_standard_error = np.std(
                total_reward_per_seed, ddof=1, axis=0
            ) / np.sqrt(len(total_reward_per_seed))

            print()
            # Collect results for JSON output
            record = {
                "run_key": run_key,
                "depth": results["depth"],
                "width": results["width"],
                "avg_success_rate": avg_success_rate,
                "avg_average_episodic_reward": avg_average_episodic_reward,
                "average_reward_area_under_curve": averaged_over_seeds_total_reward,
                "average_reward_curve": list(subsampled_average_rewards_per_seed),
                "average_reward_curve_standard_error": list(
                    standard_error_reward_curve
                ),
                "per_seed_aucs": total_reward_per_seed,
                "per_seed_auc_standard_error": total_reward_per_seed_standard_error,
            }
            # old selection code omitted
        for key, value in max_learning_rate_auc.items():
            results_list.append(value["record"])
        if not testing_on:
            with open(
                "results_more_exploration.json",
                "w",
                encoding="utf-8",
            ) as json_file:
                _json.dump(results_list, json_file, indent=2)
