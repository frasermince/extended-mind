import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import pprint
import json


def walk_experiment_runs(base_dirs=None):
    """
    Recursively walks through the `experiment_runs` directory and yields
    the path to each `metrics.pkl` file, along with the associated
    hyperparameters extracted from the folder names.

    This version uses explicit nested for-loops for each hyperparameter
    layer. The traversal at each level is sorted by the value at that
    level (ascending).
    """
    if base_dirs is None:
        base_prefix = "/Users/frasermince/Downloads/runs 2/"
        base_dirs = [
            base_prefix + "generate_optimal_path_True",
            base_prefix + "generate_optimal_path_False",
        ]

    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            return

        # Sort learning rates by value (ascending)
        lr_dirs = [
            d
            for d in os.listdir(base_dir)
            if (
                d.startswith("learning_rate_")
                and os.path.isdir(os.path.join(base_dir, d))
            )
        ]

        def lr_key(d):
            try:
                return float(d.split("learning_rate_")[1])
            except (IndexError, ValueError):
                return float("inf")

        for lr_dir in sorted(lr_dirs, key=lr_key):
            lr_path = os.path.join(base_dir, lr_dir)
            learning_rate_string = lr_dir.split("learning_rate_")[1]
            try:
                learning_rate = float(learning_rate_string)
            except (IndexError, ValueError):
                continue

            # Sort network depths by value (ascending)
            depth_dirs = [
                d
                for d in os.listdir(lr_path)
                if d.startswith("network_depth_")
                and os.path.isdir(os.path.join(lr_path, d))
            ]

            def depth_key(d):
                try:
                    return int(d.split("network_depth_")[1])
                except (IndexError, ValueError):
                    return float("inf")

            for depth_dir in sorted(depth_dirs, key=depth_key):
                depth_path = os.path.join(lr_path, depth_dir)
                try:
                    network_depth = int(depth_dir.split("network_depth_")[1])
                except (IndexError, ValueError):
                    continue

                # Sort network widths by value (ascending)
                width_dirs = [
                    d
                    for d in os.listdir(depth_path)
                    if d.startswith("network_width_")
                    and os.path.isdir(os.path.join(depth_path, d))
                ]

                def width_key(d):
                    try:
                        return int(d.split("network_width_")[1])
                    except (IndexError, ValueError):
                        return float("inf")

                for width_dir in sorted(width_dirs, key=width_key):
                    width_path = os.path.join(depth_path, width_dir)
                    try:
                        network_width = int(width_dir.split("network_width_")[1])
                    except (IndexError, ValueError):
                        continue

                    # Sort seeds by value (ascending)
                    seed_dirs = [
                        d
                        for d in os.listdir(width_path)
                        if d.startswith("seed_")
                        and os.path.isdir(os.path.join(width_path, d))
                    ]

                    def seed_key(d):
                        try:
                            return int(d.split("seed_")[1])
                        except (IndexError, ValueError):
                            return float("inf")

                    for seed_dir in sorted(seed_dirs, key=seed_key):
                        seed_path = os.path.join(width_path, seed_dir)
                        try:
                            seed_num = int(seed_dir.split("seed_")[1])
                        except (IndexError, ValueError):
                            continue

                        metrics_path = os.path.join(seed_path, "metrics.pkl")
                        optimal_metrics_path = os.path.join(
                            seed_path,
                            "metrics_optimal_path.pkl",
                        )

                        if os.path.isfile(metrics_path):
                            # Yield standard metrics
                            yield {
                                "metrics_path": metrics_path,
                                "learning_rate": learning_rate,
                                "network_depth": network_depth,
                                "network_width": network_width,
                                "seed_num": seed_num,
                                "run_key": width_path,
                                "path_variant": "standard",
                                "learning_rate_str": learning_rate_string,
                            }

                        # Yield optimal-path metrics as a separate item
                        if os.path.isfile(optimal_metrics_path):
                            yield {
                                "metrics_path": optimal_metrics_path,
                                "learning_rate": learning_rate,
                                "network_depth": network_depth,
                                "network_width": network_width,
                                "seed_num": seed_num,
                                "run_key": os.path.join(
                                    width_path,
                                    "optimal_path",
                                ),
                                "path_variant": "optimal_path",
                                "learning_rate_str": learning_rate_string,
                            }


# Can skip if want to use existing results.json
if __name__ == "__main__":
    accum_results = {}
    depths = (2, 3)
    widths = (4, 8, 16, 32)
    missing_seeds = {}
    learning_rates = ("1e-05", "5e-05", "0.0001", "0.0005", "0.001", "0.005", "0.01")
    for optimal_path in (True, False):
        for depth in depths:
            for width in widths:
                for lr in learning_rates:
                    missing_seeds[(depth, width, lr, optimal_path)] = set(range(30))

    # Example usage: print all found metrics paths and hyperparams
    for result in tqdm(walk_experiment_runs(), desc="Processing experiments"):
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
                    "success_rate": last_10_success_percentage,
                    "average_episodic_reward": (
                        last_10_average_episodic_reward_percentage
                    ),
                    "average_reward_per_timestep": average_reward_per_timestep,
                }
            )

    print("Map of missing seeds:")
    for key, value in missing_seeds.items():
        print(f"{key}: {value}")

    def capacity_key(item):
        return (item[1]["depth"], item[1]["width"])

    sorted_results = sorted(accum_results.items(), key=capacity_key)
    import json

    results_list = []
    max_learning_rate_auc = {}
    for run_key, results in tqdm(sorted_results, desc="Processing results"):
        # Sort results by capacity (assuming format "depthxwidth", e.g., "3x64")
        items = results["items"]
        avg_success_rate = sum(item["success_rate"] for item in items) / len(items)
        avg_average_episodic_reward = sum(
            item["average_episodic_reward"] for item in items
        ) / len(items)

        all_avg_episodic_reward = []
        average_reward_per_timestep = []
        for i in range(len(items[0]["average_reward_per_timestep"])):
            values_at_i = [item["average_reward_per_timestep"][i] for item in items]
            all_avg_episodic_reward.append(values_at_i)
            average_reward_per_timestep.append(np.mean(values_at_i))

        standard_error = np.std(all_avg_episodic_reward, ddof=1, axis=1) / np.sqrt(
            len(all_avg_episodic_reward[0])
        )
        average_reward_area_under_curve = np.trapezoid(
            average_reward_per_timestep, dx=1
        ).item()
        pprint.pprint(f"run_key: {run_key}")
        pprint.pprint(f"avg_success_rate: {avg_success_rate}")
        pprint.pprint(f"avg_average_episodic_reward: {avg_average_episodic_reward}")
        print()
        # Collect results for JSON output
        record = {
            "run_key": run_key,
            "depth": results["depth"],
            "width": results["width"],
            "avg_success_rate": avg_success_rate,
            "avg_average_episodic_reward": avg_average_episodic_reward,
            "average_reward_area_under_curve": average_reward_area_under_curve,
            "average_reward_curve": list(average_reward_per_timestep),
            "average_reward_curve_standard_error": list(standard_error),
            # "all_avg_episodic_reward": all_avg_episodic_reward,
        }
        if run_key not in max_learning_rate_auc:
            max_learning_rate_auc[run_key] = {
                "record": record,
                "max_auc": average_reward_area_under_curve,
            }
        elif (
            average_reward_area_under_curve > max_learning_rate_auc[run_key]["max_auc"]
        ):
            max_learning_rate_auc[run_key] = {
                "record": record,
                "max_auc": average_reward_area_under_curve,
            }
    for key, value in max_learning_rate_auc.items():
        results_list.append(value["record"])

    # Write results to JSON file in the same order
    with open(
        "results_max.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(results_list, json_file, indent=2)
