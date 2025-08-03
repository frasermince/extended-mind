import os
import pickle
import pandas as pd


def walk_experiment_runs(base_dir="experiment_runs"):
    """
    Recursively walks through the experiment_runs directory and yields the path to each metrics.pkl file,
    along with the associated hyperparameters extracted from the folder names.
    This version uses explicit nested for loops for each hyperparameter layer.
    The traversal at each level is sorted by the value at that level (ascending).
    """
    if not os.path.isdir(base_dir):
        return

    # Sort learning rates by value (ascending)
    lr_dirs = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("learning_rate_") and os.path.isdir(os.path.join(base_dir, d))
    ]

    def lr_key(d):
        try:
            return float(d.split("learning_rate_")[1])
        except Exception:
            return float("inf")

    for lr_dir in sorted(lr_dirs, key=lr_key):
        lr_path = os.path.join(base_dir, lr_dir)
        try:
            learning_rate = float(lr_dir.split("learning_rate_")[1])
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
            except Exception:
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
                except Exception:
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
                    except Exception:
                        return float("inf")

                for seed_dir in sorted(seed_dirs, key=seed_key):
                    seed_path = os.path.join(width_path, seed_dir)
                    try:
                        seed_num = int(seed_dir.split("seed_")[1])
                    except (IndexError, ValueError):
                        continue

                    metrics_path = os.path.join(seed_path, "metrics.pkl")

                    if os.path.isfile(metrics_path):
                        yield {
                            "metrics_path": metrics_path,
                            "learning_rate": learning_rate,
                            "network_depth": network_depth,
                            "network_width": network_width,
                            "seed_num": seed_num,
                            "run_key": width_path,
                        }


if __name__ == "__main__":
    import pprint

    accum_results = {}

    # Example usage: print all found metrics paths and hyperparams
    for result in walk_experiment_runs():
        with open(result["metrics_path"], "rb") as f:
            metrics = pickle.load(f)
        df = pd.DataFrame(metrics["data"])
        last_10_success_rate = (
            df[df["metric"] == "charts/success_rate"].iloc[-10:].value
        )
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
        # pprint.pprint(f"run_key: {result['run_key']}")
        if result["run_key"] not in accum_results:
            accum_results[result["run_key"]] = {
                "depth": result["network_depth"],
                "width": result["network_width"],
                "items": [
                    {
                        "success_rate": last_10_success_percentage,
                        "average_episodic_reward": last_10_average_episodic_reward_percentage,
                    }
                ],
            }
        else:
            accum_results[result["run_key"]]["items"].append(
                {
                    "success_rate": last_10_success_percentage,
                    "average_episodic_reward": last_10_average_episodic_reward_percentage,
                }
            )
        # pprint.pprint(result)
        # pprint.pprint(last_10_success_rate)

    def capacity_key(result):
        return (result[1]["depth"], result[1]["width"])

    sorted_results = sorted(accum_results.items(), key=capacity_key)
    import json

    results_list = []
    for run_key, results in sorted_results:
        # Sort results by capacity (assuming format "depthxwidth", e.g., "3x64")
        items = accum_results[run_key]["items"]
        avg_success_rate = sum(item["success_rate"] for item in items) / len(items)
        avg_average_episodic_reward = sum(
            item["average_episodic_reward"] for item in items
        ) / len(items)
        pprint.pprint(f"run_key: {run_key}")
        pprint.pprint(f"avg_success_rate: {avg_success_rate}")
        pprint.pprint(f"avg_average_episodic_reward: {avg_average_episodic_reward}")
        print()
        # Collect results for JSON output
        results_list.append(
            {
                "run_key": run_key,
                "depth": results["depth"],
                "width": results["width"],
                "avg_success_rate": avg_success_rate,
                "avg_average_episodic_reward": avg_average_episodic_reward,
            }
        )

    # Write results to JSON file in the same order
    with open("hyperparam_results.json", "w") as json_file:
        json.dump(results_list, json_file, indent=2)
