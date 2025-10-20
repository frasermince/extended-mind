import yaml 
from pathlib import Path
import itertools
import os
import numpy as np
import pandas as pd
import pickle
import copy
import json

def construct_run_path(tc, run_folder=None):
    if run_folder is None:
        run_folder = "/home/esraa1/scratch/extended-mind/runs"
    agent_name = tc['agent_name']
    if(agent_name == "main_dqn"):
        path_mode = tc['path_mode']
        learning_rate = tc['training.learning_rate']
        dense_features = tc['training.dense_features']
        
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        if learning_rate < 0.0001:
            learning_rate_str = f"{learning_rate:.0e}"
        else:
            learning_rate_str = f"{learning_rate:.6f}".rstrip('0').rstrip('.')
        
        network_depth = len(dense_features)
        network_width = dense_features[0]
        
        run_path = os.path.join(
            run_folder,
            f"path_mode_{path_mode}",
            f"learning_rate_{learning_rate_str}",
            f"network_depth_{network_depth}",
            f"network_width_{network_width}",
        )
        
    elif(agent_name == "linear_qlearning"):
        agent_name = tc['agent_name']
        path_mode = tc['path_mode']
        step_size = tc['training.step_size']
        agent_pixel_view_edge_dim = tc['training.agent_pixel_view_edge_dim']
        
        if isinstance(step_size, str):
            step_size = float(step_size)
        
        if step_size < 0.0001:
            step_size_str = f"{step_size:.0e}"
        else:
            step_size_str = f"{step_size:.6f}".rstrip('0').rstrip('.')
        
        
        run_path = os.path.join(
        run_folder,
        f"agent_name_{agent_name}",
        f"path_mode_{path_mode}",
        f"step_size_{step_size_str}",
        f"agent_pixel_view_edge_dim_{agent_pixel_view_edge_dim}",
    )
    
    else:
        raise ValueError(f"Agent name {agent_name} not supported")
        
    
    return run_path

def dict_permutations(input_dict):
    keys = list(input_dict.keys())
    values_lists = list(input_dict.values())
    
    processed_values = []
    for i, values in enumerate(values_lists):
        if isinstance(values, list):
            processed_values.append(values)
        else:
            processed_values.append([values])
    
    for combination in itertools.product(*processed_values):
        yield dict(zip(keys, combination))

def load_config(config_path):
    return yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)



def generate_hyperparam_task_configs(hyperparams_to_sweep, default_hyperparams):
    sweep_dict = {}

    for key, values in default_hyperparams.items():
        if isinstance(values, dict):
            for nested_key, nested_value in values.items():
                flat_key = f"{key}.{nested_key}"
                if flat_key in hyperparams_to_sweep:
                    sweep_dict[flat_key] = hyperparams_to_sweep[flat_key]
                else:
                    sweep_dict[flat_key] = nested_value
        else:
            if key in hyperparams_to_sweep:
                sweep_dict[key] = hyperparams_to_sweep[key]
            else:
                sweep_dict[key] = values
    
    task_confs = dict_permutations(sweep_dict)
    return task_confs

def task_conf_to_path(task_conf, run_folder, agent_name):
    if agent_name == "linear_qlearning":
        return construct_run_path(task_conf, run_folder)    
    elif agent_name == "main_dqn":
        return construct_run_path(task_conf, run_folder)
    else:
        raise ValueError(f"Agent name {agent_name} not supported")

def read_config_seed_file(config_path):

    with open(config_path, 'rb') as f:
        run_data = pickle.load(f)
    
    df = pd.DataFrame(run_data['data'])
    reward_values = df[df['metric'] == 'reward_per_timestep']['value'].values
    return reward_values

def compute_total_reward_across_seeds(config_results_path, num_seeds):
    total_rewards = []
    print(f"Computing total reward across seeds for {config_results_path}")
    for seed_folder in os.listdir(config_results_path):
        seed_results_path = os.path.join(config_results_path, seed_folder, "metrics.pkl")
        reward_array = read_config_seed_file(seed_results_path)
        print(f"  {seed_folder}: {len(reward_array)} reward values")
        total_rewards.append(reward_array.sum())
    
    # Calculate mean
    mean_reward = sum(total_rewards) / len(total_rewards)
    
    # Calculate standard error
    if len(total_rewards) <= 1:
        std_error = 0
    else:
        variance = sum((x - mean_reward) ** 2 for x in total_rewards) / (len(total_rewards) - 1)
        std_error = np.sqrt(variance) / np.sqrt(len(total_rewards))
    
    return mean_reward, std_error, total_rewards

def group_task_configs_by_hyperparam(task_configs, hyperparam):
    hyperparam_to_task_configs = {}
    for tc in task_configs:
        hyperparam_value = tc[hyperparam]
        hyperparam_to_task_configs[hyperparam_value] = hyperparam_to_task_configs.get(hyperparam_value, []) + [tc]
    return hyperparam_to_task_configs


def main():
    hyperparam_sweeps_config_path = "/home/esraa1/scratch/extended-mind/run_configs/lr_sweep_linear_qlearning_config.yaml"
    default_config_path = "linear_qlearning_config.yaml"
    run_folder = "/home/esraa1/scratch/extended-mind/runs"
    json_results_path = "/home/esraa1/scratch/extended-mind/hyperparam_selection_results.json"
    num_seeds = 30
    hyperparam_sweeps_config = load_config(hyperparam_sweeps_config_path)
    default_config = load_config(default_config_path)
    task_configs = list(generate_hyperparam_task_configs(hyperparam_sweeps_config, default_config))
    
    pathmode_grouped_task_configs = group_task_configs_by_hyperparam(task_configs, "path_mode")
    agent_pixel_view_edge_nonpath = group_task_configs_by_hyperparam(pathmode_grouped_task_configs["NONE"], "training.agent_pixel_view_edge_dim")
    agent_pixel_view_edge_path = group_task_configs_by_hyperparam(pathmode_grouped_task_configs["SHORTEST_PATH"], "training.agent_pixel_view_edge_dim")
    

    # compute reward for each non path task config
    nonpath_conf_to_total_reward = {}

    for view_edge_dim, task_configs in agent_pixel_view_edge_nonpath.items():
        for tc in task_configs:
            hyperparam_path = task_conf_to_path(tc, run_folder, "linear_qlearning")
            mean, std_error, data_points = compute_total_reward_across_seeds(hyperparam_path, num_seeds)
            nonpath_conf_to_total_reward[hyperparam_path] = (mean, std_error, data_points)

    
    # compute reward for each path task config
    path_conf_to_total_reward = {}
    for view_edge_dim, task_configs in agent_pixel_view_edge_path.items():
        for tc in task_configs:
            hyperparam_path = task_conf_to_path(tc, run_folder, "linear_qlearning")
            mean, std_error, data_points = compute_total_reward_across_seeds(hyperparam_path, num_seeds)
            path_conf_to_total_reward[hyperparam_path] = (mean, std_error, data_points)
    

    # select best hyperparam config per view edge dim in non path
    nonpath_best_view_edge_dim_conf_to_total_reward = {}
    for view_edge_dim, task_configs in agent_pixel_view_edge_nonpath.items():
        best_conf = max(task_configs, key=lambda x: nonpath_conf_to_total_reward[task_conf_to_path(x, "/home/esraa1/scratch/extended-mind/runs", "linear_qlearning")][0])
        best_path = task_conf_to_path(best_conf, run_folder, "linear_qlearning")
        nonpath_best_view_edge_dim_conf_to_total_reward[view_edge_dim] = nonpath_conf_to_total_reward[best_path]
    
    # select best hyperparam config per view edge dim in path
    path_best_view_edge_dim_conf_to_total_reward = {}
    for view_edge_dim, task_configs in agent_pixel_view_edge_path.items():
        best_conf = max(task_configs, key=lambda x: path_conf_to_total_reward[task_conf_to_path(x, "/home/esraa1/scratch/extended-mind/runs", "linear_qlearning")][0])
        best_path = task_conf_to_path(best_conf, run_folder, "linear_qlearning")
        path_best_view_edge_dim_conf_to_total_reward[view_edge_dim] = path_conf_to_total_reward[best_path]


    # pretty print the results
    print("Non path best view edge dim to total reward (mean ± std_error):")
    for view_edge_dim, (total_reward, std_error, data_points) in nonpath_best_view_edge_dim_conf_to_total_reward.items():
        print(f"{view_edge_dim}: {total_reward:.1f} ± {std_error:.1f}")
        print(f"  Data points: {data_points}")
    
    print("\nPath best view edge dim to total reward (mean ± std_error):")
    for view_edge_dim, (total_reward, std_error, data_points) in path_best_view_edge_dim_conf_to_total_reward.items():
        print(f"{view_edge_dim}: {total_reward:.1f} ± {std_error:.1f}")
        print(f"  Data points: {data_points}")

    # write results to a json file
    results = {
        "nonpath_best_view_edge_dim_conf_to_total_reward": nonpath_best_view_edge_dim_conf_to_total_reward,
        "path_best_view_edge_dim_conf_to_total_reward": path_best_view_edge_dim_conf_to_total_reward
    }
    with open(json_results_path, "w") as f:
        json.dump(results, f)
    print(f"\nResults written to {json_results_path}")



    

if __name__ == "__main__":
    main()