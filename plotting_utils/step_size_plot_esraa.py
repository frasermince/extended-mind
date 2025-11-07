import matplotlib.pyplot as plt
import matplotlib
import yaml
import itertools
import pickle
import pandas as pd
import os
import numpy as np
from matplotlib.ticker import NullLocator

def read_config(conf_file_path):
    with open(conf_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def dict_permutations(input_dict):
    keys = list(input_dict.keys())
    values_lists = list(input_dict.values())
    
    # Convert singular values to lists, but keep actual lists as-is for unpacking
    processed_values = []
    for i, values in enumerate(values_lists):
        key = keys[i]
        if isinstance(values, list):
            # Unpack all lists (including dense_features) so each item becomes a separate value
            processed_values.append(values)
        else:
            processed_values.append([values])
    
    for combination in itertools.product(*processed_values):
        yield dict(zip(keys, combination))


def group_task_configs_by_agent_pixel_view_edge_dimension(task_configs):
    grouped_task_configs = {}
    print(f"Grouping task configs by agent pixel view edge dimension...")  
    for tc in task_configs:
        print(f"TC: {tc}")
        agent_pixel_view_edge_dim = tc['training.agent_pixel_view_edge_dim']
        if agent_pixel_view_edge_dim not in grouped_task_configs:
            grouped_task_configs[agent_pixel_view_edge_dim] = []
        grouped_task_configs[agent_pixel_view_edge_dim].append(tc)
    return grouped_task_configs


def group_task_configs_by_step_size(task_configs):
    print(f"Grouping task configs by step size...")
    
    grouped_task_configs = {}
    for tc in task_configs:
        print(f"TC: {tc}")
        step_size = tc['training.step_size']
        if step_size not in grouped_task_configs:
            grouped_task_configs[step_size] = []
        grouped_task_configs[step_size].append(tc)
    return grouped_task_configs

def merge_default_and_sweep_config(default_hyperparams, hyperparams_to_sweep):
    
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
    return sweep_dict


def filter_task_configs_by_path_mode(task_configs, path_mode):
    return [tc for tc in task_configs if tc['path_mode'] == path_mode]


def construct_run_path(tc, run_folder=None):
    if run_folder is None:
        # run_folder = "/home/esraa1/scratch/extended-mind/runs"
        run_folder = "/home/esraa1/scratch/extended-mind/linear_runs_oct13/runs"
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

def read_config_seed_file(config_path):

    with open(config_path, 'rb') as f:
        run_data = pickle.load(f)
    
    df = pd.DataFrame(run_data['data'])
    reward_values = df[df['metric'] == 'reward_per_timestep']['value'].values
    return reward_values

def compute_total_reward_across_seeds(config_results_path):
    total_rewards = []
    print(f"Computing total reward across seeds for {config_results_path}")
    seed_folders = os.listdir(config_results_path)
    for seed_folder in seed_folders:
        seed_results_path = os.path.join(config_results_path, seed_folder, "metrics.pkl")
        reward_array = read_config_seed_file(seed_results_path)
        print(f"  {seed_folder}: {len(reward_array)} reward values")
        total_rewards.append(reward_array.sum())
    
    mean_reward = sum(total_rewards) / len(total_rewards)
    if len(total_rewards) <= 1:
        std_error = 0
    else:
        variance = sum((x - mean_reward) ** 2 for x in total_rewards) / (len(total_rewards) - 1)
        std_error = np.sqrt(variance) / np.sqrt(len(total_rewards))
    
    return mean_reward, std_error, total_rewards

def compute_total_reward_across_seeds_for_group_task_configs(group_task_configs):
    reward_values = []
    for tc in group_task_configs:
        run_path = construct_run_path(tc)
        mean_reward, _, _ = compute_total_reward_across_seeds(run_path)
        print(f"Mean reward for run path {run_path}: {mean_reward}")
        reward_values.append(mean_reward)
    return sum(reward_values) / len(reward_values)


def get_colorblind_friendly_colors(n):
    """Generate n colorblind-friendly colors. Maximum 10 colors supported."""
    if n > 10:
        raise ValueError(f"Cannot generate more than 10 colorblind-friendly colors. Requested: {n}")
    

    colors = [
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # bluish green
        '#0072B2',  # blue
        '#D55E00',  # vermillion
        '#CC79A7',  # reddish purple
        '#000000',  # black
        '#999999',  # gray
        '#E5E5E5',  # light gray
        '#8B4513',  # brown 
    ]
    
    return colors[:n]

def plot_step_size_vs_total_reward(ax, task_configs):
    
    path_mode = next(iter(task_configs.values()))[0].get('path_mode')
    for configs in task_configs.values():
        if any(tc.get('path_mode') != path_mode for tc in configs):
            raise ValueError(f"Path mode mismatch: expected {path_mode}")
    
    all_step_sizes = []
    sorted_dims = sorted(task_configs.keys())
    num_network_sizes = len(sorted_dims)
    color_palette_list = get_colorblind_friendly_colors(num_network_sizes)
    
    for agent_view_edge_dim in sorted_dims:
        task_configs_for_view_edge_dim = task_configs[agent_view_edge_dim]
        step_size_grouped_task_configs = group_task_configs_by_step_size(task_configs_for_view_edge_dim)
        
        step_sizes = []
        mean_rewards = []
        
        for step_size_str, task_configs_for_step_size in step_size_grouped_task_configs.items():
            step_size_float = float(step_size_str)
            mean_reward = compute_total_reward_across_seeds_for_group_task_configs(task_configs_for_step_size)
            print(f"Mean reward for step size {step_size_str} and agent view edge dim {agent_view_edge_dim}: {mean_reward}")
            step_sizes.append(step_size_float)
            mean_rewards.append(mean_reward)
            all_step_sizes.append(step_size_float)
        
        sorted_indices = np.argsort(step_sizes)
        step_sizes = [step_sizes[i] for i in sorted_indices]
        mean_rewards = [mean_rewards[i] for i in sorted_indices]
        
        # Find peak (maximum reward)
        peak_idx = np.argmax(mean_rewards)
        peak_step_size = step_sizes[peak_idx]
        peak_reward = mean_rewards[peak_idx]
        
        agent_capacity = agent_view_edge_dim * agent_view_edge_dim
        network_size_label = f"{agent_capacity}"
        color_idx = sorted_dims.index(agent_view_edge_dim)
        line_color = color_palette_list[color_idx]
        ax.plot(step_sizes, mean_rewards, marker='o', label=network_size_label, 
                markersize=6, color=line_color)
        ax.plot(peak_step_size, peak_reward, marker='*', markersize=18, 
                color=line_color, zorder=5)

    ax.set_xlabel("Step-size")
    ax.set_ylabel("Total Reward")
    ax.set_xscale('log')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    min_step_size = min(all_step_sizes)
    max_step_size = max(all_step_sizes)
    ax.set_xlim(left=min_step_size * 0.8, right=max_step_size * 1.2)
    
    unique_step_sizes = sorted(set(all_step_sizes))
    ax.set_xticks(unique_step_sizes)
    
    tick_labels = []
    for step_size in unique_step_sizes:
        if step_size < 0.0001:
            tick_labels.append(f"{step_size:.0e}")
        else:
            tick_labels.append(f"{step_size:.6f}".rstrip('0').rstrip('.'))
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.xaxis.set_minor_locator(NullLocator())
    
   
    ax.legend(title="Agent Capacity", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)

    if path_mode == "NONE":
        ax.set_title("No path")
    elif path_mode == "SHORTEST_PATH":
        ax.set_title("Shortest Path")
    else:
        raise ValueError(f"Path mode {path_mode} not supported")

def main():
    
    sweep_config_path = "/home/esraa1/scratch/extended-mind/run_configs/lr_sweep_linear_qlearning_config.yaml"
    default_config_path = "/home/esraa1/scratch/extended-mind/linear_qlearning_config.yaml"
    print("Reading configs...")
    sweep_config = read_config(sweep_config_path)
    default_config = read_config(default_config_path)
    print("Merging configs...")
    merged_config = merge_default_and_sweep_config(default_config, sweep_config)
    print("Generating task configs...")
    sweep_task_configs = list(dict_permutations(merged_config))
    print("Filtering task configs by path mode...")
    none_path_task_configs = filter_task_configs_by_path_mode(sweep_task_configs, "NONE")
    shortest_path_task_configs = filter_task_configs_by_path_mode(sweep_task_configs, "SHORTEST_PATH")


    grouped_none_path_task_configs = group_task_configs_by_agent_pixel_view_edge_dimension(none_path_task_configs)
    grouped_shortest_path_task_configs = group_task_configs_by_agent_pixel_view_edge_dimension(shortest_path_task_configs)
    
    
    _, ax_none_path = plt.subplots(figsize=(16, 6))
    plot_step_size_vs_total_reward(ax_none_path, grouped_none_path_task_configs)
    plt.tight_layout()
    plt.savefig("step_size_vs_total_reward_none_path.png", bbox_inches='tight', pad_inches=0.2)
    plt.close()

    _, ax_shortest_path = plt.subplots(figsize=(16, 6))
    plot_step_size_vs_total_reward(ax_shortest_path, grouped_shortest_path_task_configs)
    plt.tight_layout()
    plt.savefig("step_size_vs_total_reward_shortest_path.png", bbox_inches='tight', pad_inches=0.2)
    plt.close()


if __name__ == "__main__":
    main()