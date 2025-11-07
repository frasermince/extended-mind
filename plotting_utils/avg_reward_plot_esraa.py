import json
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def get_colorblind_friendly_colors(n):
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


def get_best_config_paths(json_file_path: str) -> Dict[int, str]:

    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    nonpath_results = results["nonpath_best_view_edge_dim_conf_to_total_reward"]
    path_results = results["path_best_view_edge_dim_conf_to_total_reward"]
    nonpath_config_paths = {int(dim): nonpath_results[dim]["config_path"] for dim in nonpath_results.keys()}
    path_config_paths = {int(dim): path_results[dim]["config_path"] for dim in path_results.keys()}

    return nonpath_config_paths, path_config_paths


def lifetime_avg_reward_curve_data(config_path: str) -> Tuple[np.ndarray, np.ndarray]:

    if not os.path.exists(config_path):
        raise ValueError(f"Config path does not exist: {config_path}")
    
    seed_folders = [f for f in os.listdir(config_path) if os.path.isdir(os.path.join(config_path, f))]
    
    all_lifetime_avg_rewards = []
    for seed_folder in seed_folders:
        print(f"Processing seed {seed_folder} in {config_path}")
        seed_results_path = os.path.join(config_path, seed_folder, "metrics.pkl")
        
        if not os.path.exists(seed_results_path):
            print(f"Warning: metrics.pkl not found for seed {seed_folder} in {config_path}")
            continue
        
        with open(seed_results_path, 'rb') as f:
            run_data = pickle.load(f)
        
        df = pd.DataFrame(run_data['data'])
        reward_values = df[df['metric'] == 'reward_per_timestep']['value'].values
        # lifetime avg reward is the sum of reward values up to the current step divided by the number of steps
        lifetime_avg_reward = np.cumsum(reward_values) / np.arange(1, len(reward_values) + 1)
        all_lifetime_avg_rewards.append(lifetime_avg_reward)
    
    avg_curve = np.mean(all_lifetime_avg_rewards, axis=0)
    std_error = np.std(all_lifetime_avg_rewards, axis=0) / np.sqrt(len(all_lifetime_avg_rewards))
        
    return avg_curve, std_error


def plot_curves(ax, config_paths: Dict[int, str], colors: list, label_suffix: str, linestyle: str = '-'):
    sorted_dims = sorted(config_paths.keys())
    
    for idx, view_edge_dim in enumerate(sorted_dims):
        config_path = config_paths[view_edge_dim]
        avg_curve, std_error = lifetime_avg_reward_curve_data(config_path)
        agent_capacity = view_edge_dim * view_edge_dim
        color = colors[idx]
        timesteps = np.arange(1, len(avg_curve) + 1)
        
        ax.fill_between(timesteps, avg_curve - std_error, avg_curve + std_error, 
                       color=color, alpha=0.2, label='_nolegend_')
        ax.plot(timesteps, avg_curve, label=f"{agent_capacity}", 
               color=color, linestyle=linestyle)
    
    


def main():
    
    json_file_path = "/home/esraa1/code/extended-mind/hyperparam_selection_results.json"
    nonpath_config_paths, path_config_paths = get_best_config_paths(json_file_path)
    
    sorted_dims = sorted(nonpath_config_paths.keys())
    num_curves = len(sorted_dims)
    colors = get_colorblind_friendly_colors(num_curves)
    
    # Plot for No path
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    plot_curves(ax1, nonpath_config_paths, colors, "No path", linestyle='-')
    ax1.set_title("No Path")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Average Reward")
    ax1.legend(title="Agent Capacity", frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("avg_reward_plot_no_path.png", dpi=300, bbox_inches='tight')
    print("Plot saved to avg_reward_plot_no_path.png")
    plt.close()
    
    # Plot for Optimal path
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    plot_curves(ax2, path_config_paths, colors, "Optimal path", linestyle='-')
    ax2.set_title("Optimal Path")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Average Reward")
    ax2.legend(title="Agent Capacity", frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("avg_reward_plot_optimal_path.png", dpi=300, bbox_inches='tight')
    print("Plot saved to avg_reward_plot_optimal_path.png")
    plt.close()
        

if __name__ == "__main__":
    main()