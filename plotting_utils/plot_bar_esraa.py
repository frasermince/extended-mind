import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors


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

def plot_hyperparam_comparison(json_file_path, output_path):
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    nonpath_results = results["nonpath_best_view_edge_dim_conf_to_total_reward"]
    path_results = results["path_best_view_edge_dim_conf_to_total_reward"]
    
    view_edge_dims = sorted([int(k) for k in nonpath_results.keys()])
    agent_capacities = [dim * dim for dim in view_edge_dims]
    
    nonpath_rewards = [nonpath_results[str(dim)]["mean"] for dim in view_edge_dims]  # Mean values
    nonpath_errors = [nonpath_results[str(dim)]["std_error"] for dim in view_edge_dims]  # Standard errors
    nonpath_data_points = [nonpath_results[str(dim)]["data_points"] for dim in view_edge_dims]  # Individual data points
    path_rewards = [path_results[str(dim)]["mean"] for dim in view_edge_dims]  # Mean values
    path_errors = [path_results[str(dim)]["std_error"] for dim in view_edge_dims]  # Standard errors
    path_data_points = [path_results[str(dim)]["data_points"] for dim in view_edge_dims]  # Individual data points
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    x_pos = np.arange(len(view_edge_dims))
    
    colors = get_colorblind_friendly_colors(5)
    nonpath_color = colors[4]  
    path_color = colors[3]    
    
    # Draw bars without error bars (lowest layer)
    bars1 = ax.bar(x_pos - bar_width/2, nonpath_rewards, bar_width, 
                   label='No path', color=nonpath_color, zorder=3)
    bars2 = ax.bar(x_pos + bar_width/2, path_rewards, bar_width, 
                   label='Optimal path', color=path_color, zorder=3)
    
    ax.set_xlabel('Agent Capacity', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_capacities)
        
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(frameon=False)
    
    dot_colors = get_colorblind_friendly_colors(5)
    nonpath_dot_color = dot_colors[4]  # vermillion
    path_dot_color = dot_colors[3]    # blue
    
    for i, (nonpath_points, path_points) in enumerate(zip(nonpath_data_points, path_data_points)):
        x_nonpath = [x_pos[i] - bar_width/2] * len(nonpath_points)
        ax.scatter(x_nonpath, nonpath_points, color=nonpath_dot_color, s=30, zorder=4, 
                   edgecolors='white', linewidths=0.3)
        
        x_path = [x_pos[i] + bar_width/2] * len(path_points)
        ax.scatter(x_path, path_points, color=path_dot_color, s=30, zorder=4,
                   edgecolors='white', linewidths=0.3)
    
    ax.errorbar(x_pos - bar_width/2, nonpath_rewards, yerr=nonpath_errors, 
                fmt='none', capsize=5, color='black', zorder=5, linewidth=2)
    ax.errorbar(x_pos + bar_width/2, path_rewards, yerr=path_errors, 
                fmt='none', capsize=5, color='black', zorder=5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    print("\nSummary Statistics:")
    print("=" * 50)
    for i, dim in enumerate(view_edge_dims):
        nonpath_reward = nonpath_rewards[i]
        nonpath_error = nonpath_errors[i]
        path_reward = path_rewards[i]
        path_error = path_errors[i]
        improvement = path_reward - nonpath_reward
        improvement_pct = (improvement / nonpath_reward) * 100 if nonpath_reward != 0 else 0
        
        print(f"View Edge Dim {dim}:")
        print(f"  Non-Path: {nonpath_reward:.1f} ± {nonpath_error:.1f}")
        print(f"  Path:     {path_reward:.1f} ± {path_error:.1f}")
        print(f"  Improvement: {improvement:.1f} ({improvement_pct:.1f}%)")
        print()

if __name__ == "__main__":
    output_path = '/home/esraa1/code/extended-mind/hyperparam_comparison_plot.png'
    json_file_path = "/home/esraa1/code/extended-mind/hyperparam_selection_results.json"
    plot_hyperparam_comparison(json_file_path, output_path)
