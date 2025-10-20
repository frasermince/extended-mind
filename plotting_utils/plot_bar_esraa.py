import json
import matplotlib.pyplot as plt
import numpy as np

def plot_hyperparam_comparison(json_file_path, output_path):
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    nonpath_results = results["nonpath_best_view_edge_dim_conf_to_total_reward"]
    path_results = results["path_best_view_edge_dim_conf_to_total_reward"]
    
    view_edge_dims = sorted([int(k) for k in nonpath_results.keys()])
    
    nonpath_rewards = [nonpath_results[str(dim)][0] for dim in view_edge_dims]  # Mean values
    nonpath_errors = [nonpath_results[str(dim)][1] for dim in view_edge_dims]  # Standard errors
    nonpath_data_points = [nonpath_results[str(dim)][2] for dim in view_edge_dims]  # Individual data points
    path_rewards = [path_results[str(dim)][0] for dim in view_edge_dims]  # Mean values
    path_errors = [path_results[str(dim)][1] for dim in view_edge_dims]  # Standard errors
    path_data_points = [path_results[str(dim)][2] for dim in view_edge_dims]  # Individual data points
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    x_pos = np.arange(len(view_edge_dims))
    bars1 = ax.bar(x_pos - bar_width/2, nonpath_rewards, bar_width, 
                   yerr=nonpath_errors, capsize=5,
                   label='Non-Path', color='#E69F00', alpha=0.8)  # Orange
    bars2 = ax.bar(x_pos + bar_width/2, path_rewards, bar_width, 
                   yerr=path_errors, capsize=5,
                   label='Path', color='#56B4E9', alpha=0.8)  # Blue
    
    ax.set_xlabel('Agent Pixel View Edge Dimension', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Hyperparameter Comparison: Path vs Non-Path Results\nby Agent Pixel View Edge Dimension', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(view_edge_dims)
    ax.legend()
    
    def add_value_labels(bars, errors):
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}±{error:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + error),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    add_value_labels(bars1, nonpath_errors)
    add_value_labels(bars2, path_errors)
    
    # Add individual data points
    for i, (nonpath_points, path_points) in enumerate(zip(nonpath_data_points, path_data_points)):
        x_nonpath = [x_pos[i] - bar_width/2] * len(nonpath_points)
        ax.scatter(x_nonpath, nonpath_points, color='#D55E00', alpha=0.6, s=30, zorder=3)  # Dark orange
        
        x_path = [x_pos[i] + bar_width/2] * len(path_points)
        ax.scatter(x_path, path_points, color='#0072B2', alpha=0.6, s=30, zorder=3)  # Dark blue
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    output_path = '/home/esraa1/scratch/extended-mind/hyperparam_comparison_plot.png'
    json_file_path = "/home/esraa1/scratch/extended-mind/hyperparam_selection_results.json"
    plot_hyperparam_comparison(json_file_path, output_path)
