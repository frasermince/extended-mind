import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pandas as pd
import os
import yaml
import itertools

class OnlineSampleStatsElement:
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self._sum = 0.0

    def add_scalar(self, val: float) -> None:
        self.n += 1
        self._sum += val
        delta = val - self.mean
        self.mean += delta / self.n
        delta2 = val - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return np.nan
        return self.M2 / (self.n - 1)

    @property
    def stddev(self) -> float:
        return np.sqrt(self.variance)
    
    @property
    def sum(self) -> float:
        return self._sum


class OnlineSampleStats:
    
    def __init__(self):
        self.stats_per_element = []  # List of OnlineSampleStatsElement for each position
        self.expected_shape = None
    
    def _set_expected_shape(self, shape):
        if(len(shape) > 1):
            raise ValueError(f"Expected 1D or scalar shape, got {shape}")
        self.expected_shape = shape

    def add(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)

        if self.expected_shape is None:
            self._set_expected_shape(x.shape)
        else:
            if x.shape != self.expected_shape:
                raise ValueError(f"Expected shape {self.expected_shape}, got {x.shape}")

        if x.shape == ():
            if not self.stats_per_element:
                self.stats_per_element = [OnlineSampleStatsElement()]
            self.stats_per_element[0].add_scalar(float(x))
        else:
            if not self.stats_per_element:
                self.stats_per_element = [OnlineSampleStatsElement() for _ in range(len(x))]
            for i, val in enumerate(x):
                self.stats_per_element[i].add_scalar(val)

    @property
    def mean(self):
        if self.expected_shape == ():
            return self.stats_per_element[0].mean
        else:
            return np.array([stats.mean for stats in self.stats_per_element])

    @property
    def stddev(self):
        if self.expected_shape == ():
            return self.stats_per_element[0].stddev
        else:
            return np.array([stats.stddev for stats in self.stats_per_element])

    @property
    def variance(self):
        if self.expected_shape == ():
            return self.stats_per_element[0].variance
        else:
            return np.array([stats.variance for stats in self.stats_per_element])


def plot_array_stats(stats: OnlineSampleStats, title: str, plot_path, xlabel: str = "Timestep", ylabel: str = "Reward"):
    
    mean = stats.mean
    std = stats.stddev
    
    x = np.arange(len(mean))
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, 'b-', linewidth=2, label='Mean')
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, color='blue')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_path)

def read_config_seed_file(config_path):

    with open(config_path, 'rb') as f:
        run_data = pickle.load(f)
    
    df = pd.DataFrame(run_data['data'])
    reward_values = df[df['metric'] == 'reward_per_timestep']['value'].values
    return reward_values

def generate_seed_results_paths(config_folder_path):
    seed_results_paths = []
    seed_folders = os.listdir(config_folder_path)
    for seed_folder in seed_folders:
        seed_results_paths.append(os.path.join(config_folder_path, seed_folder, "metrics.pkl"))
    return seed_results_paths
   
def compute_config_stats_over_seeds(config_folder_path):
    seed_results_paths = generate_seed_results_paths(config_folder_path)
    stats = OnlineSampleStats()
    for seed_results_path in seed_results_paths:
        reward_array = read_config_seed_file(seed_results_path)
        stats.add(reward_array)
    return stats

def create_merged_config(sweep_config, default_config):
    merged_config = default_config.copy()
    for key, value in sweep_config.items():
        merged_config[key] = value # overwrite the default config with the sweep config
    return merged_config

def sweepable_config_keys(agent_name):

    if(agent_name == "linear_qlearning"):
        return ["generate_optimal_path", "training.step_size", "training.agent_pixel_view_edge_dim"]
    elif(agent_name == "main_dqn"):
        return ["generate_optimal_path", "training.learning_rate", "training.network_depth", "training.network_width"]
    else:
        raise ValueError(f"Agent name {agent_name} not supported")

def reformat_scientific_notation(value_str):
    float_value = float(value_str)
    if float_value < 0.0001:
        new_value_str = f"{float_value:.0e}"
    else:
        new_value_str = f"{float_value:.6f}".rstrip('0').rstrip('.')
    return new_value_str


def construct_run_path(agent_name, task_config, run_folder):
    p = Path(run_folder)
    p.mkdir(parents=True, exist_ok=True)
    p = p / f"agent_name_{agent_name}"
    for key, value in task_config.items():
        # Extract only the last part of composite keys (after the last dot)
        key_name = key.split('.')[-1]
        
        if isinstance(value, bool):
            p = p / f"{key_name}_{str(value).title()}"
        else:
            if key_name == "step_size" or key_name == "learning_rate":
                value = reformat_scientific_notation(value)
            p = p / f"{key_name}_{value}"
    return p

def flatten_config(config, parent_key='', sep='.'):
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)
    
def dict_combinations(input_dict):

    keys = list(input_dict.keys())
    values_lists = []
    
    for value in input_dict.values():
        if isinstance(value, list):
            values_lists.append(value)
        else:
            values_lists.append([value])  # Wrap single value in a list
    
    combinations = []
    for combination in itertools.product(*values_lists):
        combo_dict = dict(zip(keys, combination))
        combinations.append(combo_dict)
    
    return combinations

def generate_task_configs(agent_name, sweep_config_path, default_config_path):


    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    with open(default_config_path, 'r') as f:
        default_config = yaml.safe_load(f)
    
    # check that all sweep config keys are within what is sweepable
    sweepable_keys = sweepable_config_keys(agent_name)
    for key in sweep_config.keys():
        if key not in sweepable_keys:
            raise ValueError(f"Key {key} is not sweepable")
    
    merged_config = create_merged_config(sweep_config, default_config)
    merged_config = flatten_config(merged_config)
    print(merged_config)
    merged_config = {k : merged_config[k] for k in sweepable_keys}
    task_configs = dict_combinations(merged_config)
    return task_configs

def generate_config_results_paths(agent_name, sweep_config_path, default_config_path, run_folder):

    task_configs = generate_task_configs(agent_name, sweep_config_path, default_config_path)
    config_results_paths = []
    for task_config in task_configs:
        config_results_paths.append(construct_run_path(agent_name, task_config, run_folder))
    
    return config_results_paths

def total_reward_stats(config_results_path):
    stats_tracker = OnlineSampleStatsElement()
    for seed_folder in os.listdir(config_results_path):
        seed_results_path = os.path.join(config_results_path, seed_folder, "metrics.pkl")
        reward_array = read_config_seed_file(seed_results_path)
        stats_tracker.add_scalar(reward_array.sum())
    return stats_tracker.mean, stats_tracker.stddev

def group_configs_by_step_size(task_configs):

    grouped_configs = {}
    
    for config in task_configs:
        step_size = config["training.step_size"]
        
        if isinstance(step_size, str):
            step_size = float(step_size)
        
        if step_size not in grouped_configs:
            grouped_configs[step_size] = []
        grouped_configs[step_size].append(config)
    
    return grouped_configs

def plot_stats_over_step_sizes(agent_name, sweep_config_path, default_config_path, plot_dir, run_dir):

    task_configs = generate_task_configs(agent_name, sweep_config_path, default_config_path)
    grouped_configs = group_configs_by_step_size(task_configs)
    stats_per_step_size = {}
    for step_size, configs in grouped_configs.items():
        print(f"Step size: {step_size}")
        print(f"Configs: {configs}")
        step_size_stats = OnlineSampleStatsElement()
        for config in configs:
            config_results_path = construct_run_path(agent_name, config, run_dir)
            total_reward_stats_mean, total_reward_stats_stddev = total_reward_stats(config_results_path)
            step_size_stats.add_scalar(total_reward_stats_mean) # add the mean (over seeds) of total reward for a particular config
        stats_per_step_size[step_size] = (step_size_stats.mean, step_size_stats.stddev)
    
    plt.figure(figsize=(10, 6))
    
    step_sizes = list(stats_per_step_size.keys())
    means = [stats_per_step_size[step_size][0] for step_size in step_sizes]
    stds = [stats_per_step_size[step_size][1] for step_size in step_sizes]
    
    print("Step sizes:", step_sizes)
    print("Means:", means)
    print("Standard deviations:", stds) # if there is one config per steps size, then these will be nans and thats ok
    
    plt.errorbar(step_sizes, means, yerr=stds, fmt='o-', capsize=8, capthick=3, 
                linewidth=2, markersize=8, elinewidth=2, label="Mean Total Reward ± Std")
    
    for i, (step_size, mean) in enumerate(zip(step_sizes, means)):
        plt.annotate(f'({step_size:.4f}, {mean:.2f})', 
                    xy=(step_size, mean), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=8,
                    rotation=45,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    ha='left')
    
    plt.xlabel("Step Size")
    plt.ylabel("Total Reward")
    plt.title("Step Size Sensitivity")
    plt.xscale('log')  # Set logarithmic scale for x-axis
    
    max_y = max(means)
    plt.ylim(bottom=0, top=max_y + 400)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir.joinpath("step_size_sensitivity.png"))
        

if __name__ == "__main__":

    plot_dir = Path("/home/esraa1/scratch/extended-mind/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    run_dir = "/home/esraa1/scratch/extended-mind/runs"
    
    plot_stats_over_step_sizes("linear_qlearning", "/home/esraa1/scratch/extended-mind/lr_sweep_linear_qlearning_config.yaml", "/home/esraa1/scratch/extended-mind/linear_qlearning_config.yaml", plot_dir, run_dir)
