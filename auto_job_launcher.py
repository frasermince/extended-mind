'''
Ideally we would want to take tasks and put them into jobs so that they can be 
executed in sequence in one job if they can fit a specified time limit.

We are assuming single gpu jobs for the most part.
'''
import yaml
import os
from dateutil import parser as dateutil_parser
from datetime import timedelta, datetime
from time import time, sleep
import argparse
import itertools
from collections.abc import Iterable
import shlex
import copy
from job_script_templater import generate_job_script_from_template


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

def submit_bash_script(bash_script_str):
    fd = os.open("auto_slurm.sh", os.O_WRONLY | os.O_CREAT)
    os.write(fd, bash_script_str.encode())
    exit()
    # os.system("sbatch auto_slurm.sh")
    # os.close(fd)
    # os.remove("auto_slurm.sh")
    # sleep(2)

def to_seconds(val):
    t = dateutil_parser.parse(val)
    return t.hour * 3600 + t.minute * 60 + t.second

def compute_tasks_num_per_job(task_max_time, max_time_per_job, num_parallel_tasks=1):

    task_seconds = to_seconds(task_max_time)
    task_seconds = int(task_seconds * 1.2) # add 20% buffer time
    job_seconds = to_seconds(max_time_per_job)
    
    if task_seconds > job_seconds:
        print(f"ERROR: Task time with buffer ({task_seconds}s) exceeds job time ({job_seconds}s). No tasks can fit in this job.")
        return 0
    
    num_rounds = job_seconds // task_seconds
    
    total_tasks = num_parallel_tasks * num_rounds
    
    if num_parallel_tasks > 1:
        print(f"Parallel execution: {num_parallel_tasks} tasks per round, {num_rounds} rounds = {total_tasks} total tasks per job")
    else:
        print(f"Sequential execution: {total_tasks} tasks per job")
    
    return total_tasks

def generate_script(task_confs, cluster_conf, max_job_time, wandb_api_key, script_name, run_folder=None, parquet_folder=None):

    commands = []

    for a_task_conf in task_confs:
        script_params = ""
        agent_name = a_task_conf.get('agent_name', script_name)
        for key, value in a_task_conf.items():
            if key == "exp_name" or key == "wandb_api_key" or key == "run_folder" or key == "parquet_folder" or key == "agent_name":
                continue
            
            if isinstance(value, bool):
                script_params += f"{key}={str(value).lower()} "
            
            elif isinstance(value, list):
                script_params += f"\'{key}={str(value)}\' "

            else:
                shell_val = shlex.quote(str(value))
                script_params += f"{key}={shell_val} "

        
        # For linear_qlearning, don't pass run_folder parameter
        run_folder_param = f"run_folder={shlex.quote(run_folder)} " if (run_folder and agent_name != "linear_qlearning") else ""
        parquet_folder_param = f"parquet_folder={shlex.quote(parquet_folder)} " if parquet_folder else ""
        if(wandb_api_key == ""):
            print("Warning : launching without wandb tracking. Either the prgram does not have it or it has it and you have not specified the api key.")
            commands.append(f"uv run python src/{script_name}.py {run_folder_param}{parquet_folder_param}{script_params}")
        else:
            commands.append(f"uv run python src/{script_name}.py wandb_api_key={shlex.quote(wandb_api_key)} {run_folder_param}{parquet_folder_param}{script_params}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script = generate_job_script_from_template(cluster_conf['cpus-per-task'], cluster_conf['mem-per-cpu'], max_job_time, cluster_conf['account'], commands, script_dir)

    return script
    

def generate_task_configs_per_job(seeds, hyperparams_to_sweep, default_hyperparams, max_task_time, max_job_time, run_folder, parquet_folder, num_parallel_tasks=1):
        
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
    
    sweep_dict["seed"] = seeds

    task_confs = dict_permutations(sweep_dict)

    num_tasks_per_job = compute_tasks_num_per_job(max_task_time, max_job_time, num_parallel_tasks)
    print(f"Number of tasks per job: {num_tasks_per_job}")

    if num_tasks_per_job == 0:
        raise ValueError(f"Cannot fit any tasks in job with time limit {max_job_time}. Task time with buffer exceeds job time.")

    one_job_task_confs = []
    for tc in task_confs:

        run_path = construct_run_path(tc, run_folder, parquet_folder)
        print(f">>>>> Run path: {run_path}") # TODO: comment this block if needed ... it can be too slow
        if check_if_file_exists(run_path):
            
            print(f"Skipping task: {run_path} because it already exists")
            continue
        
        print(f"Adding task: {run_path}")
        one_job_task_confs.append(tc)
        if len(one_job_task_confs) == num_tasks_per_job:
            print(f"Job complete with {len(one_job_task_confs)} tasks, yielding...")
            yield one_job_task_confs
            one_job_task_confs = []
    
    if one_job_task_confs:
        yield one_job_task_confs


def check_if_file_exists(file_path):
    print(f">>>>> Checking if file exists: {file_path}")
    metrics_path = os.path.join(file_path, "metrics.pkl")
    metrics_optimal_path = os.path.join(file_path, "metrics_optimal_path.pkl")
    print(f">>>>> Metrics path: {metrics_path}")
    print(f">>>>> Metrics optimal path: {metrics_optimal_path}")
    print(f">>>>> File exists: {os.path.exists(metrics_path) or os.path.exists(metrics_optimal_path)}")
    return os.path.exists(metrics_path) or os.path.exists(metrics_optimal_path)


def construct_run_path(tc_original, run_folder, parquet_folder):
    
    tc = copy.deepcopy(tc_original)
    agent_name = tc['agent_name']
    if(agent_name == "main_dqn"):
        # If show_landmarks is true, set path_mode to LANDMARKS (matching main_dqn.py logic)
        if tc.get('show_landmarks', False):
            path_mode = "LANDMARKS"
        else:
            path_mode = tc['path_mode']
        learning_rate = tc['training.learning_rate']
        dense_features = tc['training.dense_features']
        seed = tc['seed']
        
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
            f"seed_{seed}"
        )
        
    elif(agent_name == "linear_qlearning"):
        agent_name = tc['agent_name']
        path_mode = tc['path_mode']
        step_size = tc['training.step_size']
        agent_pixel_view_edge_dim = tc['training.agent_pixel_view_edge_dim']
        seed = tc['seed']
        
        if isinstance(step_size, str):
            step_size = float(step_size)
        
        if step_size < 0.0001:
            step_size_str = f"{step_size:.0e}"
        else:
            step_size_str = f"{step_size:.6f}".rstrip('0').rstrip('.')
        
        
        # Build base path components - use parquet_folder instead of run_folder for linear_qlearning
        path_components = [
            parquet_folder,
            f"agent_name_{agent_name}",
            f"path_mode_{path_mode}",
            f"step_size_{step_size_str}",
            f"agent_pixel_view_edge_dim_{agent_pixel_view_edge_dim}",
        ]
        
        # Only include nonstationary_path fields when path_mode is VISITED_CELLS
        if path_mode == "VISITED_CELLS":
            nonstationary_path_decay_pixels = tc['nonstationary_path_decay_pixels']
            nonstationary_path_decay_chance = tc['nonstationary_path_decay_chance']
            nonstationary_path_inclusion_pixels = tc['nonstationary_path_inclusion_pixels']

            decay_chance_str = f"{float(nonstationary_path_decay_chance):.2f}".rstrip('0').rstrip('.')
            path_components.extend([
                f"nonstationary_path_decay_pixels_{nonstationary_path_decay_pixels}",
                f"nonstationary_path_decay_chance_{decay_chance_str}",
                f"nonstationary_path_inclusion_pixels_{nonstationary_path_inclusion_pixels}",
            ])
        
        path_components.append(f"seed_{seed}")
        run_path = os.path.join(*path_components)
    
    else:
        raise ValueError(f"Agent name {agent_name} not supported")
        
    
    return run_path


def main():
    '''
    Example command:
    uv run python auto_job_launcher.py --max-job-time="02:59:00" --max-task-time="00:08:00" --cluster-conf="./clusters/narval_gpu.yaml" --hyperparam-sweep-conf="run_configs/visited_sweep.yaml" --hyperparam-default-conf="src/config.yaml" --num-seeds=30  --wandb-api-key="6314198eb6e8a2350001f6bce002acb709b5bcbe" --agent-name="main_dqn" --run-folder="/home/esraa1/scratch/extended_mind_results/runs" --parquet-folder="/home/esraa1/scratch/extended_mind_results/parquet_runs"
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-job-time', type=str)
    parser.add_argument('--max-task-time', type=str)
    parser.add_argument('--cluster-conf', type=str)
    parser.add_argument('--hyperparam-sweep-conf', type=str)
    parser.add_argument('--hyperparam-default-conf', type=str, default="config.yaml")
    parser.add_argument('--num-seeds', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wandb-api-key', type=str, default="")
    parser.add_argument('--run-folder', type=str, default=None)
    parser.add_argument('--parquet-folder', type=str, default=None)
    parser.add_argument('--agent-name', type=str)

    args = parser.parse_args()


    cluster_conf = read_config(args.cluster_conf)
    hyperparam_sweep_conf = read_config(args.hyperparam_sweep_conf)
    hyperparam_default_conf = read_config(args.hyperparam_default_conf)

    num_parallel_tasks = cluster_conf['cpus-per-task']-1 # leave 1 CPU for overhead
    task_confs_per_job = generate_task_configs_per_job(list(range(args.num_seeds)), hyperparam_sweep_conf, hyperparam_default_conf, args.max_task_time, args.max_job_time, args.run_folder, args.parquet_folder, num_parallel_tasks)

    num_jobs = 0
    script_name = args.agent_name
    for one_job_task_confs in task_confs_per_job:
        print(f"Generating script for job {num_jobs+1}...")
        script = generate_script(task_confs=one_job_task_confs, cluster_conf=cluster_conf, max_job_time=args.max_job_time, wandb_api_key=args.wandb_api_key, script_name=script_name, run_folder=args.run_folder, parquet_folder=args.parquet_folder)
        print(script)
        print("--------------------------------")
        if not args.dry_run:
            submit_bash_script(script)
        num_jobs += 1
        
    print(f"Auto Job Launcher is done. There are {num_jobs} jobs.")


if __name__ in "__main__":
    main()