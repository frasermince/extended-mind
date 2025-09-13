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
    os.system("sbatch auto_slurm.sh")
    os.close(fd)
    os.remove("auto_slurm.sh")
    sleep(2)

def to_seconds(val):
    t = dateutil_parser.parse(val)
    return t.hour * 3600 + t.minute * 60 + t.second

def compute_tasks_num_per_job(task_max_time, max_time_per_job):
    task_seconds = to_seconds(task_max_time)
    task_seconds = int(task_seconds * 1.2) # add 20% buffer time
    job_seconds = to_seconds(max_time_per_job)
    return job_seconds // task_seconds

def generate_script(task_confs, cluster_conf, max_job_time, wandb_api_key, script_name, run_folder=None):
    script = f"""#!/bin/bash
#SBATCH --job-name=auto_slurm
#SBATCH --output=auto_slurm_%j.out
#SBATCH --error=auto_slurm_%j.err
#SBATCH --time={max_job_time}
#SBATCH --mem={cluster_conf['mem']}
#SBATCH --account={cluster_conf['account']}
"""

    if cluster_conf['gpus']:
        script += f"#SBATCH --gres=gpu:{cluster_conf['gpus']}"
    else:
        script += f"#SBATCH --cpus-per-task={cluster_conf['cpus_per_task']}"
    
    env_prep = f"""
echo Start!
git clone /home/esraa1/scratch/extended-mind $SLURM_TMPDIR/extended-mind
cd $SLURM_TMPDIR/extended-mind
uv sync --offline

    """

    script += env_prep

    for a_task_conf in task_confs:
        script_params = ""
        for key, value in a_task_conf.items():
            if key == "exp_name" or key == "wandb_api_key" or key == "run_folder":
                continue
            
            if isinstance(value, bool):
                script_params += f"{key}={str(value).lower()} "
            
            elif isinstance(value, list):
                script_params += f"\'{key}={str(value)}\' "

            else:
                shell_val = shlex.quote(str(value))
                script_params += f"{key}={shell_val} "

        
        run_folder_param = f"run_folder={shlex.quote(run_folder)} " if run_folder else ""
        if(wandb_api_key == ""):
            print("Warning : launching without wandb tracking. Either the prgram does not have it or it has it and you have not specified the api key.")
            script += f"\nuv run python {script_name}.py {run_folder_param}{script_params}"
        else:
            script += f"\nuv run python {script_name}.py wandb_api_key={shlex.quote(wandb_api_key)} {run_folder_param}{script_params}"

    script += "\necho Done!"

    return script


def generate_task_configs_per_job(seeds, hyperparams_to_sweep, default_hyperparams, max_task_time, max_job_time):
        
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

    num_tasks_per_job = compute_tasks_num_per_job(max_task_time, max_job_time)
    print(f"Number of tasks per job: {num_tasks_per_job}")

    one_job_task_confs = []
    for tc in task_confs:

        run_path = construct_run_path(tc)
        if check_if_file_exists(run_path):
            print(f"Skipping task: {run_path} because it already exists")
            continue
        
        print(f"Adding task: {run_path}")
        one_job_task_confs.append(tc)
        if len(one_job_task_confs) == num_tasks_per_job:
            yield one_job_task_confs
            one_job_task_confs = []
    
    if one_job_task_confs:
        yield one_job_task_confs


def check_if_file_exists(file_path):
    metrics_path = os.path.join(file_path, "metrics.pkl")
    metrics_optimal_path = os.path.join(file_path, "metrics_optimal_path.pkl")
    return os.path.exists(metrics_path) or os.path.exists(metrics_optimal_path)


def construct_run_path(tc, run_folder="/home/esraa1/scratch/extended-mind/runs"):
    agent_name = tc['agent_name']
    if(agent_name == "main_dqn"):
        generate_optimal_path = tc['generate_optimal_path']
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
            f"generate_optimal_path_{generate_optimal_path}",
            f"learning_rate_{learning_rate_str}",
            f"network_depth_{network_depth}",
            f"network_width_{network_width}",
            f"seed_{seed}"
        )
        
    elif(agent_name == "linear_qlearning"):
        agent_name = tc['agent_name']
        generate_optimal_path = tc['generate_optimal_path']
        step_size = tc['training.step_size']
        agent_pixel_view_edge_dim = tc['training.agent_pixel_view_edge_dim']
        seed = tc['seed']
        
        if isinstance(step_size, str):
            step_size = float(step_size)
        
        if step_size < 0.0001:
            step_size_str = f"{step_size:.0e}"
        else:
            step_size_str = f"{step_size:.6f}".rstrip('0').rstrip('.')
        
        
        run_path = os.path.join(
        run_folder,
        f"agent_name_{agent_name}",
        f"generate_optimal_path_{generate_optimal_path}",
        f"step_size_{step_size_str}",
        f"agent_pixel_view_edge_dim_{agent_pixel_view_edge_dim}",
        f"seed_{seed}",
    )
    
    else:
        raise ValueError(f"Agent name {agent_name} not supported")
        
    
    return run_path


def main():

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
    parser.add_argument('--agent-name', type=str)

    args = parser.parse_args()


    cluster_conf = read_config(args.cluster_conf)
    hyperparam_sweep_conf = read_config(args.hyperparam_sweep_conf)
    hyperparam_default_conf = read_config(args.hyperparam_default_conf)

    task_confs_per_job = generate_task_configs_per_job(list(range(args.num_seeds)), hyperparam_sweep_conf, hyperparam_default_conf, args.max_task_time, args.max_job_time)

    num_jobs = 0
    script_name = args.agent_name
    for one_job_task_confs in task_confs_per_job:
        print(f"Generating script for job {num_jobs+1}...")
        script = generate_script(one_job_task_confs, cluster_conf, args.max_job_time, args.wandb_api_key, script_name, args.run_folder)
        print(script)
        print("--------------------------------")
        if not args.dry_run:
            submit_bash_script(script)
        num_jobs += 1
        
    print(f"Auto Job Launcher is done. There are {num_jobs} jobs.")


if __name__ in "__main__":
    main()