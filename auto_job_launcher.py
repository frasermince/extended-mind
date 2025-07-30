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

def generate_script(task_confs, cluster_conf, env_path, exp_group_id, max_job_time, wandb_api_key):
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
source {env_path}/bin/activate
cd {os.getcwd()}
    """

    script += env_prep

    for a_task_conf in task_confs:
        script_params = ""
        for key, value in a_task_conf.items():
            if key == "exp_group_id" or key == "exp_name" or key == "wandb_api_key":
                continue
            
            if isinstance(value, bool):
                script_params += f"{key}={str(value).lower()} "
            
            elif isinstance(value, list):
                script_params += f"\'{key}={str(value)}\' "

            else:
                shell_val = shlex.quote(str(value))
                script_params += f"{key}={shell_val} "

        
        script += f"\npython main_dqn.py exp_group_id={shlex.quote(exp_group_id)} wandb_api_key={shlex.quote(wandb_api_key)} {script_params}"

    script += "\necho Done!"

    return script


def generate_task_configs_per_job(seeds, hyperparams_to_sweep, default_hyperparams, max_task_time, max_job_time):
        
    sweep_dict = {}

    for key, values in default_hyperparams.items():
        # if specified in sweeps, then use sweep values, else, default values
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
        one_job_task_confs.append(tc)
        if len(one_job_task_confs) == num_tasks_per_job:
            yield one_job_task_confs
            one_job_task_confs = []
    
    if one_job_task_confs:
        yield one_job_task_confs



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-job-time', type=str)
    parser.add_argument('--max-task-time', type=str)
    parser.add_argument('--cluster-conf', type=str)
    parser.add_argument('--hyperparam-sweep-conf', type=str)
    parser.add_argument('--hyperparam-default-conf', type=str, default="config.yaml")
    parser.add_argument('--num-seeds', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--exp-group-id', type=str, default="")
    parser.add_argument('--env-path', type=str, default=".venv")
    parser.add_argument('--wandb-api-key', type=str, default="")

    args = parser.parse_args()

    exp_group_id = args.exp_group_id
    if exp_group_id == "":
        exp_group_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cluster_conf = read_config(args.cluster_conf)
    hyperparam_sweep_conf = read_config(args.hyperparam_sweep_conf)
    hyperparam_default_conf = read_config(args.hyperparam_default_conf)

    task_confs_per_job = generate_task_configs_per_job(list(range(args.num_seeds)), hyperparam_sweep_conf, hyperparam_default_conf, args.max_task_time, args.max_job_time)

    num_jobs = 0
    for one_job_task_confs in task_confs_per_job:
        print(f"Generating script for job {num_jobs+1}...")
        script = generate_script(one_job_task_confs, cluster_conf, args.env_path, exp_group_id, args.max_job_time, args.wandb_api_key)
        print(script)
        print("--------------------------------")
        if not args.dry_run:
            submit_bash_script(script)
        num_jobs += 1
        
    print(f"Auto Job Launcher is done. There are {num_jobs} jobs.")


if __name__ in "__main__":
    main()