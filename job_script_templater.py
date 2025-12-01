# cpus_per_task = 6
# mem_per_cpu = "4G"
# time = "00:59:00"
# account = "rrg-gberseth_gpu"
# commands = ["\"command1\"", "\"command2\"", "\"command3\""].join("\n")

def sbatch_options(cpus_per_task, mem_per_cpu, time, account, gpus_zero):
    sbatch_options_str = f"""
#SBATCH --job-name=cpu_parallel_auto_slurm
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output=cpu_parallel_auto_slurm_%j.out
#SBATCH --account={account}
"""

    if gpus_zero:
        gpus_str = "#SBATCH --gpus=0"
        partition_str = "#SBATCH --partition=cpubase_bynode_b1"
        sbatch_options_str += f"{gpus_str}\n{partition_str}"
    
    return sbatch_options_str


def generate_job_script_from_template(num_parallel_tasks, threads_per_task, mem_per_cpu, time, account, commands, script_dir, gpus_zero):
    commands = [f'"{cmd}"' for cmd in commands]
    commands_str = "\n".join(commands)
    cpus_per_task = (num_parallel_tasks * threads_per_task) + 1
    sbatch_options_str = sbatch_options(cpus_per_task, mem_per_cpu, time, account, gpus_zero)

    return f"""#!/bin/bash
{sbatch_options_str}

set -euo pipefail

echo "Starting job on single node: $(hostname)"
THREADS_PER_TASK={threads_per_task}
NUM_JOBS={num_parallel_tasks}

# Clone the repository once on the single node
WORK_DIR="${{SLURM_TMPDIR:-/tmp}}/extended-mind"
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning repository to $WORK_DIR"
    git clone {script_dir} "$WORK_DIR"
    cd "$WORK_DIR"
    uv sync --offline
    echo "Clone and sync complete"
else
    echo "Directory already exists at $WORK_DIR"
    cd "$WORK_DIR"
fi

# List of commands to run in parallel
commands=(
{commands_str}
)

echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Allocated CPUs on node: ${{SLURM_CPUS_ON_NODE:-unknown}}"
echo "Node: $(hostname)"
echo ""

# Verify SLURM allocation
if [ -z "${{SLURM_CPUS_PER_TASK:-}}" ]; then
    echo "ERROR: SLURM_CPUS_PER_TASK is not set"
    exit 1
fi

echo "CPUs per task (SLURM): $SLURM_CPUS_PER_TASK"
echo "Parallel tasks: $NUM_JOBS"
echo "Threads per task: $THREADS_PER_TASK"
echo "Total CPU allocation: $NUM_JOBS tasks × $THREADS_PER_TASK threads = $((NUM_JOBS * THREADS_PER_TASK)) CPUs"
echo "Reserved CPUs for overhead: 1"
echo "Expected total: $((NUM_JOBS * THREADS_PER_TASK + 1)) CPUs"

# Use the configured number of parallel tasks
JOBS=$NUM_JOBS

echo "Running $JOBS parallel jobs with $THREADS_PER_TASK threads each"
echo "Number of commands: ${{#commands[@]}}"

# Run commands in parallel using GNU parallel
# Set OMP_NUM_THREADS and MKL_NUM_THREADS before each Python command to prevent CPU oversubscription
printf '%s\\n' "${{commands[@]}}" | parallel -j "$JOBS" --will-cite --joblog parallel.log --tag env OMP_NUM_THREADS=${{THREADS_PER_TASK}} MKL_NUM_THREADS=${{THREADS_PER_TASK}} sh -c '{{}}'

echo ""
echo "[$(date +%H:%M:%S)] All tasks completed!"

# Print memory usage summary if sacct is available
if command -v sacct >/dev/null 2>&1 && [ -n "${{SLURM_JOB_ID:-}}" ]; then
    echo ""
    echo "=== Memory Usage Summary ==="
    sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,Elapsed,State -P 2>/dev/null | tail -n +2
    echo ""
    echo "To see detailed memory stats, run: seff $SLURM_JOB_ID"
fi

"""

