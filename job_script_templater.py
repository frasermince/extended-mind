# cpus_per_task = 6
# mem_per_cpu = "4G"
# time = "00:59:00"
# account = "rrg-gberseth_gpu"
# commands = ["\"command1\"", "\"command2\"", "\"command3\""].join("\n")

def generate_job_script_from_template(cpus_per_task, mem_per_cpu, time, account, commands, script_dir):
    commands = [f'"{cmd}"' for cmd in commands]
    commands_str = "\n".join(commands) 

    return f"""#!/bin/bash
#SBATCH --job-name=cpu_parallel_auto_slurm
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time}
#SBATCH --output=cpu_parallel_auto_slurm_%j.out
#SBATCH --account={account}
#SBATCH --gpus=0
#SBATCH --partition=cpubase_bynode_b1

set -euo pipefail

echo "Starting job on single node: $(hostname)"



# Ensure required tools exist
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found in PATH"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found in PATH"
    exit 1
fi

if ! command -v parallel >/dev/null 2>&1; then
    echo "ERROR: GNU parallel not found in PATH"
    exit 1
fi

echo "Using parallel: $(parallel --version | head -n1)"

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

# Calculate number of parallel jobs: cpus-per-task - 1 (leave 1 CPU for overhead)
if [ -z "${{SLURM_CPUS_PER_TASK:-}}" ]; then
    echo "ERROR: SLURM_CPUS_PER_TASK is not set"
    exit 1
fi

JOBS=$SLURM_CPUS_PER_TASK
if (( JOBS > 1 )); then
    JOBS=$((JOBS - 1))
else
    JOBS=1
fi

echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Running $JOBS parallel jobs (using $((JOBS + 1)) CPUs: $JOBS for jobs + 1 reserved)"

# Run commands in parallel using GNU parallel
# Use explicit 'sh -c' so each line is treated as a full command,
# regardless of parallel implementation/config.
printf '%s\n' "${{commands[@]}}" | parallel -j "$JOBS" --will-cite --joblog parallel.log --tag sh -c '{{}}'

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

