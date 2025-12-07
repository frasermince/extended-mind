#!/bin/bash
parquet_dir="nonstationary_only_optimal_runs"
max_jobs=8  # Adjust to your core count

# Experiment-specific flags (add flags for this run here)
EXPERIMENT_FLAGS=(
  nonstationary_only_optimal=true
)


# If parquet dir exists, move it to a backup
if [ -d "$parquet_dir" ]; then
  backup_dir="${parquet_dir}_backup_$(date +%Y%m%d_%H%M%S)"
  echo "Moving existing $parquet_dir to $backup_dir"
  mv "$parquet_dir" "$backup_dir"
fi

# Calculate total jobs: 1 seed × 5 edge_dims × 7 lrs × 2 path_modes = 70
total_jobs=$((1 * 5 * 7 * 2))
progress_dir=$(mktemp -d)

for seed in 0; do
  for edge_dim in 24 20 16 8 4; do
    for lr in 4e-4 8e-4 2e-3 4e-3 8e-3 2e-2 4e-2; do
      for path_mode in "NONE" "VISITED_CELLS"; do
        # Wait if we've hit max concurrent jobs
        while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
          sleep 0.5
        done

        echo "Starting: edge_dim=$edge_dim lr=$lr path_mode=$path_mode"
        (
          uv run python src/linear_qlearning.py \
            parquet_folder="$parquet_dir" \
            seed=$seed \
            training.step_size="$lr" \
            training.agent_pixel_view_edge_dim="$edge_dim" \
            path_mode="$path_mode" \
            capture_video=false \
            "${EXPERIMENT_FLAGS[@]}"
          
          # Create a marker file and count completed jobs
          touch "$progress_dir/$$"
          completed=$(ls -1 "$progress_dir" | wc -l | tr -d ' ')
          echo "✓ Finished ($completed/$total_jobs): edge_dim=$edge_dim lr=$lr path_mode=$path_mode"
        ) &
      done
    done
  done
done

wait  # Wait for all remaining jobs to finish
rm -rf "$progress_dir"
echo "All $total_jobs jobs complete!"