#!/bin/sh
for seed in 0 1 2 3 4; do
  for lr in  5e-4; do
    for depth in 2; do
    # for depth in 3; do
      for width in 32; do
      # for width in 16; do
        # for opt in "path_mode=SHORTEST_PATH" "path_mode=NONE"; do
        # for opt in "generate_optimal_path=false"; do
          # Create  capacity array of length $depth, all elements $width
          capacity=$(printf "[%s]\n" "$(yes $width | head -n $depth | paste -sd, -)")
          echo CAPACITY: $capacity
          uv run python src/main_dqn.py run_folder="landmarks_black_runs" seed=$seed training.learning_rate="$lr" $opt training.dense_features="$capacity" &
        # done
      done
    done
  done
done
wait