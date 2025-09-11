#!/bin/sh
for seed in 0 1 2 3 4; do
  for lr in 1e-3 5e-4 1e-4 5e-5 1e-5; do
    for depth in 2 3; do
    # for depth in 3; do
      for width in 2 4 8 16 32; do
      # for width in 16; do
        for opt in "generate_optimal_path=true" "generate_optimal_path=false"; do
        # for opt in "generate_optimal_path=false"; do
          # Create  capacity array of length $depth, all elements $width
          capacity=$(printf "[%s]\n" "$(yes $width | head -n $depth | paste -sd, -)")
          echo CAPACITY: $capacity
          python main_dqn.py run_folder="more_exploration_runs" seed=$seed training.learning_rate="$lr" $opt training.dense_features="$capacity" &
        done
      done
    done
  done
  wait
done