#!/bin/bash
#SBATCH --job-name=Fit_Autogluon_Tabular_Only_SCF_Best_Presets
#SBATCH --output=Fit_Autogluon_Tabular_Only_SCF_Best_Presets.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

python Fit_Autogluon_Tabular_Only_SCF_Best_Presets.py

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
