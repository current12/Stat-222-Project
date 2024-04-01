#!/bin/bash
#SBATCH --job-name=Evaluate_Autogluon_Tabular_Only_SCF_ag_20240331_233723
#SBATCH --output=Evaluate_Autogluon_Tabular_Only_SCF_ag_20240331_233723.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

echo "Starting Job"

python "Evaluate_Autogluon_Tabular_Only_SCF - ag-20240331_233723.py"

echo "Completed Job"
