#!/bin/bash
#SBATCH --job-name=Fit_Autogluon_Tabular_Only_SCF
#SBATCH --output=Fit_Autogluon_Tabular_Only_SCF.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

echo "Starting Job"

python Fit_Autogluon_Tabular_Only_SCF.py

echo "Completed Job"
