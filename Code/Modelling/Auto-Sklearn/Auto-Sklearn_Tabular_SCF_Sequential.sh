#!/bin/bash
#SBATCH --job-name=Auto-Sklearn_Tabular_SCF_Sequential
#SBATCH --output=Auto-Sklearn_Tabular_SCF_Sequential.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "Starting Job"

python Auto-Sklearn_Tabular_SCF_Sequential.py

echo "Completed Job"
