#!/bin/bash
#SBATCH --job-name=Evaluate_Autogluon_Multimodal_SCF
#SBATCH --output=Evaluate_Autogluon_Multimodal_SCF.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --execute Evaluate_Autogluon_Multimodal_SCF.ipynb

echo "Completed Job"
