#!/bin/bash
#SBATCH --job-name=Fit_Autogluon_Multimodal_SCF
#SBATCH --output=Fit_Autogluon_Multimodal_SCF.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute Fit_Autogluon_Multimodal_SCF.ipynb

echo "Completed Job"
