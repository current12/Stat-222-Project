#!/bin/bash
#SBATCH --job-name=Doc2Vec_Embeddings_Train_Set_Only
#SBATCH --output=Doc2Vec_Embeddings_Train_Set_Only.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

echo "Starting Job"

python Doc2Vec_Embeddings_Train_Set_Only.py

echo "Completed Job"
