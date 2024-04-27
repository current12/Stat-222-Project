#!/bin/bash
#SBATCH --job-name=Doc2Vec_Embeddings_All_Data
#SBATCH --output=Doc2Vec_Embeddings_All_Data.out
#SBATCH --partition=epurdom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

echo "Starting Job"

python Doc2Vec_Embeddings_All_Data.py

echo "Completed Job"
