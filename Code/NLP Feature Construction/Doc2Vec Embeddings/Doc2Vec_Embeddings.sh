#!/bin/bash
#SBATCH --job-name=Doc2Vec_Embeddings
#SBATCH --output=Doc2Vec_Embeddings.out
#SBATCH --partition=lowmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

echo "Starting Job"

python Doc2Vec_Embeddings.py

echo "Completed Job"
