#!/bin/bash
#SBATCH --job-name=NER_Firm_Mentions
#SBATCH --output=NER_Firm_Mentions.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

echo "Starting NER on Company Names"

python NER_Firm_Mentions.py

echo "Completed NER"
