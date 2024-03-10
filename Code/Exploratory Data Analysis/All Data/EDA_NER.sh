#!/bin/bash
#SBATCH --job-name=transcript_ner
#SBATCH --output=transcript_ner.out
#SBATCH --error=transcript_ner.err
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

echo "Starting mamba/conda env"

mamba activate capstone_scf

echo "Starting NER on Company Names"

python NER_on_Company_Names.py

echo "Completed NER
