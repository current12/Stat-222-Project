#!/bin/bash
#SBATCH --job-name=transcript_ner
#SBATCH --output=transcript_ner.out
#SBATCH --error=transcript_ner.err
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

mamba activate capstone_scf
python 
