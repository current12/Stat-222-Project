#!/bin/bash
#SBATCH --job-name=transcript_ner_no_gpu
#SBATCH --output=transcript_ner_no_gpu.out
#SBATCH --partition=low

echo "Starting NER on Company Names"

python NER_on_Company_Names.py

echo "Completed NER"
