#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=nlp_added # job name for queue (optional)
#SBATCH --partition=jsteinhardt    # partition (optional, default=low) 
#SBATCH --error=ex.err     # file for stderr (optional)
#SBATCH --output=ex.out    # file for stdout (optional)
#SBATCH --nodes=1          # use 1 node
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=20  

###################
# Command(s) to run
###################

python nlp_features.py
