#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=finbert2 # job name for queue (optional)
#SBATCH --partition=jsteinhardt    # partition (optional, default=low) 
#SBATCH --error=ex.err2     # file for stderr (optional)
#SBATCH --output=ex.out2    # file for stdout (optional)
#SBATCH --gres=gpu:1

###################
# Command(s) to run
###################

python finbert2.py

