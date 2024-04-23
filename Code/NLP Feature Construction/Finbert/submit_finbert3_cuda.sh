#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=finbert3 # job name for queue (optional)
#SBATCH --partition=jsteinhardt    # partition (optional, default=low) 
#SBATCH --error=ex.err3     # file for stderr (optional)
#SBATCH --output=ex.out3    # file for stdout (optional)
#SBATCH --gres=gpu:1

###################
# Command(s) to run
###################

python finbert3.py

