#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=finbert1 # job name for queue (optional)
#SBATCH --partition=jsteinhardt    # partition (optional, default=low) 
#SBATCH --error=ex.err1     # file for stderr (optional)
#SBATCH --output=ex.out1    # file for stdout (optional)
#SBATCH --gres=gpu:1

###################
# Command(s) to run
###################

python finbert1.py

