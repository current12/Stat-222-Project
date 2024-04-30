#!/bin/bash

#SBATCH --job-name=Rclone_2_Way_Copy
#SBATCH --partition=lowmem
#SBATCH --output=/dev/null

echo "Making directory"

# Make directory if needed
mkdir -p ~/Box/"STAT 222 Capstone"/Autogluon

echo "Beginning Push"

# Push to remote/online version
rclone copy ~/Box/"STAT 222 Capstone"/Autogluon "Box:STAT 222 Capstone"/Autogluon --update

echo "Beginning Pull"

# Pull from remote/online version
rclone copy "Box:STAT 222 Capstone"/Autogluon ~/Box/"STAT 222 Capstone"/Autogluon --update
