#!/bin/bash
#SBATCH --job-name=Graph_NN_on_Call_Mentions_Exclude_Previous_Rating_Model_2_Inductive
#SBATCH --output=Graph_NN_on_Call_Mentions_Exclude_Previous_Rating_Model_2_Inductive.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace Graph_NN_on_Call_Mentions_Exclude_Previous_Rating_Model_2_Inductive.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
