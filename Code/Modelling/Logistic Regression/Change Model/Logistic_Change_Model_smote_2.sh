#!/bin/bash
#SBATCH --job-name=Logistic_Change_Model_smote_2
#SBATCH --output=Logistic_Change_Model_smote_2.out
#SBATCH --partition=epurdom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace Logistic_Change_Model_smote_2.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."