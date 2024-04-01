# Fit Autogluon Tabular Only SCF
# Use the Autogluon AutoML library to predict ratings using tabular data.

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os

##################################################################################################

# Load data
# list of files in '../../../../Data/All_Data/All_Data_with_NLP_Features' directory
file_list = [f for f in os.listdir(r'../../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
# read in all parquet files
df = pd.concat([pd.read_parquet(r'../../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
print('dataframe')
print(df)

# Print out column names
print('column names')
for col in df.columns:
    print(col)

# Removing columns: 'transcript', 'Investment_Grade', 'Change Direction Since Last Fixed Quarter Date', 'Change Since Last Fixed Quarter Date', 'Next Rating', 'Next Rating Date', 'next_rating_date_or_end_of_data'
df = df.drop(columns=['transcript', 
                      'Investment_Grade', 
                      'Change Direction Since Last Fixed Quarter Date', 
                      'Change Since Last Fixed Quarter Date', 
                      'Next Rating', 
                      'Next Rating Date', 
                      'next_rating_date_or_end_of_data'])

# Get train df
train_df = df[df['train_test_80_20'] == 'train'].reset_index(drop=True)

##################################################################################################

# Fit AutoGluon

# Convert from pandas to autogluon
train_data = TabularDataset(train_df)

# Create model save directory if it doesn't exist
os.makedirs(os.path.expanduser('~/Box/STAT 222 Capstone/Autogluon/Autogluon_Tabular_Only_Best_Presets_SCF'), exist_ok=True)

# Fit models
# Use the best quality preset
# Set very high time limit
time_limit = 3600
# Set seed to try to encourage stability
import numpy as np
np.random.seed(222)
# Run predictor
predictor = TabularPredictor(label='Rating', path=os.path.expanduser('~/Box/STAT 222 Capstone/Autogluon/Autogluon_Tabular_Only_Best_Presets_SCF')).fit(train_data=train_data, presets='best_quality', time_limit=time_limit)
