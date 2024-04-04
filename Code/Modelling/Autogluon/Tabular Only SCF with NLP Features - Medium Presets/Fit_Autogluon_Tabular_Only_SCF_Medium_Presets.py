# Fit Autogluon Tabular Only SCF
# Use the Autogluon AutoML library to predict ratings using tabular data.

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import json

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

# Get train df
train_df = df[df['train_test_80_20'] == 'train'].reset_index(drop=True)

##################################################################################################

# Load features - the same as the most complex logistic regression model, to ensure comparability
# Load the JSON file
with open('../../Logistic Regression/feature_columns.json') as file:
    column_data = json.load(file)
cat_feature_columns = column_data['cat_feature_columns']
nlp_feature_columns = column_data['nlp_feature_columns']
tabular_feature_columns = column_data['tabular_feature_columns']
target_column = column_data['target_column']
logistic_regression_columns = cat_feature_columns + nlp_feature_columns + tabular_feature_columns + [target_column]
print('logistic_regression_columns')
print(logistic_regression_columns)

# Limit to items in logistic regression columns
train_df = train_df[logistic_regression_columns]
# Print out column names
print('column names')
for col in train_df.columns:
    print(col)

##################################################################################################

# Fit AutoGluon

# Convert from pandas to autogluon
train_data = TabularDataset(train_df)

# Create model save directory if it doesn't exist
os.makedirs('../AutogluonModels/Autogluon_Tabular_Only_SCF_Medium_Presets', exist_ok=True)

# Fit models
# Set seed to try to encourage stability
import numpy as np
np.random.seed(222)
# Run predictor
predictor = TabularPredictor(label='Rating', path='../AutogluonModels/Autogluon_Tabular_Only_SCF_Medium_Presets').fit(train_data=train_data, presets='medium_quality')
