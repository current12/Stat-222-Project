# Auto-Sklearn on SCF
# Note this package requires Linux OS

# Packages
import numpy as np
from pprint import pprint
import sklearn.metrics
from sklearn.utils.multiclass import type_of_target
import autosklearn.classification
import pandas as pd
import os

# list of files in '../../../Data/All_Data/All_Data_Fixed_Quarter_Dates'
file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates') if f.endswith('.parquet')]
# read in all parquet files
df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates/' + f) for f in file_list])
print('dataframe')
print(df)

# Removing columns: 'Rating Rank AAA is 10', 'transcript', 'Investment_Grade', 'Change Direction Since Last Fixed Quarter Date', 'Change Since Last Fixed Quarter Date', 'Next Rating', 'Next Rating Date', 'next_rating_date_or_end_of_data'
df = df.drop(columns=['Rating Rank AAA is 10', 'transcript', 'Investment_Grade', 'Change Direction Since Last Fixed Quarter Date', 'Change Since Last Fixed Quarter Date', 'Next Rating', 'Next Rating Date', 'next_rating_date_or_end_of_data'])

# Split into train and test datasets
train_df = df[df['train_test_80_20'] == 'train'].reset_index(drop=True)
test_df = df[df['train_test_80_20'] == 'test'].reset_index(drop=True)

# Create X and y datasets
X_train = train_df.drop(columns=['Rating', 'train_test_80_20'])
y_train = train_df['Rating']
X_test = test_df.drop(columns=['Rating', 'train_test_80_20'])
y_test = test_df['Rating']

# Create classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)
automl.fit(X_train, y_train, dataset_name='Credit Ratings and Tabular Financial Data')

# Leaderboard and models found
print(automl.leaderboard())

# Print the final ensemble constructed by auto-sklearn
pprint(automl.show_models(), indent=4)

# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())

# Score final model
predictions = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
