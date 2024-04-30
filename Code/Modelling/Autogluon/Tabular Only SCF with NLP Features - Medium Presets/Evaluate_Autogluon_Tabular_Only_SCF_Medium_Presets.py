# Evaluate Autogluon Tabular Only SCF
# Use the Autogluon AutoML library to predict ratings using tabular data.

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import json

##################################################################################################

# Load Data

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

# Get test df
test_df = df[df['train_test_80_20'] == 'test'].reset_index(drop=True)

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
limited_test_df = test_df[logistic_regression_columns]
# Print out column names
print('column names')
for col in limited_test_df.columns:
    print(col)

##################################################################################################

# Load Model
predictor = TabularPredictor.load('../AutogluonModels/Autogluon_Tabular_Only_SCF_Medium_Presets')
print(predictor)

##################################################################################################

# Make Predictions

# Convert from pandas to autogluon
test_data = TabularDataset(limited_test_df)

# Apply test
predictions = predictor.predict(test_data)
# Concatenate with test data values of 'ticker' and 'fixed_quarter_date'
# Use index values to line up
predictions = pd.concat([test_df[['ticker', 'fixed_quarter_date']], predictions], axis=1)
# Save to Excel
predictions.to_excel('../../../../Data/Predictions/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_predictions.xlsx', index=False)
print(predictions)

##################################################################################################

# Evaluation and Leaderboard

# Evaluation
predictor.evaluate(test_data, silent=True)

# Leaderboard of models
leaderboard = predictor.leaderboard(test_data)
# Save to Excel
leaderboard.to_excel('../../../../Output/Modelling/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_leaderboard.xlsx', index=False)
print(leaderboard)

##################################################################################################

# Hyperparameters

# Model info including hyperparameters
pred_info = predictor.info()
# Get model hyperparameters
list_of_models = pred_info['model_info'].keys()
# List of dataframes to fill
list_of_dfs = []
# Iterate over models
for model in list_of_models:
    # Get hyperparameters
    hyperparameters = pred_info['model_info'][model]['hyperparameters']
    # Convert to dataframe
    df = pd.DataFrame.from_dict(hyperparameters, orient='index')
    # Add model name
    df['model'] = model
    # Append to list
    list_of_dfs.append(df)
# Concatenate all dataframes
hyperparameters_df = pd.concat(list_of_dfs).reset_index().rename(columns={'index': 'hyperparameter', 0: 'value'})[['model', 'hyperparameter', 'value']]
# Save to Excel
hyperparameters_df.to_excel('../../../../Output/Modelling/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_hyperparameters.xlsx', index=False)
print(hyperparameters_df)

##################################################################################################

# Feature Importance via Permutation

# Feature importance
fi = predictor.feature_importance(test_data)

# Save to Excel
fi.to_excel('../../../../Output/Modelling/Autogluon/Autogluon_Tabular_Only_SCF_Medium_Presets_feature_importance.xlsx', index=True)

# Print entire df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(fi)
