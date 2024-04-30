# Evaluate Autogluon Tabular Only SCF
# Use the Autogluon AutoML library to predict ratings using tabular data locally (mostly just financial statement variables).

##################################################################################################

# Set model name to evaluate
model_name = 'ag-20240331_233723'

##################################################################################################

# Packages
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os

##################################################################################################

# Load Data

# list of files in '../../../Data/All_Data/All_Data_with_NLP_Features' directory
file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
# read in all parquet files
df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
print('dataframe')
print(df)

# Print out column names
print('column names')
for col in df.columns:
    print(col)

# Removing columns: 'Rating Rank AAA is 10', 'transcript', 'Investment_Grade', 'Change Direction Since Last Fixed Quarter Date', 'Change Since Last Fixed Quarter Date', 'Next Rating', 'Next Rating Date', 'next_rating_date_or_end_of_data'
df = df.drop(columns=['Rating Rank AAA is 10', 
                      'transcript', 
                      'Investment_Grade', 
                      'Change Direction Since Last Fixed Quarter Date', 
                      'Change Since Last Fixed Quarter Date', 
                      'Next Rating', 
                      'Next Rating Date', 
                      'next_rating_date_or_end_of_data'])


# Get test df
test_df = df[df['train_test_80_20'] == 'test'].reset_index(drop=True)

##################################################################################################

# Load Model
predictor = TabularPredictor.load('AutogluonModels/' + model_name)
print(predictor)

##################################################################################################

# Make Predictions

# Convert from pandas to autogluon
test_data = TabularDataset(test_df)

# Apply test
predictions = predictor.predict(test_data)
# Concatenate with test data values of 'ticker' and 'fixed_quarter_date'
# Use index values to line up
predictions = pd.concat([test_df[['ticker', 'fixed_quarter_date']], predictions], axis=1)
# Save to Excel
predictions.to_excel('../../../Data/Predictions/Autogluon/' + model_name + '_predictions.xlsx', index=False)
print(predictions)

##################################################################################################

# Evaluation and Leaderboard

# Evaluation
predictor.evaluate(test_data, silent=True)

# Leaderboard of models
leaderboard = predictor.leaderboard(test_data)
# Save to Excel
leaderboard.to_excel('../../../Output/Modelling/Autogluon/' + model_name + '_leaderboard.xlsx', index=False)
print(leaderboard)

# Keep columns model, score_test and output to LaTeX
# Rename to 'Model' and 'Test Accuracy'
lt_lb = leaderboard[['model', 'score_test']].rename(columns={'model': 'Model', 'score_test': 'Test Accuracy'})
# Replace WeightedEnsemble_L2 with WeightedEnsembleL2
lt_lb['Model'] = lt_lb['Model'].str.replace('WeightedEnsemble_L2', 'WeightedEnsembleL2')
lt_lb.to_latex('../../../Output/Modelling/Autogluon/' + model_name + '_leaderboard.tex', index=False)

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
hyperparameters_df.to_excel('../../../Output/Modelling/Autogluon/' + model_name + '_hyperparameters.xlsx', index=False)
print(hyperparameters_df)

##################################################################################################

# Feature Importance via Permutation

# Feature importance
fi = predictor.feature_importance(test_data)

# Save to Excel
fi.to_excel('../../../Output/Modelling/Autogluon/' + model_name + '_feature_importance.xlsx', index=False)

# Output 10 most important items to LaTeX
# Rename importance to 'Average Drop in Accuracy'
# Rename stddev to 'Standard Deviation'
# Rename pvalue to 'P-Value'
fi.reset_index().rename(columns={'index': 'feature'})[['feature', 'importance', 'stddev', 'p_value']].rename(columns={'feature': 'Feature', 'importance': 'Average Drop in Accuracy', 'stddev': 'Standard Deviation', 'p_value': 'P-Value'}).head(10).to_latex('../../../Output/Modelling/Autogluon/' + model_name + '_feature_importance.tex', index=False)

# Print entire df
pd.set_option('display.max_rows', None)
print(fi)
