# ## get top 15 important numerical variables based on Lasso_path


nlp_feature_columns = column_data['nlp_feature_columns']
tabular_feature_columns = column_data['tabular_feature_columns']
target_column = column_data['target_column']
numeric_feature_columns = tabular_feature_columns + nlp_feature_columns

train_and_evaluate_model(df,numeric_feature_columns,[],target_column,custom_mapping,hyperparameter_settings=best_params_4,lassopath=True)
