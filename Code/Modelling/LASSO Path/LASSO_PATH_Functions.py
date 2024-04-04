
# LASSO Path Functions

# Packages
import pandas as pd
import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
# Kill warnings
import warnings
warnings.filterwarnings("ignore")

def get_lasso_path_top_features(df, numeric_feature_columns, cat_feature_columns, target_column, custom_mapping, model_name, change=False):
    """
    Get the top 15 important numerical variables based on Lasso path. Note this is a linear, not logistic model.

    Parameters:
    - df: DataFrame containing the dataset.
    - numeric_feature_columns: list of numeric columns to be used as features.
    - cat_feature_columns: list of categorical columns to be used as features.
    - target_column: column to be used as target.
    - custom_mapping: dictionary to encode the target variable.
    - change: whether problem is to predict changes
    """
   
    # Selecting features and target, and encoding target
    train_df = df[df['train_test_80_20'] == 'train']
    train_numeric_X = train_df[numeric_feature_columns].select_dtypes(include=['int64', 'float64'])
    train_cat_X = train_df[cat_feature_columns]
    X_train = pd.concat([train_numeric_X, train_cat_X], axis=1)
    if change:
        y_train = train_df[target_column].map({-2: 'downgrade', -1: 'downgrade', 0: 'no change', 1: 'upgrade', 2: 'upgrade'}).map(custom_mapping)
    else:
        y_train =  train_df[target_column].map(custom_mapping)

    # Preprocessing
    numeric_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feature_columns),
            ('cat', cat_transformer, cat_feature_columns)
        ])
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Lasso path
    alphas, coefs, _ = lasso_path(X_train_scaled, y_train, eps=1e-3, tol=0.001)
    coefs_lasso = coefs

    # Get the names of the features
    all_feature_names = np.array(numeric_feature_columns)
    
    # Get the dropping order
    dropping_order_list = []
    for i in range(len(coefs_lasso.T)-1,-1,-1):
        coef_lasso = coefs_lasso.T[i]
        zero_indices = np.where(coef_lasso == 0)[0]
        for ind in zero_indices:
            if ind not in dropping_order_list:
                dropping_order_list.append(ind)

    print("names of top 15 important variables:")
    most_important_names = all_feature_names[dropping_order_list[0:15]]
    print(most_important_names)

    # Save the LASSO path output
    joblib.dump(alphas, '../../../Output/Modelling/LASSO Path/' + model_name + '/' + model_name + '_lasso_path_alphas.pkl')
    joblib.dump(most_important_names, '../../../Output/Modelling/LASSO Path/' + model_name + '/' + model_name + '_lasso_path_most_important_names.pkl')
    joblib.dump(coefs_lasso, '../../../Output/Modelling/LASSO Path/' + model_name + '/' + model_name + '_lasso_path_coefs.pkl')
