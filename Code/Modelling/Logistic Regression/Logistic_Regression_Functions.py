# Logistic Regression Functions
# Defines functions needed for running logistic regression models on the dataset.

# Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
# Kill warnings
import warnings
warnings.filterwarnings("ignore")

def load_data():
    """
    Load the data from the Parquet file.

    Returns:
    A DataFrame containing the data.
    """
    # Read the Parquet file into a DataFrame
    # list of files in '../../../Data/All_Data/All_Data_with_NLP_Features'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
    return df

def prepare_matrices(df, numeric_feature_columns, cat_feature_columns, target_column, custom_mapping, change=False):
    """
    Prepare the feature matrices and target vector.

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
    test_df = df[df['train_test_80_20'] == 'test']
    train_numeric_X = train_df[numeric_feature_columns].select_dtypes(include=['int64', 'float64'])
    train_cat_X = train_df[cat_feature_columns]
    test_numeric_X = test_df[numeric_feature_columns].select_dtypes(include=['int64', 'float64'])
    test_cat_X = test_df[cat_feature_columns]
    X_train = pd.concat([train_numeric_X, train_cat_X], axis=1)
    X_test = pd.concat([test_numeric_X, test_cat_X], axis=1)
    if change:
        y_train = train_df[target_column].map({-2: 'downgrade', -1: 'downgrade', 0: 'no change', 1: 'upgrade', 2: 'upgrade'}).map(custom_mapping)
        y_test = test_df[target_column].map({-2: 'downgrade', -1: 'downgrade', 0: 'no change', 1: 'upgrade', 2: 'upgrade'}).map(custom_mapping)
    else:
        y_train =  train_df[target_column].map(custom_mapping)
        y_test = test_df[target_column].map(custom_mapping)

    # Preprocessing
    numeric_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feature_columns),
            ('cat', cat_transformer, cat_feature_columns)
        ])
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Return the matrices
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model_with_grid_search(X_train_scaled, y_train, model_name):
    """
    Train a logistic regression model with grid search.

    Parameters:
    - X_train_scaled: scaled feature matrix of the training set.
    - y_train: target vector of the training set.
    - model_name: name of the model to be saved - informs folder and file paths

    Returns:
    The fitted model.
    """

    # Create necessary directories if they do not exist
    if not os.path.exists('../../../Output/Modelling/Logistic Regression/' + model_name):
        os.makedirs('../../../Output/Modelling/Logistic Regression/' + model_name)
   
    # Create a preprocessing and modeling pipeline
    model = LogisticRegression(max_iter=1000) # could be 5000 max iterations but likely limited value from this many

    # Standard hyperparameter settings
    hyperparameter_settings = [
        # Non-penalized
        {'solver': ['lbfgs'], 
        'penalty': [None], 
        'C': [1],  # C is irrelevant here but required as a placeholder
        'class_weight': [None, 'balanced'], 
        'multi_class': ['ovr']},
        # ElasticNet penalty
        {'solver': ['saga'], 
        'penalty': ['elasticnet'], 
        'C': [0.001, 0.01, 0.1, 1, 10], 
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0], 
        'class_weight': [None, 'balanced'], 
        'multi_class': ['ovr']}
    ]

    # Instantiate the grid search model
    grid_search = GridSearchCV(model, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1, refit=True) # refit is on by default, but marking True here for clarity. the model is refit on the whole training dataset after the best hyperparameters are found
    # Fit the grid search to the data
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and the accuracy of the grid search
    print("Tuned hyperparameters:", grid_search.best_params_)
    print("Best mean CV accuracy:", grid_search.best_score_)
    # Coefficients
    print("Coefficients:", grid_search.best_estimator_.coef_)
    # Save these results
    joblib.dump(grid_search.best_estimator_, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_estimator.pkl')
    joblib.dump(grid_search.best_params_, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_params.pkl')
    joblib.dump(grid_search.best_score_, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_score.pkl')

    # Return fitted model
    return grid_search.best_estimator_

def evaluate_model(model, X_test_scaled, y_test, custom_mapping, model_name):
    """
    Evaluate a logistic regression model.

    Parameters:
    - model: the model to be evaluated.
    - X_test_scaled: scaled feature matrix of the test set.
    - y_test: target vector of the test set.
    - custom_mapping: dictionary to encode the target variable.
    - model_name: name of the model to be saved - informs folder and file paths
    """

    # Create necessary directories if they do not exist
    if not os.path.exists('../../../Output/Modelling/Logistic Regression/' + model_name):
        os.makedirs('../../../Output/Modelling/Logistic Regression/' + model_name)

    # Model prediction and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    majority_class_share_baseline = y_test.value_counts(normalize=True).max()
    
    # Dictionary of accuracy, F1, and majority class baseline
    acc_f1_majority = {'accuracy': accuracy, 'f1_score': f1, 'majority_baseline': majority_class_share_baseline}
    print(acc_f1_majority)
    # Save the dictionary
    joblib.dump(acc_f1_majority, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_acc_f1_majority.pkl')
    
    ### Calculate the share of predictions that are 1 or fewer ratings away from the actual ratings
    differences = np.abs(y_pred - y_test)
    close_predictions_share = np.mean(differences <= 1)
    exact_predictions_share = np.mean(differences == 0 )

    print(f"Share of predictions exactly as the actual: {exact_predictions_share:.2%}")
    print(f"Share of predictions 1 or fewer ratings away from actual: {close_predictions_share:.2%}")
    # Create and save a dictionary of the shares
    close_exact_dict = {'exact_predictions_share': exact_predictions_share, 'close_predictions_share': close_predictions_share}
    joblib.dump(close_exact_dict, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_close_exact_dict.pkl')

    # Set up display labels
    display_labels = []
    for v in np.sort(np.unique(y_test)):
        for key, value in custom_mapping.items():
            if value == v:
                display_labels.append(key)

    # detailed evaluation with classification report
    report = classification_report(y_test, y_pred, target_names=display_labels)
    # Save classification report object
    joblib.dump(report, '../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_classification_report.pkl')

    ### confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=display_labels)
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_display.plot(cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    # Save as _no_title
    plt.savefig('../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_confusion_matrix_no_title.png')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # Save
    plt.savefig('../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_confusion_matrix.png')
    plt.show()
