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
    # list of files in '../../../../Data/All_Data/All_Data_with_NLP_Features'
    file_list = [f for f in os.listdir(r'../../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
    return df

def get_column_names_and_mapping_change(unsanitized_model_name):
    '''
    Uses the variable index Excel file to get column names for the model.

    Parameters:
    - unsanitized_model_name: name of the model, in unsanitized format

    Returns:
    - numeric_feature_columns: list of numeric columns to be used as features.
    - cat_feature_columns: list of categorical columns to be used as features.
    - target_column: column to be used as target.
    - custom_mapping: dictionary to encode the target variable.
    '''
    # Load variable index excel file
    variable_index = pd.read_excel('../../../../Variable Index.xlsx')

    # Model name column
    # For rating change models:
    if 'change_model' in unsanitized_model_name:
        # Clean model name is 'Change Model' plus the number (last character)
        clean_model_name = 'Change Model ' + unsanitized_model_name[-1]

    # Numeric features
    # Values of column_name where clean_model_name is X, and Data Type is Numeric
    numeric_feature_columns = variable_index[(variable_index[clean_model_name] == 'X') & (variable_index['Data Type'] == 'Numeric')]['column_name'].tolist()
    # Categorical features
    # Values of column_name where clean_model_name is X, and Data Type is not Numeric
    cat_feature_columns = variable_index[(variable_index[clean_model_name] == 'X') & (variable_index['Data Type'] != 'Numeric')]['column_name'].tolist()
    # If include_previous_rating is unsanitized_model_name, then add values where clean_model_name is 'X (Previous Rating Models)'
    # if 'include_previous_rating' in unsanitized_model_name:
    #     cat_feature_columns.append(variable_index[variable_index[clean_model_name] == 'X (Previous Rating Models)']['column_name'].values[0])
    # Target column
    # Values of column_name where column called model_name is Y
    target_column = variable_index[variable_index[clean_model_name] == 'Y']['column_name'].values[0]

    # Mapping for target column
    # Don't need mapping since change is already in number
    if 'rating' in unsanitized_model_name:
        custom_mapping = {'Same As Last Fixed Quarter Date': 0, 'Upgrade Since Last Fixed Quarter Date': 1, "Downgrade Since Last Fixed Quarter Date": -1}

    # Return the column names
    return numeric_feature_columns, cat_feature_columns, target_column, custom_mapping

def get_column_names_and_mapping(unsanitized_model_name):
    '''
    Uses the variable index Excel file to get column names for the model.

    Parameters:
    - unsanitized_model_name: name of the model, in unsanitized format

    Returns:
    - numeric_feature_columns: list of numeric columns to be used as features.
    - cat_feature_columns: list of categorical columns to be used as features.
    - target_column: column to be used as target.
    - custom_mapping: dictionary to encode the target variable.
    '''
    # Load variable index excel file
    variable_index = pd.read_excel('../../../../Variable Index.xlsx')

    # Model name column
    # For rating models:
    if 'rating_model' in unsanitized_model_name:
        # Clean model name is 'Rating Model' plus the number (last character)
        clean_model_name = 'Rating Model ' + unsanitized_model_name[-1]

    # Numeric features
    # Values of column_name where clean_model_name is X, and Data Type is Numeric
    numeric_feature_columns = variable_index[(variable_index[clean_model_name] == 'X') & (variable_index['Data Type'] == 'Numeric')]['column_name'].tolist()
    # Categorical features
    # Values of column_name where clean_model_name is X, and Data Type is not Numeric
    cat_feature_columns = variable_index[(variable_index[clean_model_name] == 'X') & (variable_index['Data Type'] != 'Numeric')]['column_name'].tolist()
    # If include_previous_rating is unsanitized_model_name, then add values where clean_model_name is 'X (Previous Rating Models)'
    if 'include_previous_rating' in unsanitized_model_name:
        cat_feature_columns.append(variable_index[variable_index[clean_model_name] == 'X (Previous Rating Models)']['column_name'].values[0])
    # Target column
    # Values of column_name where column called model_name is Y
    target_column = variable_index[variable_index[clean_model_name] == 'Y']['column_name'].values[0]

    # Mapping for target column
    if 'rating' in unsanitized_model_name:
        custom_mapping = {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5, 'CCC': 6, "CC": 7, "C": 8, "D": 9}

    # Return the column names
    return numeric_feature_columns, cat_feature_columns, target_column, custom_mapping

def prepare_matrices(df, numeric_feature_columns, cat_feature_columns, target_column, custom_mapping):
    """
    Prepare the feature matrices and target vector.

    Parameters:
    - df: DataFrame containing the dataset.
    - numeric_feature_columns: list of numeric columns to be used as features.
    - cat_feature_columns: list of categorical columns to be used as features.
    - target_column: column to be used as target.
    - custom_mapping: dictionary to encode the target variable.
    """
   
    # Selecting features and target, and encoding target
    train_df = df[df['train_test_80_20'] == 'train'].sort_values(by=['ticker', 'fixed_quarter_date'])
    test_df = df[df['train_test_80_20'] == 'test'].sort_values(by=['ticker', 'fixed_quarter_date'])
    train_numeric_X = train_df[numeric_feature_columns]
    train_cat_X = train_df[cat_feature_columns]
    test_numeric_X = test_df[numeric_feature_columns]
    test_cat_X = test_df[cat_feature_columns]
    X_train = pd.concat([train_numeric_X, train_cat_X], axis=1)
    X_test = pd.concat([test_numeric_X, test_cat_X], axis=1)
    y_train =  train_df[target_column].map(custom_mapping)
    y_test = test_df[target_column].map(custom_mapping)

    # Preprocessing
    numeric_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feature_columns),
            ('cat', cat_transformer, cat_feature_columns)
        ]
    )
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    print('feature names: ')
    print(preprocessor.get_feature_names_out())
    feature_names = preprocessor.get_feature_names_out()

    # Return the matrices
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

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
    if not os.path.exists('../../../../Output/Modelling/Logistic Regression/' + model_name):
        os.makedirs('../../../../Output/Modelling/Logistic Regression/' + model_name)
   
    # Create a preprocessing and modeling pipeline
    model = LogisticRegression(max_iter=1000) # could be 5000 max iterations but likely limited value from this many

    # Standard hyperparameter settings
    if "smote" in model_name:
        hyperparameter_settings = [
            # Non-penalized
            {'solver': ['saga'], 
            'penalty': [None], 
            'C': [1],  # C is irrelevant here but required as a placeholder
            'class_weight': [None], 
            'multi_class': ['ovr', 'multinomial']},
            # ElasticNet penalty
            {'solver': ['saga'], 
            'penalty': ['elasticnet'], 
            'C': [0.001, 0.01, 0.1, 1, 10], 
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0], 
            'class_weight': [None], 
            'multi_class': ['ovr', 'multinomial']}
        ]
    else:
        hyperparameter_settings = [
            # Non-penalized
            {'solver': ['saga'], 
            'penalty': [None], 
            'C': [1],  # C is irrelevant here but required as a placeholder
            'class_weight': ["balanced"], 
            'multi_class': ['ovr', 'multinomial']},
            # ElasticNet penalty
            {'solver': ['saga'], 
            'penalty': ['elasticnet'], 
            'C': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0], 
            'class_weight': ['balanced'], 
            'multi_class': ['ovr', 'multinomial']}
        ]        

    # Instantiate the grid search model
    grid_search = GridSearchCV(model, hyperparameter_settings, scoring='accuracy', cv=5, n_jobs=-1, refit=True) # refit is on by default, but marking True here for clarity. the model is refit on the whole training dataset after the best hyperparameters are found
    # Fit the grid search to the data (includes the refit step since set to true above)
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and the accuracy of the grid search
    print("Tuned hyperparameters:", grid_search.best_params_)
    train_accuracy_best_model = grid_search.best_estimator_.score(X_train_scaled, y_train)
    print("Train accuracy of best model: ", train_accuracy_best_model)
    print("Best mean CV accuracy:", grid_search.best_score_)
    # Coefficients
    print("Coefficients:", grid_search.best_estimator_.coef_)
    print("Corresponding class:", grid_search.best_estimator_.classes_)
    # Save these results
    joblib.dump(grid_search.best_estimator_, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_estimator.pkl')
    joblib.dump(grid_search.best_params_, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_params.pkl')
    joblib.dump(train_accuracy_best_model, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_train_accuracy_best_model.pkl')
    joblib.dump(grid_search.best_score_, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_best_score.pkl')

    # Return fitted model
    return grid_search.best_estimator_

def get_model_predictions(model, X_test_scaled, y_test, custom_mapping, model_name, target_column, full_df):
    """
    Get predictions from a logistic regression model.

    Parameters:
    - model: the model for predictions.
    - X_test_scaled: scaled feature matrix of the test set.
    - y_test: target vector of the test set.
    - custom_mapping: dictionary to encode the target variable.
    - model_name: name of the model to be saved - informs folder and file paths
    - target_column: column to be used as target.
    - full_df: the entire dataset, used to make a predictions DataFrame.
    """

    # Check input matrices
    print('X_test_scaled shape')
    print(X_test_scaled.shape)
    print('y_test shape')
    print(y_test.shape)

    # Create necessary directories if they do not exist
    if not os.path.exists('../../../../Output/Modelling/Logistic Regression/' + model_name):
        os.makedirs('../../../../Output/Modelling/Logistic Regression/' + model_name)
    # Place for predictions in '../../../Data/Predictions/Logistic Regression/' + model_name
    if not os.path.exists('../../../../Data/Predictions/Logistic Regression/' + model_name):
        os.makedirs('../../../../Data/Predictions/Logistic Regression/' + model_name)

    # Model prediction and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Save predictions
    # Add predictions to the test_df
    #print(full_df.head())
    #print(full_df['train_test_80_20'].value_counts())
    test_df_w_pred = full_df[full_df['train_test_80_20'] == 'test'].copy().sort_values(by=['ticker', 'fixed_quarter_date'])
    test_df_w_pred[model_name + '_predictions'] = list(y_pred)
    test_df_w_pred[model_name + '_predictions'] = test_df_w_pred[model_name + '_predictions'].map({v: k for k, v in custom_mapping.items()})
    # Keep only the necessary columns
    test_df_w_pred = test_df_w_pred[['ticker', 'fixed_quarter_date', target_column, model_name + '_predictions']]
    # Assert accuracy == share of correct predictions
    print('accuracy:', accuracy)
    print('share of correct predictions:', test_df_w_pred[model_name + '_predictions'].eq(test_df_w_pred[target_column]).mean())
    print('assertion that they match:')
    assert round(accuracy, 4) == round(test_df_w_pred[model_name + '_predictions'].eq(test_df_w_pred[target_column]).mean(), 4)
    # Save the DataFrame to Excel
    test_df_w_pred.to_excel('../../../../Data/Predictions/Logistic Regression/' + model_name + '/' + model_name + '_predictions.xlsx', index=False)
    
def create_model_figure_and_table_components(model_name, target_column, custom_mapping):
    """
    Create model figure and table components to be used.

    Parameters:
    - model_name: name of the model - informs folder and file paths of loaded predictions
    """

    # Load file of predictions
    predictions_df = pd.read_excel('../../../../Data/Predictions/Logistic Regression/' + model_name + '/' + model_name + '_predictions.xlsx')

    # Create y_test and y_pred
    y_test = predictions_df[target_column]
    y_pred = predictions_df[model_name + '_predictions']
    y_test_num = predictions_df[target_column].map(custom_mapping)
    y_pred_num = predictions_df[model_name + '_predictions'].map(custom_mapping)
    
    # Get accuracy, F1, and majority class baseline
    accuracy = accuracy_score(predictions_df[target_column], predictions_df[model_name + '_predictions'])
    f1 = f1_score(predictions_df[target_column], predictions_df[model_name + '_predictions'], average='weighted')
    majority_class_share_baseline = predictions_df[target_column].value_counts(normalize=True).max()

    # Dictionary of accuracy, F1, and majority class baseline
    acc_f1_majority = {'accuracy': accuracy, 'f1_score': f1, 'majority_baseline': majority_class_share_baseline}
    print(acc_f1_majority)
    # Save the dictionary
    joblib.dump(acc_f1_majority, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_acc_f1_majority.pkl')
    
    ### Calculate the share of predictions that are 1 or fewer ratings away from the actual ratings
    differences = np.abs(y_pred_num - y_test_num)
    close_predictions_share = np.mean(differences <= 1)
    exact_predictions_share = np.mean(differences == 0 )

    print(f"Share of predictions exactly as the actual: {exact_predictions_share:.2%}")
    print(f"Share of predictions 1 or fewer ratings away from actual: {close_predictions_share:.2%}")
    # Create and save a dictionary of the shares
    close_exact_dict = {'exact_predictions_share': exact_predictions_share, 'close_predictions_share': close_predictions_share}
    joblib.dump(close_exact_dict, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_close_exact_dict.pkl')

    # detailed evaluation with classification report
    report = classification_report(y_test, y_pred, digits=4)
    print('classification report:')
    print(report)
    # Save classification report object
    joblib.dump(report, '../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_classification_report.pkl')

    ### confusion matrix
    actual_labels = list(custom_mapping.keys())
    actual_labels = [label for label in actual_labels if label in y_test.unique() or label in y_pred.unique()]
    # Recode labels - if contains 'Upgrade', set to 'Upgrade', if contains 'Downgrade', set to 'Downgrade', if contains 'Same', set to 'Same'
    actual_labels_recoded = []
    for label in actual_labels:
        if 'Upgrade' in label:
            actual_labels_recoded.append('Upgrade')
        elif 'Downgrade' in label:
            actual_labels_recoded.append('Downgrade')
        else:
            actual_labels_recoded.append('Same')
    actual_labels = actual_labels_recoded
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=actual_labels).plot(cmap='Blues')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=actual_labels)
    #plt.show()
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_display.plot(cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    # Save as _no_title
    plt.savefig('../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_confusion_matrix_no_title.png')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # Save
    plt.savefig('../../../../Output/Modelling/Logistic Regression/' + model_name + '/' + model_name + '_confusion_matrix.png')
    plt.show()
