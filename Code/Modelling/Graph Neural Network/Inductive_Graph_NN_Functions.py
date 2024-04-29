# Inductive Graph Neural Network Functions
# Functions for training, testing, evaluating, and saving inductive graph neural network models

# Packages
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from dgl.nn.pytorch import conv as dgl_conv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# GraphSAGE model
class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats, # number of input features
                 n_hidden, # hidden layer size
                 out_dim, # output size
                 n_layers, # number of total layers
                 activation, # activation function to use
                 dropout, # dropout rate
                 aggregator_type): # type of aggregator to use
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList() # starter layers

        # input layer
        self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type,
                                             feat_drop=dropout, activation=activation))
        # output layer
        # does not have activation function
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type,
                                         feat_drop=dropout, activation=None))

    # forward pass
    def forward(self, g, features):
        h = features
        # pass through layers
        for layer in self.layers:
            h = layer(g, h)
        return h

# Node classification model/layer
class NodeClassification(nn.Module):
    def __init__(self, gconv_model, n_hidden, n_classes):
        super(NodeClassification, self).__init__() # classification model
        self.gconv_model = gconv_model # underlying graph convolutional model
        self.loss_fcn = torch.nn.CrossEntropyLoss() # loss function

    def forward(self, graph, features, labels, train_mask):
        logits = self.gconv_model(graph, features) # logits from the graph convolutional model
        return self.loss_fcn(logits[train_mask], labels[train_mask]) # return loss

def load_feature_and_class_data():
    """
    Load the feature and class data from the Parquet file.

    Returns:
    A DataFrame containing the data.
    """
    # Read the Parquet file into a DataFrame
    # list of files in '../../../../Data/All_Data/All_Data_with_NLP_Features'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
    # Sort by ticker and fixed_quarter_date, and create column node as index
    df = df.sort_values(['ticker', 'fixed_quarter_date']).reset_index(drop=True)
    df['node'] = df.index
    return df

def load_src_dst_data():
    '''
    Load the source and destination data from the company mentions file.

    Returns:
    A DataFrame containing unique source and destionation pairs/edges.
    '''

    # Company mentions file
    company_mentions_with_ticker = pd.read_excel('../../../Data/Company_Mentions/Company_Mentions_With_Ticker.xlsx')

    # Get tickers and counts
    pairwise_df = (company_mentions_with_ticker[['ticker', 'matched_ticker', 'fixed_quarter_date']]
                                               .drop_duplicates()
                                               .rename(columns={'ticker': 'src_ticker', 'matched_ticker': 'dst_ticker'}))

    # Note: don't need to do anything to handle symmetry (don't need to drop half of the pairs)

    # Return the DataFrame
    return pairwise_df

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
    variable_index = pd.read_excel('../../../Variable Index.xlsx')

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
    train_df = df[df['train_test_80_20'] == 'train'].sort_values(['ticker', 'fixed_quarter_date'])
    test_df = df[df['train_test_80_20'] == 'test'].sort_values(['ticker', 'fixed_quarter_date'])
    train_ticker_by_fixed_quarter_date = train_df[['ticker', 'fixed_quarter_date']]
    test_ticker_by_fixed_quarter_date = test_df[['ticker', 'fixed_quarter_date']]
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
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, train_ticker_by_fixed_quarter_date, test_ticker_by_fixed_quarter_date

def save_model(g, model, model_dir):
    '''
    Save model and graph in appropriate directory.

    Parameters:
    - g: Graph object
    - model: Graph neural network model
    - model_dir: Directory to save model and graph
    '''
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)

def get_predictions(logits, mask, labels):
    '''
    Given logits, prediction mask, and true labels, return true labels, predicted labels, and predicted probabilities for masked items

    Parameters:
    - logits: Logits from the model
    - mask: Mask for prediction
    - labels: True labels

    Returns:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_prob: Predicted probabilities
    '''
    # Get logits for masked items
    y_pred_prob = logits[mask]
    # Get true labels for masked items
    ground_truth_labels = labels[mask]
    # Predict label based on the class with the highest probability
    _, predict_labels = torch.max(y_pred_prob, dim=1)

    # Convert true and predicted labels to numpy arrays    
    y_true = ground_truth_labels.numpy()
    y_pred = predict_labels.numpy() 
    # Compute probabiliites with softmax
    y_pred_prob = y_pred_prob.numpy().reshape(-1, 2)
    y_pred_prob = scipy.special.softmax(y_pred_prob, axis=1)
    # Return true labels, predicted labels, and predicted probabilities
    return y_true, y_pred, y_pred_prob
    
def evaluate_on_train_and_val(model, graph, features, labels, train_mask, valid_mask):
    '''
    Evaluate the model on the training and validation datasets.

    Parameters:
    - model: Graph neural network model
    - graph: Graph object mapping source and destination nodes
    - features: Features for each node
    - labels: Labels for each node
    - train_mask: Mask for training data
    - valid_mask: Mask for validation data

    Returns:
    - train_acc: Accuracy on training dataset
    - train_f1: F1 score on training dataset
    - valid_acc: Accuracy on validation dataset
    - valid_f1: F1 score on validation dataset
    '''
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        # Compute logits
        logits = model.gconv_model(graph, features)
        
        # Training and validation dataset prediction
        train_y_true, train_y_pred, _ = get_predictions(logits, train_mask, labels)
        validation_y_true, validation_y_pred, _ = get_predictions(logits, valid_mask, labels)
            
        # Return accuracy on training and validation datasets
        return accuracy_score(train_y_true, train_y_pred), f1_score(train_y_true, train_y_pred), accuracy_score(validation_y_true, validation_y_pred), f1_score(validation_y_true, validation_y_pred)
    
def evaluate_on_test(model, graph, features, labels, test_mask):
    '''
    Evaluate the model on the test dataset.

    Parameters:
    - model: Graph neural network model
    - graph: Graph object mapping source and destination nodes
    - features: Features for each node
    - labels: Labels for each node
    - test_mask: Mask for test data

    Returns:
    - classification report: Classification report
    - test_y_true: True labels for test data
    - test_y_pred: Predicted labels for test data
    - test_y_prob: Predicted probabilities for test data
    '''
    # Set to evaluation
    model.eval()
    with torch.no_grad():
        # Compute logits
        logits = model.gconv_model(graph, features)
        
        # Test dataset prediction
        test_y_true, test_y_pred, test_y_prob = get_predictions(logits, test_mask, labels)
        # Return classification report, true labels, predicted labels, and predicted probabilities
        return classification_report(test_y_true, test_y_pred, zero_division=1, output_dict=True), test_y_true, test_y_pred, test_y_prob

def train_and_get_pred(model, optimizer, graph, features, labels, train_mask, val_mask, test_mask, n_epochs, inductive):
    '''
    Train the graph neural network and get predictions for the test dataset.

    Parameters:
    - model: Graph neural network model
    - optimizer: Optimizer
    - graph: Graph object mapping source and destination nodes
    - features: Features for each node
    - labels: Labels for each node
    - train_mask: Mask for training data
    - val_mask: Mask for validation data
    - test_mask: Mask for test data
    - n_epochs: Number of epochs to run training

    Returns:
    - model: Trained model
    - acc_dict: Dictionary containing accuracy metrics
    - y_true: True labels for test data
    - y_pred_prob: Predicted probabilities for test data
    - y_pred: Predicted labels for test data
    '''
    
    # Get graph
    full_graph, full_features, full_labels = graph, features, labels

    # Inductive model
    # Limit graph to nodes in training or validation dataset
    graph = graph.subgraph(torch.nonzero(torch.logical_or(train_mask, val_mask)).flatten())
    # Update features, labels, and masks for this subgraph
    features = features[graph.ndata[dgl.NID]]
    labels = labels[graph.ndata[dgl.NID]]
    train_mask = train_mask[graph.ndata[dgl.NID]]
    val_mask = val_mask[graph.ndata[dgl.NID]]
    
    # Keep track of time for each epoch
    duration = []
    # Iterate over epochs
    for epoch in range(n_epochs):

        # Start timer
        tic = time.time()

        # Set the model to training mode
        model.train()

        # Forward pass and loss computation - no gradient computation
        loss = model(graph, features, labels, train_mask)
        optimizer.zero_grad()
        # Backward pass - backpropagation, use optimizer to update values
        loss.backward()
        optimizer.step()
        
        # Evaluate model on training and validation datasets
        train_acc, train_f1, valid_acc, valid_f1 = evaluate_on_train_and_val(model, graph, features, labels, train_mask, val_mask)
        
        # Record time taken for epoch
        duration.append(time.time() - tic)

        # Print epoch statistics
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Training Accuracy {:.4f} | Training Loss {:.4f} | Training F1 {:.4f} | Validation Accuracy {:.4f} | Validation F1 {:.4f}".format(
                epoch, np.mean(duration), train_acc, loss.item(), train_f1, valid_acc, valid_f1)
        )

    # Evaluate model on test dataset
    acc_dict, y_true, y_pred_prob, y_pred = evaluate_on_test(model, full_graph, full_features, full_labels, test_mask)

    # Return model, accuracy dictionary, true labels, predicted probabilities, and predicted labels
    return model, acc_dict, y_true, y_pred_prob, y_pred

def run_inductive_model(train_and_val_df,
                          test_df,
                          src_dst_df,
                          model_dir,
                          prediction_file_path,
                          target_column,
                          n_hidden = 64,
                          n_layers = 2,
                          dropout = 0.0,
                          weight_decay = 5e-4,
                          n_epochs = 100,
                          lr = 0.01,
                          aggregator_type = "pool"):
    '''
    Run (train and evaluate) an inductive graph neural network model.

    Parameters:
    - train_and_val_df: DataFrame containing training and validation data
    - test_df: DataFrame containing test data
    - src_dst_df: DataFrame containing source and destination nodes
    - model_dir: Directory to save model and graph
    - prediction_file_path: File path to save model predictions
    - target_column: Target column
    - n_hidden: Number of hidden units
    - n_layers: Number of layers
    - dropout: Dropout rate
    - weight_decay: Weight decay
    - n_epochs: Number of epochs
    - lr: Learning rate
    - aggregator_type: Aggregator type
    '''

    # Create model_dir if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Create prediction_file_path directory if it doesn't exist
    if not os.path.exists(os.path.dirname(prediction_file_path)):
        os.makedirs(os.path.dirname(prediction_file_path))

    # Split train and validation data
    print("Further slice the train dataset into train and validation datasets.")
    train_df, validation_df = train_test_split(train_and_val_df, test_size=0.2, random_state=42, stratify=train_and_val_df[target_column].values)
    
    print(f"The training data has shape: {train_df.shape}.")
    print(f"The validation data has shape: {validation_df.shape}.")
    print(f"The test data has shape: {test_df.shape}.")
    
    # Put all data together - will mask later
    train_val_test_df = pd.concat([train_df, validation_df, test_df], axis=0)
    train_val_test_df.sort_index(inplace=True)
    
    # Masks indicating indices of train, validation, and test data
    print("Generate train, validation, and test masks.")
    train_mask = torch.tensor([True if ix in set(train_df.index) else False for ix in train_val_test_df.index])
    val_mask = torch.tensor([True if ix in set(validation_df.index) else False for ix in train_val_test_df.index])
    test_mask = torch.tensor([True if ix in set(test_df.index) else False for ix in train_val_test_df.index])
    
    # Construct graph using source and destination nodes
    graph = dgl.graph((src_dst_df["src"].values.tolist(), src_dst_df["dst"].values.tolist()))

    # Get features to tensor
    features = torch.FloatTensor(train_val_test_df.drop(['node', target_column], axis=1).to_numpy()) 

    # Number of nodes and features
    num_nodes, num_feats = features.shape[0], features.shape[1]
    print(f"Number of nodes = {num_nodes}")
    print(f"Number of features for each node = {num_feats}")

    # Get labels/classes to tensor
    labels = torch.LongTensor(train_val_test_df[target_column].values)
    n_classes = train_val_test_df[target_column].nunique()
    print(f"Number of classes = {n_classes}.")

    # Add features to graph
    graph.ndata['feat'] = features

    # Initialize model
    print("Initializing Model")
    # Create model
    gconv_model = GraphSAGEModel(num_feats, n_hidden, n_classes, n_layers, F.relu, dropout, aggregator_type)
    # Node classification task
    model = NodeClassification(gconv_model, n_hidden, n_classes)
    print("Initialized Model")

    # Check model
    print(model)
    print(model.parameters())

    # If GPU is available, move model to GPU
    print('GPU available: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Moving model to GPU")
        model.cuda()
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train and get predictions
    print("Starting Model training and prediction")
    model, _, y_true, y_pred, _ = train_and_get_pred(model, optimizer, graph, features, labels, train_mask, val_mask, test_mask, n_epochs)
    print("Finished Model training and prediction")

    # Save model
    print("Saving model")
    save_model(graph, model, model_dir)
    
    # Output model predictions and probabilities
    print("Saving model predictions for test data")
    pd.DataFrame.from_dict(
        {
            'target': y_true.reshape(-1, ),
            'pred': y_pred.reshape(-1, ),
        }
    ).to_csv(prediction_file_path, index=False)
