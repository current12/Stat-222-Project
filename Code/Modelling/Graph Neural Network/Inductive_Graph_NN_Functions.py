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

# Save model and graph in appropriate directories
def save_model(g, model, model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(g, f)

def get_predictions(logits, mask, labels):
    '''
    Given logits, prediction mask, and true labels, return true labels, predicted labels, and predicted probabilities for masked items
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

def evaluate(model, graph, features, labels, train_mask, valid_mask, test_mask=None):
    '''
    Evaluate the model for the graph, making use of appropriate masks.
    '''
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        # Compute logits
        logits = model.gconv_model(graph, features)
        
        # Test dataset prediction
        if test_mask is not None:
            test_y_true, test_y_pred, test_y_prob = get_predictions(logits, test_mask, labels)
            # Return classification report, true labels, predicted labels, and predicted probabilities
            return classification_report(test_y_true, test_y_pred, zero_division=1, output_dict=True), test_y_true, test_y_pred, test_y_prob
        
        # Training and validation dataset prediction
        train_y_true, train_y_pred, _ = get_predictions(logits, train_mask, labels)
        validation_y_true, validation_y_pred, _ = get_predictions(logits, valid_mask, labels)
            
        # Return accuracy on training and validation datasets
        return accuracy_score(train_y_true, train_y_pred), f1_score(train_y_true, train_y_pred), accuracy_score(validation_y_true, validation_y_pred), f1_score(validation_y_true, validation_y_pred)


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
    - inductive: Boolean indicating whether the model is inductive
    '''
    
    # Get graph
    full_graph, full_features, full_labels = graph, features, labels
    # Inductive learning
    if inductive:
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
        
        # Evaluate model on training and validation datasets (don't need test_mask argument)
        train_acc, train_f1, valid_acc, valid_f1 = evaluate(model, graph, features, labels, train_mask, val_mask)
        
        # Record time taken for epoch
        duration.append(time.time() - tic)

        # Print epoch statistics
        print(
            "Epoch {:05d} | Time(s) {:.4f} | Training Accuracy {:.4f} | Training Loss {:.4f} | Training F1 {:.4f} | Validation Accuracy {:.4f} | Validation F1 {:.4f}".format(
                epoch, np.mean(duration), train_acc, loss.item(), train_f1, valid_acc, valid_f1)
        )

    # Evaluate model on test dataset
    acc_dict, y_true, y_pred_prob, y_pred = evaluate(model, full_graph, full_features, full_labels, train_mask, val_mask, test_mask)

    # Return model, accuracy dictionary, true labels, predicted probabilities, and predicted labels
    return model, acc_dict, y_true, y_pred_prob, y_pred

def inductive_entry_train(training_dir,
                          model_dir,
                          output_dir,
                          predictions_file_name,
                          features_train,
                          features_test,
                          target_column,
                          source_destination_node_index,
                          n_hidden = 64,
                          n_layers = 2,
                          dropout = 0.0,
                          weight_decay = 5e-4,
                          n_epochs = 100,
                          lr = 0.01,
                          aggregator_type = "pool",
                          inductive = True):

    # Load train and test datasets
    print("Read train and test dataset.")
    train_validation_df = pd.read_csv(os.path.join(training_dir, features_train), header=0, index_col=0)
    test_df = pd.read_csv(os.path.join(training_dir, features_test), header=0, index_col=0)
    
    src_dst_df = pd.read_csv(os.path.join(training_dir, source_destination_node_index), header=0)

    print("Further slice the train dataset into train and validation datasets.")
    train_df, validation_df = train_test_split(train_validation_df, test_size=0.2, random_state=42, stratify=train_validation_df[target_column].values)
    
    print(f"The training data has shape: {train_df.shape}.")
    print(f"The validation data has shape: {validation_df.shape}.")
    print(f"The test data has shape: {test_df.shape}.")
    
    train_val_test_df = pd.concat([train_df, validation_df, test_df], axis=0)
    train_val_test_df.sort_index(inplace=True)
    
    # Masks indicating indices of train, validation, and test data
    print("Generate train, validation, and test masks.")
    train_mask = torch.tensor([True if ix in set(train_df.index) else False for ix in train_val_test_df.index])
    val_mask = torch.tensor([True if ix in set(validation_df.index) else False for ix in train_val_test_df.index])
    test_mask = torch.tensor([True if ix in set(test_df.index) else False for ix in train_val_test_df.index])
    
    # Construct graph using source and destination nodes
    graph = dgl.graph((src_dst_df["src"].values.tolist(), src_dst_df["dst"].values.tolist()))

    features = torch.FloatTensor(train_val_test_df.drop(['node', target_column], axis=1).to_numpy()) 
    num_nodes, num_feats = features.shape[0], features.shape[1]
    print(f"Number of nodes = {num_nodes}")
    print(f"Number of features for each node = {num_feats}")

    labels = torch.LongTensor(train_val_test_df[target_column].values)
    n_classes = train_val_test_df[target_column].nunique()
    print(f"Number of classes = {n_classes}.")

    graph.ndata['feat'] = features

    print("Initializing Model")
    gconv_model = GraphSAGEModel(num_feats, n_hidden, n_classes, n_layers, F.relu, dropout, aggregator_type)
    
    # Node classification task
    model = NodeClassification(gconv_model, n_hidden, n_classes)
    print("Initialized Model")

    print(model)
    print(model.parameters())
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Starting Model training")
    model, metric_table, y_true, y_pred, y_pred_prob = train_and_get_pred(model, optimizer, graph, features, labels, train_mask, val_mask, test_mask, n_epochs, inductive)
    print("Finished Model training")

    print("Saving model")
    save_model(graph, model, model_dir)
    
    # Output model predictions and probabilities
    print("Saving model predictions for test data")
    pd.DataFrame.from_dict(
        {
            'target': y_true.reshape(-1, ),
            'pred': y_pred.reshape(-1, ),
        }
    ).to_csv(os.path.join(output_dir, predictions_file_name), index=False)
