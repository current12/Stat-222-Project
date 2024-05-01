import pandas as pd
import os
import nltk
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import math
import numpy as np

import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

import re
import warnings
warnings.filterwarnings("ignore")

import time

# Positivity Score Calculation
def calculate_pos_score(text):
    sentences = nltk.sent_tokenize(text)
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
    try:
        results = nlp(sentences)
    except:
        print("Issues when getting positivity score:")
        print(text)

        return -1

    labels = [sent['label'] for sent in results]
    positive_count = labels.count('Positive')
    negative_count = labels.count('Negative')

    #print(positive_count, negative_count)

    net_sent_score = math.log10((positive_count+1) / (negative_count+1))

    return net_sent_score

if __name__ == '__main__':
    # Change chunksize here
    start_idx = 800
    end_idx = 1500
    print(f"Calculating pos_score for index {start_idx} to {end_idx}.")

    # list of files in '../../../Data/All_Data/All_Data_with_NLP_Features'
    #file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
    # read in all parquet files
    #df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])
    file_path = '../../../Data/All_Data/df_additional.parquet'
    df = pd.read_parquet(file_path)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('set up device')
    print('device: ', device)
    ####################################################################################################
    # Load model
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

    finbert.to(device)
    print('loaded model')

    #chunk = df.iloc[start_idx:end_idx, [0, 1,16]]
    texts = df['transcript'][start_idx:end_idx].values
    #print(texts)
    #pos_results = parrallelize_lazy(texts, calculate_pos_score)
    pos_results = [calculate_pos_score(text) for text in texts]
    # Convert the list elements to string, each followed by a newline character
    #data_str = '\n'.join(str(score) for score in pos_results[0])
    data_str = '\n'.join(str(score) for score in pos_results)

    # Specify the file
    out_path = '../../../Data/Finbert_Pos_Score/'
    file_name = f"{out_path}pos_result_new_{start_idx}_{end_idx}.txt"

    # Open the file in write mode and write the string representation of the list
    with open(file_name, "w") as file:
        file.write(data_str)
