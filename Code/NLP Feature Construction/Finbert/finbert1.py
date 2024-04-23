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

if __name__ == '__main__':
    # list of files in '../../../Data/All_Data/All_Data_with_NLP_Features'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])

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

    # Positive Score Calculation
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
   
    # Chunk 1 (0-1890)
    start_idx = 1000
    end_idx = 1200
    print(f"Calculating pos_score for index {start_idx} to {end_idx}.")

    chunk = df.iloc[start_idx:end_idx, [0, 1,16]]
    texts = chunk['transcript'].values
    #print(texts)
    #pos_results = parrallelize_lazy(texts, calculate_pos_score)
    pos_results = [calculate_pos_score(text) for text in texts]
    # Convert the list elements to string, each followed by a newline character
    #data_str = '\n'.join(str(score) for score in pos_results[0])
    data_str = '\n'.join(str(score) for score in pos_results)

    # Specify the file name
    file_name = f"pos_result_{start_idx}_{end_idx}.txt"

    # Open the file in write mode and write the string representation of the list
    with open(file_name, "w") as file:
        file.write(data_str)
