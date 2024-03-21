# Train a Doc2Vec model and get the document vectors
# https://www.geeksforgeeks.org/doc2vec-in-nlp/

# Packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import time

# Flag for if you are running this on the sample dataset
sample_run = True

# Start timer
start_time = time.time()

# Load earnings calls
# Load in sample or full parquet file
print('beginning data load')
if sample_run:
    df = pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates_Sample/all_data_fixed_quarter_dates_sample.parquet', columns=['ticker', 'fixed_quarter_date', 'transcript'])
else:
    # list of files in '../../../Data/All_Data/All_Data_Fixed_Quarter_Dates'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates/' + f, columns=['ticker', 'fixed_quarter_date', 'transcript']) for f in file_list])
print('dataframe')
print(df.head())

# Data to feed Doc2Vec
data = df['transcript']
 
# preprocess the documents, and create TaggedDocuments
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]
 
# train the Doc2vec model
model = Doc2Vec(vector_size=200, min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
 
# get the document vectors
document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in data]
 
# Add document vectors to the dataframe
df['doc2vec_vector'] = document_vectors

# Unnest doc2vec_vector
df = df.explode('doc2vec_vector')

# Print head
print(df.head())

# Export to parquet
if not sample_run:
    pd.to_parquet(df, r'../../../Data/Doc2Vec_Vectors/all_data_fixed_quarter_dates_sample_doc2vec.parquet')

# End timer
end_time = time.time()
# Print time in minutes
print('Time to run:', (end_time - start_time)/60, 'minutes')

# Note sample is 100 rows, full roughly 8K rows
if sample_run:
    print('Estimated time on full dataset:', (8000 / 100) * ((end_time - start_time) / 60), 'minutes')
