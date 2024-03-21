# Train a Doc2Vec model and get the document vectors
# pieces excerpted from: https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4

# Packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import time
import utils
# Parallelization
import multiprocessing
cores = multiprocessing.cpu_count()
print('Number of cores:', cores)

# Flag for if you are running this on the sample dataset
sample_run = True
# Flag for limiting to training data only
train_set_only = True

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

# Limit to training data only
if train_set_only:
    df = df[df['train_test_80_20'] == 'train']
print('num rows')
print(df.shape[0])

# Data to feed Doc2Vec
data = df['transcript']
 
# preprocess the documents, and create TaggedDocuments
# tokenize lowercase docs
# follow best practice of using unique integer ids as tags
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]
 
# there are 2 Doc2Vec models, distributed memory (dm=1) and distributed bag of words (dm=0)
# we will train each and then concatenated, as recommended

# distributed bag of words - document vector based on words in/not in document
model_dbow = Doc2Vec(dm=0, # dbow
                     vector_size=300, # 300 dimension vector representations
                     negative=5, # 5 words to sample as negatives - ground truth not in the doc - for each positive word prediction we train with
                     hs=0, # set to zero to use negative sampling
                     min_count=2, # ignore all words with total frequency lower than this
                     sample=0, # do not downsample 
                     workers=cores) # multicore processing
model_dbow.build_vocab([x for x in tagged_data.values])
# train model
num_epochs = 30 # iterations through the dataset
for epoch in range(num_epochs):
    model_dbow.train(utils.shuffle([x for x in tagged_data.values]), total_examples=len(tagged_data.values), epochs=1)
    model_dbow.alpha -= 0.002 # learning rate decay
    model_dbow.min_alpha = model_dbow.alpha

# distributed memory - document vector context of words in document (skip-gram)
model_dmm = Doc2Vec(dm=1, # dmm
                    dm_mean=1, # use mean of context word vectors to predict skipped word - could turn on concatenated vector prediction but not doing that here, order of words in context probably not worth the cost
                    vector_size=300, # 300 dimension vector representations
                    window=10, # context window size
                    negative=5, # 5 words to sample as negatives - ground truth not in the doc - for each positive word prediction we train with
                    min_count=1, # ignore all words with total frequency lower than this
                    workers=cores, # multicore processing
                    alpha=0.065) # initial learning rate
model_dmm.build_vocab([x for x in tagged_data.values])
# train model
num_epochs = 30 # iterations through the dataset
for epoch in range(num_epochs):
    model_dmm.train(utils.shuffle([x for x in tagged_data.values]), total_examples=len(tagged_data.values), epochs=1)
    model_dmm.alpha -= 0.002 # learning rate decay
    model_dmm.min_alpha = model_dmm.alpha

# combine the models (as recommended) to improve quality
# free up memory by deleting temp training data
model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
# concatenate the two models
new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])

# get vectors by performing inference
documents = tagged_data.values
document_vectors = [new_model.infer_vector(doc.words) for doc in documents]

# Add document vectors to the dataframe
df['doc2vec_vector'] = document_vectors

# Unnest doc2vec_vector
df = df.explode('doc2vec_vector')

# Print head
print(df.head())

# Export to parquet
if not sample_run:
    pd.to_parquet(df, r'../../../Data/Doc2Vec_Vectors/train_set_only_doc2vec_vectors.parquet')

# End timer
end_time = time.time()
# Print time in minutes
print('Time to run:', (end_time - start_time)/60, 'minutes')

# Note sample is 100 rows, full roughly 8K rows
if sample_run:
    print('Estimated time on full dataset:', (8000 / 100) * ((end_time - start_time) / 60), 'minutes')
