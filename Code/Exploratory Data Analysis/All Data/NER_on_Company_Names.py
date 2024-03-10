
# NER on Company Names
# (Slow file separated out from other code)

# Flag for if you are running this on the sample dataset
sample = False
# Modify this path as needed to run on your machine
sample_path = r'~\Box\STAT 222 Capstone\Intermediate Data\all_data_sample.csv'

# Packages
import pandas as pd
import os
import time

# Load in sample csv, or full parquet file
if sample:
    df = pd.read_csv(sample_path)
else:
    # list of files in '../../../Data/All_Data/All_Data_Fixed_Quarter_Dates'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates/' + f) for f in file_list])

import spacy

# load model and disable unnecessary components
activated = spacy.prefer_gpu()
print('using gpu? ', activated)
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'lemmatizer', 'attribute_ruler', 'morphologizer', 'textcat', 'toc2vec'])

# Function to process transcript and count company mentions
def count_companies(text):
    # Process the text
    doc=nlp(text)
    # Counter
    count = 0
    # Initialize count
    for entity in doc.ents:
        # We'll select just entities that are tagged organizations (close enough to companies)
        if entity.label_ == "ORG":
            count += 1
    return count

# Test the function on one transcript
# Start timer
start_time = time.time()
print(count_companies(df['transcript'].iloc[0]))
# Record end time
end_time = time.time()
# Print time to process in minutes
print('Estimated Time to process: ', len(df) * (end_time - start_time) / 60)

# Apply the function to the transcript column
# Start timer
start_time = time.time()
# potentially look into ways to do this in parallel or make more efficient...
df['company_mentions'] = df['transcript'].progress_apply(count_companies)
# Record end time
end_time = time.time()
# Print time to process in minutes
print('Time to process: ', (end_time - start_time).seconds / 60)

# Save parquet file of ticker, fixed_quarter_date, and company_mentions
df[['ticker', 'fixed_quarter_date', 'company_mentions']].to_parquet(r'../../../Data/All_Data/Company_Mentions.parquet')
