
# NER of Firm Mentions in Each Call Transcript
# Save data of the firms mentioned

# Flag for if you are running this on the sample dataset
sample_run = False

# Packages
import pandas as pd
import os
import time
import spacy

# Spacy GPU setting
activated = spacy.prefer_gpu()
print('using gpu? ', activated)

print('beginning data load')

# Load in sample or full parquet file
if sample_run:
    df = pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates_Sample/all_data_fixed_quarter_dates_sample.parquet', columns = ['ticker', 'fixed_quarter_date', 'transcript'])
else:
    # list of files in '../../../Data/All_Data/All_Data_Fixed_Quarter_Dates'
    file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates') if f.endswith('.parquet')]
    # read in all parquet files
    df = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates/' + f, columns=['ticker', 'fixed_quarter_date', 'transcript']) for f in file_list])
print('dataframe')
print(df.head())

# load model and disable unnecessary components
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# Function to process transcript and get company mentions as a dataframe with counts
def get_companies(ticker, fixed_quarter_date, text):
    # Process the text
    doc=nlp(text)
    # List of companies mentioned
    companies = []
    # Initialize count
    for entity in doc.ents:
        # We'll select just entities that are tagged organizations (close enough to companies)
        if entity.label_ == "ORG":
            companies.append(entity.text)
    # Uppercase
    companies = [x.upper() for x in companies]
    # Not stripping punctuation, etc. for now since that might be useful information
    # Counts dataframe
    counts = pd.Series(companies).value_counts().to_frame().reset_index()
    counts.columns = ['company_mentioned', 'count']
    # Add ticker and fixed_quarter_date columns
    counts['ticker'] = ticker
    counts['fixed_quarter_date'] = fixed_quarter_date
    # Return DF
    return counts

# Test the function on one transcript
# Start timer
start_time = time.time()
print(get_companies(df['ticker'].iloc[0], df['fixed_quarter_date'].iloc[0], df['transcript'].iloc[0]))
# Record end time
end_time = time.time()
# Print time to process in minutes
print('Estimated Time to process: ', len(df) * (end_time - start_time) / 60)

# Apply the function to the transcript column
# Start timer
start_time = time.time()
# Iterate over dataframe rows
# Add to list of dataframes
list_of_dfs = []
for _, row in df.iterrows():
    list_of_dfs.append(get_companies(row['ticker'], row['fixed_quarter_date'], row['transcript']))
# Record end time
end_time = time.time()
# Print time to process in minutes
print('Time to process: ', (end_time - start_time) / 60)

# Concatenate list of dataframes
mentions_df = pd.concat(list_of_dfs)
print('combined dataframes')
print(mentions_df.head())
print(mentions_df.shape)
print(mentions_df.tail())

# Save parquet file of ticker, fixed_quarter_date, company_mentioned, count
if sample_run:
    mentions_df[['ticker', 'fixed_quarter_date', 'company_mentioned', 'count']].to_parquet(r'../../../Data/Company_Mentions/Company_Mentions_Sample.parquet')
else:
    mentions_df[['ticker', 'fixed_quarter_date', 'company_mentioned', 'count']].to_parquet(r'../../../Data/Company_Mentions/Company_Mentions.parquet')
