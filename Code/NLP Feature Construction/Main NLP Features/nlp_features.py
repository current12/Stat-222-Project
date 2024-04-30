from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import string

import pandas as pd
import os
import nltk
import dask
#import dask.dataframe as dd
import math
import numpy as np
import pysentiment as ps

import re
from readability import Readability

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
harvard = pd.read_excel("inquirerbasic.xls")
n_core = 20 # Change num of CPUs here
n_calls_to_run = 10    # Change how many calls to process

def count_consecutive_nnp(tagged_tokens):
    count = 0
    max_count = 0
    
    for _, tag in tagged_tokens:
        if tag == "NNP":
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
            
    return max_count

# Filter for calls that are too short or have no content
def get_bad_calls(call_text):
    # sentences = sent_tokenize(call_text)
    #global i
    # Now, tokenize each sentence into words and run POS tagging
    # for sentence in sentences:
    #     tokens = word_tokenize(sentence)  # Tokenize the current sentence into words
    #     tokens_no_punct = [token for token in tokens if token not in string.punctuation]
    #     tagged_tokens = pos_tag(tokens_no_punct)   # Run POS tagging on these tokens
    #     #print(tagged_tokens)
    tokens = word_tokenize(call_text)
    #print(tokens)
    tokens_no_punct = [token for token in tokens if token not in string.punctuation]
    #print(tokens_no_punct)
    tagged_tokens = pos_tag(tokens_no_punct)
    #print(tagged_tokens)
    nnp_count = count_consecutive_nnp(tagged_tokens)

    word_count = len(tokens_no_punct)
    if nnp_count > 50 or word_count <= 500:
        return True
    else:
        return False

def parrallelize_lazy(texts, function, args = None):
    tasks = []
    n = len(texts)
    if args is not None:
        for i in range(n):
            tasks.append(dask.delayed(function)(texts[i], *args))    # Lazy computing
    else:
        for i in range(n):
            tasks.append(dask.delayed(function)(texts[i]))
        #print("Current process: ", round(i/n,1))
    result = dask.compute(tasks)

    return result

def get_num_to_word(text):
    tokens = nltk.word_tokenize(text)
    types = set(tokens)

    # Get the number of word types (without numbers)
    # Matches integers (4 digits max) and floats without taking into account cases like covid-19 or 10-K
    re_pat = r'(?<![\w.-])\d{1,4}(?:,\d)*(?:\.\d+)?(?![\w.-])'

    # Delete the count of numeric vals in types to get only word types
    N_unique_number = sum(bool(re.match(re_pat, type)) for type in types)
    N_word_tokens = len(types) - N_unique_number    # Num of unique words
    N_num = sum(bool(re.match(re_pat, token)) for token in tokens)    # Count of all numeric vals

    #print(N_unique_number, N_num)
    numeric_transparency_ratio = round(N_num / N_word_tokens, 2)

    return numeric_transparency_ratio

def get_gf_score(text):
    r = Readability(text)
    try:
        gf = r.gunning_fog()
    except:
        print("Failed to calculate gs score.")
        print(text)
        return None
    return gf.score

def get_call_length(text):
    tokens = word_tokenize(text)
    #tokens = text.apply(word_tokenize)
    tokens_no_punct = [token for token in tokens if token not in string.punctuation]
    
    full_len = len(tokens_no_punct)
    return full_len

def get_positivity_score(text, model):
    tokens = word_tokenize(text)
    tokens_no_punc = [token for token in tokens if token not in string.punctuation]
    score = model.get_score(tokens_no_punc)
    
    positive_count = score['Positive']
    negative_count = score['Negative']
    #print(positive_count, negative_count)

    net_sent_score = math.log10((positive_count+1) / (negative_count+1))
    
    return net_sent_score

# Map NLTK's POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # As a default, if the POS tag is not recognized, treat it as a noun
        #print("unknown")
        return wordnet.NOUN

# Lemmatize tokens with the appropriate POS tag
def lemmatize_call(text):
    # Tokenize the corpus into sentences
    sentences = nltk.sent_tokenize(text)

    # Function to remove punctuation from tokens
    def remove_punctuation(tokens):
        return [token for token in tokens if token not in string.punctuation] 
    # Now tokenize each sentence into words, remove punctuation, and tag
    tagged_corpus = [nltk.pos_tag(remove_punctuation(word_tokenize(sentence))) for sentence in sentences]
    
    lemmatized_tokens = []
    for tokens in tagged_corpus:    # Tokens: list of tuples: (word, pos)
        for word, pos_tag in tokens: 
            #print(word, pos_tag)
            wordnet_pos = get_wordnet_pos(pos_tag)
            # Lemmatize the token with the appropriate POS tag
            lemmatized_token = lemmatizer.lemmatize(word, pos=wordnet_pos)
            #print(word, wordnet_pos)
            lemmatized_tokens.append(lemmatized_token.lower())

    return lemmatized_tokens

# Choose words with # (multiple meaning)
def get_words_variants(harvard):   
    condition = [ind for ind, val in enumerate(harvard['Entry'].str.contains('#').values) if val == True]
    harvard_filtered = harvard.loc[condition]

    def extract_frequency(defined_str):
        try:
            freq = float(defined_str.split('|')[1].split('%')[0].strip())
        except:
            #print(defined_str)
            return None
        return freq

    # Adding a column for frequency
    harvard_filtered['Frequency'] = harvard_filtered['Defined'].apply(extract_frequency)

    harvard_filtered = harvard_filtered.dropna(subset = "Frequency")

    # Splitting the 'Word' column to identify the root word and its variant number
    harvard_filtered['Root'], harvard_filtered['Variant'] = harvard_filtered['Entry'].str.split('#', 1).str[0], harvard_filtered['Entry'].str.split('#', 1).str[1]

    # Filling NaN in 'Variant' with '0' to include the base word in comparison
    harvard_filtered['Variant'] = harvard_filtered['Variant'].fillna('0')
    # Finding the word with the highest frequency for each root word
    idx = harvard_filtered.groupby('Root')['Frequency'].idxmax()

    # Filtering the DataFrame to only include the rows with the highest frequency word of each root
    filtered_df = harvard_filtered.loc[idx]

    # Dropping extra columns if desired
    filtered_df = filtered_df.drop(['Frequency', 'Variant'], axis=1)
    
    return filtered_df

def get_words_unique(harvard):
    condition = [ind for ind, val in enumerate(harvard['Entry'].str.contains('#').values) if val == False]
    harvard_unique = harvard.loc[condition]
    return harvard_unique

def get_tone_count(text):
    lemmatized_tokens = lemmatize_call(text)
    tones = ['Positiv', 'Negativ', 'Strong', 'Weak', 
             'Active', 'Passive', "Ovrst", "Undrst"]
    word_dic = {}
    
    global harvard
    df_no_hashtag = get_words_unique(harvard)
    df_hashtag = get_words_variants(harvard)
    #is_picklable(df_no_hashtag)
    #is_picklable(df_hashtag)
    def check_if_token_in_set(token, word_set):
        if token in word_set:
            #print(token)
            return +1
        else:
            return 0
        
    def get_tone_dic (tone, df_no_hashtag, df_hash_tag):
        #final_list = []
        #for df in [words_no_hashtag, words_hash_tag]:
        words_no_hashtag = df_no_hashtag.dropna(subset=tone)
        result_unique = words_no_hashtag['Entry'].values
        #final_list.append(result)
        
        words_hashtag = df_hash_tag.dropna(subset=tone)
        result_root = words_hashtag['Root'].values
        final_list = np.concatenate((result_unique, result_root))

        final_list = [w.lower() for w in final_list]

        return final_list

    for tone in tones:
        tone_word_list = get_tone_dic(tone, df_no_hashtag, df_hashtag)
        #is_picklable(tone_word_list)
        
        count = [check_if_token_in_set(tok, tone_word_list) for tok in lemmatized_tokens]
        word_dic[tone] = np.sum(count)
    
    return word_dic

# ------------------------------------------------------------------------------------------------------------------------------------------------
file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates') if f.endswith('.parquet')]
# read in all non-nlp files
df_all = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_Fixed_Quarter_Dates/' + f) for f in file_list])

file_list = [f for f in os.listdir(r'../../../Data/All_Data/All_Data_with_NLP_Features') if f.endswith('.parquet')]
# read in all nlp files
df_nlp = pd.concat([pd.read_parquet(r'../../../Data/All_Data/All_Data_with_NLP_Features/' + f) for f in file_list])

df_merge = pd.merge(df_all, df_nlp, on=["ticker", "fixed_quarter_date"], how="left", indicator=True)
df = df_merge[df_merge['_merge'] == 'left_only']

print((len(df_all) - len(df_nlp)) == len(df))
df = df.loc[:, ["ticker","transcript_x", "Rating_x", "Sector_x"]]
df.columns = ['ticker', 'transcript', 'Rating', 'Sector']
df = df.reset_index(drop=True)


# Set up parrallelization with 20 cores
dask.config.set(scheduler='processes', num_workers = n_core)

# Text chunks
# df_dd = dd.from_pandas(df, npartitions=n_core)
bad_call_results = parrallelize_lazy(df['transcript'], get_bad_calls)

bad_call_idx = [index for index, value in enumerate(bad_call_results[0]) if value]
sum(bad_call_results[0]) # How many bad calls?

# Drop all bad calls
df_clean = df.loc[~df.index.isin(bad_call_idx)]
df_clean = df_clean.reset_index()
print(f"{bad_call_idx} are bad calls and are dropped.")

texts = df_clean['transcript'][:n_calls_to_run]

# Feature: num_transparency
results = parrallelize_lazy(texts, get_num_to_word)
df_clean['num_transparency'] = results[0]
print("Numerical transparency feature added.")

# Feature: Readability using Gunning-Fog Index
results = parrallelize_lazy(texts, get_gf_score)
df_clean['readability'] = results[0]
print("Readability (Gunning-Fog Index) feature added.")

# Feature: Call length
results = parrallelize_lazy(texts, get_call_length)
df_clean['word_count'] = results[0]
print("Call length feature added.")

# Feature: Number of Questions
df_clean['num_questions'] = [text.count("?") for text in texts]
print("Number of questions feature added.")

# Feature: Tone
harvard = pd.read_excel("inquirerbasic.xls")
results = [get_tone_count(i) for i in texts]    # No parallelization
print("Tone counts calculated.")

# PCA
df_pca = pd.DataFrame(results)
col_names = ['PN', 'SW', 'AP', 'OU']
i = 0
for j in range(4):
    first = df_pca.iloc[:, i]
    second = df_pca.iloc[:, i+1]
    col_name = col_names[j]
    df_pca[col_name] = first/second
    i += 2
scaler = StandardScaler()
df_pca_standardized = scaler.fit_transform(df_pca[col_names])

pca = PCA(n_components=1)
principal_components = pca.fit_transform(df_pca_standardized)
df_pca['TONE1'] = principal_components
print("PCA on tone completed.")

# Save the datasets
df_pca.to_parquet("word_tone_new.parquet")
print("PCA data saved to 'word_tone_new.parquet'.")

df_clean.to_parquet("nlp_features_new.parquet")
print("All features saved to 'nlp_features_new.parquet'.")
