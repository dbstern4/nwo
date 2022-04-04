# Daniel Stern
# 4/4/22

# cd Google Drive/nwo


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import math

import json
from google.cloud import bigquery
from google.oauth2 import service_account

# Start timer
start = datetime.now()

# Print output to results.txt
results = open("results.txt", "w")

## Dataset

# Functions

# JSON key
with open('nwo-sample-5f8915fdc5ec.json') as info:
    access_data = json.load(info)

def get_utc(date_time):
    return datetime.timestamp(datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S'))

def get_datetime(timestamp):
    return datetime.fromtimestamp(timestamp)


# Twitter Data
credentials = service_account.Credentials.from_service_account_file('nwo-sample-5f8915fdc5ec.json')
client = bigquery.Client(credentials=credentials, project=access_data['project_id'])
sql = """
   SELECT *
   FROM nwo-sample.graph.tweets
   ORDER BY RAND ()
   LIMIT 2000
"""
df_twitter = client.query(sql).to_dataframe()

# Reddit Data
sql = """
   SELECT *
   FROM nwo-sample.graph.reddit
   ORDER BY RAND ()
   LIMIT 2000
"""
df_reddit = client.query(sql).to_dataframe()

# Combine documents (tweets and reddit posts)
documents = np.concatenate((np.array(df_twitter['tweet']), np.array(df_reddit['body'])))

# Combine time stamps
tweet_dates = np.array(df_twitter['created_at'])
tweet_stamps = []
for date in tweet_dates:
    tweet_stamps.append(get_utc(date))
tweet_stamps = np.array(tweet_stamps)

stamps = np.concatenate((tweet_stamps, np.array(df_reddit['created_utc'])))

# Date range
stamps_sorted = stamps.copy()
stamps_sorted.sort()

least_recent_stamp = stamps_sorted[0]
most_recent_stamp = stamps_sorted[len(stamps_sorted)-1]
print('First document on:', get_datetime(least_recent_stamp), file=results)
print('Last document on:', get_datetime(most_recent_stamp), file=results)
print('', file=results)

# Recency - high [most recent, high cutoff), medium [high cutoff, medium cutoff), low [medium cutoff, older)
high_recency_cutoff = most_recent_stamp - 7860600  # 3 months
medium_recency_cutoff = most_recent_stamp - 15811200  # 6 months


## Build Descriptors

# Functions

def get_words(post):
    clean = ''
    letters = []
    for letter in post:
        if letter.isalpha():
            letters.append(letter.lower())
        if letter == ' ':
            if len(letters) == 0:
                continue
            if not letters[len(letters)-1] == ' ':
                letters.append(letter)
    clean = clean.join(letters)
    return clean.split()

def get_recency_boost(utc):
    if utc > high_recency_cutoff:
        return 2
    elif utc > medium_recency_cutoff:
        return 1
    else:
        return 0

num_documents = documents.shape[0]


# OCCURRENCE INDICATORS DESCRIPTOR

# Each unique word has a vector of length = num_documents, values indicate word appearence in document
occurrence_indicators = {}

# Look at each document
for i, document in enumerate(documents):

    # Clean text
    words = get_words(document)

    # Get unique words in each document
    document_unique_words = {}
    document_unique_words_ordered = []

    # Look at each word in document
    for word in words:

        # Find unique words in document
        if word not in document_unique_words:
            document_unique_words[word] = None
            document_unique_words_ordered.append(word)

    # Do not add unique words to corpus if one-word document
    if len(document_unique_words_ordered) == 0:
        continue
    if len(document_unique_words_ordered) == 1:
        if document_unique_words_ordered[0] not in occurrence_indicators:
            continue

    # Recency of document
    boost = get_recency_boost(stamps[i])

    # Look at each unique word in document
    for word in document_unique_words_ordered:

        # Initialize embedding for new unique words in corpus
        if word not in occurrence_indicators:
            occurrence_indicators[word] = np.zeros(num_documents)

        # Update document occurrence for each unique word
        occurrence_indicators[word][i] = 1 + boost


# Master list of unique words, in order
unique_words_ordered = []
for key in occurrence_indicators.keys():
    unique_words_ordered.append(key)
num_unique_words = len(unique_words_ordered)


# CO-OCCURRENCE FREQUENCY DESCRIPTOR

# Each unique word has a vector of length = num_unique_words, values count document co-occurrence
co_occurrence_count = {}

# Initialize embeddings for unique words in corpus
for unique_word_p in unique_words_ordered:
    co_occurrence_count[unique_word_p] = {}
    for unique_word_q in unique_words_ordered:
        co_occurrence_count[unique_word_p][unique_word_q] = 0

# Look at each document
for i, document in enumerate(documents):

    # Clean text
    words = get_words(document)

    # Get unique words in each document
    document_unique_words = {}
    document_unique_words_ordered = []

    # Look at each word in document
    for word in words:

        # Find unique words in document
        if word not in document_unique_words:
            document_unique_words[word] = None
            document_unique_words_ordered.append(word)

    # Do not add unique words to corpus if one-word document
    if len(document_unique_words_ordered) == 0:
        continue
    if len(document_unique_words_ordered) == 1:
        if document_unique_words_ordered[0] not in occurrence_indicators:
            continue

    # Recency of document
    boost = get_recency_boost(stamps[i])

    # Look at each unique word in document
    for i, curr_word in enumerate(document_unique_words_ordered):

        # Stop after next word is last word
        if i == len(document_unique_words_ordered) - 1:
            break

        # For each unique pair of words in document, increase frequency count by 1 + recency boost
        for next_word in document_unique_words_ordered[i+1:]:
            co_occurrence_count[curr_word][next_word] += (1 + boost)
            co_occurrence_count[next_word][curr_word] += (1 + boost)


## Baseline Analysis

# Functions

# Baseline analysis - rank co-occurrence count
def baseline_search(query):
    co_occurrences = co_occurrence_count[query]
    return [k for k in sorted(co_occurrences, key=co_occurrences.get, reverse=True)]

baseline_result = baseline_search('iphone')
print('Baseline iphone:', file=results)
print(baseline_result[:10], file=results)
print('', file=results)


## Occurrence Indicators Analysis

# Functions

# Takes np array
def norm(vec):
    sum_of_squares = 0.0
    for x in vec:
        sum_of_squares += x * x
    return math.sqrt(sum_of_squares)

# Takes np array
def cosine_similarity(vec1, vec2):
    dot_product = 0.0
    for i in range(len(vec1)):
        dot_product += vec1[i] * vec2[i]
    return dot_product / (norm(vec1) * norm(vec2))

# Cosine Similarity of occurrence indicator embeddings
def occurrence_indicators_search(query):
    scores = {}

    for match in occurrence_indicators:
        scores[match] = cosine_similarity(occurrence_indicators[query], occurrence_indicators[match])
    del scores[query]

    return [k for k in sorted(scores, key=scores.get, reverse=True)]

occurrence_indicators_result = occurrence_indicators_search('iphone')
print('Occcurrence indicators iphone:', file=results)
print(occurrence_indicators_result[:10], file=results)
print('', file=results)


## Co-occurrence Count Analysis

# Takes dictionary
def norm_dict(vec):
    sum_of_squares = 0.0
    for x in vec:
        sum_of_squares += vec[x] * vec[x]
    return math.sqrt(sum_of_squares)

# Takes dictionary
def cosine_similarity_dict(vec1, vec2):
    dot_product = 0.0
    for x in vec1:
        if x in vec2:
            dot_product += vec1[x] * vec2[x]
    return dot_product / (norm_dict(vec1) * norm_dict(vec2))

# Cosine Similarity of co-occurrence count embeddings
def co_occurrence_count_search(query):
    scores = {}

    for match in co_occurrence_count:
        scores[match] = cosine_similarity_dict(co_occurrence_count[query], co_occurrence_count[match])
    del scores[query]

    return [k for k in sorted(scores, key=scores.get, reverse=True)]

co_occurrence_count_result = co_occurrence_count_search('iphone')
print('Co-occurrence count iphone:', file=results)
print(co_occurrence_count_result[:10], file=results)
print('', file=results)


## Combined Analysis

def combined_search(query):

    # Occurrence indicators
    scores_ind = {}
    for match in occurrence_indicators:
        scores_ind[match] = cosine_similarity(occurrence_indicators[query], occurrence_indicators[match])
    del scores_ind[query]

    # Co-occurrence counts
    scores_count = {}
    for match in co_occurrence_count:
        scores_count[match] = cosine_similarity_dict(co_occurrence_count[query], co_occurrence_count[match])
    del scores_count[query]

    # Equally weighted combined score
    scores = {}
    for word in scores_ind.keys():
        scores[word] = scores_ind[word] + scores_count[word]

    return [k for k in sorted(scores, key=scores.get, reverse=True)]

combined_result = combined_search('iphone')
print('Combined result iphone:', file=results)
print(combined_result[:10], file=results)


## Results

# Review 'iphone' combined result
for i, result in enumerate(combined_result):
    if result == 'biden' or result == 'perfume':
        print(result, 'ranked', i, 'in iphone search results', file=results)
print('', file=results)

# Find (most recent) document containing both iphone and top search result
top_result = combined_result[0]
occurrences_q = occurrence_indicators['iphone']
ocurrences_p = occurrence_indicators[top_result]

shared_doc = False
most_recent_shared_ind = 0
champion_score = 0
for i, occurrence in enumerate(occurrences_q):
    if occurrence > 0 and ocurrences_p[i] > 0:
        shared_doc = True
        curr_score = stamps[i]
        if curr_score > champion_score:
            champion_score = curr_score
            most_recent_shared_ind = i
if shared_doc:
    print('Sample document containing iphone and', top_result + ':', file=results)
    print(documents[most_recent_shared_ind],  file=results)
    print('Occurred at:', get_datetime(stamps[most_recent_shared_ind]), file=results)
    print('', file=results)

# Combined results for additional queries
queries = ['apple', 'biden', 'china', 'tesla', 'tv']
for query in queries:
    result = combined_search(query)
    print(query + ':', result[:10], file=results)
print('', file=results)


# Exploratory

print('Number of unique words:', len(unique_words_ordered), file=results)
print('', file=results)

# Stamp on 1/1 from 2010 to 2022
post_dist = {}
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
for year in years:
    post_dist[year] = 0
for stamp in stamps:
    stamp_date = get_datetime(stamp)
    for year in years:
        if stamp_date.year == year:
            post_dist[year] += 1

# Plot distribution
names = list(post_dist.keys())
values = list(post_dist.values())
plt.bar(range(len(post_dist)), values, tick_label=names)
plt.ylabel('Number of Documents')
plt.xlabel('Year')
plt.savefig('distribution.png')


# Most referenced terms
num_references = {}
for word in occurrence_indicators.keys():
    num_references[word] = np.sum(occurrence_indicators[word])
references_s = [(k, num_references[k]) for k in sorted(num_references, key=num_references.get, reverse=True)]
top_references = []
top_reference_counts = []
for reference in references_s[:10]:
    top_references.append(reference[0])
    top_reference_counts.append(reference[1])
top_references.reverse()
top_reference_counts.reverse()

# Plot most referenced terms
x = np.arange(10)
fig, ax = plt.subplots()
plt.bar(x, top_reference_counts)
plt.xticks(x, top_references)
plt.ylabel('Number of Documents')
plt.xlabel('Term')
plt.savefig('top_terms.png')


# Elapsed time
end = datetime.now()
print('Elapsed time:', end - start, file=results)

# Close output
results.close()










