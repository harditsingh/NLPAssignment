import json
import ast
import pandas as pd
import nltk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import collections
from nltk.sentiment.vader import SentimentIntensityAnalyzer as analyser

import punctuations

stopWords = set(nltk.corpus.stopwords.words('english'))
punct_list = punctuations.punctuations

print('Reading Data...\n')
review_res = pd.read_csv('processedData.csv', sep=',', encoding='utf-8')
print('Data READ\n')

print('Replacing NAN values with empty string\n')
review_res = review_res.replace(np.nan, '', regex=True)
review_res = review_res[['reviewerID','productID','overall','reviewTime','unixReviewTime','summary','reviewText']]


# end of Dataframe creation..
print('Classification of Reviews\n')
# classification of reviews according to overall rating
senti_df = review_res.copy(deep=True)
senti_positive_rev = []
senti_negative_rev = []
senti_neutral_rev = []

for index in senti_df.index:
    if (senti_df['overall'][index] >= 4.0):
        pos_list = [senti_df['reviewerID'][index],
                    senti_df['productID'][index],
                    senti_df['overall'][index],
                    senti_df['reviewTime'][index],
                    senti_df['unixReviewTime'][index],
                    senti_df['summary'][index],
                    senti_df['reviewText'][index]]
        senti_positive_rev.append(pos_list)
    elif (senti_df['overall'][index] <= 2.0):
        neg_list = [senti_df['reviewerID'][index],
                    senti_df['productID'][index],
                    senti_df['overall'][index],
                    senti_df['reviewTime'][index],
                    senti_df['unixReviewTime'][index],
                    senti_df['summary'][index],
                    senti_df['reviewText'][index]]
        senti_negative_rev.append(neg_list)
    else:
        neu_list = [senti_df['reviewerID'][index],
                    senti_df['productID'][index],
                    senti_df['overall'][index],
                    senti_df['reviewTime'][index],
                    senti_df['unixReviewTime'][index],
                    senti_df['summary'][index],
                    senti_df['reviewText'][index]]
        senti_neutral_rev.append(neu_list)
print('Classification Completed\n')

print('*******************************************************************************************\n')
print('*******************************************************************************************\n')
# To find the top-20 positive representative words
print('FIND THE TOP 20 WORDS EXPRESSING POSITIVE SENTIMENT\n')
pos_rev_tokens_list = []
pos_review = []
for item in senti_positive_rev:
    pos_review_text = item[6]
    tokens = nltk.tokenize.word_tokenize(str(pos_review_text))
    for t in tokens:
        pos_rev_tokens_list.append(t)

print(pos_rev_tokens_list)
print(' Filtering Stop words and punctuation...\n')
tokenCount = {}
filtered_pos_rev_tokens_list = []
for token in pos_rev_tokens_list:
    t_lower = token.lower()
    if t_lower not in stopWords:
        if t_lower not in punct_list:
            filtered_pos_rev_tokens_list.append(t_lower)
print(filtered_pos_rev_tokens_list)
print(' Filtering Stop words and punctuation DONE\n')

print(' Getting the positive words...\n')
sid = analyser()
pos_word_list = []

for word in filtered_pos_rev_tokens_list:
    if (sid.polarity_scores(word)['compound']) >= 0.5:
        pos_word_list.append(word)

print(' Positive :', pos_word_list)
print(' Getting the positive words DONE\n')

print(' Get the Top 20 expressing positive sentiment\n')
wordCount = {}
for w in pos_word_list:
    if w not in wordCount:
        wordCount[w] = 1
    else:
        wordCount[w] += 1
word_Counter = collections.Counter(wordCount)
for word, count in word_Counter.most_common(20):
    print(word, ": ", count)
top20_pos_word = word_Counter.most_common(20)
top20_pos_word_df = pd.DataFrame(top20_pos_word, columns=['Pos_word', 'Count'])
top20_pos_word_df.plot.bar(x='Pos_word', y='Count', color='steelblue')
plt.savefig('Top20_positive_Words.png', bbox_inches='tight')
print('TOP 20 WORDS EXPRESSING POSITIVE SENTIMENT FOUND\n')

print('*******************************************************************************************\n')
print('*******************************************************************************************\n')
# To find the top-20 negative representative words
print('FIND THE TOP 20 WORDS EXPRESSING NEGATIVE SENTIMENT\n')
neg_rev_tokens_list = []
neg_review = []
# senti_neg_tokensTypes_list =[]
for item in senti_negative_rev:
    neg_review_text = item[6]
    tokens = nltk.tokenize.word_tokenize(str(neg_review_text))
    for t in tokens:
        neg_rev_tokens_list.append(t)

print(neg_rev_tokens_list)
print(' Filtering Stop words and punctuation...\n')
tokenCount = {}
filtered_neg_rev_tokens_list = []
for token in neg_rev_tokens_list:
    t_lower = token.lower()
    if t_lower not in stopWords:
        if t_lower not in punct_list:
            filtered_neg_rev_tokens_list.append(t_lower)
print(filtered_neg_rev_tokens_list)
print('Filtering Stop words and punctuation DONE\n')

print(' Getting the negative words...\n')
sid = analyser()
neg_word_list = []
for word in filtered_neg_rev_tokens_list:
    if (sid.polarity_scores(word)['compound']) <= -0.5:
        neg_word_list.append(word)

print(' Negative :', neg_word_list)
print(' Getting the negative words DONE\n')

print(' Get the Top 20 expressing negative sentiment\n')
wordCount = {}
for w in neg_word_list:
    if w not in wordCount:
        wordCount[w] = 1
    else:
        wordCount[w] += 1
word_Counter = collections.Counter(wordCount)
for word, count in word_Counter.most_common(20):
    print(word, ": ", count)
top20_neg_word = word_Counter.most_common(20)
top20_neg_word_df = pd.DataFrame(top20_neg_word, columns=['Neg_word', 'Count'])
top20_neg_word_df.plot.bar(x='Neg_word', y='Count', color='steelblue')
plt.savefig('Top20_negative_Words.png', bbox_inches='tight')
print('TOP 20 WORDS EXPRESSING NEGATIVE SENTIMENT FOUND\n')
