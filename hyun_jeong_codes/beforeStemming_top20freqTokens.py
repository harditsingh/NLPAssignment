import pandas as pd
import nltk
import numpy as np
import collections
import matplotlib.pyplot as plt
import punctuations
stopWords = set(nltk.corpus.stopwords.words('english'))
punct_list = punctuations.punctuations

# Given a list of words, return a dictionary of
# word-frequency pairs.
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

# Sort a dictionary of word-frequency pairs in
# order of descending frequency.
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

# Given a list of words, remove any that are
# in a list of stop words.
def removeStopwords(wordlist):
    stopWords = set(nltk.corpus.stopwords.words('english'))
    wordsFiltered = []
    wordsFiltered.append(w for w in wordlist if w not in stopWords)
    return wordsFiltered


print('Reading Data...\n')

tokens_df = pd.read_csv('TokensDF.csv', sep=',', encoding='utf-8')
print('Data READ\n')

print('Replacing NAN values with empty string\n')
tokens_df = tokens_df.replace(np.nan, '', regex=True)
print(tokens_df)


print('Tokenizing\n')
#Tokenizing
t_df = tokens_df
t_list = []
for index in t_df.index:
    text = t_df['reviewText'][index]
    tokens = nltk.tokenize.word_tokenize(text)
    for token in tokens:
        t_list.append(token)

#print('Converting tokens to lowercase\n')
#t_list_lower = [x.lower() for x in t_list]
#creating another list of the same size is causing memory error

print('Filtering Stop words...\n')
tokenCount = {}

for token in t_list:
    t_lower = token.lower()
    if t_lower not in stopWords:
        #if t_lower not in punct_list:
        if t_lower not in tokenCount:
            tokenCount[t_lower] = 1
        else:
            tokenCount[t_lower] += 1

token_Counter = collections.Counter(tokenCount)
for token, count in token_Counter.most_common(20):
    print(token, ": ", count)
top20_tokens = token_Counter.most_common(20)
top20_tokens_df = pd.DataFrame(top20_tokens, columns = ['Token', 'Count'])
top20_tokens_df.plot.bar(x='Token',y='Count', color='steelblue')
plt.savefig('Top20_tokens_bf_Stem_remove_stopwords.png', bbox_inches='tight')

