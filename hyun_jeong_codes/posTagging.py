import pandas as pd
import nltk
import numpy as np
import random
import csv
import collections
import matplotlib.pyplot as plt

print('Reading Data...\n')
tokens_df = pd.read_csv('TokensDF.csv', sep=',', encoding='utf-8')
print('Data READ\n')

print('Replacing NAN values with empty string\n')
tokens_df = tokens_df.replace(np.nan, '', regex=True)
print(tokens_df)

print('Sentence Segmentation\n')
#finding the length of reviewText in number of sentences
seg_df = tokens_df
seg_list = []
for index in seg_df.index:
    sentence = seg_df['reviewText'][index]
    sentence_list = nltk.tokenize.sent_tokenize(sentence)
    for sent in sentence_list:
        seg_list.append(sent)
print('Sentence Segmentation Completed\n')

#Random number generator
random_num = random.sample(range(0, len(seg_list)), 5)
print(random_num)

random_sen_list = []
for num in random_num:
    random_sentence = seg_list[num]
    random_sen_list.append(random_sentence)

print('Random 5 sentences selected:\n')
print(random_sen_list)
print('\n')

text = ''
for s in random_sen_list:
    text += ' ' + s

p_tokens = nltk.tokenize.word_tokenize(text)
print('Sentences into tokens:\n')
print(p_tokens)
print('\n')
posTag_random5 = nltk.pos_tag(p_tokens)
print('POS Tagging on random 5 sentences:\n')
print(posTag_random5)

##after random 5, do pos tagging for the rest
text2 =""
for s in seg_list:
    text2 += " " + s

p_tokens_all = nltk.tokenize.word_tokenize(text2)
print(p_tokens_all)
print('\n')
posTag_all = nltk.pos_tag(p_tokens_all)

#print(posTag_all)

print("pos tagging all......")

