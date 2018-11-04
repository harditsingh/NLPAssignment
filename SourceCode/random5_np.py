import pandas as pd
import nltk
import numpy as np
import random
import csv
import collections
import matplotlib.pyplot as plt

from nltk import Tree, RegexpParser, ne_chunk

nltk.download('maxent_ne_chunker')

print('Reading Data...\n')
tokens_df = pd.read_csv('processedData.csv', sep=',', encoding='utf-8')
print('Data READ\n')

print('extract 5 random reviews products')
tokens_df1 = tokens_df.sample(n=1)
tokens_df2 = tokens_df.sample(n=1)
tokens_df3 = tokens_df.sample(n=1)
tokens_df4 = tokens_df.sample(n=1)
tokens_df5 = tokens_df.sample(n=1)

#save the seleccted reviews

tokens_df1.to_csv('sample1.csv', encoding='utf-8', index = False)
tokens_df2.to_csv('sample2.csv', encoding='utf-8', index = False)
tokens_df3.to_csv('sample3.csv', encoding='utf-8', index = False)
tokens_df4.to_csv('sample4.csv', encoding='utf-8', index = False)
tokens_df5.to_csv('sample5.csv', encoding='utf-8', index = False)


print('Replacing NAN values with empty string\n')
tokens_df1 = tokens_df1.replace(np.nan, '', regex=True)
tokens_df2 = tokens_df2.replace(np.nan, '', regex=True)
tokens_df3 = tokens_df3.replace(np.nan, '', regex=True)
tokens_df4 = tokens_df4.replace(np.nan, '', regex=True)
tokens_df5 = tokens_df5.replace(np.nan, '', regex=True)

print('first review----------------------------------------------')
print(tokens_df1)
print('second review----------------------------------------------')
print(tokens_df2)
print('third review----------------------------------------------')
print(tokens_df3)
print('fourth review----------------------------------------------')
print(tokens_df4)
print('fifth review----------------------------------------------')
print(tokens_df5)

#returns segmented sentences
def sentenceSeg(df):
    seg_df = df
    seg_list = []
    for index in seg_df.index:
        sentence = seg_df['reviewText'][index]
        sentence_list = nltk.tokenize.sent_tokenize(sentence)
        for sent in sentence_list:
            seg_list.append(sent)
    print('Sentence Segmentation Completed\n')
    return seg_list

def npExtract(segmentedList):
    #extract noun phrase first
    NP = "NP: {<DT>?<JJ>*<NN>}"
    extractor = nltk.RegexpParser(NP)

    npSentence = [nltk.word_tokenize(sent) for sent in segmentedList]
    npSentence2 = [nltk.pos_tag(sent) for sent in npSentence]

    noun_phrases_list = [[' '.join(leaf[0] for leaf in tree.leaves())
                          for tree in extractor.parse(sent).subtrees()
                          if tree.label() == 'NP']
                         for sent in npSentence2]
    print(noun_phrases_list)
    return noun_phrases_list

extracted1 = npExtract(sentenceSeg(tokens_df1))
extracted2 = npExtract(sentenceSeg(tokens_df2))
extracted3 = npExtract(sentenceSeg(tokens_df3))
extracted4 = npExtract(sentenceSeg(tokens_df4))
extracted5 = npExtract(sentenceSeg(tokens_df5))

with open('Random5_np1.csv', 'w') as f:
    writer = csv.writer(f)
    for i in extracted1:
        writer.writerow([i])

with open('Random5_np2.csv', 'w') as f:
    writer = csv.writer(f)
    for i in extracted2:
        writer.writerow([i])

with open('Random5_np3.csv', 'w') as f:
    writer = csv.writer(f)
    for i in extracted3:
        writer.writerow([i])

with open('Random5_np4.csv', 'w') as f:
    writer = csv.writer(f)
    for i in extracted4:
        writer.writerow([i])

with open('Random5_np5.csv', 'w') as f:
    writer = csv.writer(f)
    for i in extracted5:
        writer.writerow([i])
