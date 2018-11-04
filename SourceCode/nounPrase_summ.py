import pandas as pd
import nltk
import numpy as np
import random
import csv
import collections
import operator
import matplotlib.pyplot as plt


nltk.download('maxent_ne_chunker')

print('Reading Data...\n')
tokens_df = pd.read_csv('processedData.csv', sep=',', encoding='utf-8')
print('Data READ\n')

print('Replacing NAN values with empty string\n')
tokens_df = tokens_df.replace(np.nan, '', regex=True)
print(tokens_df)

print('Sentence Segmentation\n')
#finding the length of reviewText in number of sentences
seg_df = tokens_df
seg_list = []
for index in seg_df.index:
    sentence = seg_df['reviewText'][index].lower()
    sentence_list = nltk.tokenize.sent_tokenize(sentence)
    for sent in sentence_list:
        seg_list.append(sent)

print('Sentence Segmentation Completed\n')


print("noun phrase extraction")
#just carry out noun phrase extraction
NP = "NP: {<DT>?<JJ>*<NN>}"
extractor = nltk.RegexpParser(NP)

npSentence = [nltk.word_tokenize(sent) for sent in seg_list]
npSentence2 = [nltk.pos_tag(sent) for sent in npSentence]

noun_phrases_list = [[' '.join(leaf[0] for leaf in tree.leaves())
                      for tree in extractor.parse(sent).subtrees()
                      if tree.label()=='NP']
                      for sent in npSentence2]

with open('nounPhrase.csv', 'w') as f:
    writer = csv.writer(f)
    for i in noun_phrases_list:
        writer.writerow([i])

#dictionary to count keys
print('top 20 most frequent noun phrases')
npCount = {}

for s in noun_phrases_list:
    for phrase in s:
        if phrase in npCount:
            if phrase == '':
                break
            npCount[phrase] += 1
        else:
            npCount[phrase] = 1

print(npCount)

print('get top 20 most frequent phrases')

npCounter  = collections.Counter(npCount)
for np, count in npCounter.most_common(20):
    print(np, ": ", count)
top20_phrases = npCounter.most_common(20)
top20_phrases_df = pd.DataFrame(top20_phrases, columns = ['Noun Phrases', 'Count'])
top20_phrases_df.plot.bar(x='Noun Phrases',y='Count', color='steelblue')
plt.savefig('Top20_Noun_Phrases.png', bbox_inches='tight')


#top 3 popular reviews -   B005SUHPO6, B0042FV2SI, B008OHNZI0
print('extract 3 popular products')
#extract top 3 popular products
df1 = tokens_df.loc[tokens_df['productID'] == 'B005SUHPO6']
df2 = tokens_df.loc[tokens_df['productID'] == 'B0042FV2SI']
df3 = tokens_df.loc[tokens_df['productID'] == 'B008OHNZI0']

#returns segmented sentences
def sentenceSeg(df):
    seg_df = df
    seg_list = []
    for index in seg_df.index:
        sentence = seg_df['reviewText'][index].lower()
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

def npRepresentative(noun_phrases_list, full_list_count):
    npCount = {}
    for s in noun_phrases_list:
        for phrase in s:
            if phrase in npCount:
                if phrase == '':
                    break
                npCount[phrase] += 1
            else:
                npCount[phrase] = 1
    print(npCount)
    #if the phrase appears in other reviews, not very representative of the product.

    for npKey in npCount.keys():
        if npKey in full_list_count.keys():
            npCount[npKey] = npCount[npKey] / full_list_count[npKey]

    print('get top 10 phrases with highest probability')
    #remove words that only appear once
    npCount = {key:val for key, val in npCount.items() if val != 1.0}
    probabilityDescending = sorted(npCount.items(), key=operator.itemgetter(1), reverse=True)

    print(probabilityDescending)

    top10_list = []
    for i in range(0, 10):
        print(probabilityDescending[i])
        top10_list.append(probabilityDescending[i])

    return top10_list


seg1 = sentenceSeg(df1)
seg2 = sentenceSeg(df2)
seg3 = sentenceSeg(df3)

np1 = npExtract(seg1)
np2 = npExtract(seg2)
np3 = npExtract(seg3)

print('representative ones')

npr1 = npRepresentative(np1, npCount)
npr2 = npRepresentative(np2, npCount)
npr3 = npRepresentative(np3, npCount)

print("first product-----------------------------------------------------")
print(npr1)
print('second product-----------------------------------------------------')
print(npr2)
print('third product------------------------------------------------------')
print(npr3)

with open('Popular_product_top10_np_1.csv', 'w') as f:
    writer = csv.writer(f)
    for i in npr1:
        writer.writerow([i])

with open('Popular_product_top10_np_2.csv', 'w') as f:
    writer = csv.writer(f)
    for i in npr2:
        writer.writerow([i])

with open('Popular_product_top10_np_3.csv', 'w') as f:
    writer = csv.writer(f)
    for i in npr3:
        writer.writerow([i])

'''
npr1.to_csv('Popular_product_top10_np_1.csv', encoding='utf-8', index = False)
npr2.to_csv('Popular_product_top10_np_2.csv', encoding='utf-8', index = False)
npr2.to_csv('Popular_product_top10_np_3.csv', encoding='utf-8', index = False)
'''
