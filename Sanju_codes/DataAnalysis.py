import json
import ast
import pandas as pd
import nltk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

print('Reading file...\n')
i = 1
reviewDict = []
with open('C:/Users/Sanjusha/Desktop/Year 4 Sem1/CZ4045/NLPAssignment/NLPAssignment/Sanju_codes/CellPhoneReview.json') as f:
    for line in f:
        test_dict = ast.literal_eval(line)
        reviewDict.append(test_dict)
        i += 1
f.close()
#print(reviewDict)
print('File READ\n')

print('Creating Dataframe...\n')
review_df = pd.DataFrame.from_dict(reviewDict, orient='columns')
review_df2 = review_df[['reviewerID','asin','overall','reviewTime','unixReviewTime','summary','reviewText']]

print(review_df2)
print('Dataframe CREATED...\n')

#asin is product id
review_res = review_df2
review_res.columns = ['reviewerID','productID','overall','reviewTime','unixReviewTime','summary','reviewText']

review_res.to_csv('processedData.csv', sep=',', encoding='utf-8')

#Identify the top-10 products that attract the most number of reviews
prod_freq = review_res
prod_freq = prod_freq.groupby(['productID'])['unixReviewTime'].count()
top10_prod = prod_freq.nlargest(10)
top10_prod_df = pd.DataFrame(top10_prod)
top10_prod_df.reset_index(inplace=True)
top10_prod_df.columns = ['productID','No_of_Reviews']
print('Top 10 Products\n')
print(top10_prod_df)

#Identify the top-10 reviewers who have contributed most number of reviews
reviewer_freq = review_res
reviewer_freq = reviewer_freq.groupby(['reviewerID'])['unixReviewTime'].count()
top10_rw = reviewer_freq.nlargest(10)
top10_rw_df = pd.DataFrame(top10_rw)
top10_rw_df.reset_index(inplace=True)
top10_rw_df.columns = ['reviewerID','No_of_Reviews']
print('Top 10 Reviewers\n')
print(top10_rw_df)

print('Sentence Segmentation\n')
#finding the length of reviewText in number of sentences
seg_df = review_res.copy(deep=True)
seg_list = []
for index in seg_df.index:
    sentence = seg_df['reviewText'][index]
    sentence_list = nltk.tokenize.sent_tokenize(sentence)
    seg_list.append(len(sentence_list))
seg_df['length_of_review(No_of_Sentences)'] = seg_list
print('Sentence Segmentation Completed\n')
print(seg_df)

print('Randomly sample 5 reviews\n')
#Randomly sample 5 reviews and verify whether the sentence segmentation function/tool detects the sentence boundaries correctly.
sample_seg_df = seg_df.copy(deep=True)
sample_seg_df = sample_seg_df.sort_values(['length_of_review(No_of_Sentences)'], ascending=False)
sample_seg_df = sample_seg_df[sample_seg_df['length_of_review(No_of_Sentences)'] > 0]
sample_long = sample_seg_df[sample_seg_df['length_of_review(No_of_Sentences)'] == 10].head(3)
sample_small = sample_seg_df[sample_seg_df['length_of_review(No_of_Sentences)'] == 5].head(2)
sample_list = [sample_long, sample_small]
sample_5 = pd.concat(sample_list)
print('Random sampling completed\n')
print(sample_5)

#sample 5 data into csv file
sample_5.to_csv('Random_sample_5.csv', encoding='utf-8', index = False)

print('Distribution of length of review(No. of Sentences)\n')
#group no of reviews by length of review
review_distr = seg_df.copy(deep=True)
review_distr = review_distr.groupby(['length_of_review(No_of_Sentences)'])['unixReviewTime'].count()
review_distr_df = pd.DataFrame(review_distr)
review_distr_df.reset_index(inplace=True)
review_distr_df.columns = ['length_of_review(No_of_Sentences)','No_of_Reviews']
print('Distribution of length of review(No. of Sentences) completed\n')
print(review_distr_df)

print('length_of_review(No_of_Sentences) chart\n')
review_distr_df.plot(kind='line',x='length_of_review(No_of_Sentences)',y='No_of_Reviews')
plt.savefig('seg_distribution.png')
print('saved distribution chart of length of review(No. of Sentences)\n')

print('Tokenizing\n')
#finding the No. of Tokens & types in reviewText
tokens_df = review_res.copy(deep=True)
tokens_list = []
tokenTypes_list = []
for index in tokens_df.index:
    text = tokens_df['reviewText'][index]
    tokens = nltk.tokenize.word_tokenize(text)
    tokenTypes = list(set(tokens))
    tokens_list.append(len(tokens))
    tokenTypes_list.append(len(tokenTypes))
tokens_df['length_of_review(No_of_Tokens)'] = tokens_list
tokens_df['length_of_review(No_of_Token_Types)'] = tokenTypes_list
print('Tokenizing Completed\n')
print(tokens_df)


#group no of reviews by No. of Tokens
tokens_distr = tokens_df.copy(deep=True)
tokens_distr = tokens_distr.groupby(['length_of_review(No_of_Tokens)'])['unixReviewTime'].count()
tokens_distr_df = pd.DataFrame(tokens_distr)
tokens_distr_df.reset_index(inplace=True)
tokens_distr_df.columns = ['length_of_review(No_of_Tokens)','No_of_Reviews']
print('Distribution of length_of_review(No_of_Tokens)\n')
print(tokens_distr_df)


tokens_distr_df.plot(kind='line',x='length_of_review(No_of_Tokens)',y='No_of_Reviews')
plt.savefig('tokens_distribution.png')
print('saved distribution chart of length of review(No. of Tokens)\n')

#group no of reviews by No. of Token Types
tokenTypes_distr = tokens_df.copy(deep=True)
tokenTypes_distr = tokenTypes_distr.groupby(['length_of_review(No_of_Token_Types)'])['unixReviewTime'].count()
tokenTypes_distr_df = pd.DataFrame(tokenTypes_distr)
tokenTypes_distr_df.reset_index(inplace=True)
tokenTypes_distr_df.columns = ['length_of_review(No_of_Token_Types)','No_of_Reviews']
print('Distribution of length_of_review(No_of_Token_Types)\n')
print(tokenTypes_distr_df)

tokenTypes_distr_df.plot(kind='line',x='length_of_review(No_of_Token_Types)',y='No_of_Reviews')
plt.savefig('tokenTypes_distribution.png')
print('saved distribution chart of length of review(No. of Token Types)\n')

print('Stemming\n')
#Stemming on tokens
stem_df = tokens_df.copy(deep=True)
stemmer = nltk.stem.porter.PorterStemmer()
stemmed_types= set()
stemmed_len = []
for index in stem_df.index:
    text = stem_df['reviewText'][index]
    tokens = nltk.tokenize.word_tokenize(text)
    tokenTypes = list(set(tokens))
    stemmed_types= set()
    for token_type in tokenTypes:
        stemmed_types.add(stemmer.stem(token_type))
    stemmed_len.append(len(stemmed_types))
stem_df['length_of_review(No_of_Stemmed_Types)'] = stemmed_len
print('Stemming completed\n')
print(stem_df)


#group no of reviews by No.of Stemmed Types
stemmedTypes_distr = stem_df.copy(deep=True)
stemmedTypes_distr = stemmedTypes_distr.groupby(['length_of_review(No_of_Stemmed_Types)'])['unixReviewTime'].count()
stemmedTypes_distr_df = pd.DataFrame(stemmedTypes_distr)
stemmedTypes_distr_df.reset_index(inplace=True)
stemmedTypes_distr_df.columns = ['length_of_review(No_of_Stemmed_Types)','No_of_Reviews']
print('Distribution of length_of_review(No_of_Stemmed_Types)\n')
print(stemmedTypes_distr_df)

stemmedTypes_distr_df.plot(kind='line',x='length_of_review(No_of_Stemmed_Types)',y='No_of_Reviews')
plt.savefig('stemmedTypes_distribution.png')
print('saved distribution chart of length of review(No. of Stemmed Types)\n')

#filter out stop words
def filter_stopWords (words):
    stopWords = set(nltk.corpus.stopwords.words('english'))
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    return wordsFiltered


#testing
'''
tokenTypesFiltered = filter_stopWords(t_list)
tokenTypesFiltered_df = pd.DataFrame(tokenTypesFiltered)
tokenTypesFiltered_df.columns = ['tokenTypes_Filtered']
print(tokenTypesFiltered_df)

tokenTypesFiltered_freq = tokenTypesFiltered_df.groupby(['tokenTypes_Filtered'])['tokenTypes_Filtered'].count()
tokenTypesFiltered_freq_df = pd.DataFrame(tokenTypesFiltered_freq)
tokenTypesFiltered_freq_df.reset_index(inplace=True)
tokenTypesFiltered_freq_df.columns = ['tokenTypes_Filtered)','Freq']
print(tokenTypesFiltered_freq_df)
'''
