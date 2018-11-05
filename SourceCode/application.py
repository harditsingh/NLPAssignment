import sys
from argparse import ArgumentParser
import ast
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm

# Data is imported from the file in the following section of the code
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file_path", help="Open specified file")
args = parser.parse_args()
file_path = args.file_path

if not file_path:
    print("Invalid arguments, please specify file path for the Reviews. Usage: python3 application.py -f <file_path>")
    sys.exit(1)

review_data = []
with open(file_path) as file:
    for line in file:
        review_data.append(ast.literal_eval(line))
file.close()

# The imported data is first inserted into a DataFrame
review_data = pd.DataFrame.from_dict(review_data, orient='columns')

# And then only the necessary columns are retained. Note here that the data is truncated to 1/100th of its original
# size to quickly calculate a result. To increase accuracy, we can increase the amount of data being processed.
review_data = review_data[['overall', 'reviewText']][0: review_data.__len__() // 20]

# Data is separated into individual columns for easier processing
ratings = review_data[['overall']]
review_text = review_data[['reviewText']]

# Empty list is created for storing the results of the next segment
analyzed_sentences = []

# We loop through all the reviews that we import from the file
for index in review_data.index:
    # A dictionary is created to store the data of one sentence temporarily
    data = {'compound': 0, 'neu': 0, 'neg': 0, 'pos': 0}

    # Reviews are taken, one at a time, from the review texts list
    review = review_data['reviewText'][index]
    # And then the review is separated into sentences
    sentence_list = nltk.tokenize.sent_tokenize(review)

    # Then, Vader Analyzer from the NLTK Library is used to do a sentiment analysis of each of the sentences obtained
    #  from the review. This analyzer gives us four parameters in the result: Compound, Neutral, Positive and Negative
    vader_analyzer = SentimentIntensityAnalyzer()
    for text in sentence_list:
        temp = vader_analyzer.polarity_scores(text)
        for key in ('compound', 'neu', 'neg', 'pos'):
            # Here, an average of the parameters is taken for all the sentences obtained from the review to find the
            # Vader Analysis scores for the review
            if sentence_list.__len__() is not 0:
                data[key] += temp[key]/sentence_list.__len__()

    # We add all the analysis scores in a list for later use
    analyzed_sentences.append(data)


# In the next few lines, we separate and prepare the data for training an OLS regression model and testing it. We
# make a 50/50 split of the data, half for training and the other half for testing.
test_cases = analyzed_sentences[analyzed_sentences.__len__() // 2:analyzed_sentences.__len__()]
analyzed_sentences = pd.DataFrame.from_dict(analyzed_sentences, orient='columns')
x = analyzed_sentences[0:analyzed_sentences.__len__() // 2]
y = ratings[0:analyzed_sentences.__len__() // 2]

# Here, x contains the scores after the Vader analysis of the review text, and y contains the actual ratings that
# users gave along with the reviews. We train a regression model with the data, which we will use to predict the
# ratings for the remaining data.

# We first train a model using the said data
model = sm.OLS(y, x).fit()
# And then obtain the parameters(weights) of the equation of the regression model
params = model.params.to_dict()

# Empty list for storing results of predictions is created
predicted_ratings = [];

# In the following code segment, we predict the ratings for the remaining reviews that we segregated earlier. The
# regression equation is simple: rating = val(compound) * weight(compound) + val(neutral) * weight(neutral) + val(
# positive) * weight(positive) + val(negative) * weight(negative)

# The above operation is simply carried out in the next few lines, and the results are stored in a list
for case in test_cases:
    predicted_rating = 0
    for i in ('compound', 'neu', 'neg', 'pos'):
                predicted_rating += params[i] * case[i]
    predicted_ratings.append(predicted_rating)

# We the convert the other required data to lists in order to handle and print them in an easier way
actual_ratings = ratings['overall'][ratings.__len__()//2:ratings.__len__()].tolist()
review_text = review_text['reviewText'][review_text.__len__()//2:review_text.__len__()].tolist()

# In the following segment, we cycle through all the predicted_ratings and compare them to the actual ratings that
# users have given in their reviews. If we find a significant absolute difference in the predicted and actual values,
# we print that particular case to the console for further analysis. Currently, the threshold for printing is a 25%
# difference (or a factor of 0.25)

total_outliers = 0

for i in range(0, predicted_ratings.__len__()):
    if abs(predicted_ratings[i] - actual_ratings[i])/5 >= 0.25:
        total_outliers += 1
        print('Predicted: %.2f, Actual: %.2f' % (predicted_ratings[i], actual_ratings[i]))
        print('Significant difference in prediction. Review Text: %s\n' % (review_text[i]))

print('Total outliers: %d\nTotal Reviews Analyzed: %d' % (total_outliers, review_text.__len__()))

