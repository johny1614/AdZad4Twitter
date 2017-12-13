import textblob as tb
from textblob.classifiers import NaiveBayesClassifier
import csv
def createTrainSet(src):
    trainSet=[]
    with open(src) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            tweet = row[5]
            # tweet=tweet.decode('utf-8')
            # print(tweet.split('http'))
            if row[0] == '0':
                sentiment = 'neg'
            else:
                sentiment = 'pos'
            trainSet.append((tweet,sentiment))
    return trainSet

def getTweets(src):
    tweets=[]
    with open(src) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)  # change contents to floats
        for row in reader:
            tweet=row[2]
            tweets.append(tweet)
    return tweets

def classifyTweets(tweets,clasificator):
    positives=0
    negatives=0
    for tweet in tweets:
        sentiment=clasificator.classify(tweet)
        if(sentiment=='pos'):
            positives=positives+1
        else:
            negatives=negatives+1
    return (negatives,positives)

def printResults(negatives,positives):
    labels = ['Negative', 'Positive']
    values = [classification[0], classification[1]]
    trace = go.Pie(labels, values)
    plotly.offline.plot([trace], filename='chart.html')


# y=[(1,2),(234,324)]
# y.append((1234,2234))
# x=[2,3]
# x.append([1,4])
# print(x)
# print(y)
#
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

trainSet = createTrainSet("train1000.csv")
cl = NaiveBayesClassifier(trainSet)
tweetsSource="realDonaldTrump_tweets.csv"
tweets=getTweets(tweetsSource)
classification=classifyTweets(tweets,cl)

labels=['Negative','Positive']
values=[classification[0],classification[1]]
trace=go.Pie(labels=labels,values=values)
plotly.offline.plot([trace],filename='chart.html')



