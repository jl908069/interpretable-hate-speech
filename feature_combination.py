from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob
from profanity_check import predict_prob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from util import parse_file_A
from util import parse_file_B
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import *
import argparse
import numpy as np

class FeatureVectorizer:
    """Create a matrix where every row is a tweet and every column is a feature."""

    def __init__(self, corpus):
        """
                Instantiates FeatureVectorizer.
                :param corpus: array of strings

                """
        self.tfidf_transformer = TfidfTransformer(use_idf=True)
        self.vectorizer = CountVectorizer(stop_words='english')
    def tfidf_base(self,corpus):
        """ For task A and B:
            Reads a list of string returns
            tfidf matrix for every row of tweet
        """
        word_count=self.vectorizer.fit_transform(corpus)
        tfidf=self.tfidf_transformer.fit_transform(word_count)
        arr=tfidf.toarray() #2d array
        return arr
    def add_sentiment(self, arr):
        """For task A: Takes array of string. Adds sentiment score (ranges from -1 to 1 where
        positive 1 is the most extreme positive score) calculated by
        Vader package to tfidf_base matrix (uses self.tfidf_base)
        """
        sent_scores = []
        arrlst=arr.tolist()
        analyzer = SentimentIntensityAnalyzer()
        tfidf=self.tfidf_base(arrlst)
        for elem in arr:
            score=analyzer.polarity_scores(elem)['compound']
            sent_scores.append(score)
        sent_arr=np.array([sent_scores])
        sent_arr=sent_arr.T
        fm=np.concatenate((tfidf, sent_arr), axis=1)
        return fm
    def add_subjectivity(self,arr):
        """For task A: Takes array of string. Adds subjectivity score (ranges from 0 to 1 where
        0.0 is very objective and 1.0 is very subjective) calculated by
        Textblob package to tfidf_base matrix (uses self.tfidf_base)
        """
        subj_scores = []
        arrlst = arr.tolist()
        tfidf = self.tfidf_base(arrlst)
        for elem in arr:
            score=TextBlob(elem).sentiment.subjectivity
            subj_scores.append(score)
        subj_arr=np.array([subj_scores])
        subj_arr = subj_arr.T
        fm = np.concatenate((tfidf, subj_arr), axis=1)
        return fm
    def add_profanity(self,arr):
        """For task A:Takes array of string. Adds profanity score (0 to 1, the probability each string is offensive because of swear words)
        calculated by profanity-check package to tfidf_base matrix (uses self.tfidf_base)
        """
        arrlst = arr.tolist()
        tfidf = self.tfidf_base(arrlst)
        score = predict_prob(arrlst)
        score = np.array([score]).T
        fm = np.concatenate((tfidf, score), axis=1)
        return fm
    def get_num_user(self,arr):
        """For task B: get @user score in a tweet by counting the presence
         of @USER in a tweet and then divided by word count of the tweet. Add
         this feature into tfidf_base matrix"""
        arrlst = arr.tolist()
        tfidf = self.tfidf_base(arrlst)
        user=[]
        for s in arr:
            target = "@USER"
            count = s.count(target)
            words=len(s.split())
            p=count / words
            user.append(p)
        user_arr = np.array([user])
        user_arr = user_arr.T
        fm = np.concatenate((tfidf, user_arr), axis=1)
        return fm

def main(train_file):

    # Load tweets and labels from util.py
    tweet_a, label_a = parse_file_A(train_file) #tweet a: array of string
    tweet_b, label_b = parse_file_B(train_file)
    # Initialize FeatureVectorizer
    fv = FeatureVectorizer(tweet_a)
    fvb=FeatureVectorizer(tweet_b) # for task B

    # Get feature matrix for each feature added
    sent_a=fv.add_sentiment(tweet_a)
    subj_a = fv.add_subjectivity(tweet_a)
    prf_a = fv.add_profanity(tweet_a)
    user_b=fvb.get_num_user(tweet_b)
    # Train test split
    sentA_train, sentA_test, ysent_train, ysent_test = train_test_split(sent_a, label_a, test_size=0.2, random_state=1)
    subjA_train, subjA_test, ysubj_train, ysubj_test = train_test_split(subj_a, label_a, test_size=0.2, random_state=1)
    prfA_train, prfA_test, yprf_train, yprf_test = train_test_split(prf_a, label_a, test_size=0.2, random_state=1)
    usrB_train, usrB_test, yusr_train, yusr_test = train_test_split(user_b, label_b, test_size=0.2, random_state=1)

    # Train Logistic Regression
    clf = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=0)
    clf.fit(sentA_train,ysent_train.ravel())
    predictions_sent = clf.predict(sentA_test)
    print("Adding sentiment Report:", metrics.classification_report(ysent_test, predictions_sent))
    print("Accuracy for adding sentiment:", metrics.accuracy_score(ysent_test, predictions_sent))
    clf.fit(subjA_train, ysubj_train.ravel())
    predictions_subj = clf.predict(subjA_test)
    print("Adding subjectivity Report:", metrics.classification_report(ysubj_test, predictions_subj))
    print("Accuracy for adding subjectivity:", metrics.accuracy_score(ysubj_test, predictions_subj))
    clf.fit(prfA_train, yprf_train.ravel())
    predictions_prf = clf.predict(prfA_test)
    print("Adding profanity feature Report:", metrics.classification_report(yprf_test, predictions_prf))
    print("Accuracy for adding profanity feature:", metrics.accuracy_score(yprf_test, predictions_prf))
    clf.fit(usrB_train, yusr_train.ravel())
    predictions_usr = clf.predict(usrB_test)
    print("Add @user feature Report:", metrics.classification_report(yusr_test, predictions_usr))
    print("Accuracy for adding @user feature:", metrics.accuracy_score(yusr_test, predictions_usr))







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="olid-training-v1.tsv",
                        help="train file")

    args = parser.parse_args()

    main(args.train_file)