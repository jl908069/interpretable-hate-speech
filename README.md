Interpretable Hate Speech
--------------------------------------------
This project trains logistic regression models for hate speech detection 
using SemEval 2019 task 6 dataset OLID. Task A is to determine whether or not
the tweet is offensive. Task B is to determine whether the offensive tweet is targeted.
Data and links to task information and paper are available 
[here](https://sites.google.com/site/offensevalsharedtask/offenseval2019).

## util.py
Implements utility functions for loading data in
Task A and B. 

Example usage:

`python util.py olid-training-v1.tsv`
## logreg.py
Trains logistic regression model with tf-idf
vectors for task A and B.
Returns following results:
* classification report (using `sklearn`)
* misclassified examples
* confusion matrix
* explainable results using `shap` package

Example usage:

`python logreg.py --train_file olid-training-v1.tsv`

## feature_combination.py
Creates a FeaatureVectorizer class to add sentiment,
subjectivity, profanity, and user feature 
to the feature function. Then it trains a logistic
regression model to evaluate the results on
the following different feature combinations.
* base_tfidf + sentiment feature(`vaderSentiment` package)
* base_tfidf + subjectivity feature(`textblob` package)
* base_tfidf + profanity feature(`profanity-check` package)
* base_tfidf + @user feature (percentage of @USER in a tweet)

Example usage:

`python feature_combination.py --train_file olid-training-v1.tsv`
