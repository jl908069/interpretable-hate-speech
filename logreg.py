from util import parse_file_A
from util import parse_file_B
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import re
import argparse
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import shap

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Citiation
    ---------
    https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def main(train_file):

    tweet_a, label_a = parse_file_A(train_file)
    tweet_b, label_b=parse_file_B(train_file) #return array
    XA_train, XA_test, yA_train, yA_test = train_test_split(tweet_a, label_a, test_size = 0.2, random_state = 0)
    XB_train, XB_test, yB_train, yB_test = train_test_split(tweet_b, label_b, test_size=0.2, random_state=0)
    # Initialize CountVectorizer and TfidfTransformer
    vectorizer = CountVectorizer(stop_words='english') #use built-in stop word list
    tf_transformer = TfidfTransformer(use_idf=True)
    """For task A"""
    XA_train_lst=XA_train.tolist()
    XA_test_lst = XA_test.tolist()
    # Transform to tfidf vectors
    vec_a_training=vectorizer.fit_transform(XA_train_lst)
    vec_a_training = tf_transformer.fit_transform(vec_a_training)
    # Train Logistic Regression
    clf = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=0)
    clf.fit(vec_a_training,yA_train.ravel())
    vec_valid = tf_transformer.transform(vectorizer.transform(XA_test_lst))
    predictions = clf.predict(vec_valid)
    # Used to generate shap results
    explainer = shap.LinearExplainer(clf,
                                     vec_a_training,
                                     feature_dependence="independent")
    shap_values = explainer.shap_values(vec_valid)
    vec_valid_array = vec_valid.toarray()
    shap.summary_plot(shap_values,
                      vec_valid_array,
                      feature_names=vectorizer.get_feature_names())
    print("Task A Report:",metrics.classification_report(yA_test, predictions))
    print("Accuracy for task A:", metrics.accuracy_score(yA_test, predictions))
    print("Misclassified examples in task A:",np.where(yA_test != predictions))
    cm_A=metrics.confusion_matrix(yA_test, predictions)
    plot_confusion_matrix(cm_A, [0,1], title='Confusion matrix for task A')
    """For task B"""
    XB_train_lst = XB_train.tolist()
    XB_test_lst = XB_test.tolist()
    # Transform to tfidf vectors
    vec_b_training = vectorizer.fit_transform(XB_train_lst)
    vec_b_training = tf_transformer.fit_transform(vec_b_training)
    clf.fit(vec_b_training, yB_train.ravel())
    vec_b_valid = tf_transformer.transform(vectorizer.transform(XB_test_lst))
    predictions_B = clf.predict(vec_b_valid)
    # Used to generate shap results
    bexplainer = shap.LinearExplainer(clf,
                                     vec_b_training,
                                     feature_dependence="independent")
    shap_b_values = bexplainer.shap_values(vec_b_valid)
    vec_valid_array = vec_b_valid.toarray()
    shap.summary_plot(shap_b_values,
                      vec_valid_array,
                      feature_names=vectorizer.get_feature_names())
    print("Task B Report:", metrics.classification_report(yB_test, predictions_B))
    print("Accuracy for task B:", metrics.accuracy_score(yB_test, predictions_B))
    print("Misclassified examples in task B:", np.where(yB_test != predictions_B))
    cm_B=metrics.confusion_matrix(yB_test, predictions_B)
    plot_confusion_matrix(cm_B, [0, 1], title='Confusion matrix for task B')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="olid-training-v1.tsv",
                        help="train file")

    args = parser.parse_args()

    main(args.train_file)

