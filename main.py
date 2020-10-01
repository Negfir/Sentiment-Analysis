# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



import sys
import sys
import os
import time

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import nltk
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize,sent_tokenize
from sklearn.naive_bayes import MultinomialNB
if __name__ == '__main__':
    #Raw data is converted to seperate files for each comment, inorder to read them easily
    # file_neg = open('./rt-polaritydata/rt-polaritydata/rt-polarity.neg')
    # file_pos = open('./rt-polaritydata/rt-polaritydata/rt-polarity.pos')
    # contents_Pos= file_pos.read()
    # comments_Pos=contents_Pos.splitlines()
    # contents_Neg= file_neg.read()
    # conmments_Neg=contents_Neg.splitlines()
    # cnt=0
    # for comment in comments_Pos:
    #     print(comment, "\n")
    #     f1 = open("comments/pos/comment-pos%s.txt" % cnt, "w")
    #     f1.write(comment)
    #     cnt += 1
    # cnt=0
    #
    # for comment in conmments_Neg:
    #     print(comment, "\n")
    #     f2 = open("comments/neg/comment-neg%s.txt" % cnt, "w")
    #     f2.write(comment)
    #     cnt += 1

    data = []
    data_labels = []
    with open("./rt-polaritydata/rt-polaritydata/rt-polarity.neg") as f:
        for i in f:
            data.append(i.strip("\n"))
            #print(i)
            data_labels.append('pos')

    with open("./rt-polaritydata/rt-polaritydata/rt-polarity.pos") as f:
        for i in f:
            data.append(i.strip("\n"))
            data_labels.append('neg')

    # x_train, x_test, y_train, y_test = train_test_split(data, data_labels, test_size=0.20, random_state=12)
    # movieVectorizer = CountVectorizer(min_df=2,tokenizer=nltk.word_tokenize,max_df = 1.0, analyzer='word', lowercase=False)
    #
    # features = movieVectorizer.fit_transform(data)
    # x_test_counts=docs_test_counts = movieVectorizer.transform(x_test)
    # print(movieVectorizer.vocabulary_.get("the"))
    # features_nd = features.toarray()  # for easy usage
    # print(features.shape)
    # print(y_train.shape)
    #
    #
    #
    # clf = MultinomialNB()
    # clf.fit(features, y_train)
    #
    # y_pred = clf.predict(x_test_counts)
    # print(sklearn.metrics.accuracy_score(y_test, y_pred))



    vectorizer = CountVectorizer(min_df=5,tokenizer=nltk.word_tokenize,max_df = 0.9,
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()  # for easy usage

    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

    from sklearn.linear_model import LogisticRegression

    #log_model = LogisticRegression()
    log_model = MultinomialNB()
    #log_model=svm.SVC(kernel='linear')
    t0 = time.time()
    log_model = log_model.fit(X=X_train, y=y_train)
    t1 = time.time()
    y_pred = log_model.predict(X_test)
    t2 = time.time()
    time_train = t1 - t0
    time_predict = t2 - t1
    # import random
    #
    # j = random.randint(0, len(X_test) - 7)
    # for i in range(j, j + 7):
    #     print(y_pred[0])
    #     ind = features_nd.tolist().index(X_test[i].tolist())
    #     print(data[ind].strip())

    print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))

    print(accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)




