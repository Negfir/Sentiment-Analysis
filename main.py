# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



import sys
import sys
import os
import time

import numpy as np
from nltk import tokenize
from nltk.stem import *
from nltk.tokenize import word_tokenize
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import nltk
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize,sent_tokenize
from sklearn.naive_bayes import *
from sklearn.tree import *
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

if __name__ == '__main__':
    #Raw data is converted to seperate files for each comment, inorder to read them easily



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


    print(data[8])

    print(data[89])

    import string

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def lemma_tokens(tokens, lemmatizer):
        lemma = []
        for item in tokens:
            lemma.append(lemmatizer.lemmatize(item))
        return lemma

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text):
        stops = set(stopwords.words("english"))
        text = "".join([ch for ch in text if (ch not in string.punctuation and ch not in string.digits)])
        tokens = nltk.word_tokenize(text)
        #stems = stem_tokens(tokens, stemmer)

        lemma = lemma_tokens(tokens, lemmatizer)
        #posTag = nltk.pos_tag(lemma)
        return lemma


    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data_labels,
        train_size=0.80,
        random_state=1234)

    vectorizer = CountVectorizer(min_df=2,max_df = 0.5,
        analyzer='word',
        lowercase=False,tokenizer=tokenize
    )
    features_train = vectorizer.fit_transform(
        X_train
    )
    features_test = vectorizer.transform(
        X_test
    )
    TFvectorizer = TfidfTransformer()
    TFfeatures_train = TFvectorizer.fit_transform(features_train)
    TFfeatures_test = TFvectorizer.transform(features_test)

  #  features_nd = TFfeatures.toarray()  # for easy usage
    #print(vectorizer.vocabulary_)


    from sklearn.linear_model import LogisticRegression

    #log_model = LogisticRegression()
    #log_model = MultinomialNB(alpha=0.85,fit_prior=True, class_prior=None)
    log_model = BernoulliNB(alpha=0.85)
    #log_model = MLPClassifier(hidden_layer_sizes=(500,)) #0.73
    #log_model = DecisionTreeClassifier() #0.63 very bad

    #log_model=svm.SVC(kernel='linear')
    #log_model=svm.LinearSVC(C=1.0, tol=0.0001, max_iter=1000, penalty='l2', loss='squared_hinge', dual=True, multi_class='ovr', fit_intercept=True, intercept_scaling=1)
    #log_model = RandomForestClassifier(n_estimators=200, random_state=0) #0.71

    tuned_parameters = [{'alpha': [0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            BernoulliNB(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(TFfeatures_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(TFfeatures_test)
        print(classification_report(y_true, y_pred))
        print()


    # t0 = time.time()
    # log_model = log_model.fit(X=TFfeatures_train, y=y_train)
    # t1 = time.time()
    # y_pred = log_model.predict(TFfeatures_test)
    # t2 = time.time()
    # time_train = t1 - t0
    # time_predict = t2 - t1
    #
    #
    #
    # print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
    #
    # print(accuracy_score(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)




