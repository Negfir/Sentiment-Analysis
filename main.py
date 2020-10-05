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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
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
from sklearn.decomposition import PCA, TruncatedSVD
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
import string
from nltk.corpus import wordnet
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    # Reading Files
    data = []
    data_labels = []
    with open("./rt-polaritydata/rt-polarity.neg",encoding="ISO-8859-1") as f:
        for i in f:
            data.append(i.strip("\n"))
            #print(i)
            data_labels.append('pos')

    with open("./rt-polaritydata/rt-polarity.pos",encoding="ISO-8859-1") as f:
        for i in f:
            data.append(i.strip("\n"))
            data_labels.append('neg')



    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()


    #spliting train set and test+dev set
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data_labels,
        train_size=0.80,
        random_state=1234)
    #spliting test set and dev set
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_test,
        y_test,
        train_size=0.50,
        random_state=4321)

    def lemma_tokens(tokens, lemmatizer):
        lemma = []
        # for item in tokens:
        #     word = tokens[0]
        #     word_pos = tokens[1]
        #     lemma.append(lemmatizer.lemmatize(word,pos=word_pos))
        # return lemma
        for word, tag in tokens:
            if tag is None:
                # if there is no available tag, append the token as is
                lemma.append(word)
            else:
                # else use the tag to lemmatize the token
                lemma.append(lemmatizer.lemmatize(word, tag))
        return lemma

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed


    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def tokenize(text):
        stops = set(stopwords.words("english"))
        text = "".join([ch for ch in text if (ch not in string.punctuation and ch not in string.digits)])
        tokens = nltk.word_tokenize(text)
        # for i in tokens:
        #     if i not in stops:
        #         posTag = nltk.pos_tag(tokens)
        posTag = nltk.pos_tag(tokens)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), posTag)

        lemma = lemma_tokens(wordnet_tagged, lemmatizer)
        #p
        return lemma

    #verctorizing and feature extraction
    vectorizer = CountVectorizer(min_df=2,max_df = 0.6,
        analyzer='word',
        lowercase=False,tokenizer=tokenize
    )
    features_train = vectorizer.fit_transform(
        X_train
    )
    features_test = vectorizer.transform(
        X_test
    )
    features_dev = vectorizer.transform(
        X_dev
    )
    # TFvectorizer = TfidfTransformer()
    # features_train = TFvectorizer.fit_transform(features_train)
    # features_test = TFvectorizer.transform(features_test)





    final_clf = LogisticRegression(max_iter=2000)
    clf2 = svm.SVC(kernel='linear')
    final_clf = MultinomialNB(alpha=0.8,fit_prior=True, class_prior=None)
    clf5 = BernoulliNB(alpha=0.8)

    clf5 = KNeighborsClassifier(n_neighbors=1)
    clf6 = MLPClassifier(hidden_layer_sizes=(500,)) #0.73
    clf8 = RandomForestClassifier(n_estimators=200, random_state=0)  # 0.71
    clf9=svm.LinearSVC(C=1.0, tol=0.0001, max_iter=1000, penalty='l2', loss='squared_hinge', dual=True, multi_class='ovr', fit_intercept=True, intercept_scaling=1)





    # t0 = time.time()
    # log_model = clf4.fit(features_train, y_train)
    # t1 = time.time()
    # y_pred = log_model.predict(features_test)
    # t2 = time.time()
    # time_train = t1 - t0
    # time_predict = t2 - t1

    LogReg_clf = LogisticRegression(max_iter=2000)
    NB1_clf = MultinomialNB(alpha=0.8,fit_prior=True, class_prior=None)
    NB2_clf = BernoulliNB(alpha=0.8)
    # #voting_clf = VotingClassifier(estimators=[('DTree', DTree_clf), ('LogReg', LogReg_clf), ('nb', SVC_clf)],voting='soft')
    final_clf = StackingClassifier(estimators=[('DTree', LogReg_clf), ('nb1', NB1_clf), ('nb', NB2_clf)])
    #
    final_clf.fit(features_train, y_train)
    preds = final_clf.predict(features_test)

    print(accuracy_score(y_test, preds)) #0.762
    print(metrics.classification_report(y_test, preds)) #0.79
    cm = confusion_matrix(y_test, preds)
    print(cm)




    # tuned_parameters = [{'alpha': [0.0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
    #
    # scores = ['precision', 'recall']
    #
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(
    #         BernoulliNB(), tuned_parameters, scoring='%s_macro' % score
    #     )
    #     clf.fit(TFfeatures_train, y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #
    #     print("Detailed classification report:")
    #     print()
    #     print("The model is trained on the full development set.")
    #     print("The scores are computed on the full evaluation set.")
    #     print()
    #     y_true, y_pred = y_test, clf.predict(TFfeatures_test)
    #     print(classification_report(y_true, y_pred))
    #     print()
