
import sys
import os
import time

from nltk.stem import *
import sklearn
from sklearn.ensemble import VotingClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from nltk.stem import WordNetLemmatizer
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





    def lemma_tokens(tokens, lemmatizer):
        lemma = []
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


    # function to convert nltk tag to wordnet tag
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
        for i in tokens:
            if i not in stops:
                posTag = nltk.pos_tag(tokens)
        # posTag = nltk.pos_tag(tokens)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), posTag)

        lemma = lemma_tokens(wordnet_tagged, lemmatizer)
        return lemma


    def LogisticReg(features_train,features_test,y_train,y_test):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(features_train, y_train)
        preds = clf.predict(features_test)

        print("Performance of Logistic Regression Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def SVM_linear(features_train,features_test,y_train,y_test):
        clf = svm.SVC(kernel='linear', C=0.1, gamma='auto')
        clf.fit(features_train, y_train)
        preds = clf.predict(features_test)

        print("Performance of SVM Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def NaiveBayes_Bernoulli(features_train,features_test,y_train,y_test):
        Normalize = Normalizer()
        features_train = Normalize.fit_transform(features_train)
        features_test = Normalize.transform(features_test)

        clf = BernoulliNB(alpha=0.8,fit_prior=True, class_prior=None)
        clf.fit(features_train, y_train)
        preds = clf.predict(features_test)

        print("Performance of NaiveBayes Bernoulli Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def NaiveBayes_Multinomial(features_train,features_test,y_train,y_test):
        Normalize = Normalizer()
        features_train = Normalize.fit_transform(features_train)
        features_test = Normalize.transform(features_test)

        clf = MultinomialNB(alpha=0.8,fit_prior=True, class_prior=None)
        clf.fit(features_train, y_train)
        preds = clf.predict(features_test)

        print("Performance of NaiveBayes Multinomial Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def fourthClassifier(features_train, features_test, y_train, y_test):
        Normalize = Normalizer()
        features_train = Normalize.fit_transform(features_train)
        features_test = Normalize.transform(features_test)

        LogReg_clf = LogisticRegression(max_iter=2000, C=1.0)
        NB1_clf = MultinomialNB(alpha=0.8,fit_prior=True, class_prior=None)
        NB2_clf = BernoulliNB(alpha=0.8, fit_prior=True, class_prior=None)

        final_clf = VotingClassifier(estimators=[('LogReg', LogReg_clf), ('nb1', NB1_clf), ('nb2', NB2_clf)],voting='soft')
        final_clf.fit(features_train, y_train)
        preds = final_clf.predict(features_test)

        print("Performance of Fourth Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def Random_Baseline(features_train, features_test, y_train, y_test):

        final_clf = sklearn.dummy.DummyClassifier(strategy='uniform')
        final_clf.fit(features_train, y_train)
        preds = final_clf.predict(features_test)

        print("Performance of Random Classifier:")
        print("Accuracy:" , accuracy_score(y_test, preds))
        print(classification_report(y_test, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))


    def Tune_NaiveBayes(features_train, y_train):
        tuned_parameters = [{'alpha': [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]}]

        scores = ['precision', 'recall']

        for score in scores:
            clf = GridSearchCV(
                BernoulliNB(), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(features_train, y_train)

            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))

    def Tune_LogReg(features_train, y_train):
        tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }]

        scores = ['precision', 'recall']

        for score in scores:
            clf = GridSearchCV(
                LogisticRegression(max_iter=500), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(features_train, y_train)

            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))

    def Tune_SVM(features_train, y_train):
        tuned_parameters = [{'C': [ 0.001, 0.01, 0.1, 1, 10] }, {'gamma' : ['scale', 'auto']}]

        scores = ['precision', 'recall']

        for score in scores:
            clf = GridSearchCV(
                svm.SVC(kernel='linear'), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(features_train, y_train)

            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))


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


    #verctorizing and feature extraction
    vectorizer = CountVectorizer(min_df=2,max_df = 0.7, analyzer='word', lowercase=False,tokenizer=tokenize)

    features_train = vectorizer.fit_transform(X_train)
    features_test = vectorizer.transform(X_test)
    features_dev = vectorizer.transform(X_dev)


    """ These function can be uncommented to change the classifier """

    # Random_Baseline(features_train, features_test, y_train, y_test)
    # Tune_LogReg(features_train, y_train)
    # LogisticReg(features_train, features_test, y_train, y_test)
    # Tune_SVM(features_train, y_train)
    # SVM_linear(features_train, features_test, y_train, y_test)
    # Tune_NaiveBayes(features_train, y_train)
    # NaiveBayes_Bernoulli(features_train, features_test, y_train, y_test)
    # NaiveBayes_Multinomial(features_train, features_test, y_train, y_test)

    fourthClassifier(features_train, features_test, y_train, y_test)