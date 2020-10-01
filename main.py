# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



import sys
import sys
import os
import time

from nltk import tokenize
from nltk.stem import *
from nltk.tokenize import word_tokenize
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
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
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize,sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

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

    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.neg") as f:
    #
    #     for i in f:
    #         p=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in words:
    #             p=p+ps.stem(w)+" "
    #         data.append(p)
    #         #print(i)
    #         data_labels.append('pos')
    #
    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.pos") as f:
    #     for i in f:
    #         n=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in words:
    #             p=p+ps.stem(w)+" "
    #         data.append(n)
    #         data_labels.append('neg')

    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.neg") as f:
    #
    #     for i in f:
    #         p=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in words:
    #             p=p+lemmatizer.lemmatize(w)+" "
    #         data.append(p)
    #         #print(i)
    #         data_labels.append('pos')
    #
    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.pos") as f:
    #     for i in f:
    #         n=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in words:
    #             p=p+lemmatizer.lemmatize(w)+" "
    #         data.append(n)
    #         data_labels.append('neg')

    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.neg") as f:
    #
    #     for i in f:
    #         p=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in i.split(' '):
    #             p=p+w+" "
    #         data.append(p)
    #         #print(i)
    #         data_labels.append('pos')
    #
    # with open("./rt-polaritydata/rt-polaritydata/rt-polarity.pos") as f:
    #     for i in f:
    #         n=""
    #         words = tokenize.word_tokenize(i.strip("\n"))
    #         for w in i.split(' '):
    #             p=p+w+" "
    #         data.append(n)
    #         data_labels.append('neg')

    data_list = ['the gamers playing games', 'higher scores', 'sports']
    print(data[8])
    # words = tokenize.word_tokenize(data[8])
    # for w in words:
    #     print(ps.stem(w))

    # from nltk.stem.snowball import SnowballStemmer
    #
    # stemmer = SnowballStemmer('english')
    # new_corpus = [' '.join([stemmer.stem(word) for word in data_list.split(' ')])
    #               for text in words]
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
        text = "".join([ch for ch in text if (ch not in string.punctuation and ch not in string.digits)])
        tokens = nltk.word_tokenize(text)
        #stems = stem_tokens(tokens, stemmer)
        lemma = lemma_tokens(tokens, lemmatizer)
        return lemma

    vectorizer = CountVectorizer(min_df=2,max_df = 0.4,
        analyzer='word',
        lowercase=False,tokenizer=tokenize
    )
    features = vectorizer.fit_transform(
        data
    )

    TFvectorizer = TfidfTransformer()
    TFfeatures = TFvectorizer.fit_transform(features)

    features_nd = TFfeatures.toarray()  # for easy usage
    #print(vectorizer.vocabulary_)
    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

    from sklearn.linear_model import LogisticRegression

    #log_model = LogisticRegression()
    log_model = MultinomialNB()
    #log_model=svm.SVC(kernel='linear')
    #log_model = RandomForestClassifier(n_estimators=200, random_state=0)
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




