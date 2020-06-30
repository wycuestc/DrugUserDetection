import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def printResult(y_test, predictions):
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions))
    print('F1 score: ', f1_score(y_test, predictions))
    print('-----------------------------------------------------------------------------')

def returnDiffResult(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return acc, prec, rec, f1

def getResultLR(X_train_cv, X_test_cv, y_train, y_test):
    lr = LogisticRegression()
    lr.fit(X_train_cv, y_train)
    predictions = lr.predict(X_test_cv)

    acc, prec, rec, f1 = returnDiffResult(y_test, predictions)
    return acc, prec, rec, f1

def getResultNB(X_train_cv, X_test_cv, y_train, y_test):
    X_train_cv = X_train_cv.toarray()
    X_test_cv = X_test_cv.toarray()
    # y_train = y_train.toarray()
    # y_test = y_test.toarray()

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train_cv, y_train)
    predictions = naive_bayes.predict(X_test_cv)

    acc, prec, rec, f1 = returnDiffResult(y_test, predictions)
    return acc, prec, rec, f1

def getResultTree(X_train_cv, X_test_cv, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train_cv, y_train)
    predictions = dt.predict(X_test_cv)

    acc, prec, rec, f1 = returnDiffResult(y_test, predictions)
    return acc, prec, rec, f1

def getResultRF(X_train_cv, X_test_cv, y_train, y_test):
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train_cv, y_train)
    predictions = rf.predict(X_test_cv)

    acc, prec, rec, f1 = returnDiffResult(y_test, predictions)
    return acc, prec, rec, f1

def getResultSVM(X_train_cv, X_test_cv, y_train, y_test):
    svm = LinearSVC()
    svm.fit(X_train_cv, y_train)
    predictions = svm.predict(X_test_cv)

    acc, prec, rec, f1 = returnDiffResult(y_test, predictions)
    return acc, prec, rec, f1

def iterExperiment(X, Y):
    numIters = 3
    num_splits = 5

    accLR_sum = 0
    precLR_sum = 0
    recLR_sum = 0
    f1LR_sum = 0

    accNB_sum = 0
    precNB_sum = 0
    recNB_sum = 0
    f1NB_sum = 0

    accTree_sum = 0
    precTree_sum = 0
    recTree_sum = 0
    f1Tree_sum = 0

    accRF_sum = 0
    precRF_sum = 0
    recRF_sum = 0
    f1RF_sum = 0

    accSVM_sum = 0
    precSVM_sum = 0
    recSVM_sum = 0
    f1SVM_sum = 0

    kfold = StratifiedKFold(n_splits=num_splits, shuffle=False, random_state=1)

    for i in range(0,numIters):
        for train, test in kfold.split(X, Y):
            X_train = X[train]
            X_test = X[test]
            y_train = Y[train]
            y_test = Y[test]
            cv = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
            X_train_cv = cv.fit_transform(X_train)
            X_test_cv = cv.transform(X_test)

            accLR, precLR, recLR, f1LR = getResultLR(X_train_cv, X_test_cv, y_train, y_test)
            accNB, precNB, recNB, f1NB = getResultNB(X_train_cv, X_test_cv, y_train, y_test)
            accTree, precTree, recTree, f1Tree = getResultTree(X_train_cv, X_test_cv, y_train, y_test)
            accRF, precRF, recRF, f1RF = getResultRF(X_train_cv, X_test_cv, y_train, y_test)
            accSVM, precSVM, recSVM, f1SVM = getResultSVM(X_train_cv, X_test_cv, y_train, y_test)

            accLR_sum += accLR
            precLR_sum += precLR
            recLR_sum += recLR
            f1LR_sum += f1LR

            accNB_sum += accNB
            precNB_sum += precNB
            recNB_sum += recNB
            f1NB_sum += f1NB

            accTree_sum += accTree
            precTree_sum += precTree
            recTree_sum += recTree
            f1Tree_sum += f1Tree

            accRF_sum += accRF
            precRF_sum += precRF
            recRF_sum += recRF
            f1RF_sum += f1RF

            accSVM_sum += accSVM
            precSVM_sum += precSVM
            recSVM_sum += recSVM
            f1SVM_sum += f1SVM

    accLR_sum /= numIters*num_splits
    precLR_sum /= numIters*num_splits
    recLR_sum /= numIters*num_splits
    f1LR_sum /= numIters*num_splits

    accNB_sum /= numIters*num_splits
    precNB_sum /= numIters*num_splits
    recNB_sum /= numIters*num_splits
    f1NB_sum /= numIters*num_splits

    accTree_sum /= numIters*num_splits
    precTree_sum /= numIters*num_splits
    recTree_sum /= numIters*num_splits
    f1Tree_sum /= numIters*num_splits

    accRF_sum /= numIters*num_splits
    precRF_sum /= numIters*num_splits
    recRF_sum /= numIters*num_splits
    f1RF_sum /= numIters*num_splits

    accSVM_sum /= numIters*num_splits
    precSVM_sum /= numIters*num_splits
    recSVM_sum /= numIters*num_splits
    f1SVM_sum /= numIters*num_splits

    print("Logistic Regression:")
    print('Accuracy score: ', accLR_sum)
    print('Precision score: ', precLR_sum)
    print('Recall score: ', recLR_sum)
    print('F1 score: ', f1LR_sum)
    print('-----------------------------------------------------------------------------')

    print("Naive Bayes:")
    print('Accuracy score: ', accNB_sum)
    print('Precision score: ', precNB_sum)
    print('Recall score: ', recNB_sum)
    print('F1 score: ', f1NB_sum)
    print('-----------------------------------------------------------------------------')

    print("Decision Tree:")
    print('Accuracy score: ', accTree_sum)
    print('Precision score: ', precTree_sum)
    print('Recall score: ', recTree_sum)
    print('F1 score: ', f1Tree_sum)
    print('-----------------------------------------------------------------------------')

    print("Random Forest:")
    print('Accuracy score: ', accRF_sum)
    print('Precision score: ', precRF_sum)
    print('Recall score: ', recRF_sum)
    print('F1 score: ', f1RF_sum)
    print('-----------------------------------------------------------------------------')

    print("SVM:")
    print('Accuracy score: ', accSVM_sum)
    print('Precision score: ', precSVM_sum)
    print('Recall score: ', recSVM_sum)
    print('F1 score: ', f1SVM_sum)
    print('-----------------------------------------------------------------------------')

def main():
    scriptDir = os.path.dirname(__file__)
    inputFile = os.path.join(scriptDir, "dataCollection/dataset1000/bidataOpiumStreet1020.csv")
    df = pd.read_csv(inputFile, delimiter=',', encoding='latin-1')

    #preprocess the data
    stop = stopwords.words('english')
    X = df.iloc[:,5].apply(lambda x: [item for item in x.split() if item not in stop])
    lemma = WordNetLemmatizer()
    X = X.apply(lambda x: [lemma.lemmatize(item) for item in x])

    Y = df.iloc[:,2]
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    iterExperiment(X, Y)




if __name__ == '__main__':
    main()