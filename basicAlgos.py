# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:27:53 2017

@author: raghu
"""

import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def prepareData():
    """ reads datafile and parses to a dataframe """
    df_text = pd.read_csv('training_text', sep = '\|\|', index_col= 'ID',skip_blank_lines =True, nrows = None, header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
    print("ROWS IN TEXT - " + str(df_text.count()))
    print("MISSING DATA IN TEXT")
    missing_text = df_text.isnull().sum()
    print(missing_text)

    df_variants = pd.read_csv('training_variants', skip_blank_lines =True, nrows = None, index_col= 'ID', header = None, skiprows = 1, names = ['ID','Gene','Variation','Class'], engine = 'python', encoding = 'utf-8')
    print("ROWS IN VARIANTS - " + str(df_variants.count()))
    print("MISSING DATA IN VARIANTS")
    missing_variants = df_variants.isnull().sum()
    print(missing_variants)
    df = pd.concat([df_text, df_variants], axis = 1)
    return df

def encodeCols(df):
    """ encodes categorical features """
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == type(object):
            df[column] = le.fit_transform(df[column])
    return df

def getXAndY(df):
    """ splits dataframe to X and y and encodes X """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 3].values
    oht = OneHotEncoder(categorical_features = [0,1,2])
    X = oht.fit_transform(X).toarray()
    return X, y

def runKnnClassifier(X_train, X_test, y_train, y_test, n):
    """ Runs Knn Classifier for given n neighbors and returns accuracy """
    classifier = KNeighborsClassifier(n_neighbors = n)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    print("KNN ACCURACY FOR ", n)
    print(accuracy)
    cm = confusion_matrix(y_test, y_pred)
    print("KNN CONFUSION MATRIX")
    print(cm)
    return accuracy

def runKnnForRange(n1, n2):
    """ runs Knn Classifier over a range of n values and finds most accurate n value """
    knnAccuracy = {}
    for x in range(n1, n2 + 1):
        accuracy = runKnnClassifier(X_train, X_test, y_train, y_test, x)
        knnAccuracy.update({x: accuracy})
    maximum = max(knnAccuracy, key=knnAccuracy.get)
    plt.plot(list(knnAccuracy.keys()), list(knnAccuracy.values()))
    print("max accuracy for ", maximum, knnAccuracy[maximum])
    
def runDecTree():
    """ runs Decision Tree Classifier """
    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(criterion = 'entropy')
    dtc.fit(X_train, y_train)
    ydtc_pred = dtc.predict(X_test)
    acc_dtc = dtc.score(X_test, y_test)
    cm_dtc = confusion_matrix(y_test, ydtc_pred)
    return acc_dtc

def runRandomForest(e):
    """ runs Random Forest Classifier for given number of estimators """
    from sklearn.ensemble import RandomForestClassifier
    classifier_en = RandomForestClassifier(n_estimators = e, criterion = 'entropy')
    classifier_en.fit(X_train, y_train)
    y_pred_en = classifier_en.predict(X_test)
    cm_en = confusion_matrix(y_test, y_pred_en)
    acc_en = classifier_en.score(X_test, y_test)
    return acc_en

def runRandomForestForRange(e1, e2):
    """ runs Random Forest Classifier for a range of estimators """
    acc_en = {}
    for e in range(e1, e2 + 1):
        accuracy_e = runRandomForest(e)
        acc_en.update({e: accuracy_e})
    max_en = max(acc_en, key = acc_en.get)
    plt.plot(list(acc_en.keys()), list(acc_en.values()))
    print("max accuracy for " , max_en, acc_en[max_en])

##########################################################################

df = prepareData()

df = encodeCols(df)
X, y = getXAndY(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

runKnnForRange(10, 15)

runRandomForestForRange(10, 20)