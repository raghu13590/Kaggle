# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def prepareData():
    """ reads datafile and returns dataframe"""
    df_text = pd.read_csv('training_text', sep = '\|\|', index_col= 'ID',skip_blank_lines =True, nrows = None, header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
    print("TEXT COUNT - " + str(df_text.count()))
    print("MISSING TEXT")
    missing_text = df_text.isnull().sum()
    print(missing_text)

    df_variants = pd.read_csv('training_variants', skip_blank_lines =True, nrows = None, index_col= 'ID', header = None, skiprows = 1, names = ['ID','Gene','Variation','Class'], engine = 'python', encoding = 'utf-8')
    print("VARIANTS COUNT - " + str(df_variants.count()))
    print("MISSING VARIANTS")
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
    print(knnAccuracy)
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
    print(" DECSISION TREE CONFUSION MATRIX", cm_dtc)
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

def modifyDf(df, n):
    """extracts words in 'Text' column and converts them into column with its frequency as value"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    from collections import Counter

    stop_words = stopwords.words('english') + list(string.punctuation) + ['...' , '“', '”',]

    for index, row in df.iterrows():
        text = row['Text']
        text = word_tokenize(text)
        text = [x for x in text if x not in stop_words]
        text = Counter(text)
        text_top_ten = text.most_common(n)
        for i, j in text_top_ten:
            if i in df.columns:
                df.loc[index,i] = j
            else:
                df[i] = 0
                df.loc[index,i] = j
#    df2 = df
#    df2.drop('Text', axis=1, inplace=True)
#    df2.to_csv("text_tokenized.csv", sep=',')
    return df

##########################################################################

df = prepareData()

df = modifyDf(df, None)
print(df.head())

df = encodeCols(df)
X, y = getXAndY(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
ev = pca.explained_variance_ratio_
print("Variance - ", sum(ev))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(15, 15, 15))
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
acc_nn = mlp.score(X_test, y_test)
print("NEURAL NTWRK ACC - ", acc_nn)

runKnnForRange(10, 20)

runRandomForestForRange(10, 20)