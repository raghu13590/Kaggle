# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 01:31:07 2018

@author: raghu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def getTestAndTrain(df):
    """remove true events from training set"""
    #split dataset into test and training
    df_rowsize, df_colsize = df.shape
    df_train_rowsize = int(0.75*df_rowsize)
    df_train = df.iloc[0:df_train_rowsize, :]
    df_test = df.iloc[df_train_rowsize: df_rowsize, :]
    
    #remove true values in training set
    df_train = df_train[df_train.is_attributed != True]
    
    #split train and test into X and y
    y_train = df_train['is_attributed'] 
    X_train = df_train.drop(columns = ['is_attributed'])
    y_test = df_test['is_attributed']
    X_test = df_test.drop(columns = ['is_attributed'])
    return X_train, y_train, X_test, y_test

def runDecTree():
    """ runs Decision Tree Classifier """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix
    dtc = DecisionTreeClassifier(criterion = 'entrophy')
    dtc.fit(X_train, y_train)
    ydtc_pred = dtc.predict(X_test)
    acc_dtc = dtc.score(X_test, y_test)
    cm_dtc = confusion_matrix(y_test, ydtc_pred)
    return acc_dtc, cm_dtc

def runRandomForest(e):
    """ runs Random Forest Classifier for given number of estimators """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    classifier_en = RandomForestClassifier(n_estimators = e, criterion = 'entropy')
    classifier_en.fit(X_train, y_train)
    y_pred_en = classifier_en.predict(X_test)
    cm_en = confusion_matrix(y_test, y_pred_en)
    acc_en = classifier_en.score(X_test, y_test)
    return acc_en, cm_en

def timeFeatures(df):
    # Make some new features with click_time column
    df['dow'] = df['click_time'].dt.dayofweek
    df["doy"] = df["click_time"].dt.dayofyear
    df["hod"] = df['click_time'].dt.hour
    return df

#####################################################################

# read dataset
starttime = time.time()
df = pd.read_hdf('train_hdf', key = 'train_features', mode = 'r')

# add features hour of day, day of week, day of year
df = timeFeatures(df)

#remove columns not required
df = df.drop(columns = ['ip', 'click_time', 'attributed_time'])

# create dummy variables for categorical data
df = pd.get_dummies(df, columns = ['app', 'device', 'os', 'channel', 'dow', 'doy', 'hod'])
print(df.head())

#remove empty rows from dataframe
#df = df.loc[:, (df != 0).any(axis=0)]

#remove trues from training set
#print("\nsplitting test and train data")
#X_train, y_train, X_test, y_test = getTestAndTrain(df)

## split data into X and y
y = df.loc[:, 'is_attributed'].values
X = df.drop(columns = ['is_attributed'])

## split data set into test and train
print("\nsplitting test and train data")
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

#apply xbBoost model
print("\nrunning xgBoost model...")
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate = 0.6, max_depth = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# plot feature importance
from xgboost import plot_importance
plot_importance(classifier)
plt.show()

#confusion matrix
print("\nconfusion matrix")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
cm = confusion_matrix(y_test, y_pred)
print(cm)
precision, recall, fscore, support = score(y_test, y_pred)
print("\nprecision " + str(precision))
print("\nrecall " + str(recall))
print("\nf1score " + str(fscore))
print("\nsupport " + str(support))

print("\ntotal run time " + str(time.time() - starttime))