# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:33:21 2018

@author: raghu
"""


import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

def prepareData():
    """ reads datafile and returns dataframe"""
    pd.set_option('display.expand_frame_repr', False)
    # read data from training_text
    df_text = pd.read_csv('training_text', sep = '\|\|', index_col= 'ID',skip_blank_lines =True, nrows = None, header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8', dtype = str)
    print("TEXT COUNT - " + str(df_text.count()))
    print("MISSING TEXT")
    missing_text = df_text.isnull().sum()
    print(missing_text)
    
    # read data from training_variants
    df_variants = pd.read_csv('training_variants', skip_blank_lines =True, nrows = None, index_col= 'ID', header = None, skiprows = 1, names = ['ID','Gene','Variation','Class'], engine = 'python', encoding = 'utf-8', dtype = str)
    print("VARIANTS COUNT - " + str(df_variants.count()))
    print("MISSING VARIANTS")
    missing_variants = df_variants.isnull().sum()
    print(missing_variants)
    # merge both datasets
    df = pd.concat([df_text, df_variants], axis = 1)
    return df

def visualizeData(df):
    """plots histogram of all columns in passed in dataframe"""
    for column in df:
        df[column].value_counts().plot(kind = 'bar', rot = 'vertical', use_index = False)

def modifyDf(df, n):
    """extracts words in 'Text' column and converts them into column with its frequency as value"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    from collections import Counter
    from nltk.stem.porter import PorterStemmer
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english') + list(string.punctuation) + ['...' , ',' , '“', '”', '.', 'fig', '.fig'])
    
    dfDict = {}
    dfLength = len(df.index)
    for i in range(1, dfLength):
        text = df['Text'][i]
        if dfDict.__contains__(text):
            df['Text'][i] = dfDict.get(text)
        else:
            textModified = text.lower()
            textModified = textModified.replace(',', '')
            textModified = textModified.replace('.', '')
            textModified = textModified.split()
            textModified = [ps.stem(word) for word in textModified if not word in set(stop_words) and len(word) > 1]
            textModified = ' '.join(textModified)
            df['Text'][i] = textModified
            dfDict[text] = textModified
        
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(df['Text']).toarray()
    dfX = pd.DataFrame(X)
    df = df.drop(['Text'], axis = 1)
    df = pd.concat([df, dfX], axis = 1)
    
    df.to_csv('bagOfWordsDataset.csv', encoding = 'utf-8')
    return df, X

###############################################################
df = prepareData()
#visualizeData(df)

df, X = modifyDf(df, None)
print(df.head())
