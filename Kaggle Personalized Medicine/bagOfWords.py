# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:33:21 2018

@author: raghu
"""


import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import time

def readData():
    """ reads datafile and returns dataframe"""
    pd.set_option('display.expand_frame_repr', False)
    # read data from training_text
    df_text = pd.read_csv('training_text', sep = '\|\|', index_col= 'ID',skip_blank_lines =True, nrows = 10, header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8', dtype = str)
    print("TEXT COUNT - " + str(df_text.count()))
    print("MISSING TEXT")
    missing_text = df_text.isnull().sum()
    print(missing_text)
    
    # read data from training_variants
    df_variants = pd.read_csv('training_variants', skip_blank_lines =True, nrows = 10, index_col= 'ID', header = None, skiprows = 1, names = ['ID','Gene','Variation','Class'], engine = 'python', encoding = 'utf-8', dtype = str)
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

def createBagOfWords(df, n):
    """extracts words in 'Text' column and converts them into column with its frequency as value"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    from collections import Counter
    from nltk.stem.porter import PorterStemmer
    
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english') + list(string.punctuation) + ['...' , ',' , '“', '”', '.', 'fig', '.fig'])
    
    dfDict = {}
#    df['count'] = df.groupby('Text')['Text'].transform(pd.Series.value_counts)
#    df.sort_values('count', ascending = False)
#    df.drop('count', axis = 1)
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
    textBagOfWords = cv.fit_transform(df['Text']).toarray()
    textBagOfWords = pd.DataFrame(textBagOfWords)
    df = df.drop(['Text'], axis = 1)
    df = pd.concat([df, textBagOfWords], axis = 1)
    return df

def writeToFile(df, fileName):
    df.to_csv(fileName, encoding = 'utf-8')
    
def readFileInChunks(filePath, chunkSize):
    df = pd.DataFrame()
    for chunk in pd.read_csv(filePath, chunksize = chunkSize):
        df = df.append(chunk)
    return df
###############################################################
startTime = time.time()
df = readData()
#visualizeData(df)
df = createBagOfWords(df, None)
writeToFile(df, 'bagOfWordsDataset.csv')

#df = readFileInChunks('bagOfWordsDataset.csv', 500)

endTime = time.time()
print("Execution time - " + str(endTime - startTime))