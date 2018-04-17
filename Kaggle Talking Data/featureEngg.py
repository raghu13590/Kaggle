# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:48:58 2018

@author: raghu
"""

import pandas as pd
import numpy as np
import time
import gc

#check if difference is within last min, hour, 24hrs or earlier
def getpastclicks(diff, index):
    df1.at[index, 'time_diff'] = diff
    if diff.days >= 1:
        df1.at[index, 'clicks_in_last_dayorearlier'] += 1
    else:
        hours, remainder = divmod(diff.seconds, 3600)
        mins, secs = divmod(remainder, 60)
        if hours >= 1 and hours <= 24:
            df1.at[index, 'clicks_in_last_24hrs'] += 1
        else:
            if mins >= 1 and mins <= 60:
                df1.at[index, 'clicks_in_last_onehr'] +=1
            else:
                if secs >= 1 and secs <= 60:
                    df1.at[index, 'clicks_in_last_onemin'] += 1

def iterate_groups(group):
    # get indices in group to a list and iterate over it
    indices_in_group = group.index.tolist()
    # no need to iterate if theres just one element in group
    index_len = len(indices_in_group)
    # if index_len > 1:
    #iterate indices in group to calculate time diff
    for i in range(0, index_len):
        t1 = group.iat[i, 5]
        for j in range(i + 1, index_len):
            t2index = indices_in_group[j]
            t2 = group.iat[j, 5]
            if t1 == t2:
                df1.at[t2index, 'clicks_in_last_onemin'] += 1
            else:
                diff = t2 - t1
                getpastclicks(diff, t2index)

###############################################################################

starttime = time.time()

dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'bool'
        }

# sampling by reading every kth index
file = "train.csv"
n = sum(1 for l in open(file))       # number of rows in file
print("\nnumber of lines in file " + str(n))
s = 1000000          # approx sample size
k = int(n/s)
skip_ids = [x for x in range(1, n) if x % k != 0]
n_rows_skipping = len(skip_ids)
print("\nsample size " + str(n - n_rows_skipping - 1))
print("\nreading file...")
df = pd.read_csv(file, dtype = dtypes, parse_dates=['click_time', 'attributed_time'], infer_datetime_format=True, skiprows = skip_ids)

del skip_ids
gc.collect

print("time taken loading file - " + str(time.time() - starttime))
print('\nmemory usage - ')
df.info(memory_usage = 'deep')
print(df.memory_usage(deep = True)/1024)
print('\ndataframe head - ')
print(df.head())

# slice on ip and click_time to iterate over and add extract features 
df1 = df[['ip', 'click_time']]
df2 = df.drop(columns = ['ip', 'click_time'])
del df
gc.collect
# add new columns
df1['time_diff'] = 0
df1['clicks_in_last_dayorearlier'] = 0
df1['clicks_in_last_24hrs'] = 0
df1['clicks_in_last_onehr'] = 0
df1['clicks_in_last_onemin'] = 0

newcol_dtypes = {
        'time_diff': 'uint16',
        'clicks_in_last_dayorearlier': 'uint16',
        'clicks_in_last_24hrs': 'uint16',
        'clicks_in_last_onehr': 'uint16',
        'clicks_in_last_onemin': 'uint16'
        }

df1 = df1.astype(dtype = newcol_dtypes) # set datatypes for new columns added

# sort by ip and clicktime
#df.sort_values(by = ['ip', 'click_time'], inplace = True)

# group by ip, get similar ip chunks, calculate time difference, can be changed to ip and device
print("\nadding features\n")
featuretime = time.time()
#grouped = df.groupby('ip')
#for key, group in grouped:
#    iterate_groups(group)
df1.groupby('ip').apply(lambda group: iterate_groups(group) if len(group.index.tolist()) > 1 else group)

print("\nadded new features")
print("time taken adding features " + str(time.time() - featuretime))

df = pd.concat([df1, df2], axis=1, join='inner')
df = df.drop(columns = ['time_diff'], axis = 1)

del df1, df2
gc.collect

#write dataframe to hdf file
print("\nwriting to file")
df.to_hdf('train_hdf', key = 'train_features', mode = 'w', format = 'table')

print('\ntotal run time ' + str(time.time() - starttime))

#df_hdf = pd.read_hdf('train_hdf', key = 'train_features', mode = 'r')