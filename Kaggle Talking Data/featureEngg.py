# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:48:58 2018

@author: raghu
"""

import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix

#check if difference is within last min, hour, 24hrs or earlier
def getpastclicks(diff, index):
    df.loc[index, 'time_diff'] = diff
    if diff.days >= 1:
        df.at[index, 'clicks_in_last_dayorearlier'] += 1
    else:
        hours, remainder = divmod(diff.seconds, 3600)
        mins, secs = divmod(remainder, 60)
        if hours >= 1 and hours <= 24:
            df.at[index, 'clicks_in_last_24hrs'] += 1
        else:
            if mins >= 1 and mins <= 60:
                df.at[index, 'clicks_in_last_onehr'] +=1
            else:
                if secs >= 1 and secs <= 60:
                    df.at[index, 'clicks_in_last_onemin'] += 1

def iterate_groups(x):
    # get indices in group to a list and iterate over it
    indices_in_group = x.index.tolist()
    # no need to iterate if theres just one element in group
    if len(indices_in_group) > 1:
        #iterate indices in group to calculate time diff
        for i in range(0, len(indices_in_group)):
            t1 = x.iloc[i, 5]
            for j in range(i + 1, len(indices_in_group)):
                t2index = indices_in_group[j]
                t2 = x.iloc[j, 5]
                if t1 == t2:
                    df.at[t2index, 'clicks_in_last_onemin'] += 1
                else:
                    diff = t2 - t1
                    getpastclicks(diff, t2index)

###############################################################################

starttime = time.time()
print("reading file...")
#df = pd.read_csv('train.csv', dtype = dtypes, skiprows=range(6000000,124903891), nrows = 5000, parse_dates=['click_time', 'attributed_time'],infer_datetime_format=True)

dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'bool'
        }

c = 0
for chunk in pd.read_csv('train.csv', dtype = dtypes, parse_dates=['click_time', 'attributed_time'], infer_datetime_format=True, skiprows=range(1,43000000), chunksize = 1000, nrows = 5000):
     chunk = chunk.drop(columns = ['attributed_time'], axis = 1)
     if c==0:
         df = chunk
         c += 1
     else:
         df = df.append(chunk, ignore_index = True)
         c += 1

del chunk
print("time taken loading file - " + str(time.time() - starttime))
print('memory usage - ')
df.info(memory_usage = 'deep')
print(df.memory_usage(deep = True)/1024)
print('dataframe head - ')
print(df.head())

# sort by ip and clicktime
#df.set_index('click_time')
df.sort_values(by = ['ip', 'click_time'], inplace = True)

# add new columns
df['time_diff'] = 0
df['clicks_in_last_dayorearlier'] = 0
df['clicks_in_last_24hrs'] = 0
df['clicks_in_last_onehr'] = 0
df['clicks_in_last_onemin'] = 0

# group by ip, get similar ip chunks, calculate time diff, can be changed to ip and device
print("\ngrouping data\n")
grouped = df.groupby('ip')
for key, group in grouped:
    #print("working on ip " + str(key))
    iterate_groups(group)

#iterate rows and calculate how many clicks ip had in last min, last hr, last day
#for index, row in df.iterrows():
#    ip_iter = row['ip']
#    #dataframe above iterated row
#    ip_index = df.columns.get_loc('ip')
#    click_time_index = df.columns.get_loc('click_time')
#    upperSlice = df.iloc[0:index,[ip_index, click_time_index]]
#    # select rows only where ip matches
#    upperSlice = upperSlice[upperSlice.ip == ip_iter]
#    for upperSliceIndex, upperSliceRow in upperSlice.iterrows():
#        t1 = upperSliceRow['click_time']
#        t2 = row['click_time']
#        diff = t2 - t1
#        col_index = df.columns.get_loc("time_diff")
#        df.iloc[index, col_index] = diff
#        getpastclicks(diff)

print("\nadded new features")

print("\nwriting to file")
df = df.drop(columns = ['time_diff'], axis = 1)
df.to_hdf('train_hdf', key = 'train_features', mode = 'w', format = 'table')

endtime = time.time()
print('\ntotal run time ' + str(endtime - starttime))

#df_hdf = pd.read_hdf('train_hdf', key = 'train_features', mode = 'r')
