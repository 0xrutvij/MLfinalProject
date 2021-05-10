'''
/* Filename:  dataProc.py
 * Date:      04/25/2021
 * Author:    Rutvij Shah
 * Email:     rutvij.shah@utdallas.edu
 * Course:    CS6375 Spring 2021
 * Version:   1.0
 * Copyright: 2021, All Rights Reserved
 *
 * Description:
 *     Pre-processes the data and generates the following files in data folder.
 *      - One csv file for each device the reading device came in contact with (4)
 *      - A csv file with all the data aggregated (1)
 *      - A .json file containing the threshold mappings for discretized data. (1)
 */
'''


import pandas as pd
import numpy as np
from dateutil import parser
import json

preFix = '../rawData/cb-devices-main('
postFix = ').csv'
files = []
dfList = []

DEBUG = False

for i in range(1, 11):
    filename = preFix + str(i) + postFix
    df = pd.read_csv(filename)
    dfList.append(df)

df = pd.concat(dfList, axis=0, ignore_index=True)

setOfDevicesContacted = set(df[['contact_id (S)']].to_numpy().transpose().tolist()[0])

if DEBUG:
    print(setOfDevicesContacted)

#convert timestamp strings to datetime objects
df['server_time (S)'] = df['server_time (S)'].apply(parser.parse)
df['message_sent_at (S)'] = df['message_sent_at (S)'].apply(parser.parse)

#time difference between when the message was sent and when the contact was detected
df['msg_delay'] = (df['message_sent_at (S)'] - df['server_time (S)']).dt.seconds

#time difference between when the current message at stamp and the contact reported at stamp
df['msg_delay_stamps'] =  df['current_msg_atstamp (N)'] - df['contact_reported_atstamp (N)']


df = df[['contact_type (S)', 'contact_class_score_diff (N)','contact_id (S)',
    'counter (S)', 'delay (N)', 'msg_delay', 'msg_delay_stamps']]

# Convert close contact into the positive class and proximate into the negative class
# i.e. close contact = 1 and proximate = 0

df.loc[df['contact_type (S)'] != 1, 'contact_type (S)'] = 0

# Rename columns for future ease.
df.rename(columns={'contact_type (S)':'contact_type',
'contact_class_score_diff (N)':'contact_class_score_diff',
    'contact_id (S)':'contact_id', 'counter (S)':'counter', 'delay (N)':'delay'}, inplace=True)

# TODO: Discretize the continuous classes.

# Discretizing delay,
# and since it is the difference of the other to delays it should capture their effect.
# [0, 8, 32, 141, 255] min, quartiles and max of delay.
#compVals = [8, 32, 141, 255]

compVals = df['delay'].describe().to_list()[-4:]
compVals = [int(i) for i in compVals]
delayThresholdMapping = {}
delayThresholdMapping['thresholdVals'] = compVals
prev = ''
prevVal = -1
for i, val in enumerate(compVals):
    df.loc[(df['delay'] > prevVal) & (df['delay'] <= val), 'delay'] = i
    delayThresholdMapping[prev + 'delay <= ' + str(val)] = i
    prevVal = val
    prev = str(val) + ' < '

if DEBUG:
    print(df.head())
    print(df.describe())
    print(delayThresholdMapping)

# Discretizing counter, but unsure of the relationship of it to other vars.
# Using min quartiles and max, [0, 7, 17, 28, 116]
# compVals = [7, 17, 28, 116]

compVals = df['counter'].describe().to_list()[-4:]
compVals = [int(i) for i in compVals]
prev = ''
prevVal = -1
counterThresholdMapping = {}
counterThresholdMapping['thresholdVals'] = compVals
for i, val in enumerate(compVals):
    df.loc[(df['counter'] > prevVal) & (df['counter'] <= val), 'counter'] = i
    counterThresholdMapping[prev + 'counter <= ' + str(val)] = i
    prevVal = val
    prev = str(val) + ' < '


if DEBUG:
    print(df.describe())
    print(counterThresholdMapping)

# Convert device ids to integers [0-3]
grouped = df.groupby('contact_id')
deviceIdToIntMapping = {}

for i, id in enumerate(setOfDevicesContacted):
    df1 = grouped.get_group(id)
    df1.to_csv('../processedData/' + id + '.csv')
    df.loc[df['contact_id'] == id, 'contact_id'] = i
    deviceIdToIntMapping[id] = i

if DEBUG:
    print(df.describe())
    print(deviceIdToIntMapping)

mappings = {
    'delayThresholdMapping':delayThresholdMapping,
    'deviceIdToIntMapping':deviceIdToIntMapping,
    'counterThresholdMapping':counterThresholdMapping
    }

with open('../processedData/mappings.json', 'w') as output:
    json.dump(mappings, output, indent=4)

df.to_csv('../processedData/aggregatedAndProcessed.csv')
