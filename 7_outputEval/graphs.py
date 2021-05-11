from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from dateutil import parser

preFix = '../1_rawData/cb-devices-main('
postFix = ').csv'
files = []
dfList = []

DEBUG = False

for i in range(1, 11):
    filename = preFix + str(i) + postFix
    df = pd.read_csv(filename)
    dfList.append(df)

df = pd.concat(dfList, axis=0, ignore_index=True)

df.rename(columns={'contact_type (S)':'contact_type',
'contact_class_score_diff (N)':'contact_class_score_diff',
    'contact_id (S)':'contact_id', 'counter (S)':'counter', 'delay (N)':'delay'}, inplace=True)

#convert timestamp strings to datetime objects
df['server_time'] = df['server_time (S)'].apply(parser.parse)
startTime = df['server_time'][0]


df['server_time'] = (df['server_time'] - startTime).dt.seconds/60
df.sort_values(['server_time'], inplace=True, ignore_index=True)
print(df['server_time'])

size = 50
list_of_dfs = [df.loc[i:i+size-1,:] for i in range(0, len(df),size)]

for df in list_of_dfs:
    #df.plot.scatter(x=['server_time'], y=['contact_type'])
    #plt.show()
    pass


df = pd.read_csv('../3_processedData/aggregatedAndProcessed.csv')
df = df[['contact_type', 'contact_class_score_diff', 'contact_id','counter', 'delay', 'msg_delay_stamps', 'msg_delay']]
df.rename(columns={
"contact_type":"contactType",
"contact_class_score_diff": "classScore", 
"contact_id": "deviceContacted", 
"msg_delay_stamps":"serverDelay", 
"msg_delay":"deviceDelay"}, inplace=True)

#print graph of contact_type vs delay
df.plot.scatter(x=['delay'], y=['classScore'], c='contactType', colormap='bwr')
plt.show()
#device IDs and contact type
df.plot.scatter(x=['deviceContacted'], y=['classScore'], c='contactType', colormap='bwr')
plt.show()
#delay vs deviceID
df.plot.scatter(x=['classScore'], y=['contactType'], c='contactType', colormap='bwr')
plt.show()