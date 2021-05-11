import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts

DEBUG = False
TEST_SIZE = 0.30

# load the data
df = pd.read_csv('../3_processedData/aggregatedAndProcessed.csv')
df = df[['contact_type', 'contact_class_score_diff', 'contact_id','counter', 'delay']]

proxCount, closeCount = df['contact_type'].value_counts()

proxRows = df[df['contact_type']==0]
closeRows = df[df['contact_type']==1]

if DEBUG:
    print('Proximate Contacts:', proxRows.shape,'; Close Contacts: ', closeRows.shape)

# convert dataframe to numpy array
wholeSet = df.to_numpy()

# separate label and feature vectors.
y = wholeSet[:, 0]
x = wholeSet[:, 1:]



##########################
## Random Oversampling ##
##########################

# create random over sample class
sampler1 = ROS(random_state=6375)

# complete the re-sample
xROS, yROS = sampler1.fit_resample(x, y)

# split into training and test data 75/25 split
xROStrain, xROStest, yROStrain, yROStest = tts(xROS, yROS, test_size=TEST_SIZE, random_state=6375)

x_pr = np.concatenate([xROStrain, xROStest])
y_pr = np.concatenate([yROStrain, yROStest])
dfProto = np.column_stack([y_pr,x_pr])
df = pd.DataFrame(data=dfProto, columns=['contact_type', 'contact_class_score_diff', 'contact_id','counter', 'delay'])

import plotly.express as px
df["contact_type"] = df["contact_type"].astype(str) #convert to string
df.sort_values(['contact_id', 'contact_type'], inplace=True)
df["contact_id"] = df["contact_id"].astype(str) #convert to string
fig = px.scatter(df, x='contact_class_score_diff', y='contact_id', color='contact_type', title="ROS",)
df["contact_type"] = df["contact_type"].astype(int) #convert back to numeric
df["contact_id"] = df["contact_id"].astype(int) #convert to numeric
#title='Contact Class Score Diff vs Contact Type'
fig.show()

# save the training and test data into their respective files.
M = np.column_stack([yROStrain,xROStrain])
np.savetxt('../4_learningData/ROStrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([yROStest,xROStest])
np.savetxt('../4_learningData/ROStest.csv', M, delimiter=',', fmt='%d')



##########################
##        SMOTE         ##
##########################

# create SMOTE class
smoteClass = SMOTE()

# complete the re-sample
xSMOTE, ySMOTE = smoteClass.fit_resample(x, y)

# split into training and test data 75/25 split
xSMOTEtrain, xSMOTEtest, ySMOTEtrain, ySMOTEtest = tts(xSMOTE, ySMOTE, test_size=TEST_SIZE, random_state=6375)

x_pr = np.concatenate([xSMOTEtrain, xSMOTEtest])
y_pr = np.concatenate([ySMOTEtrain, ySMOTEtest])
dfProto = np.column_stack([y_pr,x_pr])
df = pd.DataFrame(data=dfProto, columns=['contact_type', 'contact_class_score_diff', 'contact_id','counter', 'delay'])

import plotly.express as px
df["contact_type"] = df["contact_type"].astype(str) #convert to string
df.sort_values(['contact_id', 'contact_type'], inplace=True)
df["contact_id"] = df["contact_id"].astype(str) #convert to string
fig = px.scatter(df, x='contact_class_score_diff', y='contact_id', color='contact_type', title="SMOTE",)
df["contact_type"] = df["contact_type"].astype(int) #convert back to numeric
df["contact_id"] = df["contact_id"].astype(int) #convert to numeric
#title='Contact Class Score Diff vs Contact Type'
fig.show()


# save the training and test data into their respective files.
M = np.column_stack([ySMOTEtrain,xSMOTEtrain])
np.savetxt('../4_learningData/SMOTEtrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([ySMOTEtest,xSMOTEtest])
np.savetxt('../4_learningData/SMOTEtest.csv', M, delimiter=',', fmt='%d')
