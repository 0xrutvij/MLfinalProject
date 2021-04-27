import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts

DEBUG = False

# load the data
df = pd.read_csv('../processedData/aggregatedAndProcessed.csv')
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
xROStrain, xROStest, yROStrain, yROStest = tts(xROS, yROS, test_size=0.25, random_state=6375)

# save the training and test data into their respective files.
M = np.column_stack([yROStrain,xROStrain])
np.savetxt('../learningData/ROStrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([yROStest,xROStest])
np.savetxt('../learningData/ROStest.csv', M, delimiter=',', fmt='%d')



##########################
##        SMOTE         ##
##########################

# create SMOTE class
smoteClass = SMOTE()

# complete the re-sample
xSMOTE, ySMOTE = smoteClass.fit_resample(x, y)

# split into training and test data 75/25 split
xSMOTEtrain, xSMOTEtest, ySMOTEtrain, ySMOTEtest = tts(xSMOTE, ySMOTE, test_size=0.25, random_state=6375)

# save the training and test data into their respective files.
M = np.column_stack([ySMOTEtrain,xSMOTEtrain])
np.savetxt('../learningData/SMOTEtrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([ySMOTEtest,xSMOTEtest])
np.savetxt('../learningData/SMOTEtest.csv', M, delimiter=',', fmt='%d')
