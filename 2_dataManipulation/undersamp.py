import pandas as pd
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.under_sampling import TomekLinks as TL
from imblearn.under_sampling import NearMiss as NM
from sklearn.model_selection import train_test_split as tts

DEBUG = False
TEST_SIZE = 0.30
NN = False

if NN:
    df = pd.read_csv('../3_processedData/aggregatedAndProcessedNN.csv')
else:
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
##     Direct Split     ##
##########################

# split into training and test data 75/25 split
xDStrain, xDStest, yDStrain, yDStest = tts(x, y, test_size=TEST_SIZE, random_state=6375)

# save the training and test data into their respective files.

M = np.column_stack([yDStrain,xDStrain])
if NN:
    np.savetxt('../4_learningData/DStrainNN.csv', M, delimiter=',', fmt='%d')
else:
    np.savetxt('../4_learningData/DStrain.csv', M, delimiter=',', fmt='%d')

M = np.column_stack([yDStest,xDStest])
if NN:
    np.savetxt('../4_learningData/DStestNN.csv', M, delimiter=',', fmt='%d')
else:
    np.savetxt('../4_learningData/DStest.csv', M, delimiter=',', fmt='%d')

##########################
## Random Undersampling ##
##########################

# create the random undersampler object
sampler1 = RUS(random_state=6375, replacement=True)

# random undersamplinggg
xRUS, yRUS = sampler1.fit_resample(x, y)

if DEBUG:
    print(x.shape)
    print(y.shape)
    print(sorted(Counter(yRUS).items()))

# split into training and test data 75/25 split
xRUStrain, xRUStest, yRUStrain, yRUStest = tts(xRUS, yRUS, test_size=TEST_SIZE, random_state=6375)

# save the training and test data into their respective files.
M = np.column_stack([yRUStrain,xRUStrain])
if NN:
    np.savetxt('../4_learningData/RUStrainNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/RUStrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([yRUStest,xRUStest])
if NN:
    np.savetxt('../4_learningData/RUStestNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/RUStest.csv', M, delimiter=',', fmt='%d')



##########################
##     Tomek Links      ##
##########################

# create the tomek links undersampler object
sampler2 = TL()

# tomek link undersampling
xTL, yTL = sampler2.fit_resample(x,y)

if DEBUG:
    print(xTL.shape)
    print(yTL.shape)
    print(sorted(Counter(yTL).items()))

# split into training and test data 75/25 split
xTLtrain, xTLtest, yTLtrain, yTLtest = tts(xTL, yTL, test_size=TEST_SIZE, random_state=6375)

# save the training and test data into their respective files.
M = np.column_stack([yTLtrain,xTLtrain])
if NN:
    np.savetxt('../4_learningData/TLtrainNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/TLtrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([yTLtest,xTLtest])
if NN:
    np.savetxt('../4_learningData/TLtestNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/TLtest.csv', M, delimiter=',', fmt='%d')



##########################
##     Near Miss        ##
##########################

# create the near miss undersampler object, use version 2. [Note: version 3 fails, add explanation]
sampler3 = NM(version=2)

# near miss undersampling
xNM, yNM = sampler3.fit_resample(x,y)

if DEBUG:
    print(xNM.shape)
    print(yNM.shape)
    print(sorted(Counter(yNM).items()))

# split into training and test data 75/25 split
xNMtrain, xNMtest, yNMtrain, yNMtest = tts(xNM, yNM, test_size=TEST_SIZE, random_state=6375)

# save the training and test data into their respective files.
M = np.column_stack([yNMtrain,xNMtrain])
if NN:
    np.savetxt('../4_learningData/NMtrainNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/NMtrain.csv', M, delimiter=',', fmt='%d')
M = np.column_stack([yNMtest,xNMtest])
if NN:
    np.savetxt('../4_learningData/NMtestNN.csv', M, delimiter=',', fmt='%d')
else: 
    np.savetxt('../4_learningData/NMtest.csv', M, delimiter=',', fmt='%d')