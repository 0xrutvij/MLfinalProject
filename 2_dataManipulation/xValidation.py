import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.under_sampling import RandomUnderSampler as RUS


DEBUG = False

## Stratified Cross Validation

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

skf = StratifiedKFold(n_splits=8)

for i, z in enumerate(skf.split(x, y)):
    train_index, test_index = z
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    M = np.column_stack([y_train,x_train])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'DStrainFold.csv', M, delimiter=',', fmt='%d')
    M = np.column_stack([y_test,x_test])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'DStestFold.csv', M, delimiter=',', fmt='%d')

    # create random over sample class
    sampler1 = ROS(random_state=6375)

    # complete the re-sample
    x_testO, y_testO = sampler1.fit_resample(x_test, y_test)
    x_trainO, y_trainO = sampler1.fit_resample(x_train, y_train)

    M = np.column_stack([y_trainO,x_trainO])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'ROStrainFold.csv', M, delimiter=',', fmt='%d')
    M = np.column_stack([y_testO,x_testO])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'ROStestFold.csv', M, delimiter=',', fmt='%d')

    # create the random undersampler object
    sampler1 = RUS(random_state=6375, replacement=True)

    # random undersamplinggg
    xRUS, yRUS = sampler1.fit_resample(x_test, y_test)
    xRUSt, yRUSt = sampler1.fit_resample(x_train, y_train)

    M = np.column_stack([yRUSt,xRUSt])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'RUStrainFold.csv', M, delimiter=',', fmt='%d')
    M = np.column_stack([yRUS,xRUS])
    np.savetxt('../4_learningData/stratFolds/'+str(i)+'RUStestFold.csv', M, delimiter=',', fmt='%d')
