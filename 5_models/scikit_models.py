from sklearn.neural_network import MLPClassifier
import sys
import numpy as np

# Find the number of True positives, false positive, etc
def evaluation(result,ytest):
    numMiss = 0
    numHit = 0
    FN = TN = TP = FP = 0
    for i in range(0,result.shape[0]):
        if(ytest[i] == result[i]):
            numHit+=1
            if(ytest[i]==1):
                TP+=1
            else:
                TN+=1
        else:
            numMiss+=1
            if(ytest[i]==1):
                FN+=1
            else:
                FP+=1

    error = (numMiss)/(numMiss+numHit)
    # make a confusion matrix
    '''
        PV ->
    AV    _P_ _N_
    | |P[TP][FN]      2|1
    V |N[FP][TN]      3|4
    '''

    print(str(TP).rjust(3, '0'), '|', str(FN).rjust(3, '0'))
    print(str(FP).rjust(3, '0'), '|', str(TN).rjust(3, '0'))

    # Precision
    if TP+FP == 0:
        p = 0
    else:
        p = TP/(TP+FP)
    print('Precision:', p)

    # Recall
    if TP+FN == 0:
        r = 0
    else:
        r = TP/(TP+FN)

    print('Recall:', r)

    # F1 Score
    if r+p == 0:
        f1 = 0
    else:
        f1 = (2*r*p)/(r+p)
        
    print('F1 Score:', f1)


# the main
if __name__ == '__main__':

    keys = ['NM', 'ROS', 'RUS', 'TL', 'SMOTE', 'DS']

    for fileKey in keys:
        
        sys.stdout = open('Scikit_output'+ fileKey +'.txt', 'w')

        print(fileKey)
        print("")

        # Load the training data
        M = np.genfromtxt('C:/Users/Hallie/Source/Repos/MLfinalProject/learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrain = M[:, 0]
        xtrain = M[:, 1:]

        # Load the test data
        M = np.genfromtxt('C:/Users/Hallie/Source/Repos/MLfinalProject/learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        xtest = M[:, 1:]
    
        # Create Neural Networks
        max_epoch = 300
        step_sizes = [0.001, 0.01, 0.1, 1]
         
        for step in step_sizes:
        
          print("For a step size of ",step)
          model = MLPClassifier(random_state=1, max_iter=max_epoch,learning_rate=step,activation=‘logistic’).fit(xtrain, ytrain)
          prediction = model.predict(xtest)
          
          evaluation(prediction,ytest)
          
