from sklearn.neural_network import MLPClassifier
from sklearn import tree
import sys
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt



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
        
        sys.stdout = open('../6_output/Scikit/Scikit_output'+ fileKey +'.txt', 'w')

        print(fileKey)
        print("")

        # Load the training data
        M = np.genfromtxt('../4_learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrain = M[:, 0]
        xtrain = M[:, 1:]

        # Load the test data
        M = np.genfromtxt('../4_learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        xtest = M[:, 1:]

        # Create Binary Decision Tree
        max_depths = [1, 3, 5]

        print("Scikit's Binary Decision Trees:")
        treeDict = {}

        for depth in max_depths:
            print("For Max Depth of ", depth)
            print("---------------------------------------")

            print("Using Gini Impurity")
            tree1 = tree.DecisionTreeClassifier(criterion="gini", max_depth = depth).fit(xtrain,ytrain)
            result1 = tree1.predict(xtest)
            evaluation(result1,ytest)
            pred = tree1.predict_proba(xtest)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(ytest, pred)
            roc_auc = metrics.auc(fpr, tpr)
            treeDict[str(depth)+'Gini'] = {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc}
            print("")

            print("Using Entropy")
            tree2 = tree.DecisionTreeClassifier(criterion="entropy", max_depth = depth).fit(xtrain,ytrain)
            result2 = tree2.predict(xtest)
            evaluation(result2,ytest)
            pred = tree1.predict_proba(xtest)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(ytest, pred)
            roc_auc = metrics.auc(fpr, tpr)
            treeDict[str(depth)+'Entropy'] = {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc}
            print("")

        ax = plt.gca()
        for key in treeDict:
            info = treeDict[key]
            fpr = info['fpr']
            tpr = info['tpr']
            roc_auc = info['auc']
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=key + fileKey)
            display.plot(ax=ax)

        plt.savefig('../6_output/Scikit/rocCurves/'+ fileKey + '_SK_DTs')
        plt.close()
    
        # Create Neural Networks
        max_epoch = 500
        step_sizes = [0.001, 0.01, 0.1, 1]
        print("************************************")
        print("Scikit's Neural Networks:")
        NNDict = {}
         
        for step in step_sizes:
        
            print("For a step size of ",step)
            print("---------------------------------------")
            print("Neural Network Using Sigmoid Activation:")
            model1 = MLPClassifier(random_state=1, max_iter=max_epoch,learning_rate_init=step,activation='logistic').fit(xtrain, ytrain)
            prediction1 = model1.predict(xtest)

            pred = model1.predict_proba(xtest)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(ytest, pred)
            roc_auc = metrics.auc(fpr, tpr)
            NNDict[str(step)+'_Sigmoid'] = {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc}


            evaluation(prediction1,ytest)

            print("Neural Network Using tanh Activation:")
            model2 = MLPClassifier(random_state=1, max_iter=max_epoch,learning_rate_init=step,activation='tanh').fit(xtrain, ytrain)

            pred = model2.predict_proba(xtest)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(ytest, pred)
            roc_auc = metrics.auc(fpr, tpr)
            NNDict[str(step)+'_tanh'] = {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc}

            prediction2 = model2.predict(xtest)
            evaluation(prediction2,ytest)

            print("Neural Network Using RELU Activation:")
            model3 = MLPClassifier(random_state=1, max_iter=max_epoch,learning_rate_init=step,activation='relu').fit(xtrain, ytrain)

            pred = model3.predict_proba(xtest)[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(ytest, pred)
            roc_auc = metrics.auc(fpr, tpr)
            NNDict[str(step)+'_relu'] = {'fpr':fpr, 'tpr':tpr, 'auc':roc_auc}

            prediction3 = model3.predict(xtest)
            evaluation(prediction3,ytest)
            print("")
        
        ax = plt.gca()
        for key in NNDict:
            info = NNDict[key]
            fpr = info['fpr']
            tpr = info['tpr']
            roc_auc = info['auc']
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=key + fileKey)
            display.plot(ax=ax)

        plt.savefig('../6_output/Scikit/rocCurves/'+ fileKey + '_SK_NNs')
        plt.close()
            
