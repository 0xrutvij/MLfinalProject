from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import utils



if __name__ == '__main__':
    
    #keys = ['NM', 'ROS', 'RUS', 'TL', 'SMOTE', 'DS']
    keys = ['NM','ROS','RUS','TL','DS']
    for fileKey in keys:
        print(fileKey)
        # Load the training data
        M = np.genfromtxt('../learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrain = M[:, 0]
        ytrain = np.ascontiguousarray(ytrain)
        xtrain = M[:, 1:]
        xtrain = np.ascontiguousarray(xtrain)

        # Load the test data
        M = np.genfromtxt('../learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        ytest = np.ascontiguousarray(ytest)
        xtest = M[:, 1:]
        xtest = np.ascontiguousarray(xtest)

        model0 = XGBClassifier(n_estimators=10, use_label_encoder=False, eval_metric='error', verbosity = 0).fit(xtrain, ytrain)
        pred0 = model0.predict_proba(xtest)
        preds = pred0[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(ytest, preds)
        roc_auc = metrics.auc(fpr, tpr)
        #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        #display.plot()
        #plt.show()
        pred0 = model0.predict(xtest)
        tst_err = utils.compute_error(ytest, pred0)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        print('-*-'*10)
        model1 = XGBClassifier(n_estimators=10, scale_pos_weight=10000, use_label_encoder=False, eval_metric='error', verbosity = 0).fit(xtrain, ytrain)
        pred1 = model1.predict_proba(xtest)
        preds = pred1[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(ytest, preds)
        roc_auc = metrics.auc(fpr, tpr)
        #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
        #display.plot()
        #plt.show()
        pred1 = model1.predict(xtest)
        tst_err = utils.compute_error(ytest, pred1)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        print('-+-'*10)
        print('-+-'*10)