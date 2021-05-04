    # decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 2 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu)

from binTree import *
from ensembles import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


def construct_eval_model(xtrn, ytrn, xtest, ytest, max_depth, option = 3, attribute_value_pairs = None, bag_size=1):
    """
    creates the requested model, trains and tests the model, and then displays the results.
    """
    print('-'*30)
    # use our bagging or boosting function
    if option == 0 or option == 1:
        # create ensemble model
        if option == 0:
            start = time.process_time()
            model = bagging(xtrn, ytrn, max_depth, attribute_value_pairs, bag_size)
            end = time.process_time() - start
        else:
            start = time.process_time()
            model = boosting(xtrn, ytrn, max_depth, bag_size, attribute_value_pairs)
            end = time.process_time() - start

        # Compute the test error and display the confusion matrix
        y_pred = [predict_example(x, model, probMode=True) for x in xtest]
    
        modelName = 'Bagging' if option==0 else 'AdaBoost'
        probMode = True
        if probMode:
            fpr, tpr, thresholds = metrics.roc_curve(list(ytest), y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=modelName)
            display.plot()
            plt.show()
        numberOf = ': Number of bags =' if option==0 else ": Number of learners ="
        print(modelName, numberOf, bag_size, ", Max Depth =", max_depth)
        tst_err = compute_error(list(ytest), y_pred, probMode=True)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        #print('CPU Runtime: {0}'.format(end))

    # use scikit learners
    if option == 2 or option == 3:
        # bagging classifier
        if option == 2:
            start = time.process_time()
            model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=bag_size, random_state=0).fit(xtrn, ytrn)
            end = time.process_time() - start
        # boosting classifier
        else:
            start = time.process_time()
            model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth),n_estimators=bag_size, random_state=0).fit(xtrn, ytrn)
            end = time.process_time() - start

        # Compute the test error
        y_pred = model.predict(xtest)
        modelName = 'Scikit-Learn Bagging' if option==2 else 'SciKit-Learn AdaBoostClassifier'
        numberOf = ': Number of bags =' if option==2 else ": Number of learners ="
        print(modelName, numberOf, bag_size, ", Max Depth =", max_depth)
        tst_err = compute_error(list(ytest), y_pred)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

        #print('CPU Runtime: {0}'.format(end))

    if option == 5:
        tree = id3(np.transpose(xtrn), ytrn, attribute_value_pairs=attribute_value_pairs, max_depth=bag_size)
        model = [[1, tree]]
        y_pred = [predict_example(x, model) for x in xtest]
        modelName = 'Decision Tree Classifier, '
        numberOf = 'max depth of the tree:'
        print(modelName, numberOf, bag_size)
        tst_err = compute_error(list(ytest), y_pred)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    print('-'*30)




if __name__ == '__main__':

    keys = ['NM', 'ROS', 'RUS', 'TL', 'SMOTE', 'DS']

    for fileKey in keys:
        sys.stdout = open('../output/output'+ fileKey +'.txt', 'w')

        # Load the training data
        M = np.genfromtxt('../learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt('../learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        xtest = M[:, 1:]

        # Restructure data
        xtrn = np.transpose(xtrn)

        featureRangeList = []
        for feature in xtrn:
            dictValues = Counter(feature)
            uniqueValues = list(dictValues.keys())
            featureRangeList.append(uniqueValues)

        # create the attribute-value pairs
        attribute_value_pairs = []
        for i, featVals in enumerate(featureRangeList):
            for val in featVals:
                attribute_value_pairs.append((i, val))


        # Construct and test four bagging models for each combination of maximum depth d = 3, 5 and bag size = 10, 20
        construct_eval_model(xtrn, ytrn, xtest, ytest, 3, option = 0, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=5)
        # Construct and test four boosting models for each combination of maximum depth d = 1, 2 and bag size = 20, 40
        construct_eval_model(xtrn, ytrn, xtest, ytest, 1, option = 1, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=10)
        construct_eval_model(xtrn, ytrn, xtest, ytest, 3, option = 5, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=10)

        # Use scikit-learn’s bagging and AdaBoost learners and repeat the experiments above
        # Bagging
        #construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 3, option = 2, bag_size=10)
        #construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 1, option = 3, bag_size=10)
