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

import numpy as np
import random
import os, sys
import time
from math import log2, exp, log
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

uniques =[]

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    valIndexDict = {}
    for i, v in enumerate(x):
        if v not in valIndexDict:
            valIndexDict[v] = [i]
        else:
            valIndexDict[v].append(i)

    return valIndexDict


def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    # in weighted mode the count totals will act as weight totals
    dictValCounts = {}
    wMode = False
    agg = 0
    if w is not None:
        wMode = True

    # counts of the unique values (weighted mode available)
    for i,val in enumerate(y):
        if val not in dictValCounts:
            dictValCounts[val] = 0
        if wMode:
            dictValCounts[val]+=w[i]
            agg += w[i]
        else:
            dictValCounts[val]+=1

    if wMode:
        total = agg
    else:
        total = len(y)

    Hz = 0
    for k,v in dictValCounts.items():
        Hz += (v/total) * (log2(v/total))
    return -Hz


def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    yGivenXeq = {}
    wYGivenXeq = {}
    wOfXi = {}
    wMode = False

    if w is not None:
        wMode = True
        HofY = entropy(y, w)
    else:
        HofY = entropy(y)

    # to find yGivenX for all x

    for i, val in enumerate(x):
        if val not in yGivenXeq:
            yGivenXeq[val] = [y[i]]
            if wMode:
                wYGivenXeq[val] = [w[i]]
                wOfXi[val] = w[i]
        else:
            yGivenXeq[val].append(y[i])
            if wMode:
                wYGivenXeq[val].append(w[i])
                wOfXi[val] += w[i]

    weightTotal = 0
    if wMode:
        for weightSum in wOfXi.values():
            weightTotal += weightSum

    HofYgivenX = 0

    #weighted sum of the entropies H(y|x) for all vals of x
    for xi in yGivenXeq:
        if wMode:
            HofYgivenX += entropy(yGivenXeq[xi], wYGivenXeq[xi])*(wOfXi[xi]/weightTotal)
        else:
            HofYgivenX += entropy(yGivenXeq[xi])*(len(yGivenXeq[xi])/len(x))

    return HofY - HofYgivenX


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """
    wMode = False
    if w is not None:
        wMode = True

    yVals = {}

    for yVal in y:
        if yVal not in yVals:
            yVals[yVal] = 0
        yVals[yVal]+=1

    if len(yVals) == 1:
        return list(yVals.keys())[0]
    elif len(attribute_value_pairs) == 0 or depth == max_depth:
        return max(yVals, key = lambda k: yVals[k])

    maxGain = 0
    splitPair = None

    for featureNumber, featureValue in attribute_value_pairs:
        xFeats = filterOn(x, featureNumber, featureValue)
        if xFeats is not None:
            if wMode:
                gain = mutual_information(xFeats, y, w)
            else:
                gain = mutual_information(xFeats, y)
        else:
            gain = 0

        if gain > maxGain:
            maxGain = gain
            splitPair = (featureNumber, featureValue)

    #if max Gain is zero, return label split majority
    if splitPair is None:
        return max(yVals, key = lambda k: yVals[k])

    xSplit = partition(x[splitPair[0]])

    xIsValIndices = dict((i, 0) for i in xSplit[splitPair[1]])
    xIsVal = []
    wForXisVal = None
    yForXisVal = []
    xIsNotVal = []
    yForXisNotVal = []
    wForXisNotVal = None

    if wMode:
        wForXisVal = []
        wForXisNotVal = []

    xTranspose = np.transpose(x)
    for i in range(0, len(x[splitPair[0]])):
        if i in xIsValIndices:
            xIsVal.append(xTranspose[i])
            yForXisVal.append(y[i])
            if wMode:
                wForXisVal.append(w[i])
        else:
            xIsNotVal.append(xTranspose[i])
            yForXisNotVal.append(y[i])
            if wMode:
                wForXisNotVal.append(w[i])

    attribute_value_pairs.remove(splitPair)

    retVal = {
    (splitPair[0], splitPair[1], False): id3(np.transpose(xIsNotVal), yForXisNotVal, attribute_value_pairs, depth+1, w=wForXisNotVal),
    (splitPair[0], splitPair[1], True): id3(np.transpose(xIsVal), yForXisVal, attribute_value_pairs, depth+1, w=wForXisVal)
    }

    return  retVal


def filterOn(x, xFeatNum, xFval):
    if xFeatNum >= len(x):
        return None
    l = [1 if xi==xFval else 0 for xi in x[xFeatNum]]
    return l

def bagging(x, y, max_depth, attribute_value_pairs, num_trees):

    h_ens = []
    xTranspose = np.transpose(x)
    seed=int.from_bytes(os.urandom(64), sys.byteorder)
    random.seed(seed) #re-seed generator

    for i in range(0,num_trees):

        bootstrapX = []
        bootstrapY = []
        indices = []

        for j in range(0,len(xTranspose)):
            k = random.randint(0,len(xTranspose)-1) # generate a random index
            #indices.append(k)
            # add the kth sample to the bootstrap sample set
            bootstrapX.append(xTranspose[k])
            bootstrapY.append(y[k])


        #setIndices = set(indices) # To track the proportion of unique examples in each bag
        #uniques.append(len(setIndices)/len(indices))

        # create a tree and add it to the ensemble hypothesis
        decision_tree = id3(np.transpose(bootstrapX), bootstrapY, attribute_value_pairs=attribute_value_pairs.copy(), max_depth=max_depth)
        h_ens.append([1,decision_tree]) # Note: weight of each classifier is the same

    return h_ens

def boosting(x, y, max_depth, num_stumps, attribute_value_pairs):

    h_ens = []

    #create a list to hold the weights of the training example
    #also called the distribution d
    #initialize weight for each training example to be 1/N
    n = len(y)
    d = [1/n]*n

    while len(h_ens) < num_stumps:
        h = id3(x,y,attribute_value_pairs,max_depth=max_depth, w=d)
        epsilon, preds = weighted_errorAndPreds(h, x, y, d)
        alpha = calc_alpha(epsilon)
        wSum = 0
        for i in range(len(d)):
            if preds[i]:
                factor = exp(-alpha)
            else:
                factor = exp(alpha)
            d[i] = d[i]*factor
            wSum += d[i]

        d[:] = [w/wSum for w in d]
        h_ens.append([alpha, h])

    return h_ens


def weighted_errorAndPreds(hypo, xV, yV, weights):

    xVc = xV.copy()
    xVc = np.transpose(xVc)
    preds = []
    error = 0

    for i, (x, w) in enumerate(zip(xVc, weights)):
        pred = (predict_one_tree(x, hypo) == yV[i])
        preds.append(pred)
        if not pred:
            error += weights[i]

    return (error, preds)

def calc_alpha(e):
    ratio = (1-e)/e
    lnRatio = log(ratio)
    return 0.5*lnRatio

def predict_one_tree(x, tree):
    """
    Predicts the classification label produced by one tree for a single example x by recursively
    descending the tree until a label/leaf node is reached.

    Returns the predicted label of x, as a 1 or 0, according to tree
    """
    if type(tree) == np.int32 or type(tree) == np.int64:
        return tree

    xIdx, xVal, cond = list(tree.keys())[0]

    if x[xIdx] == xVal:
        subtree = tree[(xIdx, xVal, True)]
    else:
        subtree = tree[(xIdx, xVal, False)]

    return predict_one_tree(x, subtree)


def predict_example(x, h_ens):
    """
    Predicts the final classification label for a single example x.

    h_ens is an ensemble of weighted hypotheses.
    The ensemble is represented as [[alpha_i, h_i]]
    """

    # The ith index is the weighted total for the classification label = i
    weightedTotals = [0,0]

    # for each hypothesis
    for hypo in h_ens:
        # find the prediction from each tree and adapt the appropriate weigthed total
        pred = predict_one_tree(x, hypo[1])
        weightedTotals[pred] = weightedTotals[pred]+ hypo[0]

    # The label with the highest total is the predicted classification of x
    if weightedTotals[1] > weightedTotals[0]:
        return 1
    else:
        return 0


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels and the predicted labels

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    numMiss = 0
    numHit = 0
    FN = TN = TP = FP = 0

    for i in range(0,len(y_pred)):

        if(y_true[i] == y_pred[i]):
            numHit+=1
            if(y_pred[i]):
                TP+=1
            else:
                TN+=1
        else:
            numMiss+=1
            if(y_pred[i]):
                FP+=1
            else:
                FN+=1

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

    return error

def construct_eval_model(xtrn, ytrn, xtest, ytest, max_depth, option = 3, attribute_value_pairs = None, bag_size=1):
    """
    creates the requested model, trains and tests the model, and then displays the results.
    """

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
        y_pred = [predict_example(x, model) for x in xtest]
        modelName = 'Bagging' if option==0 else 'AdaBoost'
        numberOf = ': Number of bags =' if option==0 else ": Number of learners ="
        print(modelName, numberOf, bag_size, ", Max Depth =", max_depth)
        tst_err = compute_error(list(ytest), y_pred)
        print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
        print('CPU Runtime: {0}'.format(end))

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
        print('CPU Runtime: {0}'.format(end))



if __name__ == '__main__':

    sys.stdout = open('output.txt', 'w')

    # Load the training data
    M = np.genfromtxt('./p2data/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./p2data/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
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
    construct_eval_model(xtrn, ytrn, xtest, ytest, 3, option = 0, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=10)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 5, option = 0, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=10)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 3, option = 0, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=20)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 5, option = 0, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=20)
    # Construct and test four boosting models for each combination of maximum depth d = 1, 2 and bag size = 20, 40
    construct_eval_model(xtrn, ytrn, xtest, ytest, 1, option = 1, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=20)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 2, option = 1, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=20)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 1, option = 1, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=40)
    construct_eval_model(xtrn, ytrn, xtest, ytest, 2, option = 1, attribute_value_pairs = attribute_value_pairs.copy(), bag_size=40)


    # Use scikit-learn’s bagging and AdaBoost learners and repeat the experiments above
    # Bagging
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 3, option = 2, bag_size=10)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 5, option = 2, bag_size=10)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 3, option = 2, bag_size=20)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 5, option = 2, bag_size=20)

    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 1, option = 3, bag_size=20)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 2, option = 3, bag_size=20)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 1, option = 3, bag_size=40)
    construct_eval_model(np.transpose(xtrn), ytrn, xtest, ytest, 2, option = 3, bag_size=40)
