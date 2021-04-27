import numpy as np
import random
import os, sys
import time
from math import log2, exp, log
from collections import Counter

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
    # Precision indicates how accurate a model is for positive preds.
    # Useful when the cost of a false positive is high. Higher precision means lower FP
    if TP+FP == 0:
        p = 0
    else:
        p = TP/(TP+FP)
    print('Precision:', p)
    # Recall indicates how many of the models positive predictions are actually correct
    # Useful when the cost of a false negative is high. Higher recall means lower FN.
    if TP+FN == 0:
        r = 0
    else:
        r = TP/(TP+FN)

    print('Recall:', r)
    # F1 Score balances both precision and recall. For models that have class imbalance
    # and a large number of TNs contribute to the accuracy, thus F1 focuses more on
    # TP vs FN/FP.
    if r+p == 0:
        f1 = 0
    else:
        f1 = (2*r*p)/(r+p)
        
    print('F1 Score:', f1)

    return error
