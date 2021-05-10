import numpy as np
import random
import os, sys
import time
from math import log2, exp, log
from collections import Counter

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
