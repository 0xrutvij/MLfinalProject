from typing import List
import random
from progress.bar import Bar
import math
import pandas as pd
import sys
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

class trainingExample:
    """docstring for trainingExample. A class to hold
    a single training Example/test example. i.e. one row
    from the data file """

    def __init__(self, label, featureVector):
        #constructor for the trainingExample class
        #class label of the example
        self.label = label
        #a List containing all the feature values
        self.featureVector = featureVector

class treeNode:
    """docstring for treeNode. A class representing a single node of the tree"""

    def __init__(self):
        #constructor for the treeNode class
        #Value of the node, in our context the value to which
        #we must match ancestor node's deciding feature.
        self.val = None
        #If this the node at which we stop looking, what label will we assign to
        #the example in question
        self.label = None
        #A boolean representing whether or not the node is a leaf node
        self.isLeaf = False
        #If the node isn't a leaf, the feature on which we will split the children
        self.decidingFeature = None
        #List of children node for this node
        self.children = []


def filterOn(attNo, attVal, td):
    """An helper function to filter the training data by the current nodes
    decidingFeature, the data passed to the children is the data whose value matches
    the value of the child node's, where the decidingFeature is defined by the parent"""
    retList = []
    for i in td:
        if(i.featureVector[attNo] == attVal):
            retList.append(i)
        else:
            continue
    return retList

class decisionTree:
    """docstring for decisionTree."""

    def __init__(self, t: List[trainingExample], maxDepth: int):
        #constructor for the decisionTree class
        #params are a list of training data and the max depth of the tree

        #the root node
        self.root = None

        #a var to hold the list of trainingExamples
        self.trainingData = t #a list of training examples
        self.maxDepth = maxDepth

        #since all examples are of the same length, we check the first
        #example in the list to find how many features does our examples have
        self.numFeatures = len(t[0].featureVector)

    def train(self):
        #class function to train our classifier and build the decisionTree

        #the set of features, from 0 through n, stored as a list
        featureSet = list(range(0, self.numFeatures))

        #the set of labels, stored as a label. Most often it is 0 and 1
        #but this allows easy extension to multi-class cases
        labelSet = list(set([x.label for x in self.trainingData]))

        #a copy of the trainingData
        data = self.trainingData

        #a call on the recursive method to start the actual training process
        #starting at the root node.
        self.root = decisionTree.train_rec(data, labelSet, featureSet, self.maxDepth)

    @classmethod
    def train_rec(self, data, labelSet_l, featureSet, depth, nodeVal=None):
        #a recursive class method to help train the decision tree.

        #create a new tree node
        currNode = treeNode()

        #the value of the node is supplied by the parent
        #if the node is a root node, the default value of None is used.
        currNode.val = nodeVal

        #create a list of 0's the size of the label set
        c_list = [0] * len(labelSet_l)

        #we zip the list with the list representing the set of labels to form a dict
        labelSet = dict(zip(labelSet_l, c_list))
        #the dict will look something like {label1:0, label2:0}

        #for all the keys in the label set
        for x in labelSet.keys():
            #if all the examples in our data have the same label
            #we're done, this node's label will be that label
            #and this is also a leaf node.
            if(all(i.label == x for i in data)):
                currNode.label = x
                currNode.isLeaf = True
                return currNode
            #otherwise, we will store the number of labels of each kind in the dict
            else:
                labelSet[x] = [i.label for i in data].count(x)

        #if we have reached this point without returning, our dict might look
        #something like {label1: 10, label2: 5}

        #the current node's label is the majority label's val,
        # in this case label1
        currNode.label = max(labelSet, key=labelSet.get)

        #if there were no features left to classify on in the feature set
        #or if we have touched the maximum depth
        #make this node a leaf node and return.
        if(len(featureSet) == 0 or depth == 0):
            currNode.isLeaf = True
            return currNode

        #Find the best attribute/feature from the set of attributes/features.
        currFeat = self.selectFeature(featureSet, data)
        currNode.decidingFeature = currFeat

        #create a set of all possible value for that attribute/feature

        currFeatVals = list(set([i.featureVector[currFeat] for i in data]))

        #for each value in the decidingFeature's set, we create a child node
        #the child node is a sub-tree of our decisionTree and in itself
        #a decision tree. Thus a recursive call is made to train it.

        for z in currFeatVals:
            #send only those data points which have the value z for the decidingFeature
            td = filterOn(currFeat, z, data)

            #create a shallow copy of our set of features
            nfs = featureSet.copy()

            #remove the decidingFeature from the set of features left to classify on
            #only for this sub-tree
            nfs.remove(currFeat)

            #make a recursive call to the training function
            #params are : td = the filtered training data, labelSet_l = the set of
            #labels/classes, nfs = a modified copy of the feature set,
            #depth = depth reduced by 1 since a node has been added.
            #nodeVal for the child is the decidingFeature's filtered on val
            x = decisionTree.train_rec(td, labelSet_l, nfs, depth-1, nodeVal=z)

            #once the recursive call to train returns, we append the child + its
            #sub-tree to the list of children for this node.
            currNode.children.append(x)

        #a bit of pruning
        #if this node ends up with only 1 child and that child has leq 1 child,
        #we take the child's label value and set it to be this node's label value.
        #and turn this node is turned into a leaf node.
        if(len(currNode.children) == 1):
            if(len(currNode.children[0].children) <= 1):
                currNode.label = currNode.children[0].label
                currNode.isLeaf = True

        return currNode

    def __str__(self, level=0, node=None):
        #A function to turn our decisionTree into a string for printing

        #since we don't need to print trees of depth greater than 2, I have
        #made this filter to prevent a print in those cases. Can be removed
        #to print trees of arbitary depth when we need to debug
        if(self.maxDepth > 2):
            return ""

        #if we're just beginning traversal, we set the currNode to root
        if(level == 0):
            node = self.root

        #if we're at a leaf node, only add the node's value and its label val
        if(node.isLeaf):
            ret = "\t"*level+'Node value = '+str(node.val)+"\n"
            ret += "\t"*level+'Label value = '+str(node.label)+"\n\n"

        #else add the node's value, its decidingFeature and the set of values possible
        #for that feature.
        else:
            ret = "\t"*level+'Node value = '+str(node.val)+"\n"
            ret+= "\t"*level+' Feature f'+str(node.decidingFeature+1)
            ret+= str(set([i.featureVector[node.decidingFeature] for i in self.trainingData]))+"\n\n"
        #for each child node recursively call the str function.
            for child in node.children:
                ret+= self.__str__(level+1, child)

        return ret

    def testSingle(self, t: trainingExample, node=None):
        #a function to test a single test example

        #check if a val has been provided for the node param
        if(node):
            #if yes, set the current node to be that node
            currNode = node
        else:
            #else the current node is the root node
            currNode = self.root

        #while the current node isn't None
        while(currNode):

            #attribute/feature to filter on is the current node's decidingFeature
            attNo = currNode.decidingFeature

            #a copy of the node we begin at.
            startNode = currNode

            if(currNode.isLeaf):
                #if the current node is a leaf node, return the following info
                return (currNode.label, t.label, currNode.label == t.label)

            for child in currNode.children:
                #else we check each child to see if its val matches the
                #decidingFeature's val for the given training example
                if(t.featureVector[attNo] == child.val):
                    currNode = child
                    #if such a child is found we break and go back to the while loop
                    #and begin our check again.
                    break

            #if no such child is found, then our startNode is still the current node
            if(startNode == currNode):
                #in such a case, we break the while loop.
                break

        #if we exit the while loop, we return the following info
        return (currNode.label, t.label, currNode.label == t.label)


    def testBatch(self, data):
        #a function to test a batch of testExamples
        retList = []

        #create a progress bar, it tracks the progress of our batch processing
        bar = Bar('Processing', max=len(data))

        #for each example in our test examples
        for x in data:
            #call the function to test a single example
            root = self.root
            retList.append(self.testSingle(t=x, node=root))
            #progress the bar after each example is processed.
            bar.next()

        #finish the the bar.
        bar.finish()

        #return the results for all examples as a list
        return retList


    @classmethod
    # Use entropy/information gain to select the feature to split on
    def selectFeature(self, featureSet, data):
        # Create a list to hold the entropy values
        entropyVals = []

        # Number of data points in the current node
        dataSize = len(data)

        # for each feature
        for index in range(0,len(featureSet)):

            currFeat = featureSet[index]

            #create a set of all possible values for that attribute/feature
            currFeatVals = list(set([j.featureVector[index] for j in data]))

            # Count for how many times each value appears for that feature appears
            valueCounts = []

            # for each possible value for that attribute/feature
            conditEntro = []
            for x in currFeatVals:

                # create a subset of the data points which have the value x for the Feature
                td = filterOn(currFeat, x, data)
                valueCounts.append(len(td)) # how many data points have that value

                # determine the new labelSet
                newlabelSet = list(set([y.label for y in td]))

                # Count how many times each label appears for that feature value
                labelCounts = []
                for label in newlabelSet:
                    count = 0
                    # for each data point 
                    for i in td:
                        if(i.label == label):
                            count+=1
                    labelCounts.append(count)

                # Calculate the conditional entropies
                sum = 0
                for labelCount in labelCounts: # for each label
                    prob = labelCount/len(td)
                    sum = sum - prob*math.log(prob,2)

                # conditional entropies for one value of the feature
                conditEntro.append(sum)

            #print((len(conditEntro) == len(valueCounts)))

            # calculate the feature's entropy by taking weighted average
            entropyForFeat = 0
            for k in range(0,len(currFeatVals)):
                entropyForFeat = entropyForFeat + (valueCounts[k]/dataSize)*conditEntro[k]

            # Save the feature's entropy to the list of entropy values
            entropyVals.append(entropyForFeat)

        # Now we have the entropy value for each feature,
        # The feature with the lowest entropy has the highest information gain
            selectedFeatIndex = entropyVals.index(min(entropyVals))
            selectedFeat = featureSet[selectedFeatIndex]

            return selectedFeat
        
# function to create the training data
def createTrainingData(trainingData, df, SPECTmode=False):


    return trainingData

# function to create the test data
def createTestData(testData, df, SPECTmode=False):

    return testData

# function to compute the training and test results. If makeConfusion = 1, then
# a confusion matrix will be produced
def findResults(testResult):

    numMiss = 0
    numHit = 0
    FN = TN = TP = FP = 0

    for x in testResult:

        if(x[2]):
            numHit+=1
            if(x[0]):
                TP+=1
            else:
                TN+=1
        else:
            numMiss+=1
            if(x[0]):
                FP+=1
            else:
                FN+=1

    error = (numMiss*100)/(numMiss+numHit)

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

# function to create the decision tree and find the training and test errors
# makeReport = 1 indicates that the tree needs to be printed and
# a confusion matrix should be produced
def trainAndTest(trainingData, x, testData, makeReport):

    # create the decision tree
    someTree = dt.decisionTree(trainingData, x)

    # train the tree
    someTree.train()

    if makeReport == 1:
        print('Decision Tree of Depth', x)
        print(str(someTree))

    # find training error and make confusion matrix if needed
    testResult = someTree.testBatch(trainingData)
    if makeReport ==1:
        print('The Confusion Maxtrix on the Training Set for Depth of ', x, ':')
    trainingError = findResults(testResult, makeReport)

    # find test error and make confusion matrix if needed
    testResult2 = someTree.testBatch(testData)
    if makeReport ==1:
        print('The Confusion Maxtrix on the Test Set for Depth of ', x, ':')
    testError = findResults(testResult2, makeReport)

    List = [trainingError, testError]

    return List

# plotting training and testing error curves together
# with tree depth on the x-axis and error on the y-axis
def plotErrors(trainingErrorList, testErrorList, monkNum, SPECTmode=False):
    plt.plot(np.arange(1,11,1),trainingErrorList, label = 'Training Error')
    plt.plot(np.arange(1,11,1),testErrorList, label = 'Test Error')
    plt.xlabel('Depth')
    plt.ylabel('Errors')
    if SPECTmode:
        title = 'Training and Testing Error Curves for SPECT data'
        save_file = 'SPECT.png'
    else:
        title = 'Training and Testing Error Curves for Monk' + monkNum
        save_file = 'Monk' + monkNum + '.png'
    plt.title(title)
    plt.legend()
    plt.savefig('plots/'+save_file, bbox_inches='tight')
    plt.show()

# function to display the table of training and test errors per depth
def displayTable(trainingErrorList,testErrorList):
    trainingErrorList.insert(0, 'Training Error')
    testErrorList.insert(0, 'Test Error')
    print(tabulate([['Depth',1,2], trainingErrorList, testErrorList]))