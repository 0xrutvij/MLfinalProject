import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import sys
import numpy_ml


# Create the neural network class
class neuralNetwork:

  # Constructor method with instance variables
  def __init__(self, input, output,numOfHidden,actChoice,eta):
    self.numOfFeatures = input.shape[1]+1
    self.numOfSamples = input.shape[0]
    self.input = np.c_[input, np.ones(self.numOfSamples)]
    self.output = output.reshape((self.numOfSamples,1))
    self.numOfHidden = numOfHidden
    self.hiddenlayer = pd.DataFrame((np.random.rand(self.numOfFeatures,self.numOfHidden))-0.5)/2000 # Subract 0.5 to get some negative weights
    self.outputlayer = pd.DataFrame((np.random.rand(self.numOfHidden,1))-0.5)/2000
    self.actChoice = actChoice
    self.beta = .9
    self.ep = .01
    self.step =eta
  
  # Function to use proper activation function
  def actFunct(self,x):
    # define the activation functions
    if self.actChoice == 1:
        
        sigma = 1/(math.exp(-x)+1)
        if sigma > 0.5: #classification threshhold
            return 1
        else:
            return 0

    if self.actChoice == 2:
      tanh = math.tanh(x)
      if tanh<=0: # -1<tanh<1
          return 0
      else:
          return 1

    if self.actChoice == 3:
      relu = np.maximum(0,x)
      if relu == 0:
          return 0
      else:
          return 1
    else:
        return x
  
  # Function to use proper derivative for activation function
  def diff_actFunct(self,x):
    # define the activation functions
    if self.actChoice == 1:
      diff_sigma = (1/(math.exp(-x)+1))*(1-(1/(math.exp(-x)+1)))
      return diff_sigma
    if self.actChoice == 2:
      diff_tanh = 1-(math.tanh(x)**2)
      return diff_tanh
    if self.actChoice == 3:
      if x>0:
        return 1
      if x<=0:
        return 0
    

  # Function for the forward pass of the backpropagation
  def forwardPass(self,hiddenlayer,outputlayer):
    product1 = self.input.dot((hiddenlayer))
    vectorized_actFunct = np.vectorize(self.actFunct) # make activation funct
    hiddenlayer_pass = vectorized_actFunct(product1) # apply activation funct
    o = vectorized_actFunct(hiddenlayer_pass.dot(outputlayer))
    return o

  def testMyNetwork(self,xtest,ytest):
    # apply the network to the test data
    testInput = np.c_[xtest, np.ones(xtest.shape[0])]
    vectorized_actFunct = np.vectorize(self.actFunct)
    firstpass = vectorized_actFunct(testInput.dot(self.hiddenlayer))
    result = secondpass = vectorized_actFunct(firstpass.dot(self.outputlayer))
  
    # find the MSE
    MSE = mean_squared_error(ytest,result)
    print("The MSE is: ", MSE)

    #loss = (numpy_ml.neural_nets.losses.CrossEntropy.loss(ytest,result))/self.numOfSamples
    #print("The Binary Cross-Entropy Loss is: ",loss)
    
    # Find the number of True positives, false positive, etc
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


  def momentumTerm(self,v):
    return self.step/((v+self.ep)**(.5))

  # Function to complete the Backpropagation
  def Backprop(self,max_epoch):
    # Initial parameters
    hidden_layer_weights = self.hiddenlayer
    output_layer_weights = self.outputlayer
    target = self.output
    v1 = 0
    v2 = 0
    epoch = 0
    training_error = 1
    notDone = True
    
    while(notDone):

      # Complete the forword pass
      o = self.forwardPass(hidden_layer_weights,output_layer_weights)
 
      # determine if another backpass is needed
      diff = (target-o)
      
      #find training error using Binary Cross-Entropy
      if self.actChoice == 4:
          training_error = (numpy_ml.neural_nets.losses.CrossEntropy.loss(target,o))/self.numOfSamples

      # Find trainig error using MSE
      else:
          training_error = (sum(diff**2))/(self.numOfSamples)
      
      epoch+=1

      # evaluate the stopping conditions
      if ((training_error<.1) or (epoch-1>max_epoch)):
        
        notDone = False #
        self.hiddenlayer = hidden_layer_weights
        self.outputlayer = output_layer_weights
        print("Number of Epochs:", epoch)
        
      # Complete the backwards pass
      else:
        #if self.actChoice == 4:
            # Compute the gradient of the cross entropy loss
            #cost = numpy_ml.neural_nets.losses.CrossEntropy.grad(o,target)
 

        # find deltas
        vectorized_diff_actFunct = np.vectorize(self.diff_actFunct)
        delta_o = diff*vectorized_diff_actFunct(o) # (t-o)*a'(o)

        # get x values from hidden layer a(xTw)
        vectorized_actFunct = np.vectorize(self.actFunct)
        xhidden = vectorized_actFunct(self.input.dot(hidden_layer_weights))

        # a'(xhidden)*Weight_hidden*delta_o
        k = delta_o.dot(output_layer_weights.T)
        deltahidden = vectorized_diff_actFunct(xhidden)*k # a'(o)*sum(delta_o*weight)

        # Find dE/dwi
        partial1 = (self.input.T.dot(deltahidden))/self.numOfSamples
        partial2 = (xhidden.T.dot(delta_o))/self.numOfSamples
        partial1 = partial1 if partial1 < 5 or partial1 > -5 else 5 if partial1 > 5 else -5
        partial2 = partial2 if partial2 < 5 or partial2 > -5 else 5 if partial2 > 5 else -5

        # exponential moving average
        v1 = (partial1**2)*(1-self.beta)+(v1*self.beta) # hidden layer
        v2 = (partial2**2)*(1-self.beta)+(v2*self.beta) # output layer

        # delta w
        vectorized_momentum  =np.vectorize(self.momentumTerm)
        dw1 = vectorized_momentum(v1)*partial1
        dw2 = vectorized_momentum(v2)*partial2

        # update weights
        output_layer_weights = output_layer_weights-dw2
        hidden_layer_weights = hidden_layer_weights-dw1


# function to initiate backprop
def do_backprop(eta,max_epoch,xtrain,ytrain,xtest,ytest):
  
  # Construct Network using sigmoid activation
  myNetwork = neuralNetwork(xtrain, ytrain,3,1,eta)
  print("For Sigmoid Activation")
  myNetwork.Backprop(max_epoch)
  myNetwork.testMyNetwork(xtest,ytest)
  print("")

  # Construct Network using tanh activation
  myNetwork = neuralNetwork(xtrain, ytrain,3,2,eta)
  print("For Tanh Activation")
  myNetwork.Backprop(max_epoch)
  myNetwork.testMyNetwork(xtest,ytest)
  print("")

  # Construct Network using ReLu activation
  myNetwork = neuralNetwork(xtrain, ytrain,3,3,eta)
  print("For ReLu Activation")
  myNetwork.Backprop(max_epoch)
  myNetwork.testMyNetwork(xtest,ytest)
  print("")

  # Construct Network Using Binary Cross-Entropy Loss Function
  #myNetwork = neuralNetwork(xtrain, ytrain,3,4,eta)
  #print("For Binary Cross-Entropy Loss Function")
  #myNetwork.Backprop(max_epoch)
  #myNetwork.testMyNetwork(xtest,ytest)
  #print("")


# the main
if __name__ == '__main__':

    max_epoch = 200
    step_sizes = [0.001, 0.01, 0.1, 1]


    keys = ['NM', 'ROS', 'RUS', 'TL', 'SMOTE', 'DS']

    for fileKey in keys:
        
        sys.stdout = open('./6_output/NNoutput'+ fileKey +'.txt', 'w')

        print(fileKey)
        print("")

        # Load the training data
        M = np.genfromtxt('./4_learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrain = M[:, 0]
        xtrain = M[:, 1:]

        # Load the test data
        M = np.genfromtxt('./4_learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        xtest = M[:, 1:]

        for step in step_sizes:

            print("For a step size of ",step)

            # Construct Network using various parameters
            do_backprop(step,max_epoch,xtrain,ytrain,xtest,ytest)
            print("---------------------------------------")