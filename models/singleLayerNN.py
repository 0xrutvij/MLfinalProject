import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import sys

# Create the neural network class
class neuralNetwork:

  # Constructor method with instance variables
  def __init__(self, input, output,numOfHidden,actChoice,eta):
    self.numOfFeatures = input.shape[1]+1
    self.numOfSamples = input.shape[0]
    self.input = np.c_[input, np.ones(self.numOfSamples)]
    self.output = output.reshape((self.numOfSamples,1))
    self.numOfHidden = numOfHidden
    self.hiddenlayer = pd.DataFrame(np.random.rand(self.numOfFeatures,self.numOfHidden))
    self.outputlayer = pd.DataFrame(np.random.rand(self.numOfHidden,1))
    self.actChoice = actChoice
    self.beta = .9
    self.ep = .01
    self.step =eta
  
  # Function to use proper activation function
  def actFunct(self,x):
    # define the activation functions
    if self.actChoice == 1:
      sigma = 1/(math.exp(-x)+1)
      return sigma
    if self.actChoice == 2:
      tanh = math.tanh(x)
      return tanh
    if self.actChoice == 3:
      relu = np.maximum(0,x)
      return relu
  
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
    
  # Assign the class
  def classify(self,x):
      if(x<=0):
          return 0 
      else:
          return 1

  # Function for the forward pass of the backpropagation
  def forwardPass(self,hiddenlayer,outputlayer):
    product1 = self.input.dot((hiddenlayer))
    vectorized_actFunct = np.vectorize(self.actFunct) # make activation funct
    hiddenlayer_pass = vectorized_actFunct(product1) # apply activation funct
    vectorized_classify = np.vectorize(self.classify)
    o = vectorized_classify(hiddenlayer_pass.dot(outputlayer))
    return o

  def testMyNetwork(self,xtest,ytest):
    # apply the network to the test data
    testInput = np.c_[xtest, np.ones(xtest.shape[0])]
    vectorized_actFunct = np.vectorize(self.actFunct)
    firstpass = vectorized_actFunct(testInput.dot(self.hiddenlayer))
    vectorized_classify = np.vectorize(self.classify)
    result = secondpass = vectorized_classify(firstpass.dot(self.outputlayer))
  
    # find the MSE
    MSE = mean_squared_error(ytest,result)
    print("The MSE is: ")
    print(MSE)

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
    epoch = -1
    training_error = 1
    notDone = True
    
    while(notDone):

      # Complete the forword pass
      o = self.forwardPass(hidden_layer_weights,output_layer_weights)
 
      # determine if another backpass is needed
      # Find trainig error
      diff = (target-o)
      training_error = (sum(diff**2))/(self.numOfSamples)
      epoch+=1

      # evaluate the stopping conditions
      if ((training_error<.1) or (epoch-1>max_epoch)):
        notDone = False
        self.hiddenlayer = hidden_layer_weights
        self.outputlayer = output_layer_weights
        print("Number of Epochs:", epoch)
        print("Final Training Error")
        print(training_error)


      else:
        # Complete the backwards pass

        # find deltas
        vectorized_diff_actFunct = np.vectorize(self.diff_actFunct)
        delta_o = diff*vectorized_diff_actFunct(o) # (t-o)*a'(o)

        # get x values from hidden layer a(xTw)
        vectorized_actFunct = np.vectorize(self.actFunct)
        xhidden = vectorized_actFunct(self.input.dot(hidden_layer_weights))

        # a'(xhidden)*Weight_hidden*delta_o
        k = delta_o.dot(output_layer_weights.T)
        deltahidden = vectorized_diff_actFunct(xhidden)*k

        # Find dE/dwi
        partial1 = (self.input.T.dot(deltahidden))/self.numOfSamples
        partial2 = (xhidden.T.dot(delta_o))/self.numOfSamples

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

# the main
if __name__ == '__main__':

    max_epoch = 100
    eta = 0.01

    keys = ['NM', 'ROS', 'RUS', 'TL', 'SMOTE', 'DS']

    for fileKey in keys:
        
        sys.stdout = open('NNoutput'+ fileKey +'.txt', 'w')

        print(fileKey)

        # Load the training data
        M = np.genfromtxt('C:/Users/Hallie/Source/Repos/MLfinalProject/learningData/' +fileKey+ 'train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrain = M[:, 0]
        xtrain = M[:, 1:]

        # Load the test data
        M = np.genfromtxt('C:/Users/Hallie/Source/Repos/MLfinalProject/learningData/' +fileKey+ 'test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytest = M[:, 0]
        xtest = M[:, 1:]

        # Construct Network using various parameters
        do_backprop(eta,max_epoch,xtrain,ytrain,xtest,ytest)
        print("")