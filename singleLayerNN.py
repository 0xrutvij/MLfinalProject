import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

# Create the neural network class
class neuralNetwork:

  # Constructor method with instance variables
  def __init__(self, input, output,numOfHidden,actChoice):
    self.numOfInputs = input.shape[1]
    self.input = pd.DataFrame(np.ones(train_input.shape[0])).T.append(train_input.T,ignore_index=True) 
    self.output = output
    self.numOfSamples = input.shape[0]
    self.numOfHidden = numOfHidden
    self.hiddenlayer = (pd.DataFrame(np.random.rand(self.numOfHidden, self.numOfInputs+1))).div(10)
    self.outputlayer = (pd.DataFrame(np.random.rand(1, self.numOfHidden))).div(10)
    self.hiddenLayerBias = pd.DataFrame(np.ones(train_input.shape[0])).T
    self.actChoice = actChoice
  
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
      return np.maximum(0,x)
  
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
  def forwardPass(self,hiddenlayer,outputlayer,bias):
    
    product1 = hiddenlayer.dot((self.input))
    hiddenlayer_pass = product1.applymap(self.actFunct)
    o = outputlayer.dot(hiddenlayer_pass)
    
    return o

  def testMyNetwork(self,test_input,test_output):
    # apply the network to the test data
    testInput = pd.DataFrame(np.ones(test_input.shape[0])).T.append(test_input.reset_index(drop=True).T,ignore_index=True)
    firstpass = (self.hiddenlayer.dot(testInput)).applymap(self.actFunct)
    result = secondpass = (self.outputlayer.dot(firstpass)).T
  
    # find the MSE
    MSE = mean_squared_error(test_output,result)
    print("The MSE is: ")
    print(MSE)

  # Function to complete the Backpropagation
  def Backprop(self,step,max_epoch):
    # Initial parameters
    hidden_layer_weights = self.hiddenlayer
    output_layer_weights = self.outputlayer
    bias = self.hiddenLayerBias
    target = self.output
    v1 = 0
    v2 = 0
    beta = .9
    ep = .01
    epoch = -1
    training_error = 1
    notDone = True
    while(notDone):

      # Complete the forword pass
      o = self.forwardPass(hidden_layer_weights,output_layer_weights,bias)

      # determine if another backpass is needed
      # Find trainig error
      diff = (target.T-o.T)[0]
      training_error = (sum(diff**2))/(self.numOfSamples)
      
      epoch+=1

      # evaluate the stopping conditions
      if ((training_error<50) or (epoch-1>max_epoch)):
        notDone = False
        self.hiddenlayer = hidden_layer_weights
        self.outputlayer = output_layer_weights
        print("Number of Epochs:")
        print(epoch-1)
        print("Final Training Error")
        print(training_error)

      else:
        # Complete the backwards pass
        # for each training value
        for i in range(0, self.numOfInputs):

          #find deltas
          delta_o = -1*(target[0]-o[0])*self.diff_actFunct(float(o[0]))
    
          # get x values from hidden layer
          xhidden = pd.DataFrame(hidden_layer_weights.dot((self.input[0]))).applymap(self.actFunct)
          # h'(xhiddeni)*Weight_hiddeni*delta_o
          k = (xhidden.applymap(self.diff_actFunct)).div(1/delta_o)
          deltahidden = pd.DataFrame(k.values*(output_layer_weights.T.values),columns=k.columns,index=k.index)

          # get x values then apply derivative of activation
          diff_of_input = (pd.DataFrame(self.input[0].drop([0])).applymap(self.diff_actFunct)).T
          # Sum(delta_i*w_ij)
          summ = (deltahidden.T).dot(hidden_layer_weights.drop(columns=[0]))
          deltaInput=pd.DataFrame(diff_of_input.values*summ.values, columns=diff_of_input.columns, index=diff_of_input.index)

          # Find dE/dwi
          partial1 = pd.DataFrame(xhidden.values*deltahidden.values, columns=xhidden.columns, index=xhidden.index).T # 1x3
          partial2 = pd.DataFrame(deltaInput.values*self.input[0].drop([0]).T.values,columns = deltaInput.columns, index=deltaInput.index) #1x4

          # exponential moving average
          v1 = pd.DataFrame(((partial1.values)**2)*(1-beta)+v1*beta)
          v2 = pd.DataFrame(((partial2.values)**2)*(1-beta)+v2*beta)

          # delta w
          dw1 = pd.DataFrame(1/((v1.values+ep)**.5)*step*partial1.values,columns=v1.columns, index=v1.index)
          dw2 = pd.DataFrame(1/((v2.values+ep)**.5)*step*partial2.values,columns=v2.columns, index=v2.index)
          dw2mat = pd.concat([dw2]*3, ignore_index=True)
          deltabias = deltahidden.div(1/step)
          dw2mat = (deltabias.T.append(dw2mat.T,ignore_index=True)).T

          # update weights
          output_layer_weights = output_layer_weights-dw1
          hidden_layer_weights = hidden_layer_weights-dw2mat

# Function to import the data
def obtainData():

  url="https://www.dropbox.com/s/7sbixzdxo8a5m9g/DataForAssignment.csv?dl=1"
  allData=pd.read_csv(url,header=None)

  return allData

# function to initiate backprop
def do_backprop(eta,max_epoch):
  # Construct Network using sigmoid activation
  myNetwork = neuralNetwork(train_input, train_output,3,1)
  print("For Sigmoid Activation")
  myNetwork.Backprop(eta,max_epoch)
  myNetwork.testMyNetwork(test_input,test_output)
  print("")

  # Construct Network using tanh activation
  myNetwork = neuralNetwork(train_input, train_output,3,2)
  print("For Tanh Activation")
  myNetwork.Backprop(eta,max_epoch)
  myNetwork.testMyNetwork(test_input,test_output)
  print("")

  # Construct Network using ReLu activation
  myNetwork = neuralNetwork(train_input, train_output,3,3)
  print("For ReLu Activation")
  myNetwork.Backprop(eta,max_epoch)
  myNetwork.testMyNetwork(test_input,test_output)
  print("")

# the main
if __name__ == '__main__':

  # importing the dataset
  data = obtainData()

  # obtaining the training data, which will be 80% of the entire data
  trainingData = data.iloc[0:int(data.shape[0]*.8), :]
  train_input = trainingData.drop(columns=[trainingData.shape[1]-1])
  train_output = trainingData[trainingData.shape[1]-1]

  # obtaining the test data, which will be 20% of the entire data
  testData = data.iloc[trainingData.shape[0]:data.shape[0]-1, :]
  test_input = testData.drop(columns=[testData.shape[1]-1])
  test_output = testData[testData.shape[1]-1]

  # Construct Network using various parameters
  max_epoch = 100
  print("Maxium epoch is: ")
  print(max_epoch)
  print("")
  eta = 1
  print("For eta = 1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.1
  print("For eta = 0.1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.01
  print("For eta = 0.01")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.001
  print("For eta = 0.001")
  do_backprop(eta,max_epoch)
  print("")

  max_epoch = 50
  print("Maxium epoch is: ")
  print(max_epoch)
  print("")
  eta = 1
  print("For eta = 1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.1
  print("For eta = 0.1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.01
  print("For eta = 0.01")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.001
  print("For eta = 0.001")
  do_backprop(eta,max_epoch)
  print("")

  max_epoch = 100
  print("Maxium epoch is: ")
  print(max_epoch)
  print("")
  eta = 1
  print("For eta = 1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.1
  print("For eta = 0.1")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.01
  print("For eta = 0.01")
  do_backprop(eta,max_epoch)
  print("")

  eta = 0.001
  print("For eta = 0.001")
  do_backprop(eta,max_epoch)
  print("")
