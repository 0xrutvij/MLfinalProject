import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import requests


# Create the perceptron model class
class perceptronModel:

  # Constructor method with instance variables
  def __init__(self, input, output):
    self.input = input # each example is of the form <1,x1,x2,...,xn>
    self.output = output
    self.numOfSamples = input.shape[0]
    self.numOfAtts = input.shape[1]
    
    # randomly initialize the weights close to zero
    self.finalWeights = pd.DataFrame(np.random.rand(self.numOfAtts))

  # get prediction by constructing a linear function
  def findPred(self,weights,bias):
    
    pred = pd.DataFrame(np.ones(self.numOfSamples))
    f = (self.input.dot(weights))
    
    for i in range(0,pred.shape[0]):
      if f.iloc[i] < 0:
        pred.iloc[i] = -1

    return pred
  
  # find the partial derivative of J
  def findPartial(self,ywx):
    partial = pd.DataFrame(np.zeros(self.numOfAtts))
    
    # for every data point
    for i in range(0,self.input.shape[0]):
      
      xi = self.input.iloc[i]
    
      if ywx.iloc[i] < 0:
        for j in range(0, xi.shape[0]):
          partial.iloc[j] = partial.iloc[j] - self.input[j]*self.output.iloc[j]
    
    partial.multiply(self.numOfSamples)
    
    return partial
    
  # preform the RMSprop batch gradient descent
  def gradientDescent(self):
    
    # initialize the parameters
    weightsTemp = pd.DataFrame(np.zeros(self.numOfAtts))
    eta = .001 # step size
    v = pd.DataFrame(np.zeros(self.numOfAtts+1))
    beta = 0.9
    ep = .0001
    iterations = 0
    notDone = True

    # while the stopping condition is not met
    while (notDone):
      
      # find prediction
      pred = self.findPred(weights,bias)
      
      # use the modified hinge loss/ perceptron criteria
      ywx = (pred*self.output)[0]
      maximum = np.maximum((-ywx,0)
      avgLoss = sum(maximum)/(self.numOfSamples)

      # Find partial derivatives
      partial = self.findPartial(ywx)
      
      # find exponential moving average
      for i in range(0,v.shape[0]):
        v.iloc[i] = beta*v.iloc[i] + (1-beta)*((partial[i])**2)
      
      # find new bias parameter
      biasTemp = bias - (eta/(v.iloc[0]+ep)**0.5)*partial[0]

      # find new weights parameters
      for i in range(0,self.numOfAtts): 
        weightsTemp.iloc[i] = weights.iloc[i] - (eta/(v.iloc[i+1]+ep)**0.5)*partial[i+1]

      # Update parameters
      bias = biasTemp
      weights = weightsTemp

      iterations+=1

      # evaluate the stopping conditions
      if ((avgLoss<50) or (iterations>99)):
        notDone = False

    print("The final bias is: ") 
    print(bias)
    print("The final weights are: ")
    print(weights)
    print("The number of iterations was: ")
    print(iterations)
    self.finalBias = bias
    self.finalWeights = weights

  # function to run the model on test data
  def testModel(self,test_input,test_output):
      result = test_input.dot(self.finalWeights)+self.finalBias

      # find the MSE
      MSE = mean_squared_error(test_output,result)
      print("The MSE is: ")
      print(MSE)
      
# Start of main
#if __name__ == "__main__":
  
  # load the data
  df = pd.read_csv('/bank_note.csv')
  # train_input = ...
  # train_output = ...
  # test_input = ..
  # _output = ...
  
  # Create a perceptron model
  # myModel = perceptronModel(train_input,train_output) 

  # Preform the gradient descent
  #myModel.gradientDescent()

  # Use the test data to compute the accuracy of the model
  # myModel.testModel(test_input,test_output)
