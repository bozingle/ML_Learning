#Written by Joel Reznick 7/3/2019

'''
Description:
This program implements the theory of Logistic Regression in its glory. 
'''

import numpy as np#A beautiful module that the data science community is familiar with.

class LogisticRegression:
    def __init__(self,dims, lr = 0.01, epochs= 1000):
        self.lr = lr #Learning rate
        self.w = np.zeros(dims)#Weights
        self.epochs = epochs#Number of epochs for training.
    
    def sigmoid(self,x):
        return 1/(1+np.exp(np.dot(self.w,x)))#Sigmoid function.

    def training(self,x,y):
        for i in range(self.epochs):#Iterate for the number of epochs.
            for j in range(len(x[:,0])):
                self.w += self.lr*(y[j] - self.sigmoid(x[j]))*x[j]#Shift the weight vector
    
    def predict(self,x):
        return self.sigmoid(x)#Returns the probability value.