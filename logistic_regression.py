# Librairies
import pandas as pd
import math

# Load the dataset
dataframe = pd.read_csv('data/diabetes.csv')

#print(dataframe.head())

# Define variables (weights and learning rate)
w0 = 0.0
w1 = 0.0
w2 = 0.0
w3 = 0.0
w4 = 0.0
w5 = 0.0
w6 = 0.0
w7 = 0.0
w8 = 0.0
w9 = 0.0


learning_rate = 0.01

# Method linear scoring
def linear_score(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    z = w0 + x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + x9*w9
    return z

# Sigmoïd function
def sigmoid_function(z):
    p = 1/(1+math.exp(-z))
    return p

# Cross entropy
def cross_entropy(y, p):
    loss = -(y*math.log(p) + (1-y)*math.log(1-p))
    return loss

# Gradient calcul
def gradient_calcul(y,p):
    error = p-y
    return error

# Calcul of gradient by weights
def grad(error, x):
    grad = error*x
    return grad

# Weight upadate
def weight_update(learning_rate, w, grad):
    new_weight = w - learning_rate*grad
    return new_weight


for i in range(len(dataframe)):
    print(dataframe.loc[i])
    break