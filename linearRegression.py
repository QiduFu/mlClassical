#This project is based on the assignments completed for the Coursera specialization: 
#Machine Learning by Andrew Ng (Stanford)

#Title: Building linear Regression Model for predictions From Scratch
#By: Qidu(Quentin) Fu

#project description: This is directly pasted from the assignment description
# Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

# You would like to expand your business to cities that may give your restaurant higher profits.
# The chain already has restaurants in various cities and you have data for profits and populations from the cities.
# You also have data on cities that are candidates for a new restaurant.
#       For these cities, you have the city population.

# import libraries ------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from utilsRegression import *
import copy
import math

# load data -------------------------------------------------------------------
#------------------------------------------------------------------------------
X_train, y_train = load_data('ex1data1')

#view the data
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5,:])
print(y_train[0:5,:])

#visualize the data
plt.scatter(X_train, y_train, marker='x', c='r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population --- TRAINING DATA')
plt.show()

#build the model --------------------------------------------------------------
#------------------------------------------------------------------------------

#define the cost function
def compute_cost(x, y, w, b):
    num_examples = x.shape[0]
    
    cost = 0
    for i in range(num_examples):
        cost += (x[i] * w + b - y[i])**2
    cost = cost / (2 * num_examples)
    return cost

#compute the gradient
def compute_gradient(x, y, w, b):
    num_examples = x.shape[0]
    
    grad_w = 0
    grad_b = 0
    for i in range(num_examples):
        grad_w += (x[i] * w + b - y[i]) * x[i]
        grad_b += (x[i] * w + b - y[i])
    grad_w = grad_w / num_examples
    grad_b = grad_b / num_examples
    return grad_w, grad_b

#define the gradient descent function
def gradient_descent(x, y, w, b, learning_rate, num_iterations):
    num_examples = x.shape[0]
    J_history = np.zeros((num_iterations, 1))
    w_history = np.zeros((num_iterations, 1))
    b_history = np.zeros((num_iterations, 1))
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    for i in range(num_iterations):
        grad_w, grad_b = compute_gradient(x, y, w, b)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        J_history[i] = compute_cost(x, y, w, b)
        w_history[i] = w
        b_history[i] = b
    return w, b, J_history, w_history, b_history

#make predictions -------------------------------------------------------------
#------------------------------------------------------------------------------
def predict(x, w, b):
    num_examples = x.shape[0]
    y_pred = np.zeros((num_examples, 1))
    for i in range(num_examples):
        y_pred[i] = x[i] * w + b
    return y_pred

#visualize the model ----------------------------------------------------------
plt.plot(X_train, predict(X_train, w, b), c='b')
plt.scatter(X_train, y_train, marker='x', c='r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population Per City')
