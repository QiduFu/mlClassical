#This project is based on the assignments completed for the Coursera specialization: 
#Machine Learning by Andrew Ng (Stanford)

#Title: Building Logistic Regression Model for Classification From Scratch
#By: Qidu(Quentin) Fu

#Import the libraries and load the data----------------------------------------
#------------------------------------------------------------------------------
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from utilsRegression import *
import copy
import math

#load the data
X_train, y_train = load_data('ex2data1.txt')

#view the variables 
print('First 5 rows of X_train: \n', X_train[:5])
print('Type of X_train: ', type(X_train))

print('First 5 rows of y_train: \n', y_train[:5])
print('Type of y_train: ', type(y_train))

#visualize the data

plot_data(X_train, y_train)
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(['Admitted', 'Not admitted'])
plt.show()

#Build the model----------------------------------------------------------------
#------------------------------------------------------------------------------
#Define the sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
        z -- A scalar or numpy array of any size.
    Return:
        s -- sigmoid(z)"""
    return 1/(1+np.exp(-z))

#Define the cost function
def compute_cost(X, y, w, b, *argv):
    """
    Arguments:
        X -- input data of size (m, n)
        y -- label vector of size (m, 1)
        w -- weight vector of size (n, 1)
        b -- bias
    Return:
        cost -- cost function for logistic regression
    """
    
    m, n = X.shape
    
    cost = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        loss = -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
        cost += loss
    cost = cost/m
    return cost

#Define the gradient descent function
def compute_gradient(X, y, w, b, *argv):
    """Computes the gradient of the cost function 

    Arguments:
        X -- input data of size (m, n)
        y -- label vector of size (m, 1)
        w -- weight vector of size (n, 1)
        b -- bias
    Return:
        dj_dw -- gradient of w, numpy array of size (n, 1)
        dj_db -- gradient of b, scalar
    """
    m, n = X.shape
    dj_dw = np.zeros
    dj_db = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        dj_dw += (f_wb_i - y[i])*X[i]
        dj_db += (f_wb_i - y[i])

#Define the gradient descent function
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lamba_):
    #number of training examples
    m = len(X)
    #Array to store the cost function values
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        #calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lamba_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        
        #Save cost function value and w per iteration
        if i < 100000:
            J_history.append(cost_function(X, y, w_in, b_in, lamba_))
            w_history.append(w_in)
        
        #print cost every 50 iterations
        if i % 5000 == 0:
            print('Cost after iteration %i: %f' %(i, cost_function(X, y, w_in, b_in, lamba_)))
    return w_in, b_in, J_history, w_history

#Define the predict function

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    Arguments:
        X -- input data of size (m, n)
        w -- weight vector of size (n, 1)
        b -- bias
    Return:
        y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m, n = X.shape
    y_prediction = np.zeros((m, 1))
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += w[j]*X[i][j]
        z_wb += b
        y_prediction[i] = sigmoid(z_wb) >= 0.5
    return y_prediction

#Regularized logistic regression------------------------------------------------
#------------------------------------------------------------------------------
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Compute the cost function for regularized logistic regression
    Arguments:
        X -- input data of size (m, n)
        y -- label vector of size (m, 1)
        w -- weight vector of size (n, 1)
        b -- bias
        lambda_ -- regularization parameter
    Return:
        cost -- cost function for regularized logistic regression
    """
    m, n = X.shape
    
    cost_without_reg = compute_cost(X, y, w, b)
    cost_reg = 0
    for i in range(n):
        cost_reg += w[i]**2
    cost_reg = cost_reg*lambda_/(2*m)
    total_cost = cost_without_reg + cost_reg
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Compute the gradient of the cost function for regularized logistic regression
    Arguments:
        X -- input data of size (m, n)
        y -- label vector of size (m, 1)
        w -- weight vector of size (n, 1)
        b -- bias
        lambda_ -- regularization parameter
    Return:
        dj_dw -- gradient of w, numpy array of size (n, 1)
        dj_db -- gradient of b, scalar"""
        
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for i in range(n):
        dj_dw[i] += lambda_*w[i]/m
        
    return dj_db, dj_dw
