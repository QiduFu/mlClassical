# This project is based on the assignments completed for the Coursera specialization:
# Machine Learning by Andrew Ng (Stanford)

# Title: Building anomaly detection From Scratch
# By: Qidu(Quentin) Fu

# Import libraries ------------------------------------------------------------
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from utils2 import *

# problem description ----------------------------------------------------------
# ------------------------------------------------------------------------------

## 2 - Anomaly detection

### 2.1 Problem Statement

# In this exercise, you will implement an anomaly detection algorithm to
# detect anomalous behavior in server computers.

# The dataset contains two features -
#    * throughput (mb/s) and
#    * latency (ms) of response of each server.

# While your servers were operating, you collected $m=307$ examples of how they were behaving, and thus have an unlabeled dataset $\{x^{(1)}, \ldots, x^{(m)}\}$.
# * You suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

# You will use a Gaussian model to detect anomalous examples in your
# dataset.
# * You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing.
# * On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies.
# * After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions.

# load data -------------------------------------------------------------------
# ------------------------------------------------------------------------------
X_train, X_val, y_val = load_data()

# view the variables
print("The shape of X_train: ", X_train.shape)
print("The shape of X_val: ", X_val.shape)
print("The shape of y_val", y_val.shape)
print("The first few elements of X_train: ", X_train[0:5, :])
print("The first few elements of X_val: ", X_val[0:5, :])
print("The first few elements of y_val: ", y_val[0:5, :])

# visualize the data

plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c="r")
plt.title("The first dataset")
plt.ylabel("Throughout (mb/s)")
plt.xlabel("Latency (ms)")
plt.axis([0, 30, 0, 30])
plt.show()


# Gaussian distribution --------------------------------------------------------
# ------------------------------------------------------------------------------
def estimate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X, axis=0)  # by column: top to bottom 1*n
    sigma2 = np.var(X, axis=0)  # by column: top to bottom 1*n
    return mu, sigma2


def multivariate_guassian(X_train, mu, var):
    """
    Computes the probability
    density function of the examples X under the multivariate gaussian
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """
    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)

    X = X_train - mu
    p = (
        (2 * np.pi) ** (-k / 2)
        * np.linalg.det(var) ** (-0.5)
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(var)) * X, axis=1))
    )
    return p


# Selecting the threshold, $\epsilon$ ------------------------------------------
# ------------------------------------------------------------------------------
# The threshold is based on the F1 score, which is a measure of accuracy of the classifier.
def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    stepsize = (np.max(p_val) - np.min(p_val)) / 1000
    epsilons = np.arange(np.min(p_val), np.max(p_val), stepsize)

    for epsilon in epsilons:
        predictions = (p_val < epsilon)[:, np.newaxis]
        tp = np.sum(predictions[y_val == 1] == 1)
        fp = np.sum(predictions[y_val == 0] == 1)
        fn = np.sum(predictions[y_val == 1] == 0)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon
    return best_epsilon, best_f1


p_val = multivariate_guassian(X_val, mu, var)
epsilon, _ = select_threshold(y_val, p_val)
outliers = np.where(p < epsilon)

# high dimensional dataset -----------------------------------------------------
# ------------------------------------------------------------------------------
X_train_high, y_train_high, X_val_high, y_val_high = load_data_multi()
# Check the dimensions
print("The shape of X_train_high: ", X_train_high.shape)
print("The shape of y_train_high: ", y_train_high.shape)
print("The shape of X_val_high: ", X_val_high.shape)
print("The shape of y_val_high: ", y_val_high.shape)

# apply the anomaly detection algorithm to the high dimensional dataset


def apply_anomaly_detection(X_train, X_val, y_val):
    # Estimate the Gaussian parameters
    mu, var = estimate_gaussian(X_train)
    # Evaluate the probabilities for both the training data X_train and the cross-validation data X_val
    p = multivariate_guassian(X_train, mu, var)
    p_val = multivariate_guassian(X_val, mu, var)
    # Select the threshold
    epsilon, _ = select_threshold(y_val, p_val)
    # Find the outliers in the training data X_train
    outliers = np.where(p < epsilon)
    return outliers
