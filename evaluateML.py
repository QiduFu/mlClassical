# This project is based on the assignments completed for the Coursera specialization:
# Machine Learning by Andrew Ng (Stanford)

# Title: Evaluate and improve the models
# By: Qidu(Quentin) Fu

# import libraries-------------------------------------------------------------
# --------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import linear, relu, sigmoid
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from utilsRegression import *

# Evaluate a ML model (polynomial regression)------------------------------------
# --------------------------------------------------------------------------------
# split the data into training and testing sets
# Generate the data
X, y, x_ideal, y_ideal = gen_data(18, 2, 0.7)
print("x.shape: ", X.shape, "y.shape: ", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train.shape: ", X_train.shape, "y_train.shape: ", y_train.shape)

# plot the training data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "k--", label="ideal")
ax.set_title("Train, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_test, y_test, color="blue", label="test")
ax.legend(loc="upper left")
plt.show()


# calculate error for model evaluation, linear regression


def eval_mse(y, yaht):
    m = len(y)
    err = np.sum((y - yaht) ** 2) / (2 * m)
    return err


# compare performance on training and test data
degree = 0
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)

yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

# plot predictions over data range
x = np.linspace(0, int(X.max()), 100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1, 1)

plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

# Generate  data
X, y, x_ideal, y_ideal = gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

# split the data using sklearn routine
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

# bias and variance -------------------------------------------------------------
# --------------------------------------------------------------------------------
# plot the training, cross-validation and test data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(x_ideal, y_ideal, "k--", label="ideal")
ax.set_title("Train, CV, Test", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color="red", label="train")
ax.scatter(X_cv, y_cv, color="blue", label="cv")
ax.scatter(X_test, y_test, color="green", label="test")
ax.legend(loc="upper left")
plt.show()

# Find the optimal degree for the polynomial model

max_degree = 10
err_train = np.zeros(max_degree)
err_cv = np.zeros(max_degree)
x = np.linspace(0, int(X.max()), 100)  # predict values for plot
y_pred = np.zeros((100, max_degree))

for degree in range(max_degree):
    lmodel = lin_model(degree + 1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:, degree] = lmodel.predict(x)

optimal_degree = np.argmin(err_cv) + 1

# plot the result
plt.close("all")
plt_optimal_degree(
    X_train,
    y_train,
    X_cv,
    y_cv,
    x,
    y_pred,
    x_ideal,
    y_ideal,
    err_train,
    err_cv,
    optimal_degree,
    max_degree,
)

# tune the regularization parameter lambda

lambda_range = np.array([0.01, 0.1, 1, 10, 100, 1000])
num_steps = len(lambda_range)
err_train = np.zeros(num_steps)
err_cv = np.zeros(num_steps)
degree = 10
x = np.linspace(0, int(X.max()), 100)  # predict values for plot
y_pred = np.zeros((100, num_steps))

for i in range(num_steps):
    lambda_ = lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:, i] = lmodel.predict(x)

optimal_reg_idx = np.argmin(err_cv)

plt.close("all")
plt_tune_regularization(
    X_train,
    y_train,
    X_cv,
    y_cv,
    x,
    y_pred,
    err_train,
    err_cv,
    optimal_reg_idx,
    lambda_range,
)

# get more data: increasing training set size-------------------------------------
# --------------------------------------------------------------------------------
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)

# Evaluate ML (NN) -----------------------------------------------------------------
# --------------------------------------------------------------------------------

# Generate and split data set
X, y, centers, classes, std = gen_blobs()

# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.20, random_state=1)
print(
    "X_train.shape:",
    X_train.shape,
    "X_cv.shape:",
    X_cv.shape,
    "X_test.shape:",
    X_test.shape,
)

plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)

#evaluate a categorical mdoel by classification error
def eval_classification_error(y, yhat):
    m = len(y)
    err = np.sum(y != yhat) / m
    return err


#model complexity --------------------------------------------------------------
# --------------------------------------------------------------------------------
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

tf.random.set_seed(0)
model = Sequential(
    [
        Dense(120, activation=relu,),
        Dense(40, activation=relu),
        Dense(6, activation=linear),
    ]
)
model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.001),
)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")

#regularization --------------------------------------------------------------
# --------------------------------------------------------------------------------

tf.random.set_seed(1234)
model_r = Sequential(
    [
        Dense(units=120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name='Layer_1'),
        Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name='Layer_2'),
        Dense(units=6, activation='linear', name='Layer_3'),
    ], name= None
)
model_r.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(0.01),
)
model_r.fit(
    X_train, y_train,
    epochs=1000
)

training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )

#tune the regularization parameter lambda
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)
for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")

plt_compare(X_test,y_test, classes, model_predict_s, model_predict_r, centers)

