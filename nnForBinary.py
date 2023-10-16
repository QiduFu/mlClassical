#This project is based on the assignments completed for the Coursera specialization: 
#Machine Learning by Andrew Ng (Stanford)

#Title: Building Neural Networks for Binary Classification From Scratch
#By: Qidu(Quentin) Fu


#problem statement-------------------------------------------------------------
"""You will use a neural network to recognize two handwritten digits, 
zero and one. This is a binary classification task. 
Automated handwritten digit recognition is widely used today - 
from recognizing zip codes (postal codes) on mail envelopes 
to recognizing amounts written on bank checks. 
You will extend this network to recognize all 10 digits (0-9) in a future assignment.
This exercise will show you how the methods 
you have learned can be used for this classification task."""

#import libraries ---------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#define load data
def load_data():
    X = np.loadtxt('X.npy')
    y = np.load('y.npy')
    X = X[0:1000]
    y = y[0:1000]
    return X, y

#load the data
X, y = load_data()

#view the variables
print('view the first row of the X: ', X[0])
print('view the first row of the y: ', y[0])
print('view the shape of the X: ', X.shape)
print('view the shape of the y: ', y.shape)

#visualize the data
def view_the_data(X):
    num_samples, num_features = len(X)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)
    for i, ax in enumerate(axes.flat):
        #select random indices
        random_idx = np.random.randint(num_samples)
        #select rows corresponding to the random indices
        X_random_reshaped = X[random_idx].reshape(20, 20).T
        #plot the image
        ax.imshow(X_random_reshaped, cmap='gray')
        #Display the label above the image
        ax.set_title(y[random_idx], 0)
        ax.set_axis_off()
        plt.show()
    
#build the model----------------------------------------------------------------
#------------------------------------------------------------------------------

#build the model
def build_model():
    model = Sequential(
        [
            tf.keras.Input(shape=(400,)), #specify input size
            Dense(units=25, activation='relu', name='Layer_1'),
            Dense(units=10, activation='relu', name='Layer_2'),
            Dense(units=1, activation='sigmoid', name='Output_Layer'),
        ], name = 'my_model'
    )
    return model

model = build_model()

#compile the model
def compile_and_fit_model(model):
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)
    return model

#define the prediction function
def predict(model, X):
    y_pred = model.predict(X)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred


