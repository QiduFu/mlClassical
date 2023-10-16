#This project is based on the assignments completed for the Coursera specialization: 
#Machine Learning by Andrew Ng (Stanford)

#Title: Building Neural Networks for Multiclass-Classification From Scratch
#By: Qidu(Quentin) Fu

#problem statement-------------------------------------------------------------
"""
you will use a neural network to recognize ten handwritten digits, 0-9. 
This is a multiclass classification task where one of n choices is selected. 
Automated handwritten digit recognition is widely used today - 
from recognizing zip codes (postal codes) on mail envelopes 
to recognizing amounts written on bank checks.
"""

#import libraries -------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt

#load the data ---------------------------------------------------------------
#--------------------------------------------------------------------------------

#define the load-data function
def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

X, y = load_data()
#view the variables
print('view the first row of the X: ', X[0])
print('view the first row of the y: ', y[0])
print('view the shape of the X: ', X.shape)
print('view the shape of the y: ', y.shape)

#visualize the data
def view_the_data(X):
    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    #fig.tight_layout(pad=0.5)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)

#build the model --------------------------------------------------------------
#--------------------------------------------------------------------------------
#define the softmax function
def my_softmax(z):
    ez = np.exp(z)
    a = ez/np.sum(ez)
    return a

tf.random.set_seed(0)

def build_model():
    model = Sequential(
        [
            keras.layers.InputLayer(shape=(400,)),
            keras.layers.Dense(units=25, activation='relu', name='Layer_1'),
            keras.layers.Dense(units=15, activation='relu', name='Layer_2'),
            keras.layers.Dense(units=10, activation='linear', name='Layer_3'),
            
        ], name = 'my_model'
    )
    return model

model = build_model()

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
#reshape the image to (1,400) since the input layer is shape=(400,)
prediction = model.predict(image_of_two.reshape(1,400)) 

#If for selection, prediction is enough, 
#however, to see the actual probabilities, 
#we need to apply softmax function to the prediction
prediction_p = tf.nn.softmax(prediction)
#to select the highest probability
prob_highest_idx = np.argmax(prediction_p)

#

