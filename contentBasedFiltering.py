# This project is based on the assignments completed for the Coursera specialization:
# Machine Learning by Andrew Ng (Stanford)

# Title: Deep learning for content-based filtering From Scratch
# By: Qidu(Quentin) Fu

import numpy as np
import numpy.ma as ma
from numpy import genfromtxt
from collections import defaultdict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from utilsUnsupervise import *

# Load Data, set configuration variables
(
    item_train,
    user_train,
    y_train,
    item_features,
    user_features,
    item_vecs,
    movie_dict,
    user_to_genre,
) = load_data()

num_user_features = (
    user_train.shape[1] - 3
)  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
scaledata = True  # applies the standard scalar to data if true
print(f"Number of training vectors: {len(item_train)}")

# scale training data
if scaledata:
    item_train_save = item_train
    user_train_save = user_train

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    print(np.allclose(item_train_save, scalerItem.inverse_transform(item_train)))
    print(np.allclose(user_train_save, scalerUser.inverse_transform(user_train)))

item_train, item_test = train_test_split(
    item_train, train_size=0.80, shuffle=True, random_state=1
)
user_train, user_test = train_test_split(
    user_train, train_size=0.80, shuffle=True, random_state=1
)
y_train, y_test = train_test_split(
    y_train, train_size=0.80, shuffle=True, random_state=1
)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test  data shape: {item_test.shape}")


num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential(
    [
        ### START CODE HERE ###
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_outputs, activation="linear"),
        ### END CODE HERE ###
    ]
)

item_NN = tf.keras.models.Sequential(
    [
        ### START CODE HERE ###
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_outputs, activation="linear"),
        ### END CODE HERE ###
    ]
)

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = Model([input_user, input_item], output)

model.summary()

# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec, len(item_vecs))

# scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(
    user_vecs,
    item_vecs,
    model,
    u_s,
    i_s,
    scaler,
    scalerUser,
    scalerItem,
    scaledata=scaledata,
)

print_pred_movies(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount=10)

uid = 36
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(
    uid, scalerUser.inverse_transform(user_train), item_vecs, user_to_genre
)

# scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(
    user_vecs,
    item_vecs,
    model,
    u_s,
    i_s,
    scaler,
    scalerUser,
    scalerItem,
    scaledata=scaledata,
)
sorted_y = y_vecs[sorted_index]

# print sorted predictions
print_existing_user(
    sorted_ypu,
    sorted_y.reshape(-1, 1),
    sorted_user,
    sorted_items,
    item_features,
    ivs,
    uvs,
    movie_dict,
    maxcount=10,
)


def sq_dist(a, b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    ### START CODE HERE ###
    d = sum(np.square(a - b))
    ### END CODE HERE ###
    return d


input_item_m = tf.keras.layers.Input(shape=(num_item_features))  # input layer
vm_m = item_NN(input_item_m)  # use the trained item_NN
vm_m = tf.linalg.l2_normalize(
    vm_m, axis=1
)  # incorporate normalization as was done in the original model
model_m = Model(input_item_m, vm_m)
model_m.summary()

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:, i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")

count = 50
dim = len(vms)
dist = np.zeros((dim, dim))

for i in range(dim):
    for j in range(dim):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i, 0])
    movie2_id = int(item_vecs[min_idx, 0])
    genre1, _ = get_item_genre(item_vecs[i, :], ivs, item_features)
    genre2, _ = get_item_genre(item_vecs[min_idx, :], ivs, item_features)

    disp.append(
        [movie_dict[movie1_id]["title"], genre1, movie_dict[movie2_id]["title"], genre2]
    )
table = tabulate.tabulate(
    disp,
    tablefmt="html",
    headers="firstrow",
    floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"],
)
table

