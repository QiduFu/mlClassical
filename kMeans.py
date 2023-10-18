#This project is based on the assignments completed for the Coursera specialization: 
#Machine Learning by Andrew Ng (Stanford)

#Title: Building kMeans for clustering From Scratch
#By: Qidu(Quentin) Fu

#Import libraries ------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#Implement kMeans ------------------------------------------------------------
#------------------------------------------------------------------------------

#Find the closest centroid for each example
def find_closest_centroids(X, centroids):
    #Set K
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    
    for i in range(X.shape[0]):
        dist = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i,:] - centroids[j,:])
            dist.append(norm_ij)
        idx[i] = np.argmin(dist)
    return idx

#Compute the mean of the data points assigned to each centroid
def compute_cetroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k, :]
        centroids[k,:] = np.mean(points, axis = 0)
    return centroids

#Run kMeans
def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    plt.figure(figsize=(10,6))
    for i in range(max_iters):
        print('kMeans iteration {}/{}'.format(i+1, max_iters))
        idx = find_closest_centroids(X, centroids)
        centroids = compute_cetroids(X, idx, K)
        if plot_progress:
            plt.subplot(2,5,i+1)
            plt.scatter(X[:,0], X[:,1], c=idx[:,0], cmap='rainbow')
            plt.scatter(centroids[:,0], centroids[:,1], marker='x', c='black')
            plt.title('Iteration {}'.format(i+1))
    plt.show()
    return centroids, idx

#Random initialization
def kMeans_init_centroids(X, K):
    m, n = X.shape
    rand_idx = np.random.permutation(m)
    centroids = X[rand_idx[0:K], :]
    return centroids

#
