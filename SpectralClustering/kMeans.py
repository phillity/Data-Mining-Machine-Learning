# Tyler Phillips
# CSCI57300 Data Mining
# k-Means

import sys
import numpy as np

# k-Means function
# Args:
#   D - nxd data matrix
#   mu - kxd centroid matrix
#   k - cluster count
#   eps - convergence tolarence
def kMeans(D, k, mu=None, eps=0.0001):
    # Get dimensions of nxd D matrix
    n, d = D.shape 

    # If mu is not preset
    if mu is None:
        # Randomly intialize k centroids in kxd mu matrix
        mu = np.zeros((k,d))
        mu_list = np.random.choice(n,size=k,replace=False)
        for i,id in enumerate(mu_list):
            mu[i,:] = D[id,:]

    # Intialize previous mu matrix
    prev_mu = np.zeros((k,d))
    
    # Intialize iteration count
    iter = 1

    while True:
        # Clusters as list of k lists
        C = [[] for i in range(k)]
        labels = [[] for i in range(k)]

        # Cluster assignment step
        for i in range(n):
            min_dist = sys.float_info.max
            min_idx = -1
            # Get distances between x_i (D[i,:]) and each centriod in mu
            for j in range(k):
                dist = np.linalg.norm(D[i,:] - mu[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            # Add x_i to cluster corresponding to minimum distance
            C[min_idx].append(D[i,:])
            labels[min_idx].append(i+1)
         
        # Centroid update step3
        for i in range(k):
            # Update centriod mu_i as average of cluster C_i elements
            if len(C[i]) > 1:
                mu[i,:] = np.sum(C[i], axis=0) / len(C[i])

        # Check for convergence
        check = np.linalg.norm(mu - prev_mu)
        if check <= eps:
            return C, labels, mu, iter

        # Print update
        #print("Iteration " + str(iter) + ":")
        #for i in range(k):
            #print("c_" + str(i) + ":" + str(C[i]) + "     mu_" + str(i) + ":" + str(mu[i,:]))
        #print("||mu - mu_prev||:" + str(check) + "\n")    

        # Update iteration count
        iter += 1
        # Update previous mu
        prev_mu = np.copy(mu)