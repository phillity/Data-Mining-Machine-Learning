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
         
        # Centroid update step
        for i in range(k):
            # Update centriod mu_i as average of cluster C_i elements
            if len(C[i]) > 1:
                mu[i,:] = np.sum(C[i], axis=0) / len(C[i])

        # Check for convergence
        if np.linalg.norm(mu - prev_mu) <= eps:
            return C, labels, mu, iter  

        # Update iteration count
        iter += 1
        # Update previous mu
        prev_mu = np.copy(mu)

# Sum of squared error helper function
def SSE(C, mu):
    sse = 0
    for c_i in C:
        sse = sse + np.linalg.norm(c_i - mu) ** 2
    return sse





# Get the arguments list 
argv = str(sys.argv)
print(str(argv))

# Get number of arguments
argc = len(sys.argv)

# Print error if not enough arguments
if argc < 3:
    sys.exit("Datafile and k arguments are required!")

# Read in D data matrix
if sys.argv[1] == "iris.data.txt" or sys.argv[1] == "iris.txt":
    D = np.loadtxt(sys.argv[1],delimiter=',',usecols=(0,1,2,3))
else:
    D = np.loadtxt(sys.argv[1],delimiter=',')
if len(D.shape) < 2:
    D = D.reshape((D.shape[0],1))

# Read in k cluster count
k = int(sys.argv[2])

# Read in mu centroid id list if given
mu = np.zeros((k,D.shape[1]))
mu_list = None
if argc > 3:
    mu_list = np.loadtxt(sys.argv[3],dtype=int,delimiter=',')
    for i,id in enumerate(mu_list):
        mu[i,:] = D[id-1,:]

C, labels, mu, iter = kMeans(D, k, mu)
total_sse = 0

# Print input information
n, d = D.shape
print("Number of Datapoints: n=" + str(n))
print("Number of Dimensions: d=" + str(d))
print("Number of Clusters: k=" + str(k))
print("\n")

# Print results
print("Convergence after " + str(iter) + " iterations:")
for i in range(k):
    print("-----Cluster " + str(i) + "---------------")
    print(str(len(C[i])) + " elements:" + str(labels[i]))
    print("mu_" + str(i) + ":" + str(mu[i,:]))
    print("SSE_" + str(i) + ":" + str(SSE(C[i],mu[i,:])))
    total_sse = total_sse + SSE(C[i],mu[i,:])
print("/n")
print("Total SSE :" + str(total_sse))