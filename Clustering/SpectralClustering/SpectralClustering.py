# Tyler Phillips
# CSCI57300 Data Mining
# Spectral Clustering

import sys
import numpy as np
from kMeans import kMeans
import matplotlib.pyplot as plt

# Spectral Clustering function
# Args:
#   D - nxd data matrix
#   k - cluster count
#   ratio - flag if ratio cut should be used, otherwise assume normalized cut
#   sig - sigma used in similarity computation
def Spectral_Clustering(D,k,ratio=True,sig=1):
    # Get dimensions of nxd D matrix
    n, d = D.shape 

    # Compute nxn adjacency (similarity) matrix
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = 0.0
            else:
                A[i][j] = similarity(D[i,:],D[j,:],sig)

    # Compute degree matrix
    deg = np.identity(n) * np.sum(A,axis=1)

    # Compute laplacian matrix
    L = deg - A

    # Set B accoring to ratio cut/normalized cut
    if ratio == True:
        B = L
    else:
        B = np.linalg.inv(deg) @ L

    # Compute eigenvalues and eigenvectors of B
    w, v = np.linalg.eig(B)

    # Get reduced basis 
    v = v[:,:k]

    # Normalize basis to obtain new dataset
    Y = np.zeros((n,k))
    for i in range(n):
        Y[i,:] = v[i,:] * (1. / (sum(v[i,:] ** 2) ** (1./2)))

    # Run kMeans on new dataset
    return kMeans(Y,k)

# Similarity helper function
def similarity(x_i,x_j,sig):
    return np.exp(np.linalg.norm(x_i-x_j)/(2*sig**2))

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
    sys.exit("Datafile and k arguments are required!");

# Read in D data matrix
if sys.argv[1] == "iris.data.txt" or sys.argv[1] == "iris.txt":
    D = np.loadtxt(sys.argv[1],delimiter=',',usecols=(0,1,2,3))
else:
    D = np.loadtxt(sys.argv[1],delimiter=',')
if len(D.shape) < 2:
    D = D.reshape((D.shape[0],1))

# Read in k centroid count
k = int(sys.argv[2])

# Print input information
n, d = D.shape
print("Number of Datapoints: n=" + str(n))
print("Number of Dimensions: d=" + str(d))
print("Number of Clusters: k=" + str(k))
print("\n")

C, labels, mu, iter = Spectral_Clustering(D,k,ratio=False)

# Print results
print("Convergence after " + str(iter) + " iterations:")
for i in range(k):
    print("-----Cluster " + str(i) + "---------------")
    print(str(len(labels[i])) + " elements:" + str(labels[i]))
    print("mu_" + str(i) + ":" + str(mu[i,:]))
    print("SSE_" + str(i) + ":" + str(SSE(C[i],mu[i,:])))


