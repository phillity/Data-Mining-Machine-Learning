# Tyler Phillips
# CSCI57300 Data Mining
# Linear Discriminant Analysis (LDA)

import numpy as np
import sys

# LDA function
# Args:
#   D - nxd data matrix (last column is class labels)
#   num_components - parameter to set number of ouput dimensions
def LDA(D, num_components):
    # Get class information
    Y = D[:,-1]
    c = np.unique(Y).shape[0]

    D = D[:,:-1]
    n, d = D.shape

    # Get class subsets
    C = []
    for i in range(c):
        C.append(D[Y == i])

    # Get class means
    mu_c = np.zeros((c,d))
    for i in range(c):
        mu_c[i,:] = np.sum(C[i],axis=0) / len(C[i])

    # Get mean of class means
    mu = np.sum(mu_c,axis=0) / c

    # Get between class scatter
    B = np.zeros((d,d))
    for i in range(c):
        B = B + np.outer((mu_c[i,:] - mu).T,(mu_c[i,:] - mu))
    B = B / c

    # Center class matricies
    for i in range(c):
        C[i] = C[i] - mu_c[i,:]

    # Get class scatter matricies
    W = []
    for i in range(c):
        Z_i = C[i].T @ C[i]
        W.append(Z_i)

    # Compute within class scatter matrix
    S = np.zeros((d,d))
    for i in range(c):
        S = S + W[i]

    # Compute eigenvectors and values of S^-1 @ B
    w, v = np.linalg.eig(np.linalg.inv(S) @ B)

    # Sort descending eigenvalues and respective eigenvectors
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]

    # Get num_components dominant eigenvalues and vectors
    w = w[:num_components]
    v = v[:,:num_components]

    # Project data into eigenvector basis subspace 
    A = D @ v

    return A
    

# Get the arguments list 
argv = str(sys.argv)
print(str(argv))

# Get number of arguments
argc = len(sys.argv)

# Print error if not enough arguments
if argc < 3:
    sys.exit("Datafile and k arguments are required!")

# Read in D data matrix
D = np.loadtxt(sys.argv[1],delimiter=',')

# Read in num_components
num_components = int(sys.argv[2])

A = LDA(D, num_components)