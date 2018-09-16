# Tyler Phillips
# CSCI57300 Data Mining
# Principal Component Analysis (PCA)

import numpy as np
import sys

# PCA function
# Args:
#   D - nxd data matrix
#   alpha - variance threshold
#   num_components - optional parameter to manually set number of ouput dimensions
def PCA(D, alpha=0.9, num_components=-1):
    n, d = D.shape

    # Compute mean
    mu = np.sum(D, axis=0) / n

    # Center data
    Z = D - mu

    # Compute covariance matrix
    cov = Z.T @ Z / n

    # Compute eigenvalues and eigenvectors
    w, v = np.linalg.eig(cov)

    # Sort descending eigenvalues and respective eigenvectors
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:,idx]

    # Find fraction of total variance using 1 to d dimsensions
    f_r = np.zeros((d,1))
    for i in range(d):
        f_r[i] = np.sum(w[:i+1]) / np.sum(w)

    
    # Choose smallest amount of dimensions so that the fraction of variance meets variance threshold
    if num_components == -1:
        idx = np.where(f_r >= alpha)[0]
        r = idx[f_r[idx].argmin()]
    # Use number of components if given manually
    else:
        r = num_components-1

    # Get reduced basis 
    v = v[:,:r+1]

    # Get reduced dimensionality data
    A = np.zeros((n,r+1))
    for i in range(n):
        A[i,:] = (v.T @ Z[i,:]).T

    return A



# Get the arguments list 
argv = str(sys.argv)
print(str(argv))

# Get number of arguments
argc = len(sys.argv)

# Print error if not enough arguments
if argc < 2:
    sys.exit("Datafile argument required!")

# Read in D data matrix
if sys.argv[1] == "iris.data.txt" or sys.argv[1] == "iris.txt":
    D = np.loadtxt(sys.argv[1],delimiter=',',usecols=(0,1,2,3))
else:
    D = np.loadtxt(sys.argv[1],delimiter=',')
if len(D.shape) < 2:
    D = D.reshape((D.shape[0],1))

# Read in num_components
num_components =-1
if argc > 2:
    num_components = int(sys.argv[2])

A = PCA(D, num_components=num_components)

# Print input information
n, d = D.shape
print("Number of D Datapoints: n=" + str(n))
print("Number of D Dimensions: d=" + str(d))
print("\n")

# Print results
n, d = A.shape
print("Number of A Datapoints: n=" + str(n))
print("Number of A Dimensions: d=" + str(d))

print("A:")
print(str(A))