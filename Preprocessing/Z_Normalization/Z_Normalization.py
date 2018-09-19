# Tyler Phillips
# CSCI57300 Data Mining
# Z-Score Normalization

import numpy as np
import sys

# Z_Normalization
# Args:
#   D - nxd data matrix
def Z_Normalization(D):
    n, d = D.shape

    # Compute mean
    mu = np.sum(D, axis=0) / n

    # Compute standard deviation
    std = np.std(D, axis=0)

    # Compute z-score normalization
    Z = (D - mu) / std

    return Z

# Get the arguments list 
argv = str(sys.argv)
print(str(argv))

# Get number of arguments
argc = len(sys.argv)

# Print error if not enough arguments
if argc < 2:
    sys.exit("Datafile is required!")

# Read in D data matrix
if sys.argv[1] == "iris.data.txt" or sys.argv[1] == "iris.txt":
    D = np.loadtxt(sys.argv[1],delimiter=',',usecols=(0,1,2,3))
else:
    D = np.loadtxt(sys.argv[1],delimiter=',')
if len(D.shape) < 2:
    D = D.reshape((D.shape[0],1))

Z = Z_Normalization(D)

np.savetxt(sys.argv[1][:-4] + "_norm.txt",Z)