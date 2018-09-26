# Tyler Phillips
# CSCI57300 Data Mining
# Power Method

import numpy as np
import sys

# Power_Method
# Args:
#   A - dxd diagonalizable matrix
#   eps - convergence criteria
def Power_Method(A, eps=0.000001):
    d, d = A.shape

    # Initialize x_i
    x_i = np.random.rand(d,1)
    x_i_m_i = 0

    # Initialize x_i_prev
    x_i_prev = np.zeros((d,1))

    # Perform power nethod
    while True:
        x_i = A @ x_i
        x_i_m_i = np.amax(x_i)
        x_i = x_i / x_i_m_i

        # Check for convergece
        if np.linalg.norm(x_i - x_i_prev) < eps:
            # Get found eigenvector and eigenvalue
            v = x_i / np.linalg.norm(x_i)
            
            # Rayleigh Quotient
            w = A @ v
            w = ((w.T @ v) / (v.T @ v))[0,0]

            return w, v

        # Update x_i_prev
        x_i_prev = np.copy(x_i)


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

n, d = D.shape

# Compute mean
mu = np.sum(D,axis=0) / n

# Center data
Z = D - mu

# Compute covariance matrix
cov = Z.T @ Z / n

w, v = Power_Method(cov)