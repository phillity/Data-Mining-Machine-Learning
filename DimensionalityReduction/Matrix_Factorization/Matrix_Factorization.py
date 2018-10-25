# Tyler Phillips
# CSCI57300 Data Mining
# Matrix Factorization

import numpy as np

# Matrix Factorization function
# Args:
#   R - nxd rating matrix
#   K - number of latent factors
#   alpha - learning rate
#   beta - regularization coefficient
#   steps - iteration threshold
#   eps - convergence threshold
def Matrix_Factorization(R, K, alpha=0.0002, beta=0.02, steps=5000, eps=0.001):
    n,d = R.shape

    # Intialize random P and Q matrices
    P = np.random.rand(n,K)
    Q = np.random.rand(d,K)
    Q = Q.T

    # Intialize iteration count
    iter = 1
    
    while True:
        # Gradient Descent
        # L = (R_ij - P_i*Q_j)^2 + beta*(||P_i||^2 + ||Q_j||^2)
        # dL/dP_i = -2*E_ij*Q_j + 2*beta*P_i
        # dL/dQ_j = -2*E_ij*P_i + 2*beta*Q_j
        # P_i_new = P_i_old + 2*alpha*(E_ij*Q_j - beta*P_i)
        # Q_j_new = Q_j_old + 2*alpha*(E_ij*P_i - beta*Q_j)
        for i in range(n):
            for j in range(d):
                if R[i,j] > 0:
                    E_ij = R[i,j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i,k] = P[i,k] + 2 * alpha * (E_ij * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + 2 * alpha * (E_ij * P[i,k] - beta * Q[k,j])
        
        # Calculate error
        # Frobenius norm of difference between R and P*Q.T (only considering known elements in R)
        # Plus regularization term (sum of squared norms of P and Q.T, multiplied by regularization factor)
        err = 0
        for i in range(n):
            for j in range(d):
                if R[i,j] > 0:
                    err = err + (R[i,j] - np.dot(P[i,:],Q[:,j])) ** 2
                    err = err + beta * (np.linalg.norm(P[i,:]) ** 2 + np.linalg.norm(Q[:,j]) ** 2)
        
        # Print iteration update
        print("Iteration: " + str(iter) + "   Error: " + str(err))
        iter = iter + 1

        # Check for convergence
        if err < eps or iter > steps:
            break

    return P,Q.T


# Rating matrix
R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]
R = np.array(R)

# Number of latent factors
K = 2

# Perform matrix factorization
P,Q = Matrix_Factorization(R, K)

# Print results
print("")
print("R:")
print(str(R))
print("P:")
print(str(P))
print("Q.T:")
print(str(Q.T))
print("Predicted R:")
print(np.dot(P,Q.T))