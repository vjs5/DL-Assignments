import numpy as np


# ---------------------------------------------------
# Linear Kernel
# k(x_i,x_j) = x_i^T x_j
# ---------------------------------------------------

def linear_kernel(X1, X2):
    return X1 @ X2.T


# ---------------------------------------------------
# Polynomial Kernel
# k(x_i,x_j) = (x_i^T x_j + c)^d
# ---------------------------------------------------

def polynomial_kernel(X1, X2, degree=2, c=1):
    return (X1 @ X2.T + c) ** degree


# ---------------------------------------------------
# RBF (Gaussian) Kernel
# k(x_i,x_j) = exp(-γ ||x_i-x_j||^2)
# ---------------------------------------------------

def rbf_kernel(X1, X2, gamma=0.1):

    sq1 = np.sum(X1**2, axis=1).reshape(-1,1)
    sq2 = np.sum(X2**2, axis=1).reshape(1,-1)

    dist_sq = sq1 + sq2 - 2 * (X1 @ X2.T)

    return np.exp(-gamma * dist_sq)


# ---------------------------------------------------
# Neural Kernel (single pair version)
# Used to match assignment snippet
# ---------------------------------------------------

def neural_kernel(x_i, x_j, feature_extractor):

    phi_i = feature_extractor(x_i.reshape(1,-1))[0]
    phi_j = feature_extractor(x_j.reshape(1,-1))[0]

    return np.dot(phi_i, phi_j)


# ---------------------------------------------------
# Neural Kernel Matrix (efficient version)
# ---------------------------------------------------

def neural_kernel_matrix(phi1, phi2):
    return phi1 @ phi2.T