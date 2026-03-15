import numpy as np

def sigmoid(z):
    """Computes the sigmoid function with clipping to prevent overflow."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_gradient(X, y, h, w, inv_Sigma):
    """
    Computes the gradient of the regularized negative log-likelihood.
    
    Parameters:
    X: Augmented feature matrix (N x D)
    y: True labels (N,)
    h: Predicted probabilities (N,)
    w: Weight vector (D,)
    inv_Sigma: Inverse covariance matrix for the Gaussian prior (D x D)
    """
    grad_loss = X.T @ (h - y)
    grad_reg = inv_Sigma @ w
    return grad_loss + grad_reg

def compute_hessian(X, h, inv_Sigma):
    """
    Computes the Hessian matrix for Newton-Raphson optimization.
    Optimized to avoid creating an N x N diagonal matrix in memory.
    
    Parameters:
    X: Augmented feature matrix (N x D)
    h: Predicted probabilities (N,)
    inv_Sigma: Inverse covariance matrix for the Gaussian prior (D x D)
    """
    S = h * (1.0 - h)
    # Broadcasting X with S instead of full diagonal matrix multiplication
    hess_loss = X.T @ (X * S[:, np.newaxis])
    return hess_loss + inv_Sigma

def compute_cost(X, y, h, w, inv_Sigma):
    """
    Computes the regularized negative log-likelihood cost.
    """
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1.0 - epsilon)
    loss = -np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
    reg = 0.5 * w.T @ inv_Sigma @ w
    return loss + reg