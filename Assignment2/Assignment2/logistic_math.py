import numpy as np

def sigmoid(z):

#    Computes the logistic sigmoid function.
#    Formula: \sigma(z) = 1 / (1 + \exp(-z))
#    
#    NumPy Techniques:
#    - np.clip(z, -500, 500): Bounds the input array values. This prevents 
#      RuntimeWarning: overflow encountered in exp, which occurs when z 
#      is a large negative number.
#    - np.exp(-z): Vectorized exponential function applied element-wise.

    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_gradient(X, y, h, w, inv_Sigma):

#    Computes the gradient of the regularized negative log-likelihood.
#    Formula: \nabla \ell(w) = \sum_{n=1}^N (\sigma(w^T x_n) - y_n) x_n + \Sigma^{-1} w
#             (where \Sigma^{-1} represents \frac{1}{\lambda}I)
#    
#    Parameters:
#    X: Augmented feature matrix (N x D)
#    y: True labels (N,)
#    h: Predicted probabilities \sigma(w^T x) (N,)
#    w: Weight vector (D,)
#    inv_Sigma: Inverse covariance matrix for the Gaussian prior (D x D)
    
#    NumPy Techniques:
#    - (h - y): Computes the element-wise prediction error \sigma_n - y_n.
#    - X.T @ (h - y): The matrix multiplication (@) projects the N-dimensional 
#      error vector back into the D-dimensional feature space. This operation 
#      implicitly executes the \sum_{n=1}^N summation without Python-level loops.

    grad_loss = X.T @ (h - y)
    grad_reg = inv_Sigma @ w
    return grad_loss + grad_reg

def compute_hessian(X, h, inv_Sigma):
#    Computes the Hessian matrix for Newton-Raphson optimization.
#    Formula: H(w) = \sum_{n=1}^N \sigma_n(1 - \sigma_n) x_n x_n^T + \Sigma^{-1}
#    
#    Parameters:
#    X: Augmented feature matrix (N x D)
#    h: Predicted probabilities (N,)
#    inv_Sigma: Inverse covariance matrix for the Gaussian prior (D x D)
#    
#    NumPy Techniques (Highly Optimized):
#    - S = h * (1.0 - h): Calculates the Bernoulli variance \sigma_n(1 - \sigma_n) 
#      element-wise for all N samples.
#    - S[:, np.newaxis]: Reshapes the 1D array of shape (N,) to a 2D column vector 
#      of shape (N, 1). 
#    - (X * S[:, np.newaxis]): Utilizes NumPy broadcasting to multiply each row of X 
#      by its corresponding scalar variance in S. This avoids the severe memory 
#      overhead of constructing an (N x N) diagonal matrix (np.diag(S)), bringing 
#      space complexity down from O(N^2) to O(N * D).
#    - X.T @ (...): Completes the outer product and summation over N.

    S = h * (1.0 - h)
    # Broadcasting X with S instead of full diagonal matrix multiplication
    hess_loss = X.T @ (X * S[:, np.newaxis])
    return hess_loss + inv_Sigma

def compute_cost(X, y, h, w, inv_Sigma):
#    Computes the Maximum A Posteriori (MAP) cost function (regularized negative log-likelihood).
#    Formula: \ell(w) = -\sum_{n=1}^N [y_n \ln(h_n) + (1 - y_n) \ln(1 - h_n)] + \frac{1}{2} w^T \Sigma^{-1} w
#    
#    NumPy Techniques:
#    - np.clip(h, epsilon, 1.0 - epsilon): Strictly bounds the predicted probabilities 
#      slightly away from 0.0 and 1.0 to prevent np.log(0) resulting in -inf or NaN.
#    - np.sum(...): Aggregates the vector of cross-entropy losses into a single scalar sum, 
#      matching the exact formulation in the chapter notes.
#    - w.T @ inv_Sigma @ w: Computes the quadratic penalty term \frac{1}{2\lambda}w^T w 
#      using chained matrix multiplication.

    epsilon = 1e-15
    h = np.clip(h, epsilon, 1.0 - epsilon)
    loss = -np.sum(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
    reg = 0.5 * w.T @ inv_Sigma @ w
    return loss + reg