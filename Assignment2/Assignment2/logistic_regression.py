import numpy as np
from logistic_math import sigmoid, compute_gradient, compute_hessian, compute_cost

class BayesianLogisticRegression:
    def __init__(self, lambda_reg=1e-8, max_iter=100, tol=1e-4, fit_intercept=True):
        # Initializes the Bayesian Logistic Regression classifier using MAP estimation.
        #
        # Parameters:
        # lambda_reg: Regularization parameter (equivalent to 1/\lambda, acting on the diagonal of the prior precision matrix).
        # max_iter: Maximum number of Newton-Raphson iterations.
        # tol: Convergence tolerance for the weight updates.
        # fit_intercept: Boolean indicating whether to augment data with a bias term (w_0).
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w = None
        self.inv_Sigma_ = None
        self.cost_history_ = []

    def _augment_data(self, X):
        # Augments the feature matrix with a column of ones to absorb the bias term w_0 into the weight vector w.
        # Formula: x -> [1, x_1, ..., x_d]^T
        #
        # NumPy Techniques:
        # - np.ones((X.shape[0], 1)): Creates an N x 1 column vector of ones.
        # - np.hstack([ones, X]): Horizontally concatenates the ones vector with the feature matrix X 
        #   to enable a single matrix multiplication for both bias and feature weights.
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X

    def fit(self, X, y):
        # Fits the model to the training data using the Newton-Raphson method to minimize the negative log-posterior.
        X_aug = self._augment_data(X)
        N, D = X_aug.shape
        
        # Initialize weight vector w to zeros
        self.w = np.zeros(D)
        
        # Formula: Prior precision matrix \Sigma^{-1} = (1/\lambda)I
        # NumPy: np.eye(D) generates the identity matrix I. Multiplied by lambda_reg scalar.
        self.inv_Sigma_ = np.eye(D) * self.lambda_reg
        
        if self.fit_intercept:
            # The bias parameter w_0 shifts the decision boundary w^T x + w_0 = 0 but does not 
            # control model complexity (overfitting). Therefore, it should not be regularized.
            self.inv_Sigma_[0, 0] = 0.0

        # Newton-Raphson Optimization Loop
        # Iteratively solves: H(w^{(k)}) (w^{(k+1)} - w^{(k)}) = -\nabla \ell(w^{(k)})
        for _ in range(self.max_iter):
            # Formula: a(x) = w^T x
            # NumPy: The '@' operator executes highly optimized matrix multiplication in C.
            z = X_aug @ self.w
            
            # Formula: P(C_1 | x, w) = \sigma(w^T x)
            h = sigmoid(z)
            
            grad = compute_gradient(X_aug, y, h, self.w, self.inv_Sigma_)
            hess = compute_hessian(X_aug, h, self.inv_Sigma_)
            
            # Solve the linear system H * step = grad to find the Newton-Raphson step.
            # NumPy Techniques:
            # - np.linalg.solve(hess, grad) is computationally faster and numerically more stable 
            #   than explicitly computing the inverse (np.linalg.inv(hess) @ grad).
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # Fallback: If the Hessian is singular or ill-conditioned despite the prior \Sigma^{-1},
                # use the Moore-Penrose pseudo-inverse via Singular Value Decomposition (SVD).
                step = np.linalg.pinv(hess) @ grad
                
            # Formula: w^{(k+1)} = w^{(k)} - H^{-1} \nabla \ell(w^{(k)})
            w_new = self.w - step
            
            cost = compute_cost(X_aug, y, h, self.w, self.inv_Sigma_)
            self.cost_history_.append(cost)
            
            # Convergence check: stop if the magnitude (L2 norm) of the update step is below the tolerance limit.
            # NumPy: np.linalg.norm computes the Euclidean norm of the vector.
            if np.linalg.norm(step) < self.tol:
                self.w = w_new
                break
                
            self.w = w_new
            
        return self

    def predict_proba(self, X):
        # Returns the probability estimates for the input data.
        # Formula: P(C_1 | x, w) = \sigma(w^T x)
        if self.w is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        X_aug = self._augment_data(X)
        return sigmoid(X_aug @ self.w)

    def predict(self, X, threshold=0.5):
        # Returns the binary class predictions using the specified decision boundary threshold.
        # Decision boundary formula: w^T x = 0 (when threshold is 0.5)
        # 
        # NumPy Techniques:
        # - (self.predict_proba(X) >= threshold): Creates a boolean array based on the condition.
        # - .astype(int): Casts the True/False boolean array directly into 1/0 integer class labels.
        return (self.predict_proba(X) >= threshold).astype(int)