import numpy as np
from logistic_math import sigmoid, compute_gradient, compute_hessian, compute_cost

class BayesianLogisticRegression:
    def __init__(self, Sigma=None, max_iter=100, tol=1e-4, fit_intercept=True):
        """
        Initializes the logistic regression classifier.
        
        Parameters:
        Sigma: Covariance matrix for the Gaussian prior. If None, assumes a weak isotropic prior.
        max_iter: Maximum number of Newton-Raphson iterations.
        tol: Convergence tolerance for the weight updates.
        fit_intercept: Boolean indicating whether to augment data with a bias term.
        """
        self.Sigma = Sigma
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.w = None
        self.inv_Sigma_ = None
        self.cost_history_ = []

    def _augment_data(self, X):
        """Augments the feature matrix with a column of ones for the bias term."""
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack([ones, X])
        return X

    def fit(self, X, y):
        """
        Fits the model to the training data using the Newton-Raphson method.
        """
        X_aug = self._augment_data(X)
        N, D = X_aug.shape
        
        self.w = np.zeros(D)
        
        # Configure the inverse covariance matrix (Precision matrix)
        if self.Sigma is None:
            # Default to a weak isotropic prior (large variance)
            feature_dim = D - 1 if self.fit_intercept else D
            self.Sigma = np.eye(feature_dim) * 100.0

        inv_Sigma_features = np.linalg.inv(self.Sigma)
        
        # Construct the full inverse Sigma matrix padded for the intercept
        self.inv_Sigma_ = np.zeros((D, D))
        if self.fit_intercept:
            # Unregularized bias term (0 penalty in the precision matrix)
            self.inv_Sigma_[1:, 1:] = inv_Sigma_features
        else:
            self.inv_Sigma_ = inv_Sigma_features

        # Newton-Raphson Optimization Loop
        for _ in range(self.max_iter):
            z = X_aug @ self.w
            h = sigmoid(z)
            
            grad = compute_gradient(X_aug, y, h, self.w, self.inv_Sigma_)
            hess = compute_hessian(X_aug, h, self.inv_Sigma_)
            
            # Pseudo-inverse used to guarantee stability if Hessian approaches singularity
            step = np.linalg.pinv(hess) @ grad
            w_new = self.w - step
            
            cost = compute_cost(X_aug, y, h, self.w, self.inv_Sigma_)
            self.cost_history_.append(cost)
            
            if np.linalg.norm(w_new - self.w) < self.tol:
                self.w = w_new
                break
                
            self.w = w_new
            
        return self

    def predict_proba(self, X):
        """Returns the probability estimates for the input data."""
        if self.w is None:
            raise ValueError("Model must be fitted before calling predict_proba.")
        X_aug = self._augment_data(X)
        return sigmoid(X_aug @ self.w)

    def predict(self, X, threshold=0.5):
        """Returns the binary class predictions using the specified threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)