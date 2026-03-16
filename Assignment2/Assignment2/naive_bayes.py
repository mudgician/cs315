import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier.
    """
    def __init__(self):
        self.classes = None
        self.prior_prob = None
        self.mu_nj = None       
        self.sigma_nj = None    

    def fit(self, X, Y):
        # NumPy: np.asarray converts inputs to contiguous C-arrays for optimized internal C-loop execution.
        X = np.asarray(X)
        Y = np.asarray(Y)
        N = len(Y)
        
        # NumPy: np.unique extracts the distinct class labels to define the model's dimensions.
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.prior_prob = np.zeros(n_classes)
        self.mu_nj = np.zeros((n_classes, n_features))
        self.sigma_nj = np.zeros((n_classes, n_features))
        
        # NumPy: np.var with axis=0 calculates variance across rows for each column.
        global_var = np.var(X, axis=0)
        epsilon = 1e-9 * np.max(global_var) + 1e-9
        
        for i, c in enumerate(self.classes):
            # NumPy: Boolean indexing (Y == c) creates a boolean mask to filter rows without Python-level loops.
            X_i = X[Y == c]
            
            # Formula: P(C_j) = N_j / N
            self.prior_prob[i] = len(X_i) / N
            
            # Formula: \mu_{nj} = (1/N_j) \sum_{x \in C_j} x_{nj}
            # NumPy: axis=0 vectorizes the summation and division across all features simultaneously.
            self.mu_nj[i] = np.mean(X_i, axis=0)
            
            # Formula: \sigma_{nj}^2 = (1/N_j) \sum_{x \in C_j} (x_{nj} - \mu_{nj})^2
            # NumPy: np.var computes population variance efficiently. Epsilon is added to prevent log(0) in predict.
            self.sigma_nj[i] = np.var(X_i, axis=0) + epsilon

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        
        log_unnormalized_posterior = np.zeros((n_samples, n_classes))
        
        # Formula: \ln P(C_j)
        # NumPy: Vectorized natural logarithm applied to the entire 1D array of priors.
        log_prior = np.log(self.prior_prob + 1e-9)
        
        for i in range(n_classes):
            mean = self.mu_nj[i]
            var = self.sigma_nj[i]
            
            # Formula: \ln p(x | C_j) = \sum_{n=1}^d \left[ -\frac{1}{2} \ln(\sigma_{nj}^2) - \frac{(x_n - \mu_{nj})^2}{2\sigma_{nj}^2} \right]
            # NumPy Broadcasting: (X_test - mean) subtracts a 1D array of shape (n_features,) from a 2D array 
            # of shape (n_samples, n_features). NumPy automatically replicates 'mean' across all sample rows.
            # NumPy: np.sum with axis=1 collapses the feature dimension, summing log-likelihoods per sample.
            class_log_likelihood = np.sum(-0.5 * np.log(var) - (0.5 * ((X_test - mean) ** 2) / var), axis=1)
            
            # Formula: \ln P(C_j | x) \propto \ln p(x | C_j) + \ln P(C_j)
            log_unnormalized_posterior[:, i] = class_log_likelihood + log_prior[i]
            
        # Formula: \hat{y} = \arg\max_{C_j} [ \ln p(x | C_j) + \ln P(C_j) ]
        # NumPy: np.argmax with axis=1 returns the index of the maximum log posterior for each row (sample).
        predicted_indices = np.argmax(log_unnormalized_posterior, axis=1)
        
        # NumPy: Fancy indexing maps the array of maximum indices back to the original class labels.
        return self.classes[predicted_indices]