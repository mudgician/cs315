import numpy as np

class nb:
    def __init__(self, X, Y):
        # Automatically reshape 1D arrays to 2D (samples, 1 feature)
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = X
            
        self.Y = Y
        self.N = len(Y)
        self.classes = np.unique(Y)
        
    def fit(self):
        n_classes = len(self.classes)
        n_features = self.X.shape[1]
        
        self.prior_prob = np.zeros(n_classes)
        self.u_nj = np.zeros((n_classes, n_features))
        self.sigma_nj = np.zeros((n_classes, n_features))
        
        for i in range(n_classes):
            # Isolate data for class C_j
            X_i = self.X[self.Y == self.classes[i]]
            N_j = len(X_i)
            
            # Phase 1: Prior probability
            self.prior_prob[i] = N_j / self.N
            
            # Phase 2: Parameter Estimation (Mean and Variance per feature)
            self.u_nj[i] = np.mean(X_i, axis=0)
            self.sigma_nj[i] = np.var(X_i, axis=0) + 1e-9  # Epsilon for numerical stability

    def predict(self, X_test):
        # Automatically reshape 1D test arrays to 2D
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
            
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        n_features = X_test.shape[1]
        
        # Phase 3: Density Calculation
        log_pdf = np.zeros((n_samples, n_classes, n_features))
        
        for i in range(n_classes):
            mean = self.u_nj[i]
            var = self.sigma_nj[i]
            
            # Calculate log PDF for each feature independently
            log_pdf[:, i, :] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - (0.5 * ((X_test - mean) ** 2) / var)
            
        # Phase 4: Posterior Computation and Classification
        joint_log_likelihood = np.sum(log_pdf, axis=2)
        log_prior = np.log(self.prior_prob + 1e-9)
        unnormalized_posterior = joint_log_likelihood + log_prior
        
        predicted_indices = np.argmax(unnormalized_posterior, axis=1)
        
        return self.classes[predicted_indices]