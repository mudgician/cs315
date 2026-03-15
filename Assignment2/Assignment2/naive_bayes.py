import numpy as np

class nb:
    def __init__(self):
        self.classes = None
        self.prior_prob = None
        self.u_nj = None
        self.sigma_nj = None

    def fit(self, X, Y):
        # Enforce numpy arrays
        X = np.asarray(X)
        Y = np.asarray(Y)
        N = len(Y)
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        #Initialisation of parameter arrays
        self.prior_prob = np.zeros(n_classes)
        self.u_nj = np.zeros((n_classes, n_features))
        self.sigma_nj = np.zeros((n_classes, n_features))
        
        # Compute epsilon for variance smoothing
        global_var = np.var(X, axis=0)
        epsilon = 1e-9 * np.max(global_var) + 1e-9
        
        for i, c in enumerate(self.classes):
            X_i = X[Y == c]
            
            # Phase 1: Prior probability
            self.prior_prob[i] = len(X_i) / N
            
            # Phase 2: Parameter Estimation
            self.u_nj[i] = np.mean(X_i, axis=0)
            self.sigma_nj[i] = np.var(X_i, axis=0) + epsilon

    def predict(self, X_test):
        # Enforce numpy arrays
        X_test = np.asarray(X_test)
        n_samples = X_test.shape[0]
        n_classes = len(self.classes)
        
        unnormalized_posterior = np.zeros((n_samples, n_classes))
        log_prior = np.log(self.prior_prob + 1e-9)
        
        # Phase 3 & 4: Density Calculation and Classification
        for i in range(n_classes):
            mean = self.u_nj[i]
            var = self.sigma_nj[i]
            # The constant -0.5 * np.log(2 * np.pi) is removed as it does not affect argmax
            class_log_likelihood = np.sum(-0.5 * np.log(var) - (0.5 * ((X_test - mean) ** 2) / var), axis=1)
            
            unnormalized_posterior[:, i] = class_log_likelihood + log_prior[i]
            
        predicted_indices = np.argmax(unnormalized_posterior, axis=1)
        
        return self.classes[predicted_indices]