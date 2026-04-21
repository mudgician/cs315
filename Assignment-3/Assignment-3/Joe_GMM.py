import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Joe_kmeans import kmeans

def eval_gauss(x, u, s):
    """
    Evaluates the multivariate Gaussian probability density function.
    x: (N, D) array of samples
    u: (D,) array representing the mean vector
    s: (D, D) array representing the covariance matrix
    """
    D = x.shape[1]
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    norm_const = 1.0 / (np.power((2 * np.pi), D / 2.0) * np.sqrt(det))
    x_centered = x - u
    
    exponent = np.exp(-0.5 * np.sum(np.dot(x_centered, inv) * x_centered, axis=1))
    return norm_const * exponent

def gamma(x, p, u, s):
    """
    E-step: Calculates the responsibilities (gamma).
    """
    N, D = x.shape
    K = len(p)
    gamma_var = np.zeros((N, K), dtype=float)

    for k in range(K):
        gamma_var[:, k] = p[k] * eval_gauss(x, u[k], s[k])

    sum_gamma = np.sum(gamma_var, axis=1, keepdims=True)
    gamma_var = gamma_var / (sum_gamma + 1e-8)
    
    return gamma_var

def new_mean(x, gamma_var):
    """
    M-step: Updates the component means.
    """
    N_k = np.sum(gamma_var, axis=0)
    u_new = np.dot(gamma_var.T, x) / N_k[:, np.newaxis]
    return u_new

def new_covar(x, gamma_var, u_new):
    """
    M-step: Updates the component covariance matrices.
    """
    N, D = x.shape
    K = gamma_var.shape[1]
    s_new = np.zeros((K, D, D))
    N_k = np.sum(gamma_var, axis=0)

    for k in range(K):
        x_centered = x - u_new[k]
        weighted_cov = np.dot(x_centered.T, (gamma_var[:, k:k+1] * x_centered)) / N_k[k]
        s_new[k] = weighted_cov + np.eye(D) * 1e-6
        
    return s_new

def new_priors(gamma_var):
    """
    M-step: Updates the component priors (weights).
    """
    N = gamma_var.shape[0]
    N_k = np.sum(gamma_var, axis=0)
    p_new = N_k / N
    return p_new

def plot_gmm_2d(x, u, s, gamma_var, iteration):
    """
    Plots the 2D GMM state. Draws scatter points colored by responsibility (soft assignment)
    and ellipses representing the 95% confidence interval of the covariances.
    """
    if x.shape[1] != 2:
        print("Plotting skipped: Data is not 2-dimensional.")
        return

    plt.clf()
    ax = plt.gca()
    K = len(u)
    
    # Extract RGB values for K distinct colors (drop the alpha channel)
    base_colors = plt.get_cmap('tab10')(np.linspace(0, 1, K))[:, :3]
    
    # Map responsibilities to colors via dot product: (N, K) @ (K, 3) -> (N, 3)
    point_colors = np.dot(gamma_var, base_colors)
    
    # Plot all data points with soft assignment blending
    ax.scatter(x[:, 0], x[:, 1], s=15, c=point_colors, alpha=0.7, zorder=1)

    # Chi-Square 95% confidence interval multiplier for 2 degrees of freedom
    scale_factor = 2.4477 

    for k in range(K):
        mean = u[k]
        covar = s[k]
        
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # Calculate width and height using the 95% CI scale factor
        width, height = 2 * scale_factor * np.sqrt(np.maximum(eigenvalues, 1e-12))
        
        ell = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                      edgecolor='black', facecolor='none', linewidth=2, zorder=2)
        ax.add_artist(ell)
        ax.scatter(mean[0], mean[1], marker='X', color='black', s=100, zorder=3)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(f"Iteration: {iteration}")
    plt.pause(0.5)

def fit_gmm(x, K, max_iters=100, tol=1e-4, plot_interval=0):
    """
    Full Expectation-Maximization execution loop with K-Means initialization.
    x: (N, D) array of samples
    K: number of mixture components
    plot_interval: If > 0, plots the GMM state every N iterations. (Requires D=2)
    gamma_var: (N, K) array of responsibilities
    """
    N, D = x.shape
    
    # Initialize parameters
    p = np.ones(K) / K
    
    # Use K-Means for initial cluster centroids
    u = kmeans(x, K)
    
    # Initialize covariances as empirical covariance of the full dataset
    empirical_cov = np.cov(x, rowvar=False) + np.eye(D) * 1e-6
    s = np.array([empirical_cov for _ in range(K)])
    
    log_likelihood = 0
    
    if plot_interval > 0:
        plt.ion()
        plt.figure(figsize=(8, 6))
    
    for i in range(max_iters):
        # E-step
        gamma_var = gamma(x, p, u, s)
        
        # Plotting
        if plot_interval > 0 and i % plot_interval == 0:
            plot_gmm_2d(x, u, s, gamma_var, iteration=i)
            
        # Convergence Check
        ll_new = np.sum(np.log(np.sum([p[k] * eval_gauss(x, u[k], s[k]) for k in range(K)], axis=0) + 1e-8))
        if np.abs(ll_new - log_likelihood) < tol:
            if plot_interval > 0:
                print(f"Converged at iteration {i}")
            break
        log_likelihood = ll_new
        
        # M-step
        u_new = new_mean(x, gamma_var)
        s_new = new_covar(x, gamma_var, u_new)
        p_new = new_priors(gamma_var)
        
        u, s, p = u_new, s_new, p_new
        
    if plot_interval > 0:
        plot_gmm_2d(x, u, s, gamma_var, iteration="Final")
        plt.ioff()
        plt.show()
        
    return p, u, s, gamma_var