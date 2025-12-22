import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def poincare_distance(u, v, epsilon=1e-5):
    """
    Calculates distance in the Poincaré disk model.
    Formula: d(u, v) = arccosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    """
    sq_dist = np.sum((u - v)**2)
    sq_norm_u = np.sum(u**2)
    sq_norm_v = np.sum(v**2)
    
    # Clip norms to ensure stability (must be < 1)
    sq_norm_u = np.clip(sq_norm_u, 0, 1 - epsilon)
    sq_norm_v = np.clip(sq_norm_v, 0, 1 - epsilon)
    
    val = 1 + 2 * sq_dist / ((1 - sq_norm_u) * (1 - sq_norm_v))
    return np.arccosh(np.maximum(val, 1.0 + epsilon))

def generate_synthetic_data(n_nodes, geometry='euclidean', alpha=2.0):
    """
    Generates a synthetic network with known ground-truth positions.
    """
    np.random.seed(42)
    dim = 2
    
    if geometry == 'euclidean':
        Z_true = np.random.normal(0, 1, (n_nodes, dim))
        get_dist = lambda u, v: np.linalg.norm(u - v)
        
    elif geometry == 'hyperbolic':
        # Generate points in a disk (radius < 1)
        Z_true = np.zeros((n_nodes, dim))
        for i in range(n_nodes):
            r = np.random.uniform(0, 0.9)  # Keep away from boundary
            theta = np.random.uniform(0, 2*np.pi)
            Z_true[i] = [r*np.cos(theta), r*np.sin(theta)]
        get_dist = poincare_distance

    # Generate Adjacency Matrix
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = get_dist(Z_true[i], Z_true[j])
            # Model: logit(p) = alpha - distance
            prob = 1 / (1 + np.exp(-(alpha - d)))
            if np.random.random() < prob:
                adj[i, j] = adj[j, i] = 1
                
    return Z_true, adj

def neg_log_likelihood(params, adj, n_nodes):
    """
    The function we want to minimize to find latent positions.
    """
    Z = params.reshape((n_nodes, 2))
    ll = 0
    alpha = 2.0  # Assume we know alpha for simplicity
    epsilon = 1e-9
    
    # Calculate for all pairs (naive loop for clarity)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = np.linalg.norm(Z[i] - Z[j])  # Fitting Euclidean for now
            logit = alpha - d
            prob = 1 / (1 + np.exp(-logit))
            
            if adj[i, j] == 1:
                ll += np.log(prob + epsilon)
            else:
                ll += np.log(1 - prob + epsilon)
    return -ll

# --- RUN THE STUDY ---
if __name__ == "__main__":
    n_nodes = 30
    
    print("1. Generating Synthetic Data (Euclidean Ground Truth)...")
    Z_true, adj = generate_synthetic_data(n_nodes=n_nodes, geometry='euclidean')
    print(f"   Generated {n_nodes} nodes with {int(np.sum(adj)/2)} edges.")

    print("2. Attempting to Recover Positions (Optimization)...")
    Z_init = np.random.normal(0, 0.1, (n_nodes, 2)).flatten()
    res = minimize(neg_log_likelihood, Z_init, args=(adj, n_nodes), 
                   method='L-BFGS-B', options={'maxiter': 200})

    print(f"3. Results: Converged = {res.success}")
    
    # Evaluate recovery quality
    Z_est = res.x.reshape((n_nodes, 2))
    
    # Compute distance matrices
    D_true = np.zeros((n_nodes, n_nodes))
    D_est = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            D_true[i, j] = D_true[j, i] = np.linalg.norm(Z_true[i] - Z_true[j])
            D_est[i, j] = D_est[j, i] = np.linalg.norm(Z_est[i] - Z_est[j])
    
    # Compute probabilities from estimated distances
    alpha = 2.0
    est_probs = 1 / (1 + np.exp(-(alpha - D_est)))
    upper_mask = np.triu(np.ones((n_nodes, n_nodes)), k=1).astype(bool)
    
    auc_score = roc_auc_score(adj[upper_mask], est_probs[upper_mask])
    print(f"   AUC Score: {auc_score:.4f}")
    
    # Test hyperbolic recovery
    print("\n4. Testing Hyperbolic Geometry Recovery...")
    Z_true_hyp, adj_hyp = generate_synthetic_data(n_nodes=30, geometry='hyperbolic')
    print(f"   Generated {int(np.sum(adj_hyp)/2)} edges from hyperbolic ground truth.")
    
    print("5. Attempting to Recover Hyperbolic Positions...")
    Z_init_hyp = np.random.uniform(-0.5, 0.5, (n_nodes, 2))
    # Ensure initial positions are within disk
    norms = np.linalg.norm(Z_init_hyp, axis=1)
    Z_init_hyp = Z_init_hyp / (norms[:, np.newaxis] + 1e-6) * 0.8
    Z_init_hyp = Z_init_hyp.flatten()
    
    def neg_log_likelihood_hyp(params, adj, n_nodes):
        """Negative log-likelihood with hyperbolic distance constraint."""
        Z = params.reshape((n_nodes, 2))
        # Penalty if outside disk
        norms = np.linalg.norm(Z, axis=1)
        if np.any(norms >= 1.0):
            return 1e9
        
        ll = 0
        alpha = 2.0
        epsilon = 1e-9
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                d = poincare_distance(Z[i], Z[j])
                logit = alpha - d
                prob = 1 / (1 + np.exp(-logit))
                
                if adj[i, j] == 1:
                    ll += np.log(prob + epsilon)
                else:
                    ll += np.log(1 - prob + epsilon)
        return -ll
    
    res_hyp = minimize(neg_log_likelihood_hyp, Z_init_hyp, args=(adj_hyp, n_nodes),
                      method='L-BFGS-B', bounds=[(-0.99, 0.99)] * (n_nodes * 2),
                      options={'maxiter': 200})
    
    print(f"   Hyperbolic Recovery: Converged = {res_hyp.success}")
    if res_hyp.success:
        Z_est_hyp = res_hyp.x.reshape((n_nodes, 2))
        # Compute AUC for hyperbolic
        D_est_hyp = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                D_est_hyp[i, j] = D_est_hyp[j, i] = poincare_distance(Z_est_hyp[i], Z_est_hyp[j])
        
        est_probs_hyp = 1 / (1 + np.exp(-(2.0 - D_est_hyp)))
        auc_hyp = roc_auc_score(adj_hyp[upper_mask], est_probs_hyp[upper_mask])
        print(f"   Hyperbolic AUC Score: {auc_hyp:.4f}")

