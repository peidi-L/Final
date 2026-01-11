import numpy as np
import matplotlib.pyplot as plt
from distance_calculator import calculate_distances

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def simulate_signed_network(n, geometry, R, alpha_pos, alpha_neg, beta, seed=42):
    """
    Simulates a signed network based on the LSM latent space assumptions.
    
    Args:
        n (int): Number of nodes.
        geometry (str): 'Euclidean', 'Manhattan', 'Chebyshev', 'Spherical', 'Hyperbolic'.
        R (float): Radius or scale of the space (for Euclidean-like spaces).
        alpha_pos (float): Intercept for positive ties (friendship).
        alpha_neg (float): Intercept for negative ties (animosity).
        beta (float): Coefficient for distance in negative ties (polarization).
        seed (int): Random seed.
        
    Returns:
        adj_matrix (np.ndarray): n x n signed adjacency matrix (+1, -1, 0).
        positions (np.ndarray): n x d latent positions.
    """
    np.random.seed(seed)
    
    # 1. Generate Latent Positions
    positions = np.zeros((n, 2))
    
    if geometry in ['Euclidean', 'Manhattan', 'Chebyshev']:
        # Uniform distribution in square [-R, R]
        positions = np.random.uniform(-R, R, (n, 2))
        
    elif geometry == 'Spherical':
        # Points on the surface of a unit sphere (circle in 2D)
        # Generate Gaussian and normalize
        raw = np.random.normal(0, 1, (n, 2))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        positions = raw / norms
        
    elif geometry == 'Hyperbolic':
        # Points in Poincaré disk (radius < 1)
        # Naive sampling: Uniform in polar coordinates, r from sqrt(U) for uniform area
        angles = np.random.uniform(0, 2*np.pi, n)
        # Sample radius to be somewhat uniform in disk, but ensure strictly < 1
        radii = np.sqrt(np.random.uniform(0, 1, n)) * 0.95 # Stay away from boundary 1.0
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)
        
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    # 2. Generate Edges
    adj_matrix = np.zeros((n, n))
    
    dist_key_map = {
        'Euclidean': 'Euclidean',
        'Manhattan': 'Manhattan',
        'Chebyshev': 'Chebyshev',
        'Spherical': 'Spherical',
        'Hyperbolic': 'Hyperbolic'
    }
    dist_key = dist_key_map.get(geometry, 'Euclidean')
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate distance in the specific geometry
            dists = calculate_distances(positions[i], positions[j])
            d = dists[dist_key]
            
            # 3. Probabilities (Log-Odds Model)
            # P(friend)/P(null) = exp(alpha_pos - d)
            # P(foe)/P(null)    = exp(alpha_neg + beta * d)
            
            logit_pos = alpha_pos - d
            logit_neg = alpha_neg + beta * d # 'Far Foe' hypothesis: foes are far apart
            logit_zero = 0.0
            
            logits = np.array([logit_zero, logit_pos, logit_neg])
            probs = softmax(logits)
            
            # Choice: 0 (null), 1 (friend), -1 (foe)
            # mapping index 0->0, 1->1, 2->-1
            choice_idx = np.random.choice([0, 1, 2], p=probs)
            
            if choice_idx == 1:
                val = 1
            elif choice_idx == 2:
                val = -1
            else:
                val = 0
                
            adj_matrix[i, j] = val
            adj_matrix[j, i] = val

    return adj_matrix, positions

def plot_network(adj, pos, title):
    plt.figure(figsize=(8, 8))
    
    # Draw edges
    # Friends (Green)
    rows, cols = np.where(adj == 1)
    for r, c in zip(rows, cols):
        if r < c:
            plt.plot([pos[r,0], pos[c,0]], [pos[r,1], pos[c,1]], c='green', alpha=0.3, lw=0.5)
            
    # Foes (Red)
    rows, cols = np.where(adj == -1)
    for r, c in zip(rows, cols):
        if r < c:
            plt.plot([pos[r,0], pos[c,0]], [pos[r,1], pos[c,1]], c='red', alpha=0.3, lw=0.5)
            
    # Draw nodes
    plt.scatter(pos[:,0], pos[:,1], c='black', s=10, zorder=5)
    plt.title(title)
    plt.axis('equal')
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    # Test Simulation
    print("Running Euclidean Simulation...")
    adj_euc, pos_euc = simulate_signed_network(
        n=50, geometry="Euclidean", R=2.0, 
        alpha_pos=1.5, alpha_neg=-1.5, beta=0.5
    )
    plot_network(adj_euc, pos_euc, "Euclidean Network")
    
    print("Running Hyperbolic Simulation...")
    adj_hyp, pos_hyp = simulate_signed_network(
        n=50, geometry="Hyperbolic", R=1.0, 
        alpha_pos=1.5, alpha_neg=-1.5, beta=0.5
    )
    plot_network(adj_hyp, pos_hyp, "Hyperbolic Network")
    print("Done. Check generated PNGs.")
