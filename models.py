import numpy as np

def initialize_positions(num_users, dimensions=2, geometry='euclidean', seed=42):
    """
    Initializes latent positions based on the geometry type.
    Fixed seed ensures reproducibility for evaluation.
    """
    np.random.seed(seed)
    
    if geometry == 'euclidean':
        # Standard normal distribution centered at 0
        # Scale = 0.1 to keep gradients stable in early iterations
        return np.random.randn(num_users, dimensions) * 0.1
        
    elif geometry == 'hyperbolic':
        # Poincaré disk initialization: points must be strictly within norm < 1
        # We project uniform random points into the disk
        radius = np.sqrt(np.random.rand(num_users)) * 0.9  # Cap at 0.9 radius
        angle = np.random.rand(num_users) * 2 * np.pi
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.column_stack((x, y))
        
    else:
        raise NotImplementedError(f"Geometry {geometry} not yet implemented")

def safe_distance(pos_i, pos_j, geometry='euclidean', epsilon=1e-5):
    """
    Compute pairwise distances with stability checks.
    """
    diff = pos_i - pos_j
    dist = np.linalg.norm(diff, axis=1)
    
    # Avoid division by zero in gradients
    return np.maximum(dist, epsilon)
