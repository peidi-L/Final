import numpy as np

def initialize_positions(num_users, dimensions, geometry='euclidean'):
    # This function creates random starting positions for nodes in the latent space
    # num_users is how many nodes we have
    # dimensions is how many dimensions (like 2 for 2D)
    # geometry tells us what type of space to use
    
    if geometry == 'euclidean':
        # Random positions in Euclidean space
        positions = np.random.randn(num_users, dimensions) * 0.1
    elif geometry == 'spherical':
        # Random positions on a sphere, normalize them to be on the unit sphere
        positions = np.random.randn(num_users, dimensions)
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        positions = positions / norms
    elif geometry == 'hyperbolic':
        # Random positions in Poincaré disk, keep them within the unit circle
        positions = np.random.randn(num_users, dimensions) * 0.1
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        # Scale to keep within unit disk
        positions = positions / (1 + norms) * 0.9
    else:
        # Default to Euclidean if we don't recognize the geometry type
        positions = np.random.randn(num_users, dimensions) * 0.1
    
    # Return the array of positions, shape is (num_users, dimensions)
    return positions

