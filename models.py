import numpy as np

def initialize_positions(num_users, dimensions, geometry='euclidean'):
    if geometry == 'euclidean':
        positions = np.random.randn(num_users, dimensions) * 0.1
    elif geometry == 'spherical':
        positions = np.random.randn(num_users, dimensions)
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        positions = positions / norms
    elif geometry == 'hyperbolic':
        positions = np.random.randn(num_users, dimensions) * 0.1
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        positions = positions / (1 + norms) * 0.9
    else:
        positions = np.random.randn(num_users, dimensions) * 0.1
    
    return positions

