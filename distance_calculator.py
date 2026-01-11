import numpy as np

def calculate_distances(point1, point2):
    """
    Calculates distances between two points in 5 different geometries:
    1. Euclidean (L2)
    2. Manhattan (L1)
    3. Chebyshev (L_inf)
    4. Spherical (Great Circle)
    5. Hyperbolic (Poincaré Disk)
    
    Args:
        point1 (array-like): Coordinates of the first point.
        point2 (array-like): Coordinates of the second point.
        
    Returns:
        dict: A dictionary containing distances in all 5 metrics.
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # 1. Euclidean Distance (L2)
    euclidean = np.linalg.norm(p1 - p2)
    
    # 2. Manhattan Distance (L1)
    manhattan = np.sum(np.abs(p1 - p2))
    
    # 3. Chebyshev Distance (L_inf)
    chebyshev = np.max(np.abs(p1 - p2))
    
    # 4. Spherical Distance (Great Circle)
    # Assumes points are on a unit sphere or projected onto one.
    # Angle = arccos(cosine_similarity)
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)
    
    if norm_p1 == 0 or norm_p2 == 0:
        spherical = 0.0
    else:
        # Cosine similarity
        cosine_sim = np.dot(p1, p2) / (norm_p1 * norm_p2)
        # Clip for numerical stability to avoid domain errors in arccos
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        spherical = np.arccos(cosine_sim)

    # 5. Hyperbolic Distance (Poincaré Disk Model)
    # d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
    # Points must be strictly inside the unit disk (norm < 1)
    if norm_p1 >= 1.0 or norm_p2 >= 1.0:
        # Fallback or error indication for points outside disk
        hyperbolic = float('inf') 
    else:
        sq_dist = np.sum((p1 - p2)**2)
        # Add epsilon to denominators to prevent division by zero
        denom = (1 - norm_p1**2) * (1 - norm_p2**2)
        denom = max(denom, 1e-10) 
        
        arg = 1 + 2 * sq_dist / denom
        # Clip arg to be >= 1.0 for arccosh domain
        arg = max(arg, 1.0)
        hyperbolic = np.arccosh(arg)
    
    return {
        "Euclidean": euclidean,
        "Manhattan": manhattan,
        "Chebyshev": chebyshev,
        "Spherical": spherical,
        "Hyperbolic": hyperbolic
    }

if __name__ == "__main__":
    # Simple test case
    p_a = [0.1, 0.2]
    p_b = [0.4, 0.5]
    dists = calculate_distances(p_a, p_b)
    print("Test Distances between", p_a, "and", p_b)
    for k, v in dists.items():
        print(f"{k}: {v:.4f}")
