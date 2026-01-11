import numpy as np
import matplotlib.pyplot as plt

# 1. Geometry and probability functions

def get_distance(p1, p2, geometry):
    """Calculates distance between two points based on the chosen geometry."""
    
    # Euclidean distance - straight line between points
    if geometry == 'Euclidean':
        return np.linalg.norm(p1 - p2)
    
    # Spherical distance - shortest path on a sphere surface
    elif geometry == 'Spherical':
        # Project points to the surface
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        dot_product = np.dot(p1, p2)
        # Clip values to handle floating point precision issues
        return np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Hyperbolic distance - using the Poincaré Disk model
    elif geometry == 'Hyperbolic':
        # Points are restricted to the unit disk
        sq_dist = np.sum((p1 - p2)**2)
        norm_1 = np.sum(p1**2)
        norm_2 = np.sum(p2**2)
        
        # Scaling formula for hyperbolic space expansion
        numerator = 2 * sq_dist
        denominator = (1 - norm_1) * (1 - norm_2) + 1e-10 # small epsilon for stability
        
        arg = 1 + numerator / denominator
        return np.arccosh(np.maximum(arg, 1.0))
    
    return 0.0

def get_probabilities(distance, alpha_pos, alpha_neg, beta):
    """
    Log-odds rules:
    - Friendship decays with distance
    - Animosity grows with distance (polarization)
    """
    
    # Get raw scores for each tie type
    score_friend = alpha_pos - distance
    score_enemy  = alpha_neg + (beta * distance)
    score_none   = 0.0  # Baseline reference
    
    # Use softmax to convert scores to probabilities
    exp_friend = np.exp(score_friend)
    exp_enemy  = np.exp(score_enemy)
    exp_none   = np.exp(score_none)
    
    total = exp_friend + exp_enemy + exp_none
    
    p_friend = exp_friend / total
    p_enemy  = exp_enemy / total
    p_none   = exp_none / total
    
    return [p_none, p_friend, p_enemy]


# 2. Main simulation logic

def run_simulation(n_people, geometry):
    np.random.seed(37) # Ensures the random numbers are the same every time
    
    # Initialize node positions
    positions = np.zeros((n_people, 2))
    
    if geometry == 'Euclidean':
        # Uniform sampling in a 2x2 square
        positions = np.random.uniform(-2, 2, (n_people, 2))
        
    elif geometry == 'Spherical':
        # Sample points on the unit circle
        raw_points = np.random.normal(0, 1, (n_people, 2))
        norms = np.linalg.norm(raw_points, axis=1, keepdims=True)
        positions = raw_points / norms
        
    elif geometry == 'Hyperbolic':
        # Sample points inside the Poincaré disk
        angles = np.random.uniform(0, 2*np.pi, n_people)
        radii = np.sqrt(np.random.uniform(0, 0.95, n_people)) 
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)
        #Computer screens and plots use a grid (X and Y). These lines use trigonometry to convert the "direction and distance" into a coordinate the computer can actually draw.

    # Generate edges
    edges = []
    
    # Model parameters
    # alpha_pos: base friendship rate
    # alpha_neg: base enmity rate
    # beta: polarization strength
    alpha_pos, alpha_neg, beta = 2.5, -1.0, 0.8
    
    # Iterate through all pairs to decide relationships
    for i in range(n_people):
        for j in range(i + 1, n_people):
            
            d = get_distance(positions[i], positions[j], geometry)
            probs = get_probabilities(d, alpha_pos, alpha_neg, beta)
            
            # Weighted random choice based on calculated probabilities
            choice = np.random.choice([0, 1, 2], p=probs)
            
            if choice == 1:
                edges.append((i, j, 1))  # Friend
            elif choice == 2:
                edges.append((i, j, -1)) # Enemy

    return positions, edges

# 3. Graph visualization

def draw_graph(positions, edges, title, geometry):
    # Increase figure size for better clarity
    plt.figure(figsize=(10, 10))
    
    # Separate edges for easier legend handling and layered plotting
    friends = [e for e in edges if e[2] == 1]
    foes = [e for e in edges if e[2] == -1]
    
    # Plot foes (Red) first so they sit in the background
    for i, (u, v, sign) in enumerate(foes):
        p1, p2 = positions[u], positions[v]
        label = "Foes (Negative)" if i == 0 else ""
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', alpha=0.2, linewidth=0.8, zorder=1, label=label)
        
    # Plot friends (Green) on top of foes
    for i, (u, v, sign) in enumerate(friends):
        p1, p2 = positions[u], positions[v]
        label = "Friends (Positive)" if i == 0 else ""
        # Make all green lines lighter (more transparent)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green', alpha=0.15, linewidth=1.2, zorder=2, label=label)
        
    # Plot nodes (People) on top of all lines
    plt.scatter(positions[:, 0], positions[:, 1], c='black', s=30, zorder=3, label='People (Nodes)')
    
    # Add boundary indicator for non-Euclidean spaces
    if geometry != 'Euclidean':
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', label='Geometry Boundary')
        plt.gca().add_patch(circle)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
    plt.title(f"{title}\nTotal: {len(edges)} | Friends: {len(friends)} | Foes: {len(foes)}", fontsize=14)
    plt.axis('equal')
    
    # Add a legend to define what the colors mean
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # Save the output images
    filename = f"{geometry}_simulation.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()

# 4. Execution entry point

if __name__ == "__main__":
    # Run for 100 people in all 3 geometries
    geometries = ['Euclidean', 'Spherical', 'Hyperbolic']

    for geo in geometries:
        print(f"Simulating {geo}...")
        pos, links = run_simulation(100, geo)
        draw_graph(pos, links, f"{geo} Party", geo)

    print("Done! You have successfully built a signed network simulation.")
