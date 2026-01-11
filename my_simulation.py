import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. THE MATH (Geometry & Probability)
# ==========================================

def get_distance(p1, p2, geometry):
    """Measures distance between two points based on the 'Room' they are in."""
    
    # EUCLIDEAN (The Square Room) - Standard straight line
    if geometry == 'Euclidean':
        return np.linalg.norm(p1 - p2)
    
    # SPHERICAL (The Beach Ball) - Angle between points
    elif geometry == 'Spherical':
        # Normalize just in case
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        dot_product = np.dot(p1, p2)
        # Clip to avoid errors if dot_product is slightly > 1.0 due to math rounding
        return np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # HYPERBOLIC (The Magical Tree) - Poincaré Disk formula
    elif geometry == 'Hyperbolic':
        # Points must be inside the disk (norm < 1)
        sq_dist = np.sum((p1 - p2)**2)
        norm_1 = np.sum(p1**2)
        norm_2 = np.sum(p2**2)
        
        # The special formula that makes space expand
        numerator = 2 * sq_dist
        denominator = (1 - norm_1) * (1 - norm_2) + 1e-10 # epsilon for safety
        
        arg = 1 + numerator / denominator
        return np.arccosh(np.maximum(arg, 1.0))
    
    return 0.0

def get_probabilities(distance, alpha_pos, alpha_neg, beta):
    """
    The Rules of Attraction (Log-Odds):
    1. Friendship: drops as distance grows (alpha_pos - d)
    2. Animosity:  GROWS as distance grows (alpha_neg + beta * d)
    """
    
    # Calculate the "score" (Logits) for each outcome
    score_friend = alpha_pos - distance
    score_enemy  = alpha_neg + (beta * distance)
    score_none   = 0.0  # Baseline
    
    # Convert scores to probabilities (Softmax)
    # We use np.exp() to turn scores into positive numbers
    exp_friend = np.exp(score_friend)
    exp_enemy  = np.exp(score_enemy)
    exp_none   = np.exp(score_none)
    
    total = exp_friend + exp_enemy + exp_none
    
    p_friend = exp_friend / total
    p_enemy  = exp_enemy / total
    p_none   = exp_none / total
    
    return [p_none, p_friend, p_enemy]

# ==========================================
# 2. THE SIMULATION (The Party)
# ==========================================

def run_simulation(n_people, geometry):
    np.random.seed(42) # Ensures the random numbers are the same every time
    
    # A. PLACE PEOPLE IN THE ROOM
    positions = np.zeros((n_people, 2))
    
    if geometry == 'Euclidean':
        # Random x,y in a 2x2 square
        positions = np.random.uniform(-2, 2, (n_people, 2))
        
    elif geometry == 'Spherical':
        # Random points on a circle (normalized)
        raw_points = np.random.normal(0, 1, (n_people, 2))
        norms = np.linalg.norm(raw_points, axis=1, keepdims=True)
        positions = raw_points / norms
        
    elif geometry == 'Hyperbolic':
        # Random points inside a circle (radius < 1)
        # We put them slightly away from the edge (0.95) to keep math stable
        angles = np.random.uniform(0, 2*np.pi, n_people)
        radii = np.sqrt(np.random.uniform(0, 0.95, n_people)) 
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)

    # B. DECIDE RELATIONSHIPS
    edges = []
    
    # Parameters (You can tweak these!)
    # alpha_pos=2.5: High base chance of friendship
    # alpha_neg=-1.0: Low base chance of enemies
    # beta=0.8: Distance makes enemies VERY likely
    alpha_pos, alpha_neg, beta = 2.5, -1.0, 0.8
    
    # Check every pair of people
    for i in range(n_people):
        for j in range(i + 1, n_people):
            
            # Measure distance
            d = get_distance(positions[i], positions[j], geometry)
            
            # Get probabilities
            probs = get_probabilities(d, alpha_pos, alpha_neg, beta)
            
            # Roll the dice (0=None, 1=Friend, 2=Enemy)
            choice = np.random.choice([0, 1, 2], p=probs)
            
            if choice == 1:
                edges.append((i, j, 1))  # Friend
            elif choice == 2:
                edges.append((i, j, -1)) # Enemy

    return positions, edges

# ==========================================
# 3. VISUALIZATION (Taking the Photo)
# ==========================================

def draw_graph(positions, edges, title, geometry):
    plt.figure(figsize=(6, 6))
    
    # Draw connections
    for (u, v, sign) in edges:
        p1, p2 = positions[u], positions[v]
        color = 'green' if sign == 1 else 'red'
        # Draw line with low opacity (alpha=0.2) so it's not messy
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.3, linewidth=1)
        
    # Draw people
    plt.scatter(positions[:, 0], positions[:, 1], c='black', s=20, zorder=5)
    
    # Draw circle boundary for non-Euclidean
    if geometry != 'Euclidean':
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        
    plt.title(f"{title}\n({len(edges)} connections)")
    plt.axis('equal')
    # Save the figure instead of showing it, for headless environments
    filename = f"{geometry}_simulation.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")
    plt.close()

# ==========================================
# 4. RUN IT!
# ==========================================

if __name__ == "__main__":
    # Run for 100 people in all 3 geometries
    geometries = ['Euclidean', 'Spherical', 'Hyperbolic']

    for geo in geometries:
        print(f"Simulating {geo}...")
        pos, links = run_simulation(100, geo)
        draw_graph(pos, links, f"{geo} Party", geo)

    print("Done! You have successfully built a signed network simulation.")
