import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. GEOMETRY & PROBABILITY (The Physics)
# ==========================================

def get_distance(p1, p2, geometry):
    """Calculates distance based on the geometry."""
    if geometry == 'Euclidean':
        return np.linalg.norm(p1 - p2)
    elif geometry == 'Spherical':
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)
        dot = np.dot(p1, p2)
        return np.arccos(np.clip(dot, -1.0, 1.0))
    elif geometry == 'Hyperbolic':
        sq_dist = np.sum((p1 - p2)**2)
        norm_1 = np.sum(p1**2)
        norm_2 = np.sum(p2**2)
        numerator = 2 * sq_dist
        denominator = (1 - norm_1) * (1 - norm_2) + 1e-10
        arg = 1 + numerator / denominator
        return np.arccosh(np.maximum(arg, 1.0))
    return 0.0

def get_probabilities(distance, alpha_pos, alpha_neg, beta):
    """Calculates the probability of Friend/Foe/None based on distance."""
    score_friend = alpha_pos - distance
    score_enemy  = alpha_neg + (beta * distance)
    score_none   = 0.0
    
    # Softmax
    exp_friend = np.exp(score_friend)
    exp_enemy  = np.exp(score_enemy)
    exp_none   = np.exp(score_none)
    total = exp_friend + exp_enemy + exp_none
    
    return {
        'friend': exp_friend / total,
        'enemy': exp_enemy / total,
        'none': exp_none / total
    }

# ==========================================
# 2. GENERATOR (The Simulation)
# ==========================================

def simulate_network(n_nodes, geometry, params):
    """Creates ground truth positions and edges."""
    np.random.seed(42)
    positions = np.zeros((n_nodes, 2))
    
    # Place nodes
    if geometry == 'Euclidean':
        positions = np.random.uniform(-2, 2, (n_nodes, 2))
    elif geometry == 'Spherical':
        raw = np.random.normal(0, 1, (n_nodes, 2))
        positions = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    elif geometry == 'Hyperbolic':
        angles = np.random.uniform(0, 2*np.pi, n_nodes)
        radii = np.sqrt(np.random.uniform(0, 0.95, n_nodes))
        positions[:, 0] = radii * np.cos(angles)
        positions[:, 1] = radii * np.sin(angles)
        
    edges = []
    # Generate True Edges based on probabilities
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = get_distance(positions[i], positions[j], geometry)
            probs = get_probabilities(d, params['alpha_pos'], params['alpha_neg'], params['beta'])
            
            # Roll dice
            choice = np.random.choice(['none', 'friend', 'enemy'], p=[probs['none'], probs['friend'], probs['enemy']])
            
            # Store edge: (node_i, node_j, type)
            if choice == 'friend':
                edges.append((i, j, 1))
            elif choice == 'enemy':
                edges.append((i, j, -1))
            else:
                edges.append((i, j, 0)) # Store non-edges too for evaluation
                
    return positions, edges

# ==========================================
# 3. DECODER & TESTER (The Analysis)
# ==========================================

def test_accuracy(n_nodes, geometry):
    print(f"\n--- Testing {geometry} Space ---")
    
    # Parameters used for generating AND decoding
    params = {'alpha_pos': 2.5, 'alpha_neg': -1.0, 'beta': 0.8}
    
    # A. Run Simulation (Get Ground Truth)
    positions, true_edges = simulate_network(n_nodes, geometry, params)
    
    # Lists to store Ground Truth vs Predictions
    y_true_class = []     # 1, -1, 0
    y_score_friend = []   # Probability of being friend
    y_score_enemy = []    # Probability of being enemy
    y_pred_class = []     # Predicted class (highest prob)
    
    # B. Decode Relationships
    # We look at the positions and ask: "What SHOULD the relationship be?"
    for (u, v, true_sign) in true_edges:
        # Recover relationship from latent space
        d = get_distance(positions[u], positions[v], geometry)
        probs = get_probabilities(d, params['alpha_pos'], params['alpha_neg'], params['beta'])
        
        # Store Truth
        y_true_class.append(true_sign)
        
        # Store Predictions
        y_score_friend.append(probs['friend'])
        y_score_enemy.append(probs['enemy'])
        
        # Pick the most likely outcome
        if probs['friend'] > probs['enemy'] and probs['friend'] > probs['none']:
            y_pred_class.append(1)
        elif probs['enemy'] > probs['friend'] and probs['enemy'] > probs['none']:
            y_pred_class.append(-1)
        else:
            y_pred_class.append(0)
            
    # C. Calculate Metrics
    # Accuracy: How often did we guess the exact link type?
    acc = accuracy_score(y_true_class, y_pred_class)
    
    # AUC (Friendship): Ability to distinguish Friends from Non-Friends
    # We treat 1 as positive class, others (0, -1) as negative
    binary_friend_truth = [1 if x == 1 else 0 for x in y_true_class]
    auc_friend = roc_auc_score(binary_friend_truth, y_score_friend)
    
    # AUC (Animosity): Ability to distinguish Enemies from Non-Enemies
    # We treat -1 as positive class, others (0, 1) as negative
    binary_enemy_truth = [1 if x == -1 else 0 for x in y_true_class]
    auc_enemy = roc_auc_score(binary_enemy_truth, y_score_enemy)
    
    print(f"  Accuracy:    {acc:.4f} (Random guess would be ~0.33)")
    print(f"  AUC Friend:  {auc_friend:.4f} (1.0 is perfect)")
    print(f"  AUC Enemy:   {auc_enemy:.4f} (1.0 is perfect)")
    
    return acc, auc_friend, auc_enemy

# ==========================================
# 4. RUN COMPARISON
# ==========================================

if __name__ == "__main__":
    # Run small test (100 nodes = ~5000 pairs)
    geometries = ['Euclidean', 'Spherical', 'Hyperbolic']
    results = {}

    for geo in geometries:
        results[geo] = test_accuracy(100, geo)

    # Print Summary
    print("\n=== FINAL SCOREBOARD ===")
    print(f"{'Geometry':<12} | {'Acc':<8} | {'AUC(Friend)':<12} | {'AUC(Enemy)':<12}")
    print("-" * 50)
    for geo, (acc, auc_f, auc_e) in results.items():
        print(f"{geo:<12} | {acc:.4f}   | {auc_f:.4f}       | {auc_e:.4f}")
