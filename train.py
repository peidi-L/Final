import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
# import time  # was using this for timing, not needed anymore
import loader
import models

# --- Configuration ---
DATA_PATH = "soc-sign-Slashdot090221.txt"
DIMENSIONS = 2
LAMBDA_FRIEND = 2.0  # Weight for positive ties
MAX_ITER = 100

def train_model():
    # 1. Load Data
    try:
        links, num_users = loader.load_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # extract edge info
    sources = links[:, 0]
    targets = links[:, 1]
    signs = links[:, 2]
    
    is_friend = signs == 1
    is_enemy = signs == -1

    print("Initializing latent positions...")
    x0 = models.initialize_positions(num_users, DIMENSIONS, 'euclidean').flatten()

    # initial attempt with SGD was too unstable
    # from torch import optim
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    def loss_function(params):
        pos = params.reshape((num_users, DIMENSIONS))
        
        # get coordinates for all edges
        pos_u = pos[sources]
        pos_v = pos[targets]
        
        # compute dists
        dists = models.safe_distance(pos_u, pos_v)
        
        # friend loss: want dists close to 0
        friend_loss = np.sum(dists[is_friend]**2)
        
        # enemy loss: push them apart
        enemy_loss = np.sum(np.exp(-dists[is_enemy]))
        
        return friend_loss + (LAMBDA_FRIEND * enemy_loss)

    print(f"Starting optimization (L-BFGS-B, max_iter={MAX_ITER})...")
    res = minimize(
        loss_function, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': MAX_ITER, 'disp': True}
    )

    final_pos = res.x.reshape((num_users, DIMENSIONS))

    evaluate_model(final_pos, sources, targets, signs)
    plot_latent_space(final_pos, signs, sources, targets)

    return final_pos

def evaluate_model(pos, sources, targets, true_signs):
    """
    Calculates Area Under Curve (AUC) to see if distances predict signs.
    """
    pos_u = pos[sources]
    pos_v = pos[targets]
    dists = np.linalg.norm(pos_u - pos_v, axis=1)
    
    # friends should have low dist, enemies high dist
    # so -dist should correlate with sign
    score = -dists 
    
    auc = roc_auc_score(true_signs, score)
    print("\n" + "="*30)
    print(f"FINAL RESULTS")
    print(f"AUC Score: {auc:.4f}")
    print("="*30 + "\n")

def plot_latent_space(pos, signs, sources, targets):
    """
    Visualizes the top 500 edges to avoid clutter.
    """
    plt.figure(figsize=(10, 10))
    
    # plot subset to avoid clutter
    subset = 500
    for i in range(min(len(signs), subset)):
        u, v = sources[i], targets[i]
        s = signs[i]
        color = 'g' if s == 1 else 'r'
        alpha = 0.1 if s == 1 else 0.3
        plt.plot([pos[u,0], pos[v,0]], [pos[u,1], pos[v,1]], c=color, alpha=alpha, lw=0.5)
        
    plt.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.6, c='blue')
    plt.title("Latent Space Embedding (Green=Friend, Red=Foe)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_file = "latent_space_viz.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    train_model()
