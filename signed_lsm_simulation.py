import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. GENERATE SYNTHETIC DATA
# ==========================================
def generate_synthetic_data(n_nodes=300, n_clusters=3):
    """
    Generates a signed network with block structure:
    - Intra-cluster edges are mostly Positive (Friends)
    - Inter-cluster edges are mostly Negative (Foes)
    """
    print(f"Generating synthetic network with {n_nodes} nodes and {n_clusters} clusters...")
    np.random.seed(42)
    
    # Assign nodes to clusters
    cluster_assign = np.random.randint(0, n_clusters, n_nodes)
    
    edges = []
    
    # Generate edges
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # Probability of edge existence
            rand_val = np.random.random()
            
            if cluster_assign[i] == cluster_assign[j]:
                # Same cluster: High chance of Friend, Low chance of Foe
                if rand_val < 0.3:  # 30% density within clusters
                    # 90% chance friend, 10% noise (foe)
                    sign = 1 if np.random.random() < 0.9 else -1
                    edges.append([i, j, sign])
            else:
                # Different cluster: Low chance of Friend, Moderate chance of Foe
                if rand_val < 0.05:  # 5% density between clusters
                    # 90% chance foe, 10% noise (friend)
                    sign = -1 if np.random.random() < 0.9 else 1
                    edges.append([i, j, sign])
    
    df = pd.DataFrame(edges, columns=['From', 'To', 'Sign'])
    print(f"Generated {len(df)} edges.")
    print(f"Positive (Friends): {len(df[df['Sign']==1])}")
    print(f"Negative (Foes):    {len(df[df['Sign']==-1])}")
    return df, cluster_assign

# Generate the data
N_NODES = 100  # Set to 100 as recommended for learning & visualization
data, true_clusters = generate_synthetic_data(N_NODES, n_clusters=3)

# Train/Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# ==========================================
# 2. DISTANCE FUNCTIONS
# ==========================================
def euclidean_distance(pos_i, pos_j):
    return np.linalg.norm(pos_i - pos_j, axis=1)

def manhattan_distance(pos_i, pos_j):
    return np.sum(np.abs(pos_i - pos_j), axis=1)

def chebyshev_distance(pos_i, pos_j):
    return np.max(np.abs(pos_i - pos_j), axis=1)

def spherical_distance(pos_i, pos_j):
    # Normalize rows to unit length
    norm_i = np.linalg.norm(pos_i, axis=1, keepdims=True) + 1e-10
    norm_j = np.linalg.norm(pos_j, axis=1, keepdims=True) + 1e-10
    
    # Dot product
    dot = np.sum((pos_i / norm_i) * (pos_j / norm_j), axis=1)
    return np.arccos(np.clip(dot, -1.0, 1.0))

def hyperbolic_distance(pos_i, pos_j):
    # Poincaré disk distance
    # Ensure norms < 1
    norm_i = np.linalg.norm(pos_i, axis=1)
    norm_j = np.linalg.norm(pos_j, axis=1)
    
    # Clip to disk (safety margin)
    if np.any(norm_i >= 0.99):
        pos_i[norm_i >= 0.99] *= (0.99 / norm_i[norm_i >= 0.99][:, None])
    if np.any(norm_j >= 0.99):
        pos_j[norm_j >= 0.99] *= (0.99 / norm_j[norm_j >= 0.99][:, None])
        
    diff_sq = np.sum((pos_i - pos_j)**2, axis=1)
    # Recompute norms after clipping
    norm_i_sq = np.sum(pos_i**2, axis=1)
    norm_j_sq = np.sum(pos_j**2, axis=1)
    
    arg = 1 + 2 * diff_sq / ((1 - norm_i_sq) * (1 - norm_j_sq) + 1e-10)
    return np.arccosh(np.clip(arg, 1.0, None))

# ==========================================
# 3. MODEL CLASS (WITH YOUR FORMULAS)
# ==========================================
class SignedLSM_Simulation:
    def __init__(self, n_nodes, dim=2, dist_func=euclidean_distance, name="LSM"):
        self.n_nodes = n_nodes
        self.dim = dim
        self.dist_func = dist_func
        self.name = name
        
        # Parameters to learn
        self.positions = None
        self.alpha_pos = 0.0
        self.alpha_neg = 0.0
        self.beta = 1.0  # The Polarization parameter
        
    def init_params(self):
        # Initialize positions
        if 'hyperbolic' in self.name.lower():
            self.positions = np.random.randn(self.n_nodes, self.dim) * 0.1
        elif 'spherical' in self.name.lower():
            self.positions = np.random.randn(self.n_nodes, self.dim)
            # Project to sphere
            norms = np.linalg.norm(self.positions, axis=1, keepdims=True)
            self.positions /= (norms + 1e-10)
        else:
            self.positions = np.random.randn(self.n_nodes, self.dim) * 0.5
            
        self.alpha_pos = 0.5
        self.alpha_neg = -0.5
        self.beta = 1.0
        
    def neg_log_likelihood(self, params, edge_i, edge_j, signs):
        # Unpack parameters
        # [x1, y1, x2, y2, ..., alpha_pos, alpha_neg, beta]
        n_pos = self.n_nodes * self.dim
        positions = params[:n_pos].reshape(self.n_nodes, self.dim)
        alpha_pos = params[n_pos]
        alpha_neg = params[n_pos+1]
        beta = params[n_pos+2]
        
        # Get coordinates
        p_i = positions[edge_i]
        p_j = positions[edge_j]
        
        # Calculate Distance d(i,j)
        d = self.dist_func(p_i, p_j)
        
        # --- YOUR FORMULAS ---
        # Eq 3: Friendship (Homophily) -> alpha - d
        logit_pos = alpha_pos - d
        
        # Eq 4: Animosity (Polarization) -> alpha + beta * f(d)
        # We assume f(d) = d for generic distances
        logit_neg = alpha_neg + (beta * d)
        
        # Probabilities (Multinomial Logit)
        # P(0) reference implies: exp(0) = 1 in denominator
        exp_pos = np.exp(np.clip(logit_pos, -20, 20))
        exp_neg = np.exp(np.clip(logit_neg, -20, 20))
        partition = 1 + exp_pos + exp_neg
        
        prob_pos = exp_pos / partition
        prob_neg = exp_neg / partition
        
        # Gather probabilities for observed signs
        # sign=1 -> prob_pos, sign=-1 -> prob_neg
        probs = np.where(signs == 1, prob_pos, prob_neg)
        
        # Negative Log Likelihood
        return -np.sum(np.log(probs + 1e-10))

    def fit(self, train_df, max_iter=30):
        print(f"Training {self.name}...")
        self.init_params()
        
        edge_i = train_df['From'].values
        edge_j = train_df['To'].values
        signs = train_df['Sign'].values
        
        # Initial guess
        x0 = np.concatenate([
            self.positions.flatten(),
            [self.alpha_pos, self.alpha_neg, self.beta]
        ])
        
        # Optimization
        res = minimize(
            self.neg_log_likelihood,
            x0,
            args=(edge_i, edge_j, signs),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'disp': False}
        )
        
        # Store results
        n_pos = self.n_nodes * self.dim
        self.positions = res.x[:n_pos].reshape(self.n_nodes, self.dim)
        self.alpha_pos = res.x[n_pos]
        self.alpha_neg = res.x[n_pos+1]
        self.beta = res.x[n_pos+2]
        
        print(f"  > Done. Loss: {res.fun:.2f} | Beta: {self.beta:.3f}")

    def predict(self, test_df):
        edge_i = test_df['From'].values
        edge_j = test_df['To'].values
        
        p_i = self.positions[edge_i]
        p_j = self.positions[edge_j]
        d = self.dist_func(p_i, p_j)
        
        logit_pos = self.alpha_pos - d
        logit_neg = self.alpha_neg + (self.beta * d)
        
        exp_pos = np.exp(np.clip(logit_pos, -20, 20))
        exp_neg = np.exp(np.clip(logit_neg, -20, 20))
        partition = 1 + exp_pos + exp_neg
        
        prob_pos = exp_pos / partition
        prob_neg = exp_neg / partition
        
        return prob_pos, prob_neg

# ==========================================
# 4. RUN SIMULATION
# ==========================================
if __name__ == "__main__":
    models = [
        SignedLSM_Simulation(N_NODES, 2, euclidean_distance, "Euclidean"),
        SignedLSM_Simulation(N_NODES, 2, spherical_distance, "Spherical"),
        SignedLSM_Simulation(N_NODES, 2, hyperbolic_distance, "Hyperbolic"),
    ]

    plt.figure(figsize=(15, 5))

    for i, model in enumerate(models):
        # Train
        model.fit(train_data)
        
        # Predict
        prob_pos, prob_neg = model.predict(test_data)
        
        # Evaluate
        # Treat 'friend' prediction as binary classification (1 vs not 1)
        y_true_friend = (test_data['Sign'] == 1).astype(int)
        auc_friend = roc_auc_score(y_true_friend, prob_pos)
        
        # Treat 'foe' prediction as binary classification (-1 vs not -1)
        y_true_foe = (test_data['Sign'] == -1).astype(int)
        auc_foe = roc_auc_score(y_true_foe, prob_neg)
        
        print(f"  > Results: AUC(Friend)={auc_friend:.3f}, AUC(Foe)={auc_foe:.3f}")
        
        # Visualize
        ax = plt.subplot(1, 3, i+1)
        pos = model.positions
        
        # Color nodes by their TRUE cluster
        scatter = ax.scatter(pos[:,0], pos[:,1], c=true_clusters, cmap='viridis', s=20, alpha=0.7)
        
        # Draw Hyperbolic boundary if needed
        if "hyperbolic" in model.name.lower():
            circle = plt.Circle((0,0), 1, fill=False, color='black', linestyle='--')
            ax.add_patch(circle)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            
        ax.set_title(f"{model.name}\nBeta={model.beta:.2f} | AUC(Foe)={auc_foe:.2f}")

    plt.tight_layout()
    plt.savefig('simulation_results.png')
    print("Simulation complete. Results saved to simulation_results.png")
