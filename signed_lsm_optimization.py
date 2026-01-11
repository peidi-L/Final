import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

class SignedLSM(BaseEstimator, ClassifierMixin):
    def __init__(self, geometry='euclidean', d=2, random_state=None):
        self.geometry = geometry
        self.d = d
        self.random_state = random_state
        self.params_ = None
        self.Z_ = None
        
    def _dist_euclidean(self, Z):
        # Efficient vectorization for Euclidean distance matrix
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 <z_i, z_j>
        sq_norms = np.sum(Z**2, axis=1).reshape(-1, 1)
        dist_sq = sq_norms + sq_norms.T - 2 * np.dot(Z, Z.T)
        dist_sq = np.maximum(dist_sq, 0) # Clip negative due to float errors
        return np.sqrt(dist_sq)

    def _dist_manhattan(self, Z):
        n = Z.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = np.sum(np.abs(Z[i] - Z[j]))
                D[i, j] = val
                D[j, i] = val
        return D

    def _dist_chebyshev(self, Z):
        n = Z.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = np.max(np.abs(Z[i] - Z[j]))
                D[i, j] = val
                D[j, i] = val
        return D

    def _dist_spherical(self, Z):
        # Great circle distance
        # Points assumed to be on unit sphere? Or we project them?
        # Standard LSM usually projects or penalizes. Here we project.
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_norm = Z / np.maximum(norms, 1e-9)
        cosine_sim = np.dot(Z_norm, Z_norm.T)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return np.arccos(cosine_sim)

    def _dist_hyperbolic(self, Z):
        # Poincaré disk distance
        # d(u, v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        # Ensure points are inside disk. If optimizer pushes out, we clip.
        norms = np.linalg.norm(Z, axis=1)
        # Clip to 0.999 radius to stay inside disk
        scale_mask = norms >= 1.0
        if np.any(scale_mask):
            # Soft clip would be better but hard clip for stability
            Z[scale_mask] = Z[scale_mask] / norms[scale_mask].reshape(-1,1) * 0.999
            norms = np.linalg.norm(Z, axis=1)
            
        sq_norms = norms**2
        
        n = Z.shape[0]
        D = np.zeros((n, n))
        
        # This double loop is slow but stable for Hyperbolic formula complexity
        # Could be vectorized but tricky with the denominator term
        for i in range(n):
            for j in range(i + 1, n):
                u, v = Z[i], Z[j]
                sq_dist = np.sum((u - v)**2)
                denom = (1 - sq_norms[i]) * (1 - sq_norms[j])
                denom = max(denom, 1e-9)
                
                arg = 1 + 2 * sq_dist / denom
                dist = np.arccosh(max(arg, 1.0))
                D[i, j] = dist
                D[j, i] = dist
        return D

    def fit(self, adj_matrix):
        np.random.seed(self.random_state)
        n = adj_matrix.shape[0]
        
        # Initialize positions
        # For Hyperbolic, start small near origin
        if self.geometry == 'hyperbolic':
            Z_init = np.random.uniform(-0.5, 0.5, size=(n, self.d))
        else:
            Z_init = np.random.uniform(-2, 2, size=(n, self.d))
            
        # Initial parameters: alpha_pos, alpha_neg, beta
        params_init = np.array([1.0, -1.0, 0.5]) 
        
        x0 = np.concatenate([params_init, Z_init.ravel()])
        
        # Bounds
        # alpha_pos, alpha_neg: unbounded
        # beta: > 0 usually
        # Z: unbounded for Euclidean, but constrained for others ideally
        bounds = [(None, None)] * 2 + [(0, None)] + [(None, None)] * (n * self.d)
        if self.geometry == 'hyperbolic':
            # Bound Z to be roughly in [-1, 1] - let optimizer handle soft constraints
            # We enforce strictly inside inside distance calc
            pass

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(adj_matrix, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': False} # Set disp=True to see progress
        )
        
        self.params_ = result.x[:3]
        self.Z_ = result.x[3:].reshape(n, self.d)
        
        return self

    def _neg_log_likelihood(self, x, adj, n):
        alpha_pos, alpha_neg, beta = x[:3]
        Z = x[3:].reshape(n, self.d)
        
        # 1. Calculate Distances
        if self.geometry == 'euclidean':
            D = self._dist_euclidean(Z)
        elif self.geometry == 'manhattan':
            D = self._dist_manhattan(Z)
        elif self.geometry == 'chebyshev':
            D = self._dist_chebyshev(Z)
        elif self.geometry == 'spherical':
            D = self._dist_spherical(Z)
        elif self.geometry == 'hyperbolic':
            D = self._dist_hyperbolic(Z)
        else:
            D = self._dist_euclidean(Z) # Fallback
            
        # 2. Log-Odds
        # theta_plus = alpha_pos - D
        # theta_minus = alpha_neg + beta * D
        theta_plus = alpha_pos - D
        theta_minus = alpha_neg + beta * D
        
        # 3. Log-Partition Function (Log Sum Exp for Denominator)
        # Denom = 1 + exp(theta_plus) + exp(theta_minus)
        # log(Denom) = logaddexp(0, logaddexp(theta_plus, theta_minus))
        
        log_Z = np.logaddexp(0, np.logaddexp(theta_plus, theta_minus))
        
        # 4. Sum Log-Likelihoods
        # Only sum over i < j
        mask_upper = np.triu_indices(n, k=1)
        
        Y_flat = adj[mask_upper]
        theta_plus_flat = theta_plus[mask_upper]
        theta_minus_flat = theta_minus[mask_upper]
        log_Z_flat = log_Z[mask_upper]
        
        # LL = Sum [ I(y=1)*theta_plus + I(y=-1)*theta_minus - log_Z ]
        ll = 0
        
        # Positives
        pos_mask = (Y_flat == 1)
        ll += np.sum(theta_plus_flat[pos_mask])
        
        # Negatives
        neg_mask = (Y_flat == -1)
        ll += np.sum(theta_minus_flat[neg_mask])
        
        # Normalization
        ll -= np.sum(log_Z_flat)
        
        # Regularization (optional, keeps Z from exploding)
        reg = 0.01 * np.sum(Z**2)
        
        return -ll + reg

    def predict_proba(self, i, j):
        """Returns (prob_zero, prob_pos, prob_neg) for node pair (i,j)"""
        if self.Z_ is None:
            raise RuntimeError("Model not fitted")
            
        z_i = self.Z_[i].reshape(1, -1)
        z_j = self.Z_[j].reshape(1, -1)
        
        # Re-calc distance for single pair
        # (Simplified for brevity, ideally reuse _dist functions)
        if self.geometry == 'hyperbolic':
            # Manual single pair hyperbolic
            norm_i = np.linalg.norm(z_i)
            norm_j = np.linalg.norm(z_j)
            
            # Safe clip
            zi_safe = z_i / max(norm_i, 1.0+1e-5) * 0.999
            zj_safe = z_j / max(norm_j, 1.0+1e-5) * 0.999
            
            sq_dist = np.sum((zi_safe - zj_safe)**2)
            z_sq_i = np.sum(zi_safe**2)
            z_sq_j = np.sum(zj_safe**2)
            
            num = 2 * sq_dist
            den = (1 - z_sq_i) * (1 - z_sq_j)
            arg = 1 + num / (den + 1e-8)
            d = np.arccosh(np.maximum(arg, 1.0))
        elif self.geometry == 'spherical':
             # Manual single pair spherical
             ni = np.linalg.norm(z_i)
             nj = np.linalg.norm(z_j)
             if ni==0 or nj==0: d=0
             else:
                 cos = np.dot(z_i.flatten(), z_j.flatten()) / (ni*nj)
                 d = np.arccos(np.clip(cos, -1, 1))
        else:
            # Euclidean fallback for others in this snippet
            d = np.linalg.norm(z_i - z_j)
            
        alpha_pos, alpha_neg, beta = self.params_
        
        tp = alpha_pos - d
        tm = alpha_neg + beta * d
        
        # Stable softmax
        logits = np.array([0, tp, tm])
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        return probs
