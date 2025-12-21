# Project Documentation: Latent Space Models for Signed Networks

**Author:** Peidi Li  
**Supervisor:** Cornelius Fritz

## Overview

I'm working on latent space models for signed social networks. The goal is to compare how different geometric spaces affect network structure. I'm testing Euclidean, Spherical, and Hyperbolic spaces. I'm also looking at different distance metrics.

This builds on work by Hoff et al. (2002) and Smith et al. (2019). I'm implementing their ideas in Python.

## What I've Built So Far

### 1. Data Loading (`loader.py`)

I created a module to process the Slashdot dataset. The main function is `get_top_1000_core_nodes()`.

Here's what it does:
1. Counts how many connections each node has
2. Sorts nodes by connection count (most connected first)
3. Picks the top 1000 nodes
4. Renumbers them from 0 to 999
5. Filters edges to only keep connections between these 1000 nodes
6. Returns the edges with new IDs and signs

The output is a list of edges. Each edge looks like `[source_id, target_id, sign]`. The sign is +1 for friends and -1 for enemies. All node IDs are sequential (0-999).

### 2. Position Initialization (`models.py`)

This module creates random starting positions for nodes. The function is `initialize_positions()`.

It supports three geometries:
- **Euclidean:** Random positions from a Gaussian distribution
- **Spherical:** Random positions on a unit sphere (normalized)
- **Hyperbolic:** Random positions inside a Poincaré disk

Right now I'm using Euclidean for the baseline. I'll add the others later.

### 3. Two-Stage Training (`train.py`)

This is the main training script. It positions nodes so friends are close and enemies are far apart.

The process works like this:
1. Loads the top 1000 nodes using `loader.py`
2. Creates random 2D starting positions using `models.py`
3. **Stage 1:** Clusters friends together. It tries to make friend connections have distance around 1.0
4. **Stage 2:** Handles both friends and enemies:
   - Friends stay close (distance ~1.0) with a stronger penalty
   - Enemies get pushed apart by penalizing small distances
5. Returns the final 2D positions for all nodes

## How This Matches the Theory

I've checked my code against the papers. It matches what Hoff et al. (2002) and Smith et al. (2019) describe.

### Initialization

The initialization function enforces geometric constraints from Smith et al. (2019).

**Euclidean:**
- My code uses `np.random.randn()` to create random positions
- This matches Hoff et al. (2002). They assume positions come from a Gaussian distribution
- No boundary constraints needed

**Spherical:**
- My code normalizes positions to unit length
- This matches Smith et al. (2019). Points must have length exactly 1
- All points lie on the unit sphere

**Hyperbolic:**
- My code scales positions to stay inside the unit disk
- This matches the Poincaré Disk model from Smith et al. (2019)
- Points must be inside the disk (not on the edge)
- I cap the radius at 0.9 to avoid infinity issues

### Training Logic

The training matches Hoff et al. (2002). I'm using a distance-based loss function.

**Friends:**
- My code: `total_error += (dist - 1.0) ** 2`
- Theory: Hoff's model says link probability is high when distance is low
- Why it works: Maximizing likelihood equals minimizing squared error. By making distance 1.0, I maximize friendship probability

Stage 1 only looks at friends. It clusters them at distance 1.0.

Stage 2 keeps friends close (with stronger penalty) and also handles enemies.

**Enemies:**
- My code: `total_error -= capped_dist` for enemies
- Theory: This matches Structural Balance Theory. Enemies should be far apart
- Why it works: Subtracting distance means small distances increase error. The optimizer pushes enemies apart
- I cap distance at 10.0 to avoid extreme values

### Optimization

I'm using L-BFGS-B for optimization. It's efficient for high-dimensional problems. It can also handle constraints, which I'll need for spherical and hyperbolic geometries.

- Stage 1: 30 iterations (focus on friends)
- Stage 2: 50 iterations (refine both friends and enemies)

## Implementation Challenges

### Performance Optimization

The initial implementation used Python loops to compute distances for each edge pair, which was computationally expensive. With ~1000 nodes and tens of thousands of edges, each optimization step was taking several seconds. I refactored the loss functions to use vectorized numpy operations, computing all distances in a single matrix operation. This reduced per-iteration time by roughly 10x.

The key insight was pre-computing source and target indices as numpy arrays, then using advanced indexing to extract all relevant positions at once. This allows `np.linalg.norm()` to compute distances across all edges simultaneously.

### Boundary Constraints in Hyperbolic Space

The Poincaré disk model requires all points to lie strictly inside the unit circle. During optimization, points can drift toward the boundary, causing numerical instability as distances approach infinity. I initially tried using `np.clip()` to constrain positions, but this created discontinuities that broke L-BFGS-B's gradient estimates.

The current approach scales initial positions by `0.9 / (1 + norm)` to keep them safely inside the disk. For future work with hyperbolic optimization, I'll need to implement proper constraint handling or use a different optimizer that respects boundaries better.

### Loss Function Balancing

The two-stage approach emerged from trial and error. Initially, I tried optimizing friends and enemies simultaneously, but the optimizer would collapse all nodes to minimize enemy distances (since there are typically fewer enemy edges). Stage 1 establishes a stable friend structure, then Stage 2 refines it while pushing enemies apart.

The 2.0 multiplier on friend errors in Stage 2 was chosen empirically—lower values allowed friends to drift too far apart, while higher values prevented enemies from separating. This suggests the loss landscape has competing objectives that need careful weighting.

### L-BFGS-B Stability

I found that L-BFGS-B sometimes fails to converge if the initial positions are too spread out. Starting with positions scaled by 0.1 (rather than unit variance) helps the optimizer find a good basin quickly. For hyperbolic space, I'll likely need to experiment with different initialization strategies or switch to SGD with momentum for the initial epochs.

## Summary

I've translated the probability theory from the papers into code. The implementation has:
1. Geometric constraints in initialization
2. Optimization rules in training

This gives me a valid Euclidean baseline. I can use it to compare different geometries and metrics later.

## File Structure

```
Final/
├── loader.py              # Data loading and filtering
├── models.py              # Position initialization
├── train.py               # Two-stage training
├── README.md              # Project overview
└── PROJECT_DOCUMENTATION.md  # This file
```

## What's Next

1. Add spherical and hyperbolic distance calculations
2. Implement L1 (Manhattan) and L∞ (Chebyshev) distance metrics
3. Add evaluation metrics (AUC, accuracy, etc.)
4. Create visualizations of the learned embeddings
5. Compare results across different geometries and metrics

## References

- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460), 1090-1098.

- Smith, A., Asta, D. M., & Calder, C. A. (2019). The geometry of continuous latent space models for network data. *Statistical Science*, 34(3), 428-453.

- Tang, M., & Zhu, J. (2025). Population-level balance in signed latent space network models. *Journal of Machine Learning Research* (forthcoming).

- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Signed networks in social media. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.
