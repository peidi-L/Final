# Project Documentation: Latent Space Models for Signed Networks

**Author:** Peidi Li  
**Supervisor:** Cornelius Fritz

## Overview

I'm working on latent space models for signed social networks. The goal is to compare how different geometric spaces affect network structure. I'm testing Euclidean, Spherical, and Hyperbolic spaces. I'm also looking at different distance metrics.

This builds on work by Hoff et al. (2002) and Smith et al. (2019). I'm implementing their ideas in Python.

## What I've Built So Far

### 1. Data Loading (`loader.py`)

I created a module to process the Slashdot dataset. The main function is `load_data()`.

It uses pandas to load the TSV file, counts connections per node, filters for the top k most connected nodes, and remaps IDs to sequential indices. The output is a numpy array of edges `[source_id, target_id, sign]` where signs are +1 for friends and -1 for enemies.

### 2. Position Initialization (`models.py`)

This module creates random starting positions for nodes. The function is `initialize_positions()`.

It supports three geometries:
- **Euclidean:** Random positions from a Gaussian distribution
- **Spherical:** Random positions on a unit sphere (normalized)
- **Hyperbolic:** Random positions inside a Poincaré disk

Right now I'm using Euclidean for the baseline. I'll add the others later.

### 3. Training (`train.py`)

The main training script optimizes latent positions using a vectorized loss function. It loads data, initializes positions, and runs L-BFGS-B optimization. After training, it evaluates performance using AUC (comparing predicted distances to true edge signs) and generates a visualization of the learned embedding.

## Model Architecture

The loss function implements log-likelihood maximization from Hoff et al. (2002), penalizing distances for positive edges. For friends, we minimize squared distance to pull connected nodes together. For enemies, we use an exponential penalty term `exp(-dist)` that increases sharply when enemies are close, pushing them apart.

The initialization follows Smith et al. (2019) geometric constraints: Euclidean uses standard normal distribution, hyperbolic uses Poincaré disk with radius capped at 0.9 to avoid boundary issues.

I'm using L-BFGS-B for optimization since it's efficient for high-dimensional problems and can handle constraints (needed for hyperbolic space later). The current implementation uses a single-stage optimization with weighted friend/enemy terms, though I initially tried a two-stage approach.

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
3. Compare results across different geometries and metrics
4. Run experiments on different subsets of the data

## References

- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460), 1090-1098.

- Smith, A., Asta, D. M., & Calder, C. A. (2019). The geometry of continuous latent space models for network data. *Statistical Science*, 34(3), 428-453.

- Tang, M., & Zhu, J. (2025). Population-level balance in signed latent space network models. *Journal of Machine Learning Research* (forthcoming).

- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Signed networks in social media. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.
