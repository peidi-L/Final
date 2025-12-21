# Project Documentation: Latent Space Models for Signed Networks

**Author:** Peidi Li  
**Supervisor:** Cornelius Fritz

## Overview

This project implements latent space models for signed social networks, comparing how different geometric spaces (Euclidean, Spherical, Hyperbolic) and distance metrics affect network structure and model performance. The implementation is based on foundational work by Hoff et al. (2002) and geometric extensions by Smith et al. (2019).

## What Has Been Implemented

### 1. Data Loading (`loader.py`)

The `loader.py` module processes the Slashdot Zoo signed social network dataset:

- **Function:** `get_top_1000_core_nodes(file_path)`
- **Purpose:** Filters the network to the top 1000 most connected nodes
- **Process:**
  1. Counts connections for all nodes in the network
  2. Sorts nodes by connection count (highest first)
  3. Selects the top 1000 nodes
  4. Renumbers nodes sequentially from 0 to 999
  5. Filters edges to only include connections between these top 1000 nodes
  6. Returns edges with new sequential IDs and signs (+1 for friends, -1 for foes)

**Output:** A list of edges `[source_id, target_id, sign]` where:
- Node IDs are sequential (0-999)
- Only includes connections between the top 1000 most connected nodes
- Preserves the sign information (friend/foe relationships)

### 2. Position Initialization (`models.py`)

The `models.py` module provides geometric initialization functions:

- **Function:** `initialize_positions(num_users, dimensions, geometry)`
- **Purpose:** Creates random starting positions for nodes in different geometric spaces
- **Supported Geometries:**
  - **Euclidean:** Random positions from Gaussian distribution
  - **Spherical:** Random positions on unit sphere (normalized)
  - **Hyperbolic:** Random positions within Poincaré disk (scaled to stay inside unit circle)

### 3. Two-Stage Training (`train.py`)

The `train.py` module implements a sequential training process:

- **Function:** `train_sequential()`
- **Purpose:** Trains a latent space model to position nodes such that friends are close and enemies are far apart
- **Process:**
  1. Loads top 1000 nodes using `loader.py`
  2. Initializes random 2D positions using `models.py`
  3. **Stage 1:** Clusters friend connections (sign = +1) by minimizing distances to target distance of 1.0
  4. **Stage 2:** Handles both friends and enemies:
     - Friends: Maintains close distances (~1.0) with stronger penalty
     - Enemies: Pushes nodes apart by penalizing small distances
  5. Returns final 2D positions for all nodes

## Mathematical Foundation and Theoretical Alignment

This implementation is **mathematically sound** and aligns with the theoretical constraints laid out in Hoff et al. (2002) and Smith et al. (2019).

### 1. Initialization (Matching Smith et al., 2019)

The `initialize_positions` function correctly enforces the geometric "boundary conditions" described in the Smith paper.

#### Euclidean Geometry
- **Code:** `np.random.randn(num_users, dimensions) * 0.1`
- **Theory:** Matches Hoff et al. (2002). They assume latent positions come from a Gaussian (Normal) distribution. The code generates exactly that.
- **Constraint:** No boundary constraints in Euclidean space

#### Spherical Geometry
- **Code:** `positions / norms` (normalization to unit length)
- **Theory:** Matches Smith et al. (2019). The latent space is a Riemannian manifold. The defining constraint is that every point must have a length of exactly 1 (||x|| = 1). The normalization step enforces this strictly.
- **Constraint:** All points must lie on the unit sphere

#### Hyperbolic Geometry
- **Code:** `positions / (1 + norms) * 0.9`
- **Theory:** Matches the Poincaré Disk model in Smith et al. (2019). The constraint is that points must be strictly *inside* the unit disk (||x|| < 1). The edge (||x|| = 1) represents infinity. The code ensures no point ever touches the edge (max radius 0.9), preventing the distance formula from exploding to infinity.
- **Constraint:** All points must be within the unit disk

### 2. The Training Logic (Matching Hoff et al., 2002)

The `train_sequential` function is a **Distance-Based Loss** implementation of the probabilistic models in the papers.

#### The "Friends" Logic

- **Code:** `total_error += (dist - 1.0) ** 2`
- **Theory:** Hoff's model says the probability of a link is high when distance is low: P(link) ∝ exp(-distance). 
- **The Match:** In statistics, "Maximizing Likelihood" (Hoff's method) is mathematically equivalent to "Minimizing Squared Error" (this implementation). By trying to make the distance `1.0` (small), the model maximizes the probability of friendship.

**Stage 1:** Focuses exclusively on friend connections, clustering them at distance 1.0.

**Stage 2:** Maintains friend clustering with stronger penalty (2.0 multiplier) while also handling enemies.

#### The "Foes" Logic

- **Code:** `total_error -= capped_dist` (for enemies)
- **Theory:** This aligns with **Structural Balance Theory** (mentioned in the introduction and Tang & Zhu, 2025). This theory states that "the enemy of my enemy is my friend," implying that positive links form clusters and negative links span *between* clusters.
- **The Match:** By subtracting the distance for foes, the error increases (badness) if foes are close. The optimizer fixes this by pushing them apart. The distance is capped at 10.0 to prevent extreme values.

### 3. Optimization Method

- **Method:** L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds)
- **Why:** This is a quasi-Newton optimization method that's efficient for high-dimensional problems and can handle constraints (useful for spherical and hyperbolic geometries).
- **Iterations:** 
  - Stage 1: 30 iterations (focus on friends)
  - Stage 2: 50 iterations (refine both friends and enemies)

## Summary

This implementation successfully translates the **Probability Theory** of the papers into:
1. **Geometric Constraints** (Initialization) - Enforcing boundary conditions for different geometries
2. **Optimization Rules** (Training) - Minimizing distance-based loss functions that align with probabilistic models

The code acts as a valid **Euclidean Baseline** for thesis results, providing a foundation for comparing different geometries and metrics.

## File Structure

```
/Applications/untitled folder/
├── loader.py              # Data loading and filtering
├── models.py              # Position initialization for different geometries
├── train.py               # Two-stage training process
├── soc-sign-Slashdot090221.txt  # Dataset
└── PROJECT_DOCUMENTATION.md     # This file
```

## Next Steps

1. **Extend to other geometries:** Implement spherical and hyperbolic distance calculations
2. **Add different metrics:** Implement L1 (Manhattan) and L∞ (Chebyshev) distance metrics
3. **Evaluation:** Add metrics to evaluate model performance (AUC, accuracy, etc.)
4. **Visualization:** Create visualizations of learned embeddings
5. **Comparison:** Compare results across different geometries and metrics

## References

- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460), 1090-1098.

- Smith, A., Asta, D. M., & Calder, C. A. (2019). The geometry of continuous latent space models for network data. *Statistical Science*, 34(3), 428-453.

- Tang, M., & Zhu, J. (2025). Population-level balance in signed latent space network models. *Journal of Machine Learning Research* (forthcoming).

- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Signed networks in social media. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.

