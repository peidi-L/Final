# An Investigation on the Impact of Geometries and Metrics in Latent Space Network Models

**Author:** Peidi Li  
**Supervisor:** Cornelius Fritz

## Introduction

Network data provide a way to represent complex relationships in areas like social science, biology, and international relations. As we have more network data available, researchers have developed statistical models to explain how networks form and what shapes their structure.

Latent space models are one powerful approach. These models represent how nodes connect by placing each node at an unobserved position in a continuous geometric space. The key idea is that the probability of a connection between two nodes depends on how close they are in this hidden space. This naturally captures important network properties like homophily (similar nodes connect) and transitivity (friends of friends are friends).

Recent work has shown that the geometry of the latent space itself matters. The curvature of the space—whether it's flat (Euclidean), curved outward (spherical), or curved inward (hyperbolic)—directly affects network features like how nodes cluster, how connections are distributed, and how centralized the network is.

This project also works with signed networks, where relationships can be positive (friends) or negative (foes). We model both types of relationships using structural balance theory, which captures ideas like "the friend of my friend is a friend" and "the enemy of my enemy is a friend."

## Literature Review

### Foundational Work

Hoff, Raftery, and Handcock (2002) introduced the first formal latent space model for networks. In their model, each node gets an unobserved position in a low-dimensional Euclidean space. The chance of an edge between two nodes goes down as the distance between them increases. This approach lets us visualize network structure and measure uncertainty in node positions.

### Geometry and Networks

Smith, Asta, and Calder (2019) looked at latent space models through a geometric lens. They showed that the geometry of the latent space plays a central role in shaping what we observe in networks. For example:
- Hyperbolic space works better for hierarchical and scale-free networks because of its negative curvature
- Euclidean space works better for networks with communities or homogeneous structures
- The curvature of the space affects network connectivity independently of network size

### Signed Networks

Tang and Zhu (2025) extended latent space models to signed networks. Their balanced inner-product model combines structural balance theory with a probabilistic framework. They showed that certain latent geometries naturally promote balanced network structures, where relationships follow predictable patterns.

## Methodological Framework

This project investigates how the choice of latent space geometry and distance metric affects network structure and model performance. Specifically, we:

1. **Compare Different Geometries**: We test Euclidean, spherical, and hyperbolic geometries, looking at how curvature affects network statistics like:
   - Clustering coefficient (how tightly nodes group together)
   - Degree heterogeneity (how varied the number of connections are)
   - Average path length (typical distance between nodes)

2. **Compare Distance Metrics**: Within Euclidean space, we test different ways to measure distance:
   - **Euclidean (L2)**: Standard straight-line distance
   - **Manhattan (L1)**: City-block distance (sum of absolute differences)
   - **Chebyshev (L∞)**: Maximum coordinate difference

3. **Spectral Analysis**: We use graph Laplacian spectra to measure how geometric curvature shapes connectivity patterns, independent of network size.

The balanced inner-product model integrates structural balance theory ("the friend of my friend is a friend", "the enemy of my enemy is a friend") into a probabilistic framework of latent space.

## Dataset

The study uses the **Slashdot Zoo signed social network** (Leskovec et al., 2010), which contains both positive (friend) and negative (foe) edges.

- **Source**: Slashdot Zoo signed social network (February 21, 2009)
- **Full dataset**: 82,144 nodes, 549,202 edges
- **Working subset**: Top 500-1000 most active nodes for computational efficiency
- **Signs**: +1 (friend), -1 (foe)
- **Dataset URL**: https://snap.stanford.edu/data/soc-sign-Slashdot090221.html

A working subset of nodes is used to ensure computational feasibility while preserving the essential structural patterns of the full network.

## Expected Contributions

This study aims to advance our understanding of how geometric and metric choices in latent space models influence network structure and interpretation. It will:

- Provide a systematic comparison of network properties under varying geometric curvatures
- Establish connections between spectral graph theory and latent space geometry
- Extend latent space models to accommodate signed relationships in a geometrically principled manner
- Offer insights into how curvature-based latent representations can improve model interpretability and prediction for real-world network data

## Implementation Notes

### Computational Considerations

The loss functions are vectorized using numpy operations to handle large networks efficiently. For the Slashdot dataset subset (1000 nodes, ~50k edges), each optimization iteration computes all pairwise distances simultaneously rather than iterating through edges.

### Optimization Strategy

The two-stage training approach addresses the competing objectives of clustering friends while separating enemies. Stage 1 establishes friend structure, then Stage 2 refines positions while enforcing enemy separation. L-BFGS-B is used for its efficiency, though boundary constraints for hyperbolic space may require alternative optimizers in future work.

### Challenges Encountered

- **Boundary constraints**: Hyperbolic space requires points to stay within the Poincaré disk. Current initialization keeps points safely inside, but optimization constraints need refinement.
- **Loss balancing**: The relative weighting of friend vs. enemy terms affects convergence. The current 2.0 multiplier on friend errors was determined empirically.
- **Numerical stability**: Distances near the disk boundary approach infinity, requiring careful initialization and potentially different distance formulations.

### Setup

### Environment Setup

1. **Create a conda environment** (recommended):
```bash
conda create -n lsm_env python=3.9
conda activate lsm_env
```

2. **Install required packages**:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

**Required dependencies:**
- `numpy` - numerical computing
- `pandas` - data manipulation
- `matplotlib` - visualization
- `seaborn` - enhanced plotting
- `scipy` - optimization algorithms
- `scikit-learn` - train/test split and evaluation metrics
- `tqdm` - progress bars

### Running the Model

```bash
conda activate lsm_env
python3 lsm_implementation.py
```
## References

- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460), 1090-1098.

- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Signed networks in social media. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.

- Smith, A., Asta, D. M., & Calder, C. A. (2019). The geometry of continuous latent space models for network data. *Statistical Science*, 34(3), 428-453.

- Tang, M., & Zhu, J. (2025). Population-level balance in signed latent space network models. *Journal of Machine Learning Research* (forthcoming).
