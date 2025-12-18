# An Investigation on the Impact of Geometries and Metrics in Latent Space Network Models

**Author:** Peidi Li  
**Supervisor:** Cornelius Fritz

## Overview

This research project investigates how the choice of geometry and metric in latent space network models affects network properties, interpretability, and inferential capacity. Network data provide a mathematical framework for representing complex relational systems in domains such as social science, biology, and international relations. Latent space models represent dependencies among dyadic bonds through unobserved positions of nodes in a continuous geometric space, where the probability of a tie between two nodes is a function of their proximity in the latent space.

## Introduction

Latent space models assume that the probability of a tie between two nodes is a function of their proximity in a latent space, naturally capturing fundamental properties of real-world networks such as homophily and transitivity. Recent developments have extended this framework to consider the role of geometry itself in shaping network structure. The underlying curvature of the latent space—whether Euclidean, spherical, or hyperbolic—directly influences emergent features of modeled networks, including degree distributions, clustering, and centralization.

This project also incorporates signed networks into the latent space framework, enabling the simultaneous modeling of positive and negative relationships under structural balance theory.

## Research Objectives

This project investigates how the choice of latent space geometry and metric affects the structural properties and inferential performance of latent space network models through:

1. **Geometric Comparison**: Comparing Euclidean, spherical, and hyperbolic geometries for continuous latent space models, examining how curvature influences emergent network statistics such as:
   - Clustering coefficient
   - Degree heterogeneity
   - Average path length

2. **Spectral Analysis**: Using graph Laplacian spectra to quantify how geometric curvature shapes connectivity patterns independently of network size, building on spectral graph theory results.

3. **Signed Networks Extension**: Extending analyses to signed networks by implementing the latent space model of the balanced inner-product, exploring how population-level balance interacts with geometric curvature.

4. **Simulation Studies**: Developing simulation studies to analyze network features arising from embeddings in spaces of different curvature, visualizing how geometric assumptions influence network formation mechanisms.

## Methodology

### Latent Space Models

The project builds upon the foundational latent space model introduced by Hoff, Raftery, and Handcock (2002), which assigns each node an unobserved position in a low-dimensional space. The likelihood of an edge between two nodes decreases with distance in that space.

### Geometric Frameworks

- **Euclidean Space**: Appropriate for homogeneous or community-based networks
- **Spherical Space**: Captures different structural properties through positive curvature
- **Hyperbolic Space**: Better captures hierarchical and scale-free structures due to negative curvature

### Signed Networks

The balanced inner-product model integrates structural balance theory ("the friend of my friend is a friend", "the enemy of my enemy is a friend") into a probabilistic framework of latent space.

## Dataset

The study uses the **Slashdot Zoo signed social network** (Leskovec et al., 2010), which contains both positive (friend) and negative (foe) edges. A working subset of nodes will be used to ensure computational feasibility while preserving the essential structural patterns of the full network.

## Expected Contributions

This study aims to advance the theoretical understanding of how geometric and metric choices in latent space models influence the structure and interpretation of network data by:

- Providing a systematic comparison of network properties under varying geometric curvatures
- Establishing connections between spectral graph theory and latent space geometry
- Extending latent space models to accommodate signed relationships in a geometrically principled manner
- Offering insights into how curvature-based latent representations can improve model interpretability and prediction for real-world network data

## Installation

### Prerequisites

- Python 3.8+ (or R, depending on implementation)
- Required packages (to be specified based on implementation):
  - Network analysis libraries (e.g., NetworkX, igraph)
  - Numerical computing (e.g., NumPy, SciPy)
  - Statistical modeling (e.g., PyTorch, TensorFlow, or Stan)
  - Visualization (e.g., Matplotlib, Plotly)

### Setup

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd Final

# Install dependencies
pip install -r requirements.txt
```

## Usage

(To be updated based on implementation)

```python
# Example usage (to be implemented)
from latent_space_models import EuclideanModel, HyperbolicModel, SphericalModel

# Load dataset
network = load_slashdot_data()

# Fit models with different geometries
euclidean_model = EuclideanModel()
hyperbolic_model = HyperbolicModel()
spherical_model = SphericalModel()

# Compare results
results = compare_geometries([euclidean_model, hyperbolic_model, spherical_model])
```

## Project Structure

```
Final/
├── README.md
├── data/
│   └── slashdot_zoo/          # Dataset files
├── src/
│   ├── models/                # Latent space model implementations
│   ├── analysis/              # Analysis scripts
│   └── visualization/         # Visualization utilities
├── notebooks/                 # Jupyter notebooks for exploration
├── results/                   # Output results and figures
└── requirements.txt           # Python dependencies
```

## References

- Hoff, P. D., Raftery, A. E., & Handcock, M. S. (2002). Latent space approaches to social network analysis. *Journal of the American Statistical Association*, 97(460), 1090-1098.

- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Signed networks in social media. *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.

- Smith, A., Asta, D. M., & Calder, C. A. (2019). The geometry of continuous latent space models for network data. *Statistical Science*, 34(3), 428-453.

- Tang, M., & Zhu, J. (2025). Population-level balance in signed latent space network models. *Journal of Machine Learning Research* (forthcoming).

## License

(To be specified)

## Contact

**Peidi Li**  
Supervisor: Cornelius Fritz

---

*This project bridges statistical modeling, geometry, and network theory, contributing to a unified understanding of how latent space geometry governs observable network complexity.*

