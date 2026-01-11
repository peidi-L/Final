# Signed Latent Space Models (LSM)

This project implements a **Signed Latent Space Model** to analyze social networks containing both positive (friends) and negative (foes) relationships. It investigates how different geometric spaces (Euclidean, Spherical, Hyperbolic) and distance metrics affect the modeling of social structure.

## Project Structure

* **`simulation_python.py`**: Generates synthetic signed networks using specific geometric rules and log-odds probability models.
* **`signed_lsm_optimization.py`**: The core model class. Uses Maximum Likelihood Estimation (MLE) and `L-BFGS-B` optimization to learn latent positions and parameters from network data.
* **`distance_calculator.py`**: A library of distance metrics (Euclidean, Manhattan, Chebyshev, Spherical, Hyperbolic) with numerical stability handling.
* **`verify_pipeline.py`**: An end-to-end verification script that generates ground-truth data, trains the model, and validates that the learned parameters match the truth.
* **`IMPLEMENTATION_SUMMARY.md`**: Detailed explanation of the simulation and modeling pipeline.

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Verification**:
   To prove the model works as expected (generating data -> training -> recovering parameters):
   ```bash
   python3 verify_pipeline.py
   ```

3. **Run Simulation**:
   To generate network visualizations in different geometries:
   ```bash
   python3 simulation_python.py
   ```

## Implementation Details

The model assumes:
* **Friendship (+1)** is driven by proximity (Homophily): $\log(P_{pos}/P_{null}) = \alpha_{pos} - d$
* **Animosity (-1)** is driven by "signed distance" (Polarization): $\log(P_{neg}/P_{null}) = \alpha_{neg} + \beta \cdot d$

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for a full breakdown of the logic.
