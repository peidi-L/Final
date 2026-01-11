# Signed Latent Space Model (LSM) Implementation Pipeline

Based on the code files, we have built a **closed-loop simulation pipeline** to verify the Signed Latent Space Model (LSM) hypothesis.

Instead of using messy real-world data (Slashdot), this code creates a "perfect" synthetic world where we control the rules, allows the model to observe only the edges, and then checks if the model can figure out the hidden rules (and node positions) that generated them.

Here is exactly what each part of the code is doing:

### 1. The Geometry Library (`distance_calculator.py`)

This is a utility file. It doesn't run any logic itself; it just provides the "rulers" to measure distance in 5 different mathematical universes.

* **Key Function:** `calculate_distances(point1, point2)`
* **What it does:** It takes two coordinates and returns their distance according to Euclidean, Manhattan, Chebyshev, Spherical, and Hyperbolic rules.
* **Important Detail:** For **Hyperbolic** distance (Poincaré disk), it includes safety checks (`epsilon` and clipping) to ensure the math doesn't crash if points get too close to the edge of the disk ($R=1$).

### 2. The "Generator" (`simulation_python.py`)

This script acts as "God" for the simulation. It creates the Ground Truth.

* **Generates Latent Positions:** It places $N$ nodes randomly in a 2D space.
    * *Euclidean:* Random points in a square.
    * *Spherical:* Random points on the surface of a circle/sphere.
    * *Hyperbolic:* Random points strictly inside the unit disk.

* **Calculates Probabilities:** It uses the exact formulas requested:
    * **Friendship (Eq 3):** $\log(P_{pos}/P_{null}) = \alpha_{pos} - d$ (Closer = higher probability).
    * **Animosity (Eq 4):** $\log(P_{neg}/P_{null}) = \alpha_{neg} + \beta \cdot d$ (Farther = higher probability, scaled by $\beta$).

* **Creates Edges:** It rolls a weighted die for every pair of nodes to decide if they are Friends (+1), Foes (-1), or Unconnected (0) based on those probabilities.

### 3. The "Learner" (`signed_lsm_optimization.py`)

This is the model that tries to reverse-engineer the Generator. It receives *only* the edges (Adjacency Matrix) and must guess the positions.

* **Initialization:** It starts by scattering nodes randomly (it doesn't know the true positions yet).
* **The Objective Function (`_neg_log_likelihood`):**
    * It calculates distances between its *guessed* positions.
    * It calculates the likelihood of the observed edges using the same formulas: $\alpha_{pos} - d$ and $\alpha_{neg} + \beta \cdot d$.
    * It returns a score (Negative Log Likelihood) telling the optimizer how "wrong" the current guesses are.

* **Optimization:** It uses the `L-BFGS-B` algorithm to repeatedly shift node positions and tweak parameters ($\alpha, \beta$) until the error is minimized.

### 4. The "Test Runner" (`verify_pipeline.py`)

This script ties everything together to prove the code works.

1. **Step 1 (Generate):** It calls the Generator to create a fake network with *known* parameters (e.g., `alpha_pos=1.5`, `beta=0.5`).
2. **Step 2 (Fit):** It creates a fresh `SignedLSM` model (which knows nothing) and trains it on that fake network.
3. **Step 3 (Verify):** It prints the parameters the model learned.
    * **Success Condition:** If the "Learned Parameters" printed at the end are close to the parameters used in the Generator (e.g., Learned $\beta \approx$ True $\beta$), the code is working correctly.

### Summary of the Workflow

1. **You define the physics:** "Friends are close, enemies are far (controlled by $\beta$)."
2. **`simulation_python.py`** builds a world obeying these physics.
3. **`signed_lsm_optimization.py`** observes the social network of that world and uses math to deduce the physics.
4. **`verify_pipeline.py`** confirms that the deduced physics match the original physics.
