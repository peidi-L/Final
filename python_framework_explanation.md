To code these models in Python, you generally follow a three-step process: **(1) Define the Distance/Similarity Function**, **(2) Define the Probability Model (Log-Likelihood)**, and **(3) Optimize** using a minimizer to find the best latent positions () and baseline parameters ().



Below is a Python framework implementing the four geometries you requested, based on the mathematical definitions in the papers.



### **1. Prerequisites & Setup**



You will need `numpy` for matrix operations and `scipy.optimize` to find the best latent positions.



```python

import numpy as np  # Import numpy for numerical operations

from scipy.optimize import minimize  # Import minimize function for optimization

from scipy.special import expit  # Import expit (sigmoid) function for probability calculation



# Sample Data: Random Adjacency Matrix (5 nodes)

# 1 = Tie, 0 = No Tie

Y_sample = np.array([  # Define a sample adjacency matrix

    [0, 1, 1, 0, 0],  # Row 0: Node 0 is connected to Node 1 and 2

    [1, 0, 1, 0, 0],  # Row 1: Node 1 is connected to Node 0 and 2

    [1, 1, 0, 1, 0],  # Row 2: Node 2 is connected to Node 0, 1, and 3

    [0, 0, 1, 0, 1],  # Row 3: Node 3 is connected to Node 2 and 4

    [0, 0, 0, 1, 0]   # Row 4: Node 4 is connected to Node 3

])

n_nodes = Y_sample.shape[0]  # Get the number of nodes (5)

dim = 2  # Set latent space dimension to 2 (2D latent space)



```



---



### **2. Euclidean Distance (Flat)**



This implements the standard **Distance Model** from Hoff et al. (2002) .



**The Math:** .



```python

def euclidean_distance(zi, zj):

    """Calculates straight-line distance."""

    return np.linalg.norm(zi - zj)  # Calculate L2 norm (Euclidean distance) between vectors zi and zj



def neg_log_likelihood_euclidean(params, Y):

    """

    params: Flat array containing alpha (1) + Z (n_nodes * dim)

    """

    alpha = params[0]  # Extract the first parameter as alpha (baseline intercept)

    Z = params[1:].reshape((n_nodes, dim))  # Reshape the rest of params into the latent position matrix Z (n_nodes x dim)

    

    nll = 0  # Initialize negative log-likelihood accumulator

    # Iterate over all unique pairs (upper triangle of adjacency matrix)

    for i in range(n_nodes):  # Loop through each node i

        for j in range(i + 1, n_nodes):  # Loop through each node j > i

            dist = euclidean_distance(Z[i], Z[j])  # Calculate Euclidean distance between node i and node j

            eta = alpha - dist  # Linear predictor: baseline alpha minus distance (closer = higher prob)

            

            # Probability via sigmoid

            prob = expit(eta)  # Convert linear predictor to probability (0 to 1) using sigmoid

            

            # Avoid log(0) errors

            epsilon = 1e-10  # Small value to prevent numerical instability

            prob = np.clip(prob, epsilon, 1 - epsilon)  # Clip probability to be within [epsilon, 1-epsilon]

            

            # Binary Cross Entropy

            if Y[i, j] == 1:  # If there is an edge (connection)

                nll -= np.log(prob)  # Add negative log-probability of edge existing

            else:  # If there is no edge

                nll -= np.log(1 - prob)  # Add negative log-probability of edge NOT existing

    return nll  # Return the total negative log-likelihood



```



---



### **3. Hyperbolic Distance (Negative Curvature)**



This implements the polar coordinate formula emphasized by Smith et al. (2019) .



**The Math:** .



*Coding Note:* To optimize this, the latent vector `Z` is treated as storing polar coordinates: Column 0 is  (radius), Column 1 is  (angle).



```python

def hyperbolic_distance_polar(zi, zj):

    """

    zi, zj: Arrays [r, phi]

    Implements the formula from Smith et al. (2019)

    """

    ri, phi_i = zi[0], zi[1]  # Extract radius and angle for node i

    rj, phi_j = zj[0], zj[1]  # Extract radius and angle for node j

    

    # Angular difference

    delta_phi = np.pi - np.abs(np.pi - np.abs(phi_i - phi_j)) # Calculate minimal angular difference considering periodicity

    

    # Hyperbolic Law of Cosines

    term = np.cosh(ri) * np.cosh(rj) - np.sinh(ri) * np.sinh(rj) * np.cos(delta_phi) # Compute the argument for arccosh

    

    # Numerical stability clip (acosh requires x >= 1)

    term = np.maximum(term, 1.0 + 1e-10)  # Ensure term is at least 1 (plus epsilon) for arccosh domain

    

    return np.arccosh(term)  # Return hyperbolic distance using inverse hyperbolic cosine



def neg_log_likelihood_hyperbolic(params, Y):

    # Unpack params

    alpha = params[0]  # Extract alpha

    Z = params[1:].reshape((n_nodes, dim))  # Reshape latent positions

    

    # Constraint: Radius r must be positive. 

    # In practice, use bounds in the optimizer or absolute value here.

    Z[:, 0] = np.abs(Z[:, 0])   # Force radii (first column) to be non-negative

    

    nll = 0  # Initialize negative log-likelihood

    for i in range(n_nodes):  # Loop through nodes

        for j in range(i + 1, n_nodes):  # Loop through unique pairs

            dist = hyperbolic_distance_polar(Z[i], Z[j])  # Calculate hyperbolic distance

            eta = alpha - dist # Linear predictor: alpha minus distance

            prob = expit(eta)  # Calculate probability

            prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Clip for stability

            

            if Y[i, j] == 1:  # If edge exists

                nll -= np.log(prob)  # Add term for edge presence

            else:  # If edge does not exist

                nll -= np.log(1 - prob)  # Add term for edge absence

    return nll  # Return total NLL



```



---



### **4. Spherical Distance (Positive Curvature)**



This implements the Great Circle distance on a hypersphere.



**The Math:** .



```python

def spherical_distance(zi, zj):

    """Calculates Great Circle distance (angle)."""

    # Normalize vectors to unit length to ensure they are on the sphere

    zi_norm = zi / (np.linalg.norm(zi) + 1e-10)  # Normalize zi

    zj_norm = zj / (np.linalg.norm(zj) + 1e-10)  # Normalize zj

    

    dot_product = np.dot(zi_norm, zj_norm)  # Calculate dot product of normalized vectors

    

    # Numerical stability clip for arccos (-1 to 1)

    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure dot product is within valid range for arccos

    

    return np.arccos(dot_product)  # Return spherical distance (angle)



# The likelihood function follows the exact same pattern as Euclidean,

# just replacing 'euclidean_distance' with 'spherical_distance'.



```



---



### **5. Inner Product (Dot Product)**



This uses the projection metric suitable for signed or directed networks.



**The Math:** .



*Coding Note:* Unlike distance models where "smaller is better," here "larger is better" (more aligned). So the predictor is , not minus.



```python

def inner_product(zi, zj):

    return np.dot(zi, zj)  # Calculate dot product similarity between zi and zj



def neg_log_likelihood_inner_product(params, Y):

    alpha = params[0]  # Extract alpha

    Z = params[1:].reshape((n_nodes, dim))  # Reshape latent positions

    

    nll = 0  # Initialize NLL

    # Note: Inner product models are often used for DIRECTED graphs, 

    # so we iterate i != j, not just i < j.

    for i in range(n_nodes):  # Loop through all nodes as source

        for j in range(n_nodes):  # Loop through all nodes as target

            if i == j: continue  # Skip self-loops

            

            similarity = inner_product(Z[i], Z[j])  # Calculate similarity

            eta = alpha + similarity  # Linear predictor: alpha PLUS similarity (higher similarity = higher prob)

            

            prob = expit(eta)  # Calculate probability

            prob = np.clip(prob, 1e-10, 1 - 1e-10)  # Clip for stability

            

            if Y[i, j] == 1:  # If edge exists

                nll -= np.log(prob)  # Add term for edge presence

            else:  # If edge does not exist

                nll -= np.log(1 - prob)  # Add term for edge absence

    return nll  # Return total NLL



```



---



### **6. Running the Optimization**



To actually get the coordinates, you minimize the chosen `neg_log_likelihood` function.



```python

# 1. Initial Guess (Random)

initial_alpha = 0.0  # Start with alpha = 0

initial_Z = np.random.normal(size=(n_nodes * dim))  # Random initial latent positions

initial_params = np.concatenate(([initial_alpha], initial_Z))  # Combine into one parameter array



# 2. Optimize (Example using Euclidean)

result = minimize(  # Use scipy.optimize.minimize

    fun=neg_log_likelihood_euclidean,  # The function to minimize (NLL)

    x0=initial_params,  # Initial parameter guess

    args=(Y_sample,),  # Extra arguments (the data Y)

    method='BFGS' # Optimization algorithm (L-BFGS-B is better for bounds)

)



# 3. Extract Results

fitted_alpha = result.x[0]  # Get the optimized alpha

fitted_Z = result.x[1:].reshape((n_nodes, dim))  # Get and reshape the optimized latent positions



print("Learned Latent Positions:\n", fitted_Z)  # Print the result



```
