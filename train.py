import numpy as np

from scipy.optimize import minimize

import loader

import models

def train_sequential():
    # Load the top 1000 nodes and their connections
    links = loader.get_top_1000_core_nodes("soc-sign-Slashdot090221.txt")

    # Find the maximum node ID to figure out how many users we have
    max_id = 0
    for row in links:
        if row[0] > max_id:
            max_id = row[0]
        if row[1] > max_id:
            max_id = row[1]

    # Number of users is max_id + 1 because IDs start from 0
    num_users = max_id + 1
    print(f"Training on {num_users} users with {len(links)} connections.")

    # Create random starting positions for all nodes in 2D Euclidean space
    initial_positions = models.initialize_positions(num_users, 2, 'euclidean')
    # Flatten the positions into a 1D array for the optimizer
    params = initial_positions.flatten()

    print("Starting Stage 1: Clustering friends...")

    def stage1_loss(current_params):
        # Reshape the flattened parameters back into (num_users, 2) shape
        pos = current_params.reshape((num_users, 2))
        total_error = 0

        # Go through all connections
        for source, target, sign in links:
            # Only look at friend connections (sign == 1)
            if sign == 1:
                # Get the positions of the two nodes
                u = pos[source]
                v = pos[target]

                # Calculate the distance between them
                diff = u - v
                dist = np.sqrt(np.sum(diff**2))

                # Add error if distance is not close to 1.0
                # We want friends to be about distance 1.0 apart
                total_error += (dist - 1.0) ** 2

        return total_error

    # Optimize to minimize the loss function
    result_1 = minimize(
        stage1_loss, 
        params, 
        method='L-BFGS-B', 
        options={'maxiter': 30, 'disp': True}
    )

    # Save the positions after stage 1
    stage_1_positions = result_1.x
    print("Stage 1 done.")

    print("Starting Stage 2: Handling enemies...")

    def stage2_loss(current_params):
        # Reshape the flattened parameters back into (num_users, 2) shape
        pos = current_params.reshape((num_users, 2))
        total_error = 0

        # Go through all connections
        for source, target, sign in links:
            # Get the positions of the two nodes
            u = pos[source]
            v = pos[target]

            # Calculate the distance between them
            diff = u - v
            dist = np.sqrt(np.sum(diff**2))

            if sign == 1:
                # For friends, keep them close (distance around 1.0)
                # Use stronger penalty (2.0) to maintain what we learned in stage 1
                total_error += 2.0 * ((dist - 1.0) ** 2)
            else:
                # For enemies, we want them far apart
                # Cap the distance at 10.0 to avoid extreme values
                if dist > 10.0:
                    capped_dist = 10.0
                else:
                    capped_dist = dist
                # Subtract distance so smaller distances increase error
                total_error -= capped_dist

        return total_error

    # Optimize to minimize the loss function, starting from stage 1 positions
    result_2 = minimize(
        stage2_loss, 
        stage_1_positions, 
        method='L-BFGS-B', 
        options={'maxiter': 50, 'disp': True}
    )

    # Reshape the final positions back into (num_users, 2) shape
    final_positions = result_2.x.reshape((num_users, 2))
    print("Stage 2 done.")

    # Return the final positions for all nodes
    return final_positions

# This part only runs if you run this file directly (not if you import it)
if __name__ == "__main__":
    # Run the training and get the final positions
    final_map = train_sequential()
    # Print the final coordinates for user 0 as an example
    print("Final coordinates for user 0:")
    print(final_map[0])

