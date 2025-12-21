import numpy as np

from scipy.optimize import minimize

import loader

import models

def train_sequential():
    links = loader.get_top_1000_core_nodes("soc-sign-Slashdot090221.txt")
    
    links_array = np.array(links)
    num_users = int(links_array[:, :2].max()) + 1
    print(f"Training on {num_users} users with {len(links)} connections.")

    initial_positions = models.initialize_positions(num_users, 2, 'euclidean')
    params = initial_positions.flatten()

    friend_mask = links_array[:, 2] == 1
    friend_links = links_array[friend_mask]
    friend_sources = friend_links[:, 0].astype(int)
    friend_targets = friend_links[:, 1].astype(int)

    print("Starting Stage 1: Clustering friends...")

    def stage1_loss(current_params):
        pos = current_params.reshape((num_users, 2))
        
        source_pos = pos[friend_sources]
        target_pos = pos[friend_targets]
        diffs = source_pos - target_pos
        distances = np.linalg.norm(diffs, axis=1)
        
        return np.sum((distances - 1.0) ** 2)

    result_1 = minimize(
        stage1_loss, 
        params, 
        method='L-BFGS-B', 
        options={'maxiter': 30, 'disp': True}
    )

    stage_1_positions = result_1.x
    print("Stage 1 done.")

    print("Starting Stage 2: Handling enemies...")

    sources = links_array[:, 0].astype(int)
    targets = links_array[:, 1].astype(int)
    signs = links_array[:, 2]
    enemy_mask = signs == -1
    enemy_sources = sources[enemy_mask]
    enemy_targets = targets[enemy_mask]

    def stage2_loss(current_params):
        pos = current_params.reshape((num_users, 2))
        
        all_source_pos = pos[sources]
        all_target_pos = pos[targets]
        all_diffs = all_source_pos - all_target_pos
        all_distances = np.linalg.norm(all_diffs, axis=1)
        
        friend_error = 2.0 * np.sum((all_distances[friend_mask] - 1.0) ** 2)
        
        enemy_distances = all_distances[enemy_mask]
        capped_enemy_distances = np.clip(enemy_distances, None, 10.0)
        enemy_error = -np.sum(capped_enemy_distances)
        
        return friend_error + enemy_error

    result_2 = minimize(
        stage2_loss, 
        stage_1_positions, 
        method='L-BFGS-B', 
        options={'maxiter': 50, 'disp': True}
    )

    final_positions = result_2.x.reshape((num_users, 2))
    print("Stage 2 done.")

    return final_positions

if __name__ == "__main__":
    final_map = train_sequential()
    print("Final coordinates for user 0:")
    print(final_map[0])

