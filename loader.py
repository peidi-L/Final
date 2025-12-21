import pandas as pd
import numpy as np
import os

def load_data(file_path, core_node_count=1000):
    """
    Loads signed network data and filters for the k-core.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load dataset using pandas
    print(f"Loading dataset: {file_path}...")
    df = pd.read_csv(file_path, 
                     sep='\t', 
                     comment='#', 
                     header=None, 
                     names=['source', 'target', 'sign'])

    # Filter for top k active nodes to ensure dense connectivity
    # This is a standard preprocessing step in network analysis
    user_counts = pd.concat([df['source'], df['target']]).value_counts()
    
    top_users = user_counts.head(core_node_count).index
    valid_users = set(top_users)
    
    # Map original IDs to continuous 0..N indices
    # (Critical for matrix operations later)
    id_map = {original: new for new, original in enumerate(top_users)}
    
    # Filter DataFrame
    mask = df['source'].isin(valid_users) & df['target'].isin(valid_users)
    df_filtered = df[mask].copy()
    
    # Remap IDs
    df_filtered['source_id'] = df_filtered['source'].map(id_map)
    df_filtered['target_id'] = df_filtered['target'].map(id_map)
    
    # Convert to numpy for the optimizer
    # Format: [source_id, target_id, sign]
    final_data = df_filtered[['source_id', 'target_id', 'sign']].values.astype(int)
    
    print(f"Data loaded. Nodes: {len(valid_users)}, Edges: {len(final_data)}")
    print(f"Class balance: {np.mean(final_data[:, 2] == 1):.2%} positive edges")
    
    return final_data, len(valid_users)

if __name__ == "__main__":
    # Quick sanity check
    data, n_users = load_data("soc-sign-Slashdot090221.txt")
