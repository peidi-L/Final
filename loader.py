def get_top_1000_core_nodes(file_path):
    # This function finds the top 1000 most popular users and give them new IDs
    print("Counting connections...")
    
    # The below will count how many friends/connections each node has
    user_counts = {}
    

    with open(file_path, 'r') as f:
        for line in f:
            # skip comment lines
            if line.startswith('#'):
                continue
            
            # Each line has information separated by spaces, we split it up into parts so we can look at each part separately
            parts = line.strip().split()
            # The first part is the node who made the connection
            source = parts[0]
            # The second part is the node being connected to
            target = parts[1]
            
            # To count how many times each node appears
            # If we've seen this source node before, we add 1 to their count
            # If we haven't seen them before, we start counting at 1
            if source in user_counts:
                user_counts[source] = user_counts[source] + 1
            else:
                user_counts[source] = 1
                
            # We do the same thing for the target node because they're also part of a connection
            # So both nodes get their count increased when we see this connection
            if target in user_counts:
                user_counts[target] = user_counts[target] + 1
            else:
                user_counts[target] = 1

    # Print out how many total different nodes we found
    print(f"Total users found: {len(user_counts)}")

    # Turn the dictionary into a list of pairs (node name, connection count)
    all_users = list(user_counts.items())
    # Sort them from highest to lowest connection count

    all_users.sort(key=lambda item: item[1], reverse=True)
    
    # Take the first 1000 nodes from our sorted list with the highest connection count
    top_1000_users = all_users[:1000]
    
    # Make a set to hold all the names of our top 1000 nodes so we can quickly check if a node is in it
    valid_users_set = set()

    # Map to convert old IDs to new IDs
    id_map = {} 
    
    # Go through our top 1000 nodes and give them new sequential IDs from 0 to 999
    for new_id, user_data in enumerate(top_1000_users):
        # The original_name is their old ID from the file
        original_name = user_data[0]
        # Add them to our set of valid nodes
        valid_users_set.add(original_name)
        # Remember that this original name maps to this new ID
        id_map[original_name] = new_id
        
    # Print out who the top node is and how many connections they have
    print(f"Top user is ID {top_1000_users[0][0]} with {top_1000_users[0][1]} connections.")

    # Store all the final connections/edges between our top 1000 nodes
    final_data = []
    
    # Read through the file again to find all the connections
    # between our top 1000 nodes and convert their IDs to the new sequential ones
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('#'):
                continue
                
            # Split the line into parts
            parts = line.strip().split()
            # Get the source node
            source = parts[0]
            # Get the target node
            target = parts[1]
            # Get the sign which tells us if they're friends (+1) or enemies (-1)
            sign = int(parts[2])
            
            # Only keep this connection if both nodes are in our top 1000 list
            if source in valid_users_set and target in valid_users_set:
                # Convert the old IDs to the new sequential IDs using our map
                new_source_id = id_map[source]
                new_target_id = id_map[target]
                
                # Add this connection to our final data list with the new IDs
                final_data.append([new_source_id, new_target_id, sign])
                
    # Print out how many connections we found between our top 1000 nodes
    print(f"Found {len(final_data)} connections between the top 1000 users.")
    
    # Return all the connections we found, with the new sequential IDs
    return final_data

# This part only runs if you run this file directly (not if you import it)
if __name__ == "__main__":
    # Call our function with the name of the data file
    my_data = get_top_1000_core_nodes("soc-sign-Slashdot090221.txt")