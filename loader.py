from collections import Counter

def get_top_1000_core_nodes(file_path):
    print("Counting connections...")
    
    user_counts = Counter()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split()
            source = parts[0]
            target = parts[1]
            
            user_counts[source] += 1
            user_counts[target] += 1

    print(f"Total users found: {len(user_counts)}")

    top_1000_users = user_counts.most_common(1000)
    
    valid_users_set = set()
    id_map = {}
    
    for new_id, (original_name, _) in enumerate(top_1000_users):
        valid_users_set.add(original_name)
        id_map[original_name] = new_id
        
    print(f"Top user is ID {top_1000_users[0][0]} with {top_1000_users[0][1]} connections.")

    final_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            parts = line.strip().split()
            source = parts[0]
            target = parts[1]
            sign = int(parts[2])
            
            if source in valid_users_set and target in valid_users_set:
                new_source_id = id_map[source]
                new_target_id = id_map[target]
                final_data.append([new_source_id, new_target_id, sign])
                
    print(f"Found {len(final_data)} connections between the top 1000 users.")
    
    return final_data

if __name__ == "__main__":
    my_data = get_top_1000_core_nodes("soc-sign-Slashdot090221.txt")