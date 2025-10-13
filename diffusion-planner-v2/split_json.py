
import json
import os
import math

# Configuration
input_file = "/root/planning-intern/Diffusion-Planner/nuplan_train.json"  # Replace with the actual path
output_dir = "/root/planning-intern/Diffusion-Planner/train_splits"       # Replace with your desired output directory
num_parts = 600                            # Number of parts to split into

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Calculate the number of items per split
total_items = len(data)
items_per_part = math.ceil(total_items / num_parts)

# Split and save each part
split_files = []
for i in range(num_parts):
    start_index = i * items_per_part
    end_index = min(start_index + items_per_part, total_items)
    part_data = data[start_index:end_index]
    
    output_file = os.path.join(output_dir, f"nuplan_train_{i:03d}.json")
    with open(output_file, 'w') as f:
        json.dump(part_data, f, indent=2)
    split_files.append(output_file)

print(f"Split complete. {num_parts} files saved in '{output_dir}' directory.")

# Verification step: Combine and compare
reconstructed = []
for file in split_files:
    with open(file, 'r') as f:
        reconstructed.extend(json.load(f))

assert reconstructed == data, "Reconstructed data does not match original!"
print("✅ Verification successful: Reconstructed data matches the original.")
