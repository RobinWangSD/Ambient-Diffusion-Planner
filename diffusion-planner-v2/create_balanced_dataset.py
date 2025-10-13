import os
import pickle
import random
from pathlib import Path
from collections import defaultdict

def load_merged_data(input_path):
    """Load the merged pickle file"""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Remove 'unknown' class
    if 'unknown' in data:
        del data['unknown']
        print("Removed 'unknown' class")
    
    return data

def load_path_mapping(mapping_path):
    """Load the filename to path mapping"""
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    print(f"Loaded path mapping with {len(mapping):,} files")
    return mapping

def get_long_tail_classes(data, threshold=10000):
    """Identify long-tail classes (< threshold samples)"""
    long_tail = {}
    head = {}
    
    for scenario_type, files in data.items():
        if len(files) < threshold:
            long_tail[scenario_type] = files
        else:
            head[scenario_type] = files
    
    return long_tail, head

def create_validation_set(data, val_types, samples_per_type, seed=42):
    """Create validation set from specific scenario types"""
    random.seed(seed)
    val_data = {}
    
    for scenario_type in val_types:
        if scenario_type in data:
            files = data[scenario_type]
            if len(files) >= samples_per_type:
                val_data[scenario_type] = random.sample(files, samples_per_type)
            else:
                val_data[scenario_type] = files
                print(f"Warning: {scenario_type} has only {len(files)} samples for validation")
        else:
            print(f"Warning: {scenario_type} not found in data")
    
    return val_data

def sample_head_classes(head_data, target_samples, seed=42):
    """Sample from head classes proportionally"""
    random.seed(seed)
    
    # Calculate total available samples
    total_available = sum(len(files) for files in head_data.values())
    
    sampled = {}
    remaining_target = target_samples
    
    # Sort by size to sample from largest first
    sorted_classes = sorted(head_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for scenario_type, files in sorted_classes:
        # Calculate proportion
        proportion = len(files) / total_available
        samples_to_take = min(int(target_samples * proportion), len(files))
        samples_to_take = min(samples_to_take, remaining_target)
        
        if samples_to_take > 0:
            sampled[scenario_type] = random.sample(files, samples_to_take)
            remaining_target -= samples_to_take
    
    # If we still need more samples, take more from the largest classes
    while remaining_target > 0:
        for scenario_type, files in sorted_classes:
            if len(sampled[scenario_type]) < len(files):
                additional = min(remaining_target, len(files) - len(sampled[scenario_type]))
                already_sampled = set(sampled[scenario_type])
                remaining_files = [f for f in files if f not in already_sampled]
                sampled[scenario_type].extend(random.sample(remaining_files, additional))
                remaining_target -= additional
                if remaining_target <= 0:
                    break
    
    return sampled

def combine_datasets(long_tail, head_sample):
    """Combine long-tail and head samples"""
    combined = {}
    
    # Add all long-tail
    for scenario_type, files in long_tail.items():
        combined[scenario_type] = files
    
    # Add head samples
    for scenario_type, files in head_sample.items():
        combined[scenario_type] = files
    
    return combined

def save_dataset(data, output_path, name, path_mapping=None):
    """Save dataset and create file list with paths"""
    # Save pickle with scenario type mapping
    with open(f"{output_path}/{name}.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    # Create file list
    all_files = []
    for files in data.values():
        all_files.extend(files)
    
    # Save simple file list
    with open(f"{output_path}/{name}_files.txt", 'w') as f:
        for filename in all_files:
            f.write(f"{filename}\n")
    
    # Save path mapping if available
    if path_mapping:
        dataset_mapping = {}
        missing_count = 0
        for filename in all_files:
            if filename in path_mapping:
                dataset_mapping[filename] = path_mapping[filename]
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"  Warning: {missing_count} files not found in path mapping")
        
        # Save dataset-specific path mapping
        mapping_path = f"{output_path}/{name}_path_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(dataset_mapping, f)
        print(f"  Saved path mapping: {len(dataset_mapping):,} files")
    
    return len(all_files)

def print_dataset_stats(data, name, long_tail_classes):
    """Print dataset statistics"""
    total = sum(len(v) for v in data.values())
    print(f"\n{name}:")
    print(f"  Total files: {total:,}")
    print(f"  Total types: {len(data)}")
    
    # Count long-tail vs head using the original long-tail class list
    long_tail_count = sum(len(data[k]) for k in long_tail_classes if k in data)
    head_count = total - long_tail_count
    if long_tail_count > 0:
        ratio = head_count / long_tail_count if long_tail_count > 0 else 0
        print(f"  Long-tail samples: {long_tail_count:,}")
        print(f"  Head samples: {head_count:,}")
        print(f"  Head:Tail ratio: {ratio:.1f}:1")

def main():
    # Configuration
    input_file = "/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/trainval_batch/metadata/merged_all_scenarios.pkl"
    path_mapping_file = "/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/trainval_batch/file_mapping.pkl"  # Adjust path as needed
    output_dir = "/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/trainval_batch/dataset_splits"
    
    # Validation set configuration
    val_scenario_types = [
        'behind_long_vehicle',
        'changing_lane_to_left',
        'changing_lane_to_right',
        'following_lane_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'near_multiple_vehicles',
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stationary_in_traffic',
        'stopping_with_lead',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross'
    ]
    val_samples_per_type = 10000 // len(val_scenario_types)  # ~714 per type
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load merged data
    print("Loading merged data...")
    data = load_merged_data(input_file)
    
    # Load path mapping
    print("\nLoading path mapping...")
    path_mapping = load_path_mapping(path_mapping_file)
    
    # Separate long-tail and head classes
    print("\nSeparating long-tail and head classes...")
    long_tail, head = get_long_tail_classes(data, threshold=10000)
    
    long_tail_total = sum(len(v) for v in long_tail.values())
    head_total = sum(len(v) for v in head.values())
    
    print(f"Long-tail classes: {len(long_tail)} classes, {long_tail_total:,} samples")
    print(f"Head classes: {len(head)} classes, {head_total:,} samples")
    
    # Create validation set
    print(f"\nCreating validation set...")
    val_data = create_validation_set(data, val_scenario_types, val_samples_per_type)
    val_count = save_dataset(val_data, output_dir, "validation", path_mapping)
    print(f"Validation set: {val_count:,} samples from {len(val_data)} types")
    
    # Define dataset configurations
    datasets = [
        ("dataset_150k", 75430),   # 1:1
        ("dataset_250k", 175430),   # 2.3:1
        ("dataset_500k", 425430),   # 5.7:1 ratio
        ("dataset_1m", 925430),     # 12.4:1
        ("dataset_3m", 2925430),    # 39.2:1
        ("dataset_10m", 9925430),   # 133.1:1
        ("dataset_20m", 19925430),  # 267.2:1
    ]
    
    print("\nCreating scaled datasets...")
    print("="*60)
    
    # Get list of long-tail class names for consistent counting
    long_tail_class_names = set(long_tail.keys())
    
    for name, head_samples in datasets:
        print(f"\nCreating {name}...")
        
        # Sample from head classes
        head_sample = sample_head_classes(head, head_samples, seed=42)
        
        # Combine with all long-tail
        combined = combine_datasets(long_tail, head_sample)
        
        # Save dataset with path mapping
        total_files = save_dataset(combined, output_dir, name, path_mapping)
        
        # Print stats with consistent long-tail counting
        print_dataset_stats(combined, name, long_tail_class_names)
    
    print("\n" + "="*60)
    print("All datasets created successfully!")
    print(f"Output directory: {output_dir}")
    print("\nCreated files for each dataset:")
    print("  - {name}.pkl: Scenario type to filenames mapping")
    print("  - {name}_files.txt: Simple list of all filenames")
    print("  - {name}_path_mapping.pkl: Filename to relative path mapping")

if __name__ == "__main__":
    main()