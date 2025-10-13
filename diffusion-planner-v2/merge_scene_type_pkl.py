import os
import pickle
from pathlib import Path
from collections import defaultdict

def merge_pkl_files(input_dir):
    """Merge all *_by_type.pkl files"""
    merged = defaultdict(list)
    
    pkl_files = list(Path(input_dir).glob("*_by_type.pkl"))
    print(f"Found {len(pkl_files)} pickle files to merge")
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        for scenario_type, filenames in data.items():
            merged[scenario_type].extend(filenames)
        
        print(f"  Merged {pkl_file.name}")
    
    # Remove duplicates
    for scenario_type in merged:
        original_count = len(merged[scenario_type])
        merged[scenario_type] = list(set(merged[scenario_type]))
        if original_count != len(merged[scenario_type]):
            print(f"  Removed {original_count - len(merged[scenario_type])} duplicates from {scenario_type}")
    
    return dict(merged)

def print_stats(data):
    """Print detailed statistics"""
    total = sum(len(v) for v in data.values())
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total scenario types: {len(data)}")
    print(f"Total scenarios: {total:,}")
    
    # Calculate min, max, mean
    sizes = [len(v) for v in data.values()]
    print(f"Min scenarios per type: {min(sizes):,}")
    print(f"Max scenarios per type: {max(sizes):,}")
    print(f"Avg scenarios per type: {sum(sizes)//len(sizes):,}")
    
    print("\nScenarios by type:")
    for scenario_type, files in sorted(data.items(), key=lambda x: len(x[1]), reverse=True):
        percentage = (len(files) / total) * 100
        print(f"  {scenario_type}: {len(files):,} ({percentage:.1f}%)")

def main():
    # Configuration
    input_dir = "/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/trainval_batch/metadata_splits"  # Directory with pkl files
    output_dir = "/data/temp/Expire180Days/automlops-fraprd/long-retention/diffusion_planner/trainval_batch/metadata"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge all pickle files
    print("Merging pickle files...")
    merged = merge_pkl_files(input_dir)
    
    # Print statistics
    print_stats(merged)
    
    # Save merged dataset
    output_path = f"{output_dir}/merged_all_scenarios.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(merged, f)
    print(f"\nSaved merged dataset to: {output_path}")
    
    # Create a simple file list
    all_files = []
    for files in merged.values():
        all_files.extend(files)
    
    list_path = f"{output_dir}/all_files.txt"
    with open(list_path, 'w') as f:
        for filename in all_files:
            f.write(f"{filename}\n")
    print(f"Saved complete file list to: {list_path} ({len(all_files):,} files)")

if __name__ == "__main__":
    main()