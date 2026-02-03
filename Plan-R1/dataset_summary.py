import os
import json
import torch
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

def process_directory(args):
    """Process a single directory and return file metadata."""
    dir_name, root, mode = args
    root_path = Path(root)
    dir_path = root_path / 'nuplan-v1.1' / 'processed' / f"{dir_name}-processed-{mode}-PlanR1"

    if not dir_path.exists():
        return dir_name, None

    files_data = []
    scenario_types = defaultdict(int)

    for file_path in dir_path.glob("*.pt"):
        parts = file_path.stem.split('-', 1)
        scenario_type = parts[0] if len(parts) == 2 else 'unknown'
        scenario_name = parts[1] if len(parts) == 2 else file_path.stem

        # Store relative path from root instead of full path
        relative_path = file_path.relative_to(root_path)

        files_data.append({
            'filename': file_path.name,
            'relative_path': str(relative_path),
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'scenario_type': scenario_type,
            'scenario_name': scenario_name
        })
        scenario_types[scenario_type] += 1

    # Store relative path for directory as well
    relative_dir_path = dir_path.relative_to(root_path)

    return dir_name, {
        'path': str(relative_dir_path),
        'num_files': len(files_data),
        'total_size_mb': sum(f['size_mb'] for f in files_data),
        'scenario_types': dict(scenario_types),
        'files': files_data  # Return ALL files for file index
    }

def analyze_sample_file(file_path):
    """Analyze structure of a single file."""
    try:
        data = torch.load(file_path, map_location='cpu')
        
        def get_structure(obj, depth=2):
            if depth <= 0:
                return type(obj).__name__
            
            if isinstance(obj, dict):
                return {'type': 'dict', 'keys': list(obj.keys()), 
                        'sample': {k: get_structure(v, depth-1) for k in list(obj.keys())[:3]}}
            elif isinstance(obj, torch.Tensor):
                return {'type': 'Tensor', 'shape': list(obj.shape), 'dtype': str(obj.dtype)}
            elif isinstance(obj, (list, tuple)):
                return {'type': type(obj).__name__, 'length': len(obj),
                        'element': get_structure(obj[0], depth-1) if obj else None}
            return type(obj).__name__
        
        return get_structure(data)
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Parallel data file summarizer')
    parser.add_argument('root_path', help='Root directory')
    parser.add_argument('--mode', default='pred', help='Mode (default: pred)')
    parser.add_argument('--workers', type=int, default=cpu_count(), help='Number of parallel workers')
    parser.add_argument('--analyze', action='store_true', help='Analyze sample file structure')
    parser.add_argument('--output', help='Output JSON path')
    
    args = parser.parse_args()
    
    dirs = ['train_boston', 'train_pittsburgh', 'train_singapore'] + \
           [f'train_vegas_{i}' for i in range(1, 7)] + \
           ['val']
    
    print(f"Processing {len(dirs)} directories with {args.workers} workers...")
    
    # Parallel directory processing
    results = {}
    scenario_type_totals = defaultdict(int)
    total_files = 0
    total_size_gb = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_directory, (d, args.root_path, args.mode)): d 
                  for d in dirs}
        
        for future in as_completed(futures):
            dir_name, data = future.result()
            if data:
                results[dir_name] = data
                total_files += data['num_files']
                total_size_gb += data['total_size_mb'] / 1024
                for st, count in data['scenario_types'].items():
                    scenario_type_totals[st] += count
                print(f"✓ {dir_name}: {data['num_files']} files, "
                      f"{data['total_size_mb']/1024:.1f} GB")
            else:
                print(f"✗ {dir_name}: Not found")
    
    # Analyze sample structure if requested
    structure = None
    if args.analyze and results:
        first_dir = next(iter(results.values()))
        if first_dir['files']:
            sample_path = Path(first_dir['path']) / first_dir['files'][0]['filename']
            print(f"\nAnalyzing structure from: {sample_path.name}")
            structure = analyze_sample_file(sample_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {total_files:,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Directories found: {len(results)}/{len(dirs)}")
    
    print(f"\nScenario types:")
    for st, count in sorted(scenario_type_totals.items()):
        print(f"  {st}: {count:,} files")
    
    print(f"\nFiles per directory:")
    for dir_name, data in sorted(results.items()):
        print(f"  {dir_name}: {data['num_files']:,}")
    
    # Save output - create file index for dataloader
    file_index = {}
    for dir_name, data in results.items():
        for file_info in data['files']:
            # Use filename as key for easy lookup
            file_index[file_info['filename']] = {
                'relative_path': file_info['relative_path'],
                'dir': dir_name,
                'scenario_type': file_info['scenario_type'],
                'scenario_name': file_info['scenario_name'],
                'size_mb': file_info['size_mb']
            }
    
    output_data = {
        'metadata': {
            'root_path': args.root_path,
            'mode': args.mode,
            'timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'total_size_gb': total_size_gb,
            'scenario_types': dict(scenario_type_totals)
        },
        'file_index': file_index,
        'structure': structure
    }
    
    # Save main index file
    output_path = Path(args.root_path) / f"file_index_{args.mode}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")

    # Save per-directory JSON files for nuplan_dataset.py compatibility
    processed_dir = Path(args.root_path) / 'nuplan-v1.1' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    for dir_name, data in results.items():
        dir_file_index = {
            'root': args.root_path,
            'mode': args.mode,
            'dir_name': dir_name,
            'num_files': data['num_files'],
            'files': [
                {
                    'filename': f['filename'],
                    'relative_path': f['relative_path'],
                    'scenario_type': f['scenario_type'],
                    'scenario_name': f['scenario_name']
                }
                for f in data['files']
            ]
        }
        dir_output_path = processed_dir / f"{dir_name}-file_index-{args.mode}-PlanR1.json"
        with open(dir_output_path, 'w') as f:
            json.dump(dir_file_index, f, indent=2)
        print(f"Saved: {dir_output_path}")

if __name__ == "__main__":
    main()