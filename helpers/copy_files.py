#!/usr/bin/env python3

import os
import sys
import pickle
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import time

def load_mapping(pkl_path):
    """Load the file mapping from pickle file"""
    print(f"Loading mapping from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        mapping = pickle.load(f)
    print(f"Loaded {len(mapping):,} file mappings")
    return mapping

def organize_by_batch(mapping):
    """Organize files by their batch directory"""
    batch_to_files = defaultdict(list)
    
    for original, new_path in mapping.items():
        batch_dir = new_path.split('/')[0]  # e.g., "batch_000001"
        batch_to_files[batch_dir].append((original, new_path))
    
    return batch_to_files

def copy_batch_directory_simple(src_dir, dest_dir, batch_name, dry_run=False):
    """Simple directory copy for entire batch folders"""
    src_batch = src_dir / batch_name
    dest_batch = dest_dir / batch_name
    
    if not src_batch.exists():
        return f"✗ {batch_name}: Source not found", False
    
    if dry_run:
        file_count = sum(1 for _ in src_batch.rglob('*') if _.is_file())
        size = sum(f.stat().st_size for f in src_batch.rglob('*') if f.is_file())
        return f"[DRY RUN] Would copy {batch_name} ({file_count} files, {format_size(size)})", True
    
    try:
        # Use rsync if available (faster for large directories)
        if shutil.which('rsync'):
            os.makedirs(dest_batch, exist_ok=True)
            result = os.system(f'rsync -a --quiet "{src_batch}/" "{dest_batch}/"')
            if result == 0:
                file_count = sum(1 for _ in dest_batch.rglob('*') if _.is_file())
                return f"✓ {batch_name} ({file_count} files)", True
        
        # Fallback to shutil
        if dest_batch.exists():
            shutil.rmtree(dest_batch)
        shutil.copytree(src_batch, dest_batch)
        
        file_count = sum(1 for _ in dest_batch.rglob('*') if _.is_file())
        return f"✓ {batch_name} ({file_count} files)", True
        
    except Exception as e:
        return f"✗ {batch_name}: {str(e)}", False

def copy_batch_with_mapping(src_dir, dest_dir, batch_name, files, dry_run=False):
    """Copy files using the mapping (preserves original structure)"""
    success_count = 0
    failed_files = []
    
    if dry_run:
        return f"[DRY RUN] Would copy {batch_name} ({len(files)} files)", True
    
    # Create batch directory
    dest_batch = dest_dir / batch_name
    dest_batch.mkdir(parents=True, exist_ok=True)
    
    for original, new_path in files:
        try:
            src_file = src_dir / new_path
            dest_file = dest_dir / new_path
            
            # Create parent directory
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_file, dest_file)
            success_count += 1
            
        except Exception as e:
            failed_files.append((original, str(e)))
    
    if failed_files:
        return f"⚠ {batch_name} ({success_count}/{len(files)} files copied)", False
    else:
        return f"✓ {batch_name} ({success_count} files)", True

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def main():
    parser = argparse.ArgumentParser(description='Copy organized batch data using pickle mapping')
    parser.add_argument('--source', required=True, help='Source directory with organized batches')
    parser.add_argument('--dest', required=True, help='Destination directory')
    parser.add_argument('--mapping-pkl', required=True, help='Path to pickle mapping file')
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel copy jobs')
    parser.add_argument('--start', type=int, help='Start from batch number (e.g., 1)')
    parser.add_argument('--end', type=int, help='End at batch number (e.g., 100)')
    parser.add_argument('--batch-list', nargs='+', help='Specific batch names to copy')
    parser.add_argument('--mode', choices=['simple', 'mapping'], default='simple',
                        help='Copy mode: simple (full directory) or mapping (use pickle file)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be copied')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    
    # Validate source
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    # Load mapping
    mapping = load_mapping(args.mapping_pkl)
    batch_to_files = organize_by_batch(mapping)
    all_batches = sorted(batch_to_files.keys())
    
    print(f"Found {len(all_batches)} batch directories in mapping")
    
    # Determine which batches to copy
    if args.batch_list:
        # Specific batches
        batches_to_copy = [b for b in args.batch_list if b in all_batches]
    elif args.start and args.end:
        # Range of batches
        batches_to_copy = []
        for i in range(args.start, args.end + 1):
            batch_name = f"batch_{i:06d}"
            if batch_name in all_batches:
                batches_to_copy.append(batch_name)
    else:
        # All batches
        batches_to_copy = all_batches
    
    if not batches_to_copy:
        print("No batches to copy based on criteria")
        sys.exit(0)
    
    print(f"\nWill copy {len(batches_to_copy)} batch directories")
    
    # Calculate statistics
    total_files = sum(len(batch_to_files[b]) for b in batches_to_copy)
    print(f"Total files to copy: {total_files:,}")
    
    if not args.dry_run:
        # Calculate size (sample first few batches for estimate)
        print("Estimating total size...")
        sample_batches = batches_to_copy[:min(1, len(batches_to_copy))]
        sample_size = 0
        sample_files = 0
        
        for batch in sample_batches:
            batch_path = source_dir / batch
            if batch_path.exists():
                for f in batch_path.rglob('*'):
                    if f.is_file():
                        sample_size += f.stat().st_size
                        sample_files += 1
        
        if sample_files > 0:
            avg_file_size = sample_size / sample_files
            estimated_total = avg_file_size * total_files
            print(f"Estimated total size: {format_size(estimated_total)}")
        
        # Confirm
        response = input(f"\nCopy {len(batches_to_copy)} batches to {dest_dir}? (y/N) ")
        if response.lower() != 'y':
            print("Copy cancelled")
            sys.exit(0)
    
    # Create destination
    if not args.dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Start copying
    start_time = time.time()
    failed = 0
    
    print(f"\n{'Starting' if not args.dry_run else 'Simulating'} copy with {args.parallel} parallel jobs...")
    print(f"Copy mode: {args.mode}")
    print("=" * 50)
    
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit copy jobs
        futures = {}
        
        for batch in batches_to_copy:
            if args.mode == 'simple':
                # Copy entire directory
                future = executor.submit(
                    copy_batch_directory_simple,
                    source_dir, dest_dir, batch, args.dry_run
                )
            else:
                # Copy using mapping
                future = executor.submit(
                    copy_batch_with_mapping,
                    source_dir, dest_dir, batch,
                    batch_to_files[batch], args.dry_run
                )
            futures[future] = batch
        
        # Progress bar
        with tqdm(total=len(batches_to_copy), desc="Copying batches") as pbar:
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    message, success = future.result()
                    if not success:
                        failed += 1
                    if args.dry_run or not success:
                        tqdm.write(message)
                except Exception as e:
                    failed += 1
                    tqdm.write(f"✗ {batch}: {str(e)}")
                pbar.update(1)
    
    # Summary
    duration = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"{'Copy' if not args.dry_run else 'Dry run'} completed!")
    print(f"Total time: {int(duration // 60)}m {int(duration % 60)}s")
    print(f"Successful: {len(batches_to_copy) - failed}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    # Verify
    if not args.dry_run and not args.no_verify:
        print("\nVerifying copy...")
        
        # Count files in destination
        dest_file_count = 0
        for batch in tqdm(batches_to_copy, desc="Verifying"):
            dest_batch = dest_dir / batch
            if dest_batch.exists():
                dest_file_count += sum(1 for _ in dest_batch.rglob('*') if _.is_file())
        
        expected_count = sum(len(batch_to_files[b]) for b in batches_to_copy)
        print(f"Expected files: {expected_count:,}")
        print(f"Copied files: {dest_file_count:,}")
        
        if dest_file_count == expected_count:
            print("✓ Copy verified successfully!")
        else:
            print(f"⚠ Warning: Missing {expected_count - dest_file_count:,} files")

if __name__ == "__main__":
    main()