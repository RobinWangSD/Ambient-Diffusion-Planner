#!/usr/bin/env python3

import os
import sys
import time
import pickle
import shutil
import argparse
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

class ParallelStreamingOrganizer:
    """Organizes files into batches with parallel processing during scanning"""
    
    def __init__(self, source_dir, dest_dir, files_per_batch=100000, num_workers=8):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.files_per_batch = files_per_batch
        self.num_workers = num_workers
        
        # Thread pool for moving files
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.pending_moves = deque(maxlen=num_workers * 2)
        
        # Thread-safe state
        self.lock = threading.Lock()
        self.current_batch_num = 1
        self.current_batch_size = 0
        self.file_mapping = {}
        
        # Statistics
        self.files_scanned = 0
        self.files_moved = 0
        self.files_failed = 0
        self.start_time = time.time()
        
        # Create destination and first batch
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._create_batch(self.current_batch_num)
    
    def _create_batch(self, batch_num):
        """Create a new batch directory"""
        batch_name = f"batch_{batch_num:06d}"
        batch_dir = self.dest_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        return batch_dir
    
    def organize(self):
        """Main method with parallel moving"""
        print(f"Starting parallel streaming organization")
        print(f"Source: {self.source_dir}")
        print(f"Destination: {self.dest_dir}")
        print(f"Files per batch: {self.files_per_batch:,}")
        print(f"Workers: {self.num_workers}")
        print("="*60)
        
        # Start progress thread
        progress_thread = threading.Thread(target=self._show_progress, daemon=True)
        progress_thread.start()
        
        # Start move result handler
        result_thread = threading.Thread(target=self._handle_move_results, daemon=True)
        result_thread.start()
        
        # Scan and queue moves
        if sys.platform != "win32" and shutil.which('find'):
            self._scan_with_find()
        else:
            self._scan_with_python()
        
        # Wait for all moves to complete
        print("\n\nWaiting for pending moves to complete...")
        self.executor.shutdown(wait=True)
        
        # Wait for results to be processed
        while self.pending_moves:
            time.sleep(0.1)
        
        # Save mapping and stats
        self._save_mapping()
        self._print_final_stats()
    
    def _scan_with_find(self):
        """Stream files using find command"""
        print("Using 'find' command for fast scanning...")
        cmd = ['find', str(self.source_dir), '-type', 'f', '-print0']
        
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            buffer = b''
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                
                buffer += chunk
                while b'\0' in buffer:
                    file_path, buffer = buffer.split(b'\0', 1)
                    if file_path:
                        self._queue_file_move(file_path.decode('utf-8'))
    
    def _scan_with_python(self):
        """Stream files using os.walk"""
        print("Using Python os.walk for scanning...")
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._queue_file_move(file_path)
    
    def _queue_file_move(self, source_path):
        """Queue a file for moving"""
        with self.lock:
            self.files_scanned += 1
            
            # Determine batch
            if self.current_batch_size >= self.files_per_batch:
                self.current_batch_num += 1
                self.current_batch_size = 0
                self._create_batch(self.current_batch_num)
            
            batch_num = self.current_batch_num
            batch_name = f"batch_{batch_num:06d}"
            
            # Create destination path
            source = Path(source_path)
            dest = self.dest_dir / batch_name / source.name
            
            # Handle name conflicts
            if dest.exists():
                base = dest.stem
                ext = dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest.parent / f"{base}_{counter}{ext}"
                    counter += 1
            
            # Submit move task
            future = self.executor.submit(self._move_file, source_path, str(dest))
            self.pending_moves.append((future, source_path, str(dest)))
            
            self.current_batch_size += 1
    
    def _move_file(self, source, dest):
        """Move a single file"""
        try:
            shutil.move(source, dest)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _handle_move_results(self):
        """Handle results from move operations"""
        while True:
            if self.pending_moves:
                future, source, dest = self.pending_moves[0]
                if future.done():
                    self.pending_moves.popleft()
                    success, error = future.result()
                    
                    with self.lock:
                        if success:
                            self.files_moved += 1
                            # Update mapping
                            source_path = Path(source)
                            dest_path = Path(dest)
                            try:
                                rel_source = source_path.relative_to(self.source_dir)
                            except ValueError:
                                rel_source = source_path.name
                            rel_dest = dest_path.relative_to(self.dest_dir)
                            self.file_mapping[str(rel_source)] = str(rel_dest)
                        else:
                            self.files_failed += 1
                            if self.files_failed <= 10:
                                print(f"\nError moving {source}: {error}")
            else:
                # Check if we're done
                if self.executor._shutdown:
                    break
            
            time.sleep(0.01)  # Small delay to prevent CPU spinning
    
    def _show_progress(self):
        """Show progress"""
        last_update = time.time()
        
        while True:
            time.sleep(2)
            
            with self.lock:
                current_time = time.time()
                elapsed = current_time - self.start_time
                scan_rate = self.files_scanned / elapsed if elapsed > 0 else 0
                move_rate = self.files_moved / elapsed if elapsed > 0 else 0
                
                # Estimate completion
                if move_rate > 0 and self.files_scanned > self.files_moved:
                    remaining = self.files_scanned - self.files_moved
                    eta = remaining / move_rate
                else:
                    eta = 0
                
                print(f"\rBatch {self.current_batch_num} | "
                      f"Scanned: {self.files_scanned:,} ({scan_rate:.0f}/s) | "
                      f"Moved: {self.files_moved:,} ({move_rate:.0f}/s) | "
                      f"Failed: {self.files_failed:,} | "
                      f"Pending: {len(self.pending_moves)} | "
                      f"ETA: {int(eta//60)}m {int(eta%60)}s", 
                      end="", flush=True)
                
                # Stop when done
                if self.executor._shutdown and not self.pending_moves:
                    break
    
    def _save_mapping(self):
        """Save mapping file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        mapping_file = self.dest_dir / f"file_mapping_{timestamp}.pkl"
        
        print(f"\nSaving mapping to {mapping_file}")
        with open(mapping_file, 'wb') as f:
            pickle.dump(self.file_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Mapping saved: {len(self.file_mapping):,} entries")
        print(f"File size: {mapping_file.stat().st_size / (1024*1024):.2f} MB")
    
    def _print_final_stats(self):
        """Print statistics"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("Organization Complete!")
        print("="*60)
        print(f"Total files scanned: {self.files_scanned:,}")
        print(f"Files moved: {self.files_moved:,}")
        print(f"Files failed: {self.files_failed:,}")
        print(f"Batches created: {self.current_batch_num}")
        print(f"Time taken: {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"Scan rate: {self.files_scanned/elapsed:.0f} files/sec")
        print(f"Move rate: {self.files_moved/elapsed:.0f} files/sec")
        
        # Show batch distribution
        print("\nBatch distribution:")
        for i in range(1, min(6, self.current_batch_num + 1)):
            batch_dir = self.dest_dir / f"batch_{i:06d}"
            if batch_dir.exists():
                file_count = len(list(batch_dir.iterdir()))
                print(f"  {batch_dir.name}: {file_count:,} files")
        if self.current_batch_num > 5:
            print(f"  ... and {self.current_batch_num - 5} more batches")


def main():
    parser = argparse.ArgumentParser(description='Parallel streaming file organizer')
    parser.add_argument('source', help='Source directory containing files')
    parser.add_argument('dest', help='Destination directory for organized batches')
    parser.add_argument('--batch-size', type=int, default=100000, 
                        help='Number of files per batch (default: 100000)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    
    args = parser.parse_args()
    
    # Validate source
    if not os.path.exists(args.source):
        print(f"Error: Source directory does not exist: {args.source}")
        sys.exit(1)
    
    # Create organizer and run
    organizer = ParallelStreamingOrganizer(
        args.source,
        args.dest,
        args.batch_size,
        args.workers
    )
    
    try:
        organizer.organize()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")
        organizer.executor.shutdown(wait=False)
        sys.exit(1)

if __name__ == "__main__":
    main()