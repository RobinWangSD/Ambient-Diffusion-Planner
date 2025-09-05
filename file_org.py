#!/usr/bin/env python3
import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse


class FileOrganizer:
   def __init__(self, source_dir, files_per_dir=100000, prefix="batch_", workers=4):
       self.source_dir = Path(source_dir)
       self.files_per_dir = files_per_dir
       self.prefix = prefix
       self.workers = workers
       # Thread-safe counters
       self.lock = Lock()
       self.moved_count = 0
       self.current_dir_count = 0
       self.current_dir_num = 1
       self.current_dir = None

   def create_next_directory(self):
       """Create the next batch directory"""
       dir_name = f"{self.prefix}{self.current_dir_num:06d}"
       dir_path = self.source_dir / dir_name
       dir_path.mkdir(exist_ok=True)
       self.current_dir = dir_path
       self.current_dir_count = 0
       self.current_dir_num += 1
       return dir_path

   def move_file_batch(self, file_batch):
       """Move a batch of files to the appropriate directories"""
       results = []
       for file_path in file_batch:
           try:
               with self.lock:
                   # Check if we need a new directory
                   if self.current_dir is None or self.current_dir_count >= self.files_per_dir:
                       self.create_next_directory()
                   target_dir = self.current_dir
                   self.current_dir_count += 1
               # Move the file (outside the lock to avoid blocking)
               target_path = target_dir / file_path.name
               shutil.move(str(file_path), str(target_path))
               with self.lock:
                   self.moved_count += 1
               results.append((True, file_path))
           except Exception as e:
               results.append((False, file_path, str(e)))
       return results

   def organize_files(self):
       """Main method to organize files"""
       print(f"Starting file organization in: {self.source_dir}")
       print(f"Organizing files into subdirectories with {self.files_per_dir:,} files each...")
       # Get all files (excluding directories)
       print("Scanning directory for files...")
       files = [f for f in self.source_dir.iterdir() if f.is_file()]
       total_files = len(files)
       if total_files == 0:
           print("No files to organize. Exiting.")
           return
       print(f"Total files found: {total_files:,}")
       dirs_needed = (total_files + self.files_per_dir - 1) // self.files_per_dir
       print(f"Will create {dirs_needed} directories")
       # Create first directory
       self.create_next_directory()
       # Split files into batches for parallel processing
       batch_size = max(100, total_files // (self.workers * 10))  # Dynamic batch size
       file_batches = [files[i:i + batch_size] for i in range(0, total_files, batch_size)]
       start_time = time.time()
       errors = []
       # Progress tracking
       last_update = time.time()
       # Process files in parallel
       with ThreadPoolExecutor(max_workers=self.workers) as executor:
           futures = {executor.submit(self.move_file_batch, batch): batch
                     for batch in file_batches}
           for future in as_completed(futures):
               results = future.result()
               # Collect errors
               for result in results:
                   if not result[0]:
                       errors.append(result)
               # Update progress
               current_time = time.time()
               if current_time - last_update > 0.5:  # Update every 0.5 seconds
                   elapsed = current_time - start_time
                   rate = self.moved_count / elapsed if elapsed > 0 else 0
                   percent = (self.moved_count / total_files) * 100
                   print(f"\rProgress: {self.moved_count:,}/{total_files:,} files "
                         f"({percent:.1f}%) | Rate: {rate:.0f} files/sec | "
                         f"Directory: {self.current_dir.name}", end="", flush=True)
                   last_update = current_time
       # Final statistics
       end_time = time.time()
       total_time = end_time - start_time
       print(f"\n\nOrganization complete!")
       print("-" * 50)
       print(f"Files organized: {self.moved_count:,}")
       print(f"Directories created: {self.current_dir_num - 1}")
       print(f"Time taken: {total_time:.2f} seconds")
       print(f"Average rate: {self.moved_count / total_time:.0f} files/sec")
       if errors:
           print(f"\nErrors encountered: {len(errors)}")
           for error in errors[:10]:  # Show first 10 errors
               print(f"  - Failed to move {error[1]}: {error[2]}")
           if len(errors) > 10:
               print(f"  ... and {len(errors) - 10} more errors")
       print("-" * 50)
       # Show directory summary
       print(f"\nDirectory summary:")
       for dir_path in sorted(self.source_dir.glob(f"{self.prefix}*")):
           if dir_path.is_dir():
               file_count = sum(1 for _ in dir_path.iterdir() if _.is_file())
               print(f"{dir_path.name:<20}: {file_count:>6,} files")

               
def main():
   parser = argparse.ArgumentParser(description="Organize files into subdirectories")
   parser.add_argument("source_dir", nargs="?", default=".",
                      help="Source directory (default: current directory)")
   parser.add_argument("-n", "--files-per-dir", type=int, default=100000,
                      help="Number of files per subdirectory (default: 100000)")
   parser.add_argument("-p", "--prefix", default="batch_",
                      help="Prefix for subdirectory names (default: batch_)")
   parser.add_argument("-w", "--workers", type=int, default=4,
                      help="Number of parallel workers (default: 4)")
   args = parser.parse_args()
   # Validate source directory
   if not os.path.isdir(args.source_dir):
       print(f"Error: Directory '{args.source_dir}' does not exist")
       sys.exit(1)
   # Create organizer and run
   organizer = FileOrganizer(
       args.source_dir,
       args.files_per_dir,
       args.prefix,
       args.workers
   )
   organizer.organize_files()
if __name__ == "__main__":
   main()