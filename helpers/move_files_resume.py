#!/usr/bin/env python3

import os
import sys
import time
import pickle
import shutil
import argparse
import logging
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import sqlite3

class MappingDatabase:
    """SQLite-based mapping storage for handling large mappings efficiently"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_mappings (
                    source_path TEXT PRIMARY KEY,
                    dest_path TEXT NOT NULL,
                    batch_num INTEGER,
                    timestamp REAL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch ON file_mappings(batch_num)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dest ON file_mappings(dest_path)')
            self.conn.commit()
    
    def add_mapping(self, source_path, dest_path, batch_num):
        """Add a single mapping"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO file_mappings (source_path, dest_path, batch_num, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (source_path, dest_path, batch_num, time.time()))
            self.conn.commit()
    
    def add_mappings_batch(self, mappings):
        """Add multiple mappings at once"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO file_mappings (source_path, dest_path, batch_num, timestamp)
                VALUES (?, ?, ?, ?)
            ''', [(m[0], m[1], m[2], time.time()) for m in mappings])
            self.conn.commit()
    
    def get_total_count(self):
        """Get total mapping count"""
        with self.lock:
            cursor = self.conn.cursor()
            result = cursor.execute('SELECT COUNT(*) FROM file_mappings').fetchone()
            return result[0] if result else 0
    
    def export_to_pickle(self, pickle_path):
        """Export database to pickle file"""
        with self.lock:
            cursor = self.conn.cursor()
            mappings = {}
            for row in cursor.execute('SELECT source_path, dest_path FROM file_mappings'):
                mappings[row[0]] = row[1]
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return len(mappings)
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class SimpleFileOrganizer:
    """Simplified file organizer that starts from a specified batch number"""
    
    def __init__(self, source_dir, dest_dir, start_batch, files_per_batch=100000, 
                 num_workers=8, log_file=None):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.start_batch = start_batch
        self.files_per_batch = files_per_batch
        self.num_workers = num_workers
        
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # Database for mapping (resume if exists)
        self.mapping_db = MappingDatabase(self.dest_dir / "file_mappings.db")
        
        # Thread pool for moving files
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.pending_moves = deque(maxlen=num_workers * 2)
        
        # Thread-safe state
        self.lock = threading.Lock()
        self.current_batch_num = start_batch
        self.current_batch_size = 0
        
        # Statistics
        self.files_scanned = 0
        self.files_moved = 0
        self.files_failed = 0
        self.start_time = time.time()
        
        # Create destination
        self.dest_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self, log_file):
        """Setup logging configuration"""
        if log_file:
            log_path = Path(log_file)
        else:
            log_path = self.dest_dir / f"organizer_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create logger
        logger = logging.getLogger('FileOrganizer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def organize(self):
        """Main organization method"""
        self.logger.info("="*60)
        self.logger.info("Starting Simple File Organization")
        self.logger.info(f"Source: {self.source_dir}")
        self.logger.info(f"Destination: {self.dest_dir}")
        self.logger.info(f"Starting batch: {self.start_batch}")
        self.logger.info(f"Files per batch: {self.files_per_batch:,}")
        self.logger.info(f"Workers: {self.num_workers}")
        self.logger.info("="*60)
        
        # Create first batch
        self._create_batch(self.current_batch_num)
        
        # Start background threads
        progress_thread = threading.Thread(target=self._progress_reporter, daemon=True)
        progress_thread.start()
        
        result_thread = threading.Thread(target=self._handle_move_results, daemon=True)
        result_thread.start()
        
        # Start scanning and organizing
        self._start_organization()
        
        # Wait for completion
        self._wait_for_pending_moves()
        
        # Export final mapping
        self._export_final_mapping()
        
        # Cleanup
        self.mapping_db.close()
        self.logger.info("Organization completed successfully")
    
    def _start_organization(self):
        """Start the main file organization"""
        self.logger.info("Starting file scan and organization...")
        
        # Choose scanning method based on platform
        if sys.platform != "win32" and shutil.which('find'):
            self._scan_with_find()
        else:
            self._scan_with_python()
    
    def _scan_with_find(self):
        """Stream files using find command"""
        self.logger.info("Using 'find' command for scanning...")
        
        cmd = ['find', str(self.source_dir), '-type', 'f', '-print0']
        
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                buffer = b''
                while True:
                    chunk = proc.stdout.read(8192)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    while b'\0' in buffer:
                        file_path, buffer = buffer.split(b'\0', 1)
                        if file_path:
                            try:
                                self._queue_file_move(file_path.decode('utf-8'))
                            except UnicodeDecodeError:
                                self.logger.warning(f"Could not decode file path: {file_path}")
                                continue
        except Exception as e:
            self.logger.error(f"Error with find command: {e}")
            self.logger.info("Falling back to Python scanning...")
            self._scan_with_python()
    
    def _scan_with_python(self):
        """Stream files using os.walk"""
        self.logger.info("Using Python os.walk for scanning...")
        
        try:
            for root, dirs, files in os.walk(self.source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    self._queue_file_move(file_path)
        except Exception as e:
            self.logger.error(f"Error during Python scan: {e}")
            raise
    
    def _queue_file_move(self, source_path):
        """Queue a file for moving"""
        with self.lock:
            self.files_scanned += 1
            
            # Check if need new batch
            if self.current_batch_size >= self.files_per_batch:
                self.current_batch_num += 1
                self.current_batch_size = 0
                self._create_batch(self.current_batch_num)
            
            # Create destination path
            source = Path(source_path)
            batch_name = f"batch_{self.current_batch_num:06d}"
            dest = self.dest_dir / batch_name / source.name
            
            # Handle filename conflicts
            if dest.exists():
                base = dest.stem
                ext = dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest.parent / f"{base}_{counter}{ext}"
                    counter += 1
            
            # Submit move operation
            future = self.executor.submit(self._move_file, source_path, str(dest), self.current_batch_num)
            self.pending_moves.append((future, source_path, str(dest), self.current_batch_num))
            
            self.current_batch_size += 1
    
    def _create_batch(self, batch_num):
        """Create a new batch directory"""
        batch_name = f"batch_{batch_num:06d}"
        batch_dir = self.dest_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created batch: {batch_name}")
        return batch_dir
    
    def _move_file(self, source, dest, batch_num):
        """Move a single file"""
        try:
            if not os.path.exists(source):
                return False, "Source file no longer exists", batch_num
            
            # Ensure destination directory exists
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(source, dest)
            return True, None, batch_num
            
        except Exception as e:
            return False, str(e), batch_num
    
    def _handle_move_results(self):
        """Handle results from move operations"""
        while True:
            if self.pending_moves:
                future, source, dest, batch_num = self.pending_moves[0]
                if future.done():
                    self.pending_moves.popleft()
                    
                    try:
                        success, error, batch_num = future.result()
                        
                        with self.lock:
                            if success:
                                self.files_moved += 1
                                
                                # Add mapping to database
                                source_path = Path(source)
                                dest_path = Path(dest)
                                
                                try:
                                    rel_source = source_path.relative_to(self.source_dir)
                                except ValueError:
                                    # If can't get relative path, use full path
                                    rel_source = source_path
                                
                                rel_dest = dest_path.relative_to(self.dest_dir)
                                self.mapping_db.add_mapping(str(rel_source), str(rel_dest), batch_num)
                                
                            else:
                                self.files_failed += 1
                                if self.files_failed <= 10:  # Log first 10 failures
                                    self.logger.error(f"Failed to move {source}: {error}")
                                elif self.files_failed == 11:
                                    self.logger.error("Further failures will not be logged individually")
                    
                    except Exception as e:
                        with self.lock:
                            self.files_failed += 1
                            self.logger.error(f"Error processing result for {source}: {e}")
            else:
                # Check if executor is shutdown and no pending moves
                if hasattr(self.executor, '_shutdown') and self.executor._shutdown:
                    break
            
            time.sleep(0.01)
    
    def _progress_reporter(self):
        """Report progress periodically"""
        last_moved = 0
        
        while True:
            time.sleep(5)  # Report every 5 seconds
            
            with self.lock:
                elapsed = time.time() - self.start_time
                scan_rate = self.files_scanned / elapsed if elapsed > 0 else 0
                move_rate = self.files_moved / elapsed if elapsed > 0 else 0
                
                # Calculate recent move rate
                recent_moved = self.files_moved - last_moved
                recent_rate = recent_moved / 5  # files per second in last 5 seconds
                last_moved = self.files_moved
                
                self.logger.info(
                    f"Batch: {self.current_batch_num} | "
                    f"Scanned: {self.files_scanned:,} ({scan_rate:.0f}/s) | "
                    f"Moved: {self.files_moved:,} (avg: {move_rate:.0f}/s, recent: {recent_rate:.0f}/s) | "
                    f"Failed: {self.files_failed:,} | "
                    f"Pending: {len(self.pending_moves)}"
                )
                
                # Check if scanning is complete and no pending moves
                if (hasattr(self.executor, '_shutdown') and self.executor._shutdown and 
                    not self.pending_moves):
                    break
    
    def _wait_for_pending_moves(self):
        """Wait for all pending moves to complete"""
        self.logger.info("Waiting for pending moves to complete...")
        self.executor.shutdown(wait=True)
        
        # Wait for result handler to process remaining moves
        while self.pending_moves:
            time.sleep(0.1)
        
        time.sleep(1)  # Give result handler time to finish
    
    def _export_final_mapping(self):
        """Export final mapping to pickle file"""
        self.logger.info("Exporting final mapping...")
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        pickle_path = self.dest_dir / f"file_mapping_{timestamp}.pkl"
        
        count = self.mapping_db.export_to_pickle(pickle_path)
        
        self.logger.info(f"Exported {count:,} mappings to {pickle_path.name}")
        
        if pickle_path.exists():
            file_size_mb = pickle_path.stat().st_size / (1024*1024)
            self.logger.info(f"Pickle file size: {file_size_mb:.2f} MB")
        
        # Final statistics
        elapsed = time.time() - self.start_time
        self.logger.info("\n" + "="*60)
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total files scanned: {self.files_scanned:,}")
        self.logger.info(f"Files moved: {self.files_moved:,}")
        self.logger.info(f"Files failed: {self.files_failed:,}")
        self.logger.info(f"Final batch number: {self.current_batch_num}")
        self.logger.info(f"Total time: {int(elapsed//60)}m {int(elapsed%60)}s")
        if self.files_moved > 0 and elapsed > 0:
            self.logger.info(f"Average rate: {self.files_moved/elapsed:.1f} files/sec")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Simple file organizer with batch numbering')
    parser.add_argument('source', help='Source directory')
    parser.add_argument('dest', help='Destination directory')
    parser.add_argument('--start-batch', type=int, default=1, 
                       help='Starting batch number (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100000, 
                       help='Files per batch (default: 100000)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--log-file', help='Log file path (default: auto-generated in dest)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.source):
        print(f"Error: Source directory does not exist: {args.source}")
        sys.exit(1)
    
    if args.start_batch < 1:
        print("Error: Start batch must be >= 1")
        sys.exit(1)
    
    if args.batch_size < 1:
        print("Error: Batch size must be >= 1")
        sys.exit(1)
    
    if args.workers < 1:
        print("Error: Number of workers must be >= 1")
        sys.exit(1)
    
    # Create organizer
    organizer = SimpleFileOrganizer(
        args.source,
        args.dest,
        args.start_batch,
        args.batch_size,
        args.workers,
        args.log_file
    )
    
    try:
        organizer.organize()
    except KeyboardInterrupt:
        organizer.logger.critical("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        organizer.logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()