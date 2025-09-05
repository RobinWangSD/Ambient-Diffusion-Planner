import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from collections import defaultdict
import time

# Configure logging
def setup_logger(name: str = "npz_mapper", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler
    file_handler = logging.FileHandler('npz_mapper.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

def scan_directory_chunk(args: Tuple[str, str]) -> Tuple[Dict[str, str], int]:
    """
    Scan a single directory for NPZ files.
    
    Args:
        args: Tuple of (directory_path, root_folder)
    
    Returns:
        Tuple of (file_mapping, duplicate_count)
    """
    directory_path, root_folder = args
    root_path = Path(root_folder)
    dir_path = Path(directory_path)
    
    file_mapping = {}
    duplicate_count = 0
    
    try:
        # Use scandir for faster directory listing
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.npz'):
                    file_path = dir_path / entry.name
                    relative_path = file_path.relative_to(root_path)
                    
                    filename = entry.name
                    if filename in file_mapping:
                        duplicate_count += 1
                        logger.debug(f"Duplicate filename in directory {directory_path}: {filename}")
                    else:
                        file_mapping[filename] = str(relative_path)
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot access directory {directory_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error scanning directory {directory_path}: {e}")
    
    return file_mapping, duplicate_count

def get_all_directories(root_folder: str) -> List[str]:
    """
    Get all immediate subdirectories that contain NPZ files.
    
    Args:
        root_folder: Root directory to scan
        
    Returns:
        List of subdirectory paths (excluding root folder)
    """
    directories = []
    
    try:
        logger.debug(f"Scanning immediate subdirectories of: {root_folder}")
        with os.scandir(root_folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    directories.append(entry.path)
                    logger.debug(f"Found subdirectory: {entry.path}")
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot access root directory {root_folder}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error discovering directories in {root_folder}: {e}")
        raise
    
    logger.debug(f"Directory discovery completed. Found {len(directories)} subdirectories.")
    return directories

def create_npz_file_mapping_parallel(root_folder: str, 
                                   output_pkl: str = "npz_file_mapping.pkl",
                                   max_workers: int = None,
                                   use_processes: bool = False,
                                   batch_size: int = 100) -> Dict[str, str]:
    """
    Scan subfolders for NPZ files using parallel processing and create a mapping.
    
    Args:
        root_folder: Path to the root folder containing subfolders with NPZ files
        output_pkl: Output pickle file name
        max_workers: Number of worker threads/processes (None = auto)
        use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        batch_size: Number of directories to process in each batch
    
    Returns:
        Dict[str, str]: Mapping from filename to relative path
    """
    
    root_path = Path(root_folder)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder '{root_folder}' does not exist")
    
    if not root_path.is_dir():
        raise NotADirectoryError(f"'{root_folder}' is not a directory")
    
    logger.info("Starting NPZ file mapping process")
    logger.info(f"Root folder: {root_folder}")
    logger.info(f"Output file: {output_pkl}")
    logger.info(f"Use processes: {use_processes}")
    logger.info(f"Batch size: {batch_size}")
    
    logger.info("Discovering directories...")
    start_time = time.time()
    
    # Get all directories first
    directories = get_all_directories(root_folder)
    discovery_time = time.time() - start_time
    
    logger.info(f"Found {len(directories):,} directories to scan in {discovery_time:.2f} seconds")
    logger.info("Processing directories in parallel...")
    
    # Prepare arguments for parallel processing
    dir_args = [(dir_path, root_folder) for dir_path in directories]
    
    # Choose executor type
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    executor_type = "ProcessPoolExecutor" if use_processes else "ThreadPoolExecutor"
    
    # Set default max_workers based on CPU count
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    logger.info(f"Using {executor_type} with {max_workers} workers")
    
    # Final mapping and counters
    final_mapping = {}
    total_duplicates = 0
    processed_dirs = 0
    
    # Process directories in parallel
    process_start = time.time()
    
    with executor_class(max_workers=max_workers) as executor:
        # Process in batches to manage memory
        for i in range(0, len(dir_args), batch_size):
            batch = dir_args[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} ({len(batch)} directories)")
            
            # Submit batch
            future_to_dir = {executor.submit(scan_directory_chunk, args): args[0] 
                           for args in batch}
            
            # Collect results
            for future in as_completed(future_to_dir):
                try:
                    file_mapping, duplicate_count = future.result()
                    
                    # Merge results (handle global duplicates)
                    for filename, path in file_mapping.items():
                        if filename in final_mapping:
                            total_duplicates += 1
                            logger.debug(f"Global duplicate filename: {filename} "
                                       f"(existing: {final_mapping[filename]}, new: {path})")
                        else:
                            final_mapping[filename] = path
                    
                    total_duplicates += duplicate_count
                    processed_dirs += 1
                    
                    # Progress update every 1000 directories
                    if processed_dirs % 1000 == 0:
                        elapsed = time.time() - process_start
                        rate = processed_dirs / elapsed
                        eta = (len(directories) - processed_dirs) / rate if rate > 0 else 0
                        logger.info(f"Progress: {processed_dirs:,}/{len(directories):,} directories "
                                  f"({rate:.1f} dirs/sec, ETA: {eta:.1f}s)")
                
                except Exception as e:
                    logger.error(f"Error processing directory {future_to_dir[future]}: {e}")
    
    processing_time = time.time() - process_start
    
    logger.info("Saving mapping to pickle file...")
    save_start = time.time()
    
    # Save mapping to pickle file
    output_path = Path(output_pkl)
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(final_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Mapping successfully saved to: {output_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save mapping to {output_path}: {e}")
        raise
    
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    
    # Final report
    logger.info("=" * 60)
    logger.info("FILE MAPPING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Total directories scanned: {len(directories):,}")
    logger.info(f"Total NPZ files processed: {len(final_mapping) + total_duplicates:,}")
    logger.info(f"Unique filenames: {len(final_mapping):,}")
    
    if total_duplicates > 0:
        logger.warning(f"DUPLICATES FOUND: {total_duplicates:,} duplicate filenames were skipped!")
        logger.warning("Only the first occurrence of each filename was kept in the mapping.")
    else:
        logger.info("No duplicate filenames found.")
    
    logger.info(f"Output file: {output_path.absolute()}")
    logger.info("")
    logger.info("TIMING BREAKDOWN:")
    logger.info(f"  Directory discovery: {discovery_time:.2f}s")
    logger.info(f"  Parallel processing: {processing_time:.2f}s")
    logger.info(f"  File saving: {save_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Processing rate: {len(directories)/processing_time:.1f} directories/second")
    logger.info("=" * 60)
    
    return final_mapping

def create_npz_file_mapping_fast(root_folder: str, 
                               output_pkl: str = "npz_file_mapping.pkl") -> Dict[str, str]:
    """
    Ultra-fast version using optimized settings for very large datasets.
    """
    logger.info("Using fast mode with optimized settings")
    return create_npz_file_mapping_parallel(
        root_folder=root_folder,
        output_pkl=output_pkl,
        max_workers=min(64, (os.cpu_count() or 1) * 2),
        use_processes=False,  # Threads are often faster for I/O bound tasks
        batch_size=500
    )

def load_npz_file_mapping(pkl_file: str) -> Dict[str, str]:
    """
    Load the NPZ file mapping from a pickle file.
    
    Args:
        pkl_file: Path to the pickle file
    
    Returns:
        Dict[str, str]: Mapping from filename to relative path
    """
    logger.info(f"Loading mapping from: {pkl_file}")
    try:
        with open(pkl_file, 'rb') as f:
            mapping = pickle.load(f)
        logger.info(f"Successfully loaded {len(mapping):,} file mappings")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load mapping from {pkl_file}: {e}")
        raise

def set_log_level(level: str):
    """
    Set the logging level.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    if level.upper() in level_map:
        logger.setLevel(level_map[level.upper()])
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level_map[level.upper()])
        logger.info(f"Log level set to {level.upper()}")
    else:
        logger.warning(f"Invalid log level: {level}. Valid options: {list(level_map.keys())}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NPZ File Mapper - Create mapping from filenames to relative paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python npz_mapper.py /path/to/data --fast
  python npz_mapper.py /path/to/data --workers 32 --batch-size 1000 --log-level DEBUG
  python npz_mapper.py /path/to/data --use-processes --output my_mapping.pkl
        """
    )
    
    parser.add_argument(
        "root_folder",
        help="Path to the root folder containing subfolders with NPZ files"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="npz_file_mapping.pkl",
        help="Output pickle file name (default: npz_file_mapping.pkl)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode with optimized settings for large datasets"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads/processes (default: auto-detect)"
    )
    
    parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of directories to process in each batch (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    set_log_level(args.log_level)
    
    logger.info("NPZ File Mapper - Parallel Processing Version")
    logger.info(f"Logs will be saved to 'npz_mapper.log'")
    
    try:
        if args.fast:
            # Fast mode
            logger.info("Using fast mode with optimized settings")
            mapping = create_npz_file_mapping_fast(
                root_folder=args.root_folder,
                output_pkl=args.output
            )
        else:
            # Custom mode with specified parameters
            mapping = create_npz_file_mapping_parallel(
                root_folder=args.root_folder,
                output_pkl=args.output,
                max_workers=args.workers,
                use_processes=args.use_processes,
                batch_size=args.batch_size
            )
        
        logger.info("Process completed successfully!")
        logger.info("To load this mapping later, use:")
        logger.info(f"mapping = load_npz_file_mapping('{args.output}')")
        
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"Directory error: {e}")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)