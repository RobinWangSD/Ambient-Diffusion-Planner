"""
Script to compute the missing states rate of neighbor agents in the future horizon.

Usage:
    python compute_missing_states.py --pkl_file mapping.pkl --dataset_dir /path/to/dataset --num_agents 50 --output_file missing_rates.npz
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime


def setup_logging(log_file=None, log_level='INFO'):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create handlers
    handlers = [logging.StreamHandler()]  # Console handler
    
    if log_file:
        # Also log to file if specified
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_mapping(pkl_file):
    """Load the mapping from pickle file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading mapping from {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        mapping = pickle.load(f)
    
    logger.info(f"Loaded mapping with {len(mapping)} entries")
    return mapping


def process_single_file(args):
    """
    Process a single file and return statistics.
    This function is designed to be called in parallel.
    
    Args:
        args: Tuple of (file_name, relative_path, dataset_dir, num_agents)
        
    Returns:
        Tuple of (missing_per_timestep, valid_agents_count, num_timesteps) or None if error
    """
    file_name, relative_path, dataset_dir, num_agents = args
    logger = logging.getLogger(__name__)
    
    # Construct full path
    full_path = Path(dataset_dir) / relative_path
    
    if not full_path.exists():
        logger.warning(f"File not found: {full_path}")
        return None
    
    try:
        data = np.load(full_path)
        
        # Get neighbor agents data
        neighbor_agents_past = data['neighbor_agents_past']
        neighbor_agents_future = data['neighbor_agents_future']
        
        # Get actual shape
        actual_num_agents = min(neighbor_agents_future.shape[0], num_agents)
        num_future_timesteps = neighbor_agents_future.shape[1]
        
        # Check which agents are present at current time (vectorized)
        if actual_num_agents > 0:
            current_states = neighbor_agents_past[:actual_num_agents, -1, :4]
            agent_valid_at_current = ~np.all(np.abs(current_states) < 1e-6, axis=1)
            
            # Check for missing future states (vectorized)
            agent_futures = neighbor_agents_future[:actual_num_agents]
            missing_states = np.all(np.abs(agent_futures) < 1e-6, axis=2)
            
            # Only count missing for valid agents
            valid_missing = missing_states[agent_valid_at_current]
            
            # Sum missing states per timestep
            missing_per_timestep = np.sum(valid_missing, axis=0)
            valid_agents_count = np.sum(agent_valid_at_current)
            
            logger.debug(f"Processed {file_name}: {valid_agents_count} valid agents")
            return (missing_per_timestep, valid_agents_count, num_future_timesteps)
        else:
            logger.debug(f"No agents found in {file_name}")
            return (np.zeros(num_future_timesteps), 0, num_future_timesteps)
            
    except Exception as e:
        logger.error(f"Error processing {full_path}: {e}")
        return None


def compute_missing_rates_parallel(pkl_file, dataset_dir, num_agents, max_files=None, num_workers=None):
    """
    Compute missing rates across all files in the mapping using parallel processing.
    
    Args:
        pkl_file: Path to pickle file with mapping
        dataset_dir: Parent directory of the dataset
        num_agents: Number of agents to compute statistics for
        max_files: Maximum number of files to process (None for all)
        num_workers: Number of parallel workers (None for cpu_count)
        
    Returns:
        missing_rates_per_timestep: Array of shape (num_future_timesteps,) with missing rates
        total_valid_agents_per_timestep: Array tracking total valid agents at each timestep
        total_missing_per_timestep: Array tracking total missing states at each timestep
    """
    logger = logging.getLogger(__name__)
    
    # Load mapping
    mapping = load_mapping(pkl_file)
    
    # Get list of files to process
    file_list = list(mapping.items())
    if max_files:
        file_list = file_list[:max_files]
        logger.info(f"Processing limited to {max_files} files")
    
    total_files = len(file_list)
    logger.info(f"Starting processing of {total_files} files")
    
    # Prepare arguments for parallel processing
    process_args = [(file_name, relative_path, dataset_dir, num_agents) 
                    for file_name, relative_path in file_list]
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # Cap at 8 workers to avoid memory issues
    
    logger.info(f"Using {num_workers} parallel workers")
    
    # Process files in parallel
    total_missing_per_timestep = None
    total_valid_agents_per_timestep = None
    num_future_timesteps = None
    
    start_time = datetime.now()
    
    # Calculate logging intervals
    log_interval = max(1, total_files // 20)  # Log approximately 20 times during processing
    
    with Pool(num_workers) as pool:
        # Use imap for memory efficiency with progress tracking
        results = []
        processed_count = 0
        last_log_count = 0
        
        # Create iterator with progress bar
        iterator = pool.imap(process_single_file, process_args, chunksize=10)
        
        # Wrap with tqdm for visual progress
        for result in tqdm(iterator, total=len(process_args), desc="Processing files"):
            results.append(result)
            processed_count += 1
            
            # Log progress at intervals
            if processed_count - last_log_count >= log_interval or processed_count == total_files:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_files - processed_count) / rate if rate > 0 else 0
                
                logger.info(f"Progress: {processed_count}/{total_files} files "
                          f"({100*processed_count/total_files:.1f}%) | "
                          f"Rate: {rate:.1f} files/sec | "
                          f"ETA: {remaining:.1f} seconds")
                last_log_count = processed_count
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Parallel processing completed in {processing_time:.2f} seconds")
    logger.info(f"Average processing rate: {total_files/processing_time:.2f} files/second")
    
    # Aggregate results
    valid_results = 0
    failed_results = 0
    
    for idx, result in enumerate(results):
        if result is None:
            failed_results += 1
            continue
            
        missing_per_timestep, valid_agents_count, timesteps = result
        valid_results += 1
        
        # Initialize arrays on first valid result
        if total_missing_per_timestep is None:
            num_future_timesteps = timesteps
            total_missing_per_timestep = np.zeros(num_future_timesteps, dtype=np.float64)
            total_valid_agents_per_timestep = np.zeros(num_future_timesteps, dtype=np.float64)
            logger.info(f"Initialized arrays for {num_future_timesteps} future timesteps")
        
        # Accumulate statistics
        total_missing_per_timestep += missing_per_timestep
        total_valid_agents_per_timestep += valid_agents_count
        
        # Log aggregation progress periodically
        if (idx + 1) % log_interval == 0 or idx == len(results) - 1:
            logger.debug(f"Aggregated {idx + 1}/{len(results)} results")
    
    logger.info(f"Aggregation complete: {valid_results} valid files, {failed_results} failed files")
    
    if failed_results > 0:
        logger.warning(f"{failed_results} files failed to process ({100*failed_results/total_files:.1f}%)")
    
    # Compute rates per timestep
    with np.errstate(divide='ignore', invalid='ignore'):
        missing_rates_per_timestep = np.where(total_valid_agents_per_timestep > 0,
                                              total_missing_per_timestep / total_valid_agents_per_timestep,
                                              np.nan)
    
    return missing_rates_per_timestep, total_valid_agents_per_timestep, total_missing_per_timestep


def main():
    parser = argparse.ArgumentParser(description='Compute missing states rate for neighbor agents')
    parser.add_argument('--pkl_file', type=str, required=True,
                       help='Path to pickle file with mapping')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Parent directory of the dataset')
    parser.add_argument('--num_agents', type=int, default=50,
                       help='Number of agents to compute statistics for')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--output_file', type=str, default='missing_rates_analysis.npz',
                       help='Output file to save results')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path (optional, logs to console by default)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    logger.info("="*60)
    logger.info("Starting Missing States Analysis")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  - Dataset directory: {args.dataset_dir}")
    logger.info(f"  - Number of agents: {args.num_agents}")
    logger.info(f"  - Max files: {args.max_files if args.max_files else 'All'}")
    logger.info(f"  - Output file: {args.output_file}")
    
    # Compute missing rates with parallel processing
    try:
        missing_rates_per_timestep, total_valid_agents, total_missing = compute_missing_rates_parallel(
            args.pkl_file, 
            args.dataset_dir, 
            args.num_agents,
            args.max_files,
            args.num_workers
        )
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}", exc_info=True)
        return 1
    
    # Log summary statistics
    logger.info("="*60)
    logger.info("ANALYSIS RESULTS")
    logger.info("="*60)
    
    logger.info(f"Shape of missing_rates array: {missing_rates_per_timestep.shape}")
    logger.info(f"Number of future timesteps: {len(missing_rates_per_timestep)}")
    logger.info(f"Total valid agents processed: {int(total_valid_agents[0]) if len(total_valid_agents) > 0 else 0}")
    
    # Overall statistics
    valid_rates = missing_rates_per_timestep[~np.isnan(missing_rates_per_timestep)]
    if len(valid_rates) > 0:
        logger.info("Overall Statistics:")
        logger.info(f"  - Mean missing rate: {np.mean(valid_rates):.4f}")
        logger.info(f"  - Median missing rate: {np.median(valid_rates):.4f}")
        logger.info(f"  - Min missing rate: {np.min(valid_rates):.4f}")
        logger.info(f"  - Max missing rate: {np.max(valid_rates):.4f}")
    
    # Per-timestep statistics (log a subset)
    logger.info("Per-Timestep Missing Rates (sampled):")
    for t in range(0, len(missing_rates_per_timestep), 10):  # Every second (10Hz)
        time_sec = t / 10.0
        rate = missing_rates_per_timestep[t]
        valid_count = int(total_valid_agents[t])
        logger.info(f"  t={time_sec:.1f}s: rate={rate:.4f}, valid_agents={valid_count}")
    
    # Trend analysis
    first_second_rate = np.mean(missing_rates_per_timestep[:10])
    last_second_rate = np.mean(missing_rates_per_timestep[-10:])
    rate_increase = last_second_rate - first_second_rate
    
    logger.info("Missing Rate Trend:")
    logger.info(f"  - First second (0-1s): {first_second_rate:.4f}")
    logger.info(f"  - Last second (7-8s): {last_second_rate:.4f}")
    logger.info(f"  - Rate increase: {rate_increase:.4f}")
    
    # Save results
    try:
        np.savez(args.output_file,
                 missing_rates_per_timestep=missing_rates_per_timestep,
                 total_valid_agents_per_timestep=total_valid_agents,
                 total_missing_per_timestep=total_missing,
                 num_agents_considered=args.num_agents,
                 num_timesteps=len(missing_rates_per_timestep))
        
        logger.info(f"Results successfully saved to: {args.output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1
    
    logger.info("Analysis completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())