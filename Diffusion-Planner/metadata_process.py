import os
import json
import pickle
import argparse
import csv
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Set, Optional
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_processed_files_from_pickle(mapping_pkl: str) -> Dict[str, str]:
    """
    Load processed file mapping from a pickle file.
    
    Args:
        mapping_pkl: Path to pickle file containing filename to path mappings
        
    Returns:
        Dictionary mapping filenames to relative paths
    """
    # Check if mapping file exists
    if not os.path.exists(mapping_pkl):
        logger.error(f"Mapping file {mapping_pkl} does not exist!")
        return {}
    
    # Load the pickle mapping
    with open(mapping_pkl, 'rb') as f:
        mapping = pickle.load(f)
    
    logger.info(f"Loaded mapping with {len(mapping):,} files")
    
    return mapping


def get_processed_file_identifiers(mapping: Dict[str, str]) -> Set[str]:
    """
    Extract file identifiers from mapping.
    
    Args:
        mapping: Dictionary mapping filenames to paths
        
    Returns:
        Set of file identifiers (without .npz extension)
    """
    processed_files = set()
    
    for filename in mapping.keys():
        if filename.endswith('.npz'):
            # Remove .npz extension to get the identifier
            file_id = filename[:-4]
            processed_files.add(file_id)
        else:
            # If no extension, use as is
            processed_files.add(filename)
    
    return processed_files


class ScenarioMetadataExtractor:
    """Class to handle scenario metadata extraction with parallel processing"""
    
    def __init__(self, processed_files: Set[str], output_path: str, split_id: Optional[int] = None, parallel: bool = True):
        self.processed_files = processed_files
        self.output_path = output_path
        self.split_id = split_id
        self.parallel = parallel
        self.scenario_type_mapping = defaultdict(list)  # scenario_type -> list of filenames
        self.matched_count = 0
        self.unmatched_scenarios = []
        self.processed_count = 0  # Track number of processed scenarios
        self.completed_batches = 0  # Track completed batches for logging
        self.total_matched = 0  # Track total matched scenarios in real-time
        
        # Create unique temporary files based on split_id and process ID
        import os
        pid = os.getpid()
        if split_id is not None:
            suffix = f"split_{split_id:03d}_pid_{pid}"
        else:
            suffix = f"pid_{pid}"
        
        # Create temporary file for streaming metadata
        self.temp_metadata_file = os.path.join(output_path, f".temp_metadata_{suffix}.jsonl")
        self.checkpoint_path = os.path.join(output_path, f".checkpoint_type_mapping_{suffix}.pkl")
        os.makedirs(output_path, exist_ok=True)
        
        # Clear temp files if they exist (in case of previous interrupted run)
        if os.path.exists(self.temp_metadata_file):
            os.remove(self.temp_metadata_file)
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        
        logger.info(f"Using temporary file: {self.temp_metadata_file}")
    
    def process_scenarios(self, scenarios: List[NuPlanScenario]) -> None:
        """
        Process all scenarios using parallel or sequential processing.
        Saves results directly to disk instead of keeping in memory.
        
        Args:
            scenarios: List of NuPlan scenarios
        """
        if self.parallel:
            batch_size = 50
            batches = [scenarios[i:i+batch_size] for i in range(0, len(scenarios), batch_size)]
            total_batches = len(batches)
            
            logger.info(f"Processing {len(scenarios):,} scenarios in {total_batches} batches with parallel processing...")
            logger.info(f"Saving results to: {self.output_path}")
            
            # Use ProcessPoolExecutor for better control over logging
            with ProcessPoolExecutor(max_workers=100) as executor:
                # Submit all batch processing tasks
                futures = {executor.submit(self.process_batch_scenario, batch): idx 
                          for idx, batch in enumerate(batches)}
                
                # Process completed batches as they finish
                with tqdm(total=total_batches, desc="Processing batches") as pbar:
                    for future in as_completed(futures):
                        batch_idx = futures[future]
                        try:
                            batch_metadata, batch_type_mapping, batch_processed = future.result()
                            
                            # Save batch results immediately to disk
                            self._save_batch_results(batch_metadata, batch_type_mapping)
                            
                            # Update counters
                            self.processed_count += batch_processed
                            self.completed_batches += 1
                            self.total_matched += len(batch_metadata)
                            
                            # Update progress bar
                            pbar.update(1)
                            
                            # Log progress at intervals
                            if self.completed_batches % 10 == 0 or self.completed_batches in [1, 50, 100, 500]:
                                elapsed_pct = (self.completed_batches / total_batches) * 100
                                logger.info(f"Progress: {self.completed_batches}/{total_batches} batches "
                                          f"({elapsed_pct:.1f}%) - Total matched: {self.total_matched:,} scenarios")
                            
                            # Periodically save the type mapping to avoid memory buildup
                            if self.completed_batches % 100 == 0:
                                self._save_type_mapping_checkpoint()
                        
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx}: {e}")
        else:
            logger.info(f"Processing {len(scenarios):,} scenarios sequentially...")
            logger.info(f"Saving results to: {self.output_path}")
            
            for idx, scenario in enumerate(tqdm(scenarios, desc="Processing scenarios"), 1):
                metadata = self.process_single_scenario(scenario)
                self.processed_count += 1
                
                if metadata:
                    # Save individual result immediately
                    self._save_single_result(metadata)
                    self.total_matched += 1
                
                # Log progress every 1000 scenarios
                if idx % 1000 == 0:
                    logger.info(f"Processed {idx:,}/{len(scenarios):,} scenarios - Matched: {self.total_matched:,}")
                    # Save checkpoint
                    self._save_type_mapping_checkpoint()
        
        # Final save of type mapping
        self._finalize_results()
        
        # Count matched scenarios
        self.matched_count = self.total_matched
        
        logger.info("="*60)
        logger.info(f"Processing complete!")
        logger.info(f"Total scenarios processed: {self.processed_count:,}")
        logger.info(f"Matched with processed files: {self.matched_count:,}")
        logger.info(f"Unmatched scenarios: {len(self.unmatched_scenarios):,}")
        if self.processed_count > 0:
            logger.info(f"Match rate: {(self.matched_count/self.processed_count)*100:.1f}%")
        logger.info("="*60)
    
    def _save_batch_results(self, batch_metadata: List[Dict], batch_type_mapping: Dict[str, List[str]]):
        """
        Save batch results to disk immediately with file locking for safety.
        
        Args:
            batch_metadata: List of metadata dictionaries from the batch
            batch_type_mapping: Type mapping from the batch
        """
        # Append metadata to temporary JSONL file (one JSON object per line)
        if batch_metadata:
            import fcntl
            with open(self.temp_metadata_file, 'a') as f:
                # Use file locking to prevent conflicts
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for metadata in batch_metadata:
                        json.dump(metadata, f)
                        f.write('\n')
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Update in-memory type mapping (more efficient than disk I/O for each update)
        for scenario_type, filenames in batch_type_mapping.items():
            self.scenario_type_mapping[scenario_type].extend(filenames)
    
    def _save_single_result(self, metadata: Dict):
        """
        Save a single result to disk with file locking.
        
        Args:
            metadata: Metadata dictionary for a single scenario
        """
        # Append to temporary JSONL file
        import fcntl
        with open(self.temp_metadata_file, 'a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(metadata, f)
                f.write('\n')
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        # Update type mapping
        scenario_type = metadata['scenario_type']
        filename = metadata['npz_file']
        self.scenario_type_mapping[scenario_type].append(filename)
    
    def _save_type_mapping_checkpoint(self):
        """
        Save current type mapping as a checkpoint.
        """
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(dict(self.scenario_type_mapping), f)
        logger.debug(f"Saved type mapping checkpoint with {len(self.scenario_type_mapping)} types")
    
    def _finalize_results(self):
        """
        Finalize and clean up results after processing.
        """
        # Remove temporary metadata file (we only need the type mapping)
        if os.path.exists(self.temp_metadata_file):
            os.remove(self.temp_metadata_file)
            logger.debug(f"Removed temporary metadata file: {self.temp_metadata_file}")
        
        # Remove checkpoint file if it exists
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            logger.debug(f"Removed checkpoint file: {self.checkpoint_path}")
    
    def process_batch_scenario(self, batch: List[NuPlanScenario]) -> tuple:
        """
        Process a batch of scenarios.
        
        Args:
            batch: List of scenarios to process
            
        Returns:
            Tuple of (metadata_list, scenario_type_mapping, processed_count) for this batch
        """
        batch_metadata = []
        batch_type_mapping = defaultdict(list)
        batch_processed = len(batch)
        
        for scenario in batch:
            metadata = self.process_single_scenario(scenario)
            if metadata:
                batch_metadata.append(metadata)
                # Add to scenario type mapping
                scenario_type = metadata['scenario_type']
                filename = metadata['npz_file']
                batch_type_mapping[scenario_type].append(filename)
        
        return batch_metadata, dict(batch_type_mapping), batch_processed
    
    def process_single_scenario(self, scenario: NuPlanScenario) -> Optional[Dict[str, str]]:
        """
        Process a single scenario and extract metadata if it has been processed.
        
        Args:
            scenario: NuPlan scenario to process
            
        Returns:
            Metadata dictionary if scenario was processed, None otherwise
        """
        # Create the identifier that matches the file naming convention
        file_identifier = f"{scenario._map_name}_{scenario.token}"
        
        # Check if this scenario has been processed
        if file_identifier in self.processed_files:
            # Extract scenario type
            scenario_type = str(scenario.scenario_type) if hasattr(scenario, 'scenario_type') else 'unknown'
            
            # Extract scenario name (if available)
            scenario_name = scenario.scenario_name if hasattr(scenario, 'scenario_name') else file_identifier
            
            # Extract log name
            log_name = scenario.log_name if hasattr(scenario, 'log_name') else 'unknown'
            
            metadata = {
                'scenario_name': scenario_name,
                'scenario_type': scenario_type,
                'map_name': scenario._map_name,
                'token': scenario.token,
                'log_name': log_name,
                'npz_file': f"{file_identifier}.npz",
                'file_identifier': file_identifier
            }
            
            return metadata
        else:
            self.unmatched_scenarios.append(file_identifier)
            return None


def save_metadata(scenario_type_mapping: Dict[str, List[str]], 
                  output_dir: str, output_prefix: str = "scenario_metadata"):
    """
    Save scenario type mapping to pickle file.
    
    Args:
        scenario_type_mapping: Dictionary mapping scenario types to filenames
        output_dir: Directory to save output files
        output_prefix: Prefix for output filenames
    """
    if not scenario_type_mapping:
        logger.warning("No metadata to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scenario type mapping as pickle
    type_mapping_pkl_path = os.path.join(output_dir, f"{output_prefix}_by_type.pkl")
    with open(type_mapping_pkl_path, 'wb') as pklfile:
        pickle.dump(dict(scenario_type_mapping), pklfile)
    logger.info(f"Scenario type mapping saved to: {type_mapping_pkl_path}")
    
    # Print summary statistics
    print_summary(scenario_type_mapping)


def print_summary(scenario_type_mapping: Dict[str, List[str]]):
    """
    Print summary statistics of the extracted metadata.
    
    Args:
        scenario_type_mapping: Dictionary mapping scenario types to filenames
    """
    total_scenarios = sum(len(files) for files in scenario_type_mapping.values())
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total processed scenarios: {total_scenarios:,}")
    
    # Scenario type statistics
    logger.info("\nScenarios by type:")
    for scenario_type in sorted(scenario_type_mapping.keys()):
        count = len(scenario_type_mapping[scenario_type])
        percentage = (count / total_scenarios) * 100 if total_scenarios > 0 else 0
        logger.info(f"  {scenario_type}: {count:,} scenarios ({percentage:.1f}%)")


def get_filter_parameters(log_names=None):
    """
    Get filter parameters for scenario builder (simplified version).
    """
    scenario_types = None
    scenario_tokens = None
    map_names = None
    num_scenarios_per_type = None
    limit_total_scenarios = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    shuffle = False  # Don't shuffle for metadata extraction
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None
    
    return (scenario_types, scenario_tokens, log_names, map_names, 
            num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, 
            ego_displacement_minimum_m, expand_scenarios, remove_invalid_goals, 
            shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, 
            speed_noise_tolerance)


def main():
    parser = argparse.ArgumentParser(description='Extract scenario metadata from processed files')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, 
                       help='path to raw NuPlan data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, 
                       help='path to NuPlan map data')
    parser.add_argument('--mapping_pkl', type=str, required=True,
                       help='path to pickle file containing filename to path mappings')
    parser.add_argument('--output_dir', default='./metadata', type=str, 
                       help='directory to save metadata files')
    parser.add_argument('--split_file', type=str, default=None,
                       help='path to train split JSON file (e.g., nuplan_train_00.json)')
    parser.add_argument('--split_id', type=int, default=None,
                       help='split ID to process (0-based), used if split_file not provided')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='use parallel processing (default: True)')
    parser.add_argument('--no_parallel', dest='parallel', action='store_false',
                       help='disable parallel processing')
    parser.add_argument('--debug', action='store_true',
                       help='enable debug mode to show sample entries')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='set logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Load processed files from pickle mapping
    logger.info(f"Loading processed files from: {args.mapping_pkl}")
    mapping = load_processed_files_from_pickle(args.mapping_pkl)
    
    if not mapping:
        logger.error("No files found in mapping!")
        return
    
    # Get file identifiers
    processed_files = get_processed_file_identifiers(mapping)
    logger.info(f"Extracted {len(processed_files):,} file identifiers")
    
    # Show sample in debug mode
    if args.debug and processed_files:
        logger.debug("\nSample of processed file identifiers:")
        for identifier in list(processed_files)[:5]:
            logger.debug(f"  - {identifier}")
        logger.debug("...\n")
    
    # Load log names if split file is provided
    log_names = None
    if args.split_file:
        with open(args.split_file, "r", encoding="utf-8") as file:
            log_names = json.load(file)
        logger.info(f"Loaded {len(log_names)} log names from {args.split_file}")
    elif args.split_id is not None:
        # Try to load using the split_id format
        split_file_path = f'/root/planning-intern/Diffusion-Planner/train_splits/nuplan_train_{args.split_id:03d}.json'
        if os.path.exists(split_file_path):
            with open(split_file_path, "r", encoding="utf-8") as file:
                log_names = json.load(file)
            logger.info(f"Loaded {len(log_names)} log names from split {args.split_id}")
        else:
            logger.warning(f"Split file not found: {split_file_path}")
    
    # Initialize NuPlan scenario builder
    logger.info("\nInitializing NuPlan scenario builder...")
    sensor_root = None
    db_files = None
    map_version = "nuplan-maps-v1.0"
    
    builder = NuPlanScenarioBuilder(
        data_root=args.data_path,
        map_root=args.map_path,
        sensor_root=sensor_root,
        db_files=db_files,
        map_version=map_version
    )
    
    # Create scenario filter
    scenario_filter = ScenarioFilter(*get_filter_parameters(log_names=log_names))
    
    # Get scenarios
    logger.info("Loading scenarios from NuPlan database...")
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    logger.info(f"Loaded {len(scenarios):,} scenarios")
    
    # Clean up
    del worker, builder
    
    # Extract metadata using parallel processing
    logger.info(f"\nStarting metadata extraction (parallel={args.parallel})...")
    extractor = ScenarioMetadataExtractor(
        processed_files, 
        args.output_dir, 
        split_id=args.split_id,
        parallel=args.parallel
    )
    extractor.process_scenarios(scenarios)
    
    # Save final results
    output_prefix = "scenario_metadata"
    if args.split_id is not None:
        output_prefix = f"scenario_metadata_split_{args.split_id:03d}"
    
    save_metadata(extractor.scenario_type_mapping, args.output_dir, output_prefix)
    
    # Print debug info if in debug mode
    if args.debug:
        logger.debug("\n" + "="*60)
        logger.debug("DEBUG INFO")
        logger.debug("="*60)
        logger.debug(f"Total processed files: {len(processed_files):,}")
        logger.debug(f"Total scenarios: {len(scenarios):,}")
        logger.debug(f"Matched scenarios: {extractor.matched_count:,}")
        logger.debug(f"Unmatched count: {len(extractor.unmatched_scenarios):,}")
        if extractor.unmatched_scenarios:
            logger.debug(f"Sample unmatched (first 10):")
            for identifier in extractor.unmatched_scenarios[:10]:
                logger.debug(f"  - {identifier}")


if __name__ == "__main__":
    main()