#!/usr/bin/env python3
"""
Optimized script to organize trajectory/episode data with associated motion images.
Creates a new raw_data directory structure with runs 0-N, each containing:
- run JSON data 
- associated motion image data directory

Performance optimizations:
- Parallel file processing using multiprocessing
- Cached file system operations
- Batch operations where possible
- Progress tracking with tqdm
- Memory-efficient JSON parsing
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
import logging

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Simple progress tracker when tqdm is not available"""
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        
    def update(self, n: int = 1):
        self.current += n
        if self.current % max(1, self.total // 20) == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            percent = (self.current / self.total) * 100
            print(f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%) - {rate:.1f} items/sec")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def create_progress_bar(total: int, desc: str = "Processing"):
    """Create progress bar (tqdm if available, otherwise simple tracker)"""
    if HAS_TQDM:
        return tqdm(total=total, desc=desc, unit="files")
    else:
        return ProgressTracker(total, desc)


def extract_session_id_from_filename(filename: str) -> str:
    """Extract session ID from filename (first UUID part before underscore)"""
    return filename.split('_')[0]


@lru_cache(maxsize=1000)
def get_directory_file_count(directory_path: str) -> int:
    """Cached function to get file count in directory"""
    try:
        path = Path(directory_path)
        if not path.exists():
            return 0
        return sum(1 for _ in path.iterdir() if _.is_file())
    except (OSError, PermissionError):
        return 0


@lru_cache(maxsize=1000)
def directory_exists_and_has_files(directory_path: str) -> bool:
    """Cached function to check if directory exists and has files"""
    try:
        path = Path(directory_path)
        return path.exists() and any(path.iterdir())
    except (OSError, PermissionError):
        return False


def check_json_has_motion_data(json_file_path: str) -> bool:
    """Check if JSON file contains motion data (optimized parsing)"""
    try:
        with open(json_file_path, 'r') as f:
            # Read only what we need - look for motion_data key
            data = json.load(f)
            motion_data = data.get('motion_data', [])
            return len(motion_data) > 0
    except (json.JSONDecodeError, KeyError, FileNotFoundError, OSError):
        return False


def validate_session_worker(args: Tuple[str, str, List[str]]) -> Optional[Tuple[str, str, str, int]]:
    """Worker function to validate a single session (for multiprocessing)"""
    json_file_path, session_id, motion_paths = args
    
    # Check if JSON has motion data first (fastest check)
    if not check_json_has_motion_data(json_file_path):
        return None
    
    # Check all motion directories for this session
    for motion_path in motion_paths:
        motion_dir_path = str(Path(motion_path) / session_id)
        if directory_exists_and_has_files(motion_dir_path):
            image_count = get_directory_file_count(motion_dir_path)
            return (json_file_path, session_id, motion_path, image_count)
    
    return None


def find_data_directories() -> Tuple[List[Path], List[Path]]:
    """Find existing data directories by checking multiple possible locations"""
    base_paths = [
        "cognitive_bt_framework/src/task_execution_data",
        "task_execution_data"
    ]
    
    found_sessions = []
    found_motion = []
    
    for base_path in base_paths:
        sessions_path = Path(base_path) / "sessions"
        motion_path = Path(base_path) / "images" / "motion"
        
        if sessions_path.exists() and any(sessions_path.glob("*.json")):
            found_sessions.append(sessions_path)
            logger.info(f"Found sessions directory: {sessions_path}")
        
        if motion_path.exists() and any(motion_path.iterdir()):
            found_motion.append(motion_path)
            logger.info(f"Found motion images directory: {motion_path}")
    
    if len(found_sessions) > 1:
        logger.info(f"Will combine data from all {len(found_sessions)} sessions directories")
    if len(found_motion) > 1:
        logger.info(f"Will combine data from all {len(found_motion)} motion directories")
    
    return found_sessions, found_motion


def collect_json_files(sessions_paths: List[str]) -> List[Path]:
    """Collect all JSON files from sessions directories"""
    all_json_files = []
    
    with create_progress_bar(len(sessions_paths), "Scanning directories") as pbar:
        for sessions_path in sessions_paths:
            sessions_dir = Path(sessions_path)
            json_files = list(sessions_dir.glob("*.json"))
            all_json_files.extend(json_files)
            logger.info(f"Found {len(json_files)} session files in {sessions_path}")
            if HAS_TQDM:
                pbar.update(1)
    
    logger.info(f"Total {len(all_json_files)} session files found")
    return all_json_files


def validate_sessions_parallel(json_files: List[Path], motion_paths: List[str], max_workers: Optional[int] = None) -> List[Tuple[str, str, str, int]]:
    """Validate sessions in parallel using multiprocessing"""
    if max_workers is None:
        max_workers = min(10, len(json_files))
    
    # Prepare arguments for worker processes
    worker_args = []
    for json_file in json_files:
        session_id = extract_session_id_from_filename(json_file.name)
        worker_args.append((str(json_file), session_id, motion_paths))
    
    valid_sessions = []
    
    logger.info(f"Validating {len(json_files)} sessions using {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with create_progress_bar(len(worker_args), "Validating sessions") as pbar:
            # Submit all tasks
            future_to_args = {executor.submit(validate_session_worker, args): args for args in worker_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                result = future.result()
                if result is not None:
                    valid_sessions.append(result)
                
                if HAS_TQDM:
                    pbar.update(1)
    
    logger.info(f"Found {len(valid_sessions)} sessions with motion image data")
    return valid_sessions


def copy_files_worker(args: Tuple[str, str, str, str, int]) -> Tuple[bool, str, int]:
    """Worker function to copy files for a single run"""
    json_file_path, session_id, motion_source_path, run_dir_path, expected_images = args
    
    try:
        run_dir = Path(run_dir_path)
        run_dir.mkdir(exist_ok=True)
        
        # Copy JSON file
        json_dest = run_dir / "session_data.json"
        shutil.copy2(json_file_path, json_dest)
        
        # Copy motion images directory
        motion_src = Path(motion_source_path) / session_id
        motion_dest = run_dir / "motion_images"
        
        if motion_src.exists():
            shutil.copytree(motion_src, motion_dest, dirs_exist_ok=True)
            actual_images = len(list(motion_dest.iterdir()))
            return (True, str(run_dir.name), actual_images)
        else:
            return (False, f"Motion images not found for {session_id}", 0)
            
    except Exception as e:
        return (False, f"Error processing {session_id}: {str(e)}", 0)


def organize_data_parallel(sessions_paths: List[str], motion_images_paths: List[str], output_path: str, max_workers: Optional[int] = None):
    """
    Organize session data and motion images into numbered run directories (parallelized version)
    
    Args:
        sessions_paths: List of paths to sessions directories containing JSON files
        motion_images_paths: List of paths to motion images directories
        output_path: Path to output raw_data directory
        max_workers: Maximum number of worker processes (None for auto-detect)
    """
    start_time = time.time()
    
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Collect all JSON files
    all_json_files = collect_json_files(sessions_paths)
    
    # Step 2: Validate sessions in parallel
    valid_sessions = validate_sessions_parallel(all_json_files, motion_images_paths, max_workers)
    
    if not valid_sessions:
        logger.warning("No valid sessions found!")
        return
    
    # Step 3: Sort by filename for consistent ordering
    valid_sessions.sort(key=lambda x: Path(x[0]).name)
    
    # Step 4: Copy files in parallel
    if max_workers is None:
        max_workers = min(cpu_count(), len(valid_sessions))
    
    logger.info(f"Copying files for {len(valid_sessions)} runs using {max_workers} workers...")
    
    # Prepare arguments for copy workers
    copy_args = []
    for run_num, (json_file_path, session_id, motion_source_path, expected_images) in enumerate(valid_sessions):
        run_dir_path = str(output_dir / f"run_{run_num:03d}")
        copy_args.append((json_file_path, session_id, motion_source_path, run_dir_path, expected_images))
    
    successful_runs = 0
    total_images = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with create_progress_bar(len(copy_args), "Copying files") as pbar:
            # Submit all copy tasks
            future_to_args = {executor.submit(copy_files_worker, args): args for args in copy_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                success, message, image_count = future.result()
                
                if success:
                    successful_runs += 1
                    total_images += image_count
                    logger.info(f"Created {message}: {image_count} images")
                else:
                    logger.warning(f"Failed: {message}")
                
                if HAS_TQDM:
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nOrganization complete!")
    logger.info(f"Created {successful_runs} runs in {output_dir}")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    logger.info(f"Average speed: {len(valid_sessions)/elapsed_time:.2f} runs/second")


def organize_data_sequential(sessions_paths: List[str], motion_images_paths: List[str], output_path: str):
    """Original sequential implementation (kept for compatibility)"""
    from organize_trajectory_data import organize_data
    organize_data(sessions_paths, motion_images_paths, output_path)


def dry_run_parallel(sessions_paths: List[str], motion_paths: List[str], max_workers: Optional[int] = None):
    """Parallel dry run implementation"""
    logger.info("DRY RUN MODE - No files will be copied")
    
    all_json_files = collect_json_files(sessions_paths)
    valid_sessions = validate_sessions_parallel(all_json_files, motion_paths, max_workers)
    
    total_images = sum(session[3] for session in valid_sessions)
    
    logger.info(f"\nDry run results:")
    logger.info(f"Would create {len(valid_sessions)} runs from {len(all_json_files)} session files")
    logger.info(f"Total images to process: {total_images}")
    
    # Show sample of what would be processed
    if valid_sessions:
        logger.info("\nSample runs that would be created:")
        for i, (json_path, session_id, motion_path, image_count) in enumerate(valid_sessions[:5]):
            json_name = Path(json_path).name
            logger.info(f"  run_{i:03d}: {json_name} ({image_count} images from {motion_path})")
        
        if len(valid_sessions) > 5:
            logger.info(f"  ... and {len(valid_sessions) - 5} more runs")


def main():
    parser = argparse.ArgumentParser(description="Organize trajectory data with motion images (optimized)")
    parser.add_argument(
        "--sessions", 
        help="Path to sessions directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--motion-images",
        help="Path to motion images directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output",
        default="./raw_data",
        help="Output directory for organized data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be organized without copying files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: auto-detect based on CPU cores)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use original sequential implementation instead of parallel"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Auto-detect data directories if not specified
    if not args.sessions or not args.motion_images:
        logger.info("Auto-detecting data directories...")
        found_sessions, found_motion = find_data_directories()
        
        if not args.sessions:
            if found_sessions:
                sessions_paths = [str(path) for path in found_sessions]
            else:
                logger.error("No sessions directories found. Please specify --sessions path.")
                return
        else:
            sessions_paths = [args.sessions]
        
        if not args.motion_images:
            if found_motion:
                motion_paths = [str(path) for path in found_motion]
            else:
                logger.error("No motion images directories found. Please specify --motion-images path.")
                return
        else:
            motion_paths = [args.motion_images]
    else:
        sessions_paths = [args.sessions]
        motion_paths = [args.motion_images]
    
    # Show configuration
    logger.info(f"Configuration:")
    logger.info(f"  Sessions paths: {sessions_paths}")
    logger.info(f"  Motion paths: {motion_paths}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Workers: {args.workers or 'auto-detect'}")
    logger.info(f"  Parallel: {not args.sequential}")
    
    if args.dry_run:
        if args.sequential:
            # Use original dry run implementation
            from organize_trajectory_data import main as original_main
            import sys
            sys.argv = ['organize_trajectory_data.py', '--dry-run'] + (
                ['--sessions', sessions_paths[0]] if len(sessions_paths) == 1 else []
            ) + (
                ['--motion-images', motion_paths[0]] if len(motion_paths) == 1 else []
            ) + ['--output', args.output]
            original_main()
        else:
            dry_run_parallel(sessions_paths, motion_paths, args.workers)
    else:
        if args.sequential:
            logger.info("Using sequential implementation...")
            organize_data_sequential(sessions_paths, motion_paths, args.output)
        else:
            logger.info("Using parallel implementation...")
            organize_data_parallel(sessions_paths, motion_paths, args.output, args.workers)


if __name__ == "__main__":
    main()