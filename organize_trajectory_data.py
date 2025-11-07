#!/usr/bin/env python3
"""
Script to organize trajectory/episode data with associated motion images.
Creates a new raw_data directory structure with runs 0-N, each containing:
- run JSON data 
- associated motion image data directory
"""

import os
import json
import shutil
from pathlib import Path
import argparse


def extract_session_id_from_filename(filename):
    """Extract session ID from filename (first UUID part before underscore)"""
    return filename.split('_')[0]


def has_motion_data_and_images(json_file_path, session_id, motion_images_base_path):
    """Check if JSON has motion data and motion images exist for this session ID"""
    # Check if motion images directory exists
    motion_dir = Path(motion_images_base_path) / session_id
    if not motion_dir.exists() or not any(motion_dir.iterdir()):
        return False
    
    # Check if JSON actually contains motion_data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            motion_data = data.get('motion_data', [])
            return len(motion_data) > 0
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False


def find_data_directories():
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
            print(f"Found sessions directory: {sessions_path}")
        
        if motion_path.exists() and any(motion_path.iterdir()):
            found_motion.append(motion_path)
            print(f"Found motion images directory: {motion_path}")
    
    if len(found_sessions) > 1:
        print(f"Will combine data from all {len(found_sessions)} sessions directories")
    if len(found_motion) > 1:
        print(f"Will combine data from all {len(found_motion)} motion directories")
    
    return found_sessions, found_motion


def organize_data(sessions_paths, motion_images_paths, output_path):
    """
    Organize session data and motion images into numbered run directories
    
    Args:
        sessions_paths: List of paths to sessions directories containing JSON files
        motion_images_paths: List of paths to motion images directories
        output_path: Path to output raw_data directory
    """
    
    output_dir = Path(output_path)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Collect all session JSON files from all directories
    all_json_files = []
    for sessions_path in sessions_paths:
        sessions_dir = Path(sessions_path)
        json_files = list(sessions_dir.glob("*.json"))
        all_json_files.extend(json_files)
        print(f"Found {len(json_files)} session files in {sessions_path}")
    
    print(f"Total {len(all_json_files)} session files found")
    
    # Filter to only sessions that have both motion data in JSON and motion image files
    valid_sessions = []
    for json_file in all_json_files:
        session_id = extract_session_id_from_filename(json_file.name)
        
        # Check all motion directories for this session
        motion_found = False
        motion_source = None
        for motion_path in motion_images_paths:
            motion_dir = Path(motion_path)
            if has_motion_data_and_images(json_file, session_id, motion_dir):
                motion_found = True
                motion_source = motion_dir
                break
        
        if motion_found:
            valid_sessions.append((json_file, session_id, motion_source))
        else:
            print(f"Skipping {json_file.name} - no motion data or images found")
    
    print(f"Found {len(valid_sessions)} sessions with motion image data")
    
    # Sort by filename for consistent ordering
    valid_sessions.sort(key=lambda x: x[0].name)
    
    # Create numbered run directories
    for run_num, (json_file, session_id, motion_source_dir) in enumerate(valid_sessions):
        run_dir = output_dir / f"run_{run_num:03d}"
        run_dir.mkdir(exist_ok=True)
        
        # Copy JSON file
        json_dest = run_dir / f"session_data.json"
        shutil.copy2(json_file, json_dest)
        
        # Copy motion images directory from the correct source
        motion_src = motion_source_dir / session_id
        motion_dest = run_dir / "motion_images"
        
        if motion_src.exists():
            shutil.copytree(motion_src, motion_dest, dirs_exist_ok=True)
            print(f"Created run_{run_num:03d}: {json_file.name} -> {len(list(motion_dest.iterdir()))} images")
        else:
            print(f"Warning: Motion images not found for {session_id}")
    
    print(f"\nOrganization complete! Created {len(valid_sessions)} runs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Organize trajectory data with motion images")
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
    
    args = parser.parse_args()
    
    # Auto-detect data directories if not specified
    if not args.sessions or not args.motion_images:
        print("Auto-detecting data directories...")
        found_sessions, found_motion = find_data_directories()
        
        if not args.sessions:
            if found_sessions:
                sessions_paths = [str(path) for path in found_sessions]
            else:
                print("ERROR: No sessions directories found. Please specify --sessions path.")
                return
        else:
            sessions_paths = [args.sessions]
        
        if not args.motion_images:
            if found_motion:
                motion_paths = [str(path) for path in found_motion]
            else:
                print("ERROR: No motion images directories found. Please specify --motion-images path.")
                return
        else:
            motion_paths = [args.motion_images]
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        
        all_json_files = []
        for sessions_path in sessions_paths:
            sessions_dir = Path(sessions_path)
            json_files = list(sessions_dir.glob("*.json"))
            all_json_files.extend(json_files)
            print(f"Found {len(json_files)} session files in {sessions_path}")
        
        valid_count = 0
        for json_file in all_json_files:
            session_id = extract_session_id_from_filename(json_file.name)
            
            # Check all motion directories for this session
            motion_found = False
            for motion_path in motion_paths:
                motion_dir = Path(motion_path)
                if has_motion_data_and_images(json_file, session_id, motion_dir):
                    motion_found = True
                    image_path = motion_dir / session_id
                    image_count = len(list(image_path.iterdir())) if image_path.exists() else 0
                    print(f"Would process: {json_file.name} ({image_count} images from {motion_path})")
                    break
            
            if motion_found:
                valid_count += 1
            else:
                print(f"Would skip: {json_file.name} (no motion data or images)")
        
        print(f"\nWould create {valid_count} runs total from {len(all_json_files)} session files")
    else:
        organize_data(sessions_paths, motion_paths, args.output)


if __name__ == "__main__":
    main()