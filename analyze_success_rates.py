#!/usr/bin/env python3
"""
Script to analyze success rates per task from trajectory data.
Extracts execution_success and user_success_rating from JSON files
and compiles success rates per task into a CSV file.
"""

import json
import csv
from pathlib import Path
import argparse
from collections import defaultdict


def find_sessions_directories():
    """Find all existing sessions directories by checking multiple possible locations"""
    base_paths = [
        "cognitive_bt_framework/src/task_execution_data",
        "task_execution_data"
    ]
    
    found_directories = []
    
    for base_path in base_paths:
        sessions_path = Path(base_path) / "sessions"
        if sessions_path.exists() and any(sessions_path.glob("*.json")):
            found_directories.append(sessions_path)
            print(f"Found sessions directory: {sessions_path}")
    
    if found_directories:
        if len(found_directories) > 1:
            print(f"Will combine data from all {len(found_directories)} sessions directories")
        return found_directories
    
    return []


def extract_task_name_from_natural_language(natural_language_task):
    """Extract a normalized task name from the natural language description"""
    # Clean and normalize the task name
    task = natural_language_task.lower().strip()
    if task.endswith('.'):
        task = task[:-1]
    return task


def analyze_success_rates(sessions_paths, output_csv_path):
    """
    Analyze success rates from session JSON files and output to CSV
    
    Args:
        sessions_paths: List of paths to sessions directories containing JSON files
        output_csv_path: Path to output CSV file
    """
    
    # Dictionary to store task statistics
    task_stats = defaultdict(lambda: {
        'total_attempts': 0,
        'user_success_count': 0,
        'user_success_with_motion_count': 0,
        'user_success_rate': 0.0,
        'user_success_with_motion_rate': 0.0,
        'planning_times': [],
        'inference_times': [],
        'execution_times': [],
        'total_times': [],
        'avg_planning_time': 0.0,
        'avg_inference_time': 0.0,
        'avg_execution_time': 0.0,
        'avg_total_time': 0.0
    })
    
    # Collect all JSON files from all sessions directories
    all_json_files = []
    for sessions_path in sessions_paths:
        sessions_dir = Path(sessions_path)
        json_files = list(sessions_dir.glob("*.json"))
        all_json_files.extend(json_files)
        print(f"Found {len(json_files)} session files in {sessions_path}")
    
    print(f"Processing {len(all_json_files)} total session files...")
    
    processed_count = 0
    error_count = 0
    
    for json_file in all_json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract required fields
            natural_language_task = data.get('natural_language_task', '')
            user_success_rating = data.get('user_success_rating', False)
            motion_data = data.get('motion_data', [])
            timing_data = data.get('timing', {})
            
            if not natural_language_task:
                print(f"Warning: No natural_language_task found in {json_file.name}")
                continue
                
            # Normalize task name
            task_name = extract_task_name_from_natural_language(natural_language_task)
            
            # Update statistics
            stats = task_stats[task_name]
            stats['total_attempts'] += 1

            if user_success_rating:
                stats['user_success_count'] += 1

            # Count user success trials that also have motion data
            if user_success_rating and len(motion_data) > 0:
                stats['user_success_with_motion_count'] += 1

            # Extract timing information if available
            if timing_data:
                planning_start = timing_data.get('planning_start')
                planning_end = timing_data.get('planning_end')
                inference_start = timing_data.get('inference_start')
                inference_end = timing_data.get('inference_end')
                execution_start = timing_data.get('execution_start')
                execution_end = timing_data.get('execution_end')

                # Calculate timing durations
                if planning_start is not None and planning_end is not None:
                    planning_time = planning_end - planning_start
                    if planning_time >= 0:  # Sanity check
                        stats['planning_times'].append(planning_time)

                if inference_start is not None and inference_end is not None:
                    inference_time = inference_end - inference_start
                    if inference_time >= 0:  # Sanity check
                        stats['inference_times'].append(inference_time)

                if execution_start is not None and execution_end is not None:
                    execution_time = execution_end - execution_start
                    if execution_time >= 0:  # Sanity check
                        stats['execution_times'].append(execution_time)

                # Calculate total time (from planning start to execution end)
                if planning_start is not None and execution_end is not None:
                    total_time = execution_end - planning_start
                    if total_time >= 0:  # Sanity check
                        stats['total_times'].append(total_time)
                
            processed_count += 1
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {json_file.name}: {e}")
            error_count += 1
            continue
    
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")
    
    # Calculate success rates and average timing
    for task_name, stats in task_stats.items():
        total = stats['total_attempts']
        if total > 0:
            stats['user_success_rate'] = stats['user_success_count'] / total * 100
            stats['user_success_with_motion_rate'] = stats['user_success_with_motion_count'] / total * 100

        # Calculate average timing metrics
        if stats['planning_times']:
            stats['avg_planning_time'] = sum(stats['planning_times']) / len(stats['planning_times'])

        if stats['inference_times']:
            stats['avg_inference_time'] = sum(stats['inference_times']) / len(stats['inference_times'])

        if stats['execution_times']:
            stats['avg_execution_time'] = sum(stats['execution_times']) / len(stats['execution_times'])

        if stats['total_times']:
            stats['avg_total_time'] = sum(stats['total_times']) / len(stats['total_times'])
    
    # Write to CSV
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'task_name',
            'total_attempts',
            'user_success_count',
            'user_success_rate',
            'user_success_with_motion_count',
            'user_success_with_motion_rate',
            'avg_planning_time',
            'avg_inference_time',
            'avg_execution_time',
            'avg_total_time'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort tasks by name for consistent output
        for task_name in sorted(task_stats.keys()):
            stats = task_stats[task_name]
            writer.writerow({
                'task_name': task_name,
                'total_attempts': stats['total_attempts'],
                'user_success_count': stats['user_success_count'],
                'user_success_rate': f"{stats['user_success_rate']:.1f}%",
                'user_success_with_motion_count': stats['user_success_with_motion_count'],
                'user_success_with_motion_rate': f"{stats['user_success_with_motion_rate']:.1f}%",
                'avg_planning_time': f"{stats['avg_planning_time']:.2f}s",
                'avg_inference_time': f"{stats['avg_inference_time']:.2f}s",
                'avg_execution_time': f"{stats['avg_execution_time']:.2f}s",
                'avg_total_time': f"{stats['avg_total_time']:.2f}s"
            })
    
    print(f"\nSuccess rate analysis saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== TASK SUCCESS RATE SUMMARY ===")
    print(f"{'Task Name':<45} {'Attempts':<8} {'User Success':<15} {'User+Motion':<15} {'Planning':<10} {'Inference':<10} {'Execution':<12} {'Total':<10}")
    print("=" * 135)

    for task_name in sorted(task_stats.keys()):
        stats = task_stats[task_name]
        user_display = f"{stats['user_success_count']}/{stats['total_attempts']} ({stats['user_success_rate']:.1f}%)"
        motion_display = f"{stats['user_success_with_motion_count']}/{stats['total_attempts']} ({stats['user_success_with_motion_rate']:.1f}%)"

        planning_display = f"{stats['avg_planning_time']:.1f}s" if stats['avg_planning_time'] > 0 else "N/A"
        inference_display = f"{stats['avg_inference_time']:.3f}s" if stats['avg_inference_time'] > 0 else "N/A"
        execution_display = f"{stats['avg_execution_time']:.1f}s" if stats['avg_execution_time'] > 0 else "N/A"
        total_display = f"{stats['avg_total_time']:.1f}s" if stats['avg_total_time'] > 0 else "N/A"

        print(f"{task_name:<45} {stats['total_attempts']:<8} {user_display:<15} {motion_display:<15} {planning_display:<10} {inference_display:<10} {execution_display:<12} {total_display:<10}")
    
    # Overall statistics
    total_attempts = sum(stats['total_attempts'] for stats in task_stats.values())
    total_user_success = sum(stats['user_success_count'] for stats in task_stats.values())
    total_user_motion_success = sum(stats['user_success_with_motion_count'] for stats in task_stats.values())

    # Calculate overall average timing
    all_planning_times = []
    all_inference_times = []
    all_execution_times = []
    all_total_times = []

    for stats in task_stats.values():
        all_planning_times.extend(stats['planning_times'])
        all_inference_times.extend(stats['inference_times'])
        all_execution_times.extend(stats['execution_times'])
        all_total_times.extend(stats['total_times'])

    overall_avg_planning = sum(all_planning_times) / len(all_planning_times) if all_planning_times else 0
    overall_avg_inference = sum(all_inference_times) / len(all_inference_times) if all_inference_times else 0
    overall_avg_execution = sum(all_execution_times) / len(all_execution_times) if all_execution_times else 0
    overall_avg_total = sum(all_total_times) / len(all_total_times) if all_total_times else 0

    overall_user_rate = (total_user_success / total_attempts * 100) if total_attempts > 0 else 0
    overall_motion_rate = (total_user_motion_success / total_attempts * 100) if total_attempts > 0 else 0

    print("=" * 135)
    overall_user_display = f"{total_user_success}/{total_attempts} ({overall_user_rate:.1f}%)"
    overall_motion_display = f"{total_user_motion_success}/{total_attempts} ({overall_motion_rate:.1f}%)"
    overall_planning_display = f"{overall_avg_planning:.1f}s" if overall_avg_planning > 0 else "N/A"
    overall_inference_display = f"{overall_avg_inference:.3f}s" if overall_avg_inference > 0 else "N/A"
    overall_execution_display = f"{overall_avg_execution:.1f}s" if overall_avg_execution > 0 else "N/A"
    overall_total_display = f"{overall_avg_total:.1f}s" if overall_avg_total > 0 else "N/A"

    print(f"{'OVERALL':<45} {total_attempts:<8} {overall_user_display:<15} {overall_motion_display:<15} {overall_planning_display:<10} {overall_inference_display:<10} {overall_execution_display:<12} {overall_total_display:<10}")


def main():
    parser = argparse.ArgumentParser(description="Analyze task success rates from trajectory data")
    parser.add_argument(
        "--sessions", 
        help="Path to sessions directory containing JSON files (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output",
        default="./task_success_rates.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview analysis without writing CSV file"
    )
    
    args = parser.parse_args()
    
    # Auto-detect sessions directories if not specified
    if not args.sessions:
        print("Auto-detecting sessions directories...")
        found_sessions = find_sessions_directories()
        if found_sessions:
            sessions_paths = [str(path) for path in found_sessions]
        else:
            print("ERROR: No sessions directories found. Please specify --sessions path.")
            return
    else:
        sessions_paths = [args.sessions]
    
    if args.dry_run:
        print("DRY RUN MODE - No CSV file will be written")
        
        all_json_files = []
        for sessions_path in sessions_paths:
            sessions_dir = Path(sessions_path)
            json_files = list(sessions_dir.glob("*.json"))
            all_json_files.extend(json_files)
            print(f"Found {len(json_files)} JSON files in {sessions_path}")
        
        task_preview = defaultdict(int)
        preview_files = all_json_files[:10]  # Preview first 10 files total
        for json_file in preview_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                natural_language_task = data.get('natural_language_task', '')
                if natural_language_task:
                    task_name = extract_task_name_from_natural_language(natural_language_task)
                    task_preview[task_name] += 1
            except Exception as e:
                print(f"Error previewing {json_file.name}: {e}")
        
        print(f"Found {len(all_json_files)} total JSON files across all directories")
        print("Sample task names from first 10 files:")
        for task_name, count in sorted(task_preview.items()):
            print(f"  {task_name} ({count} files)")
    else:
        analyze_success_rates(sessions_paths, args.output)


if __name__ == "__main__":
    main()