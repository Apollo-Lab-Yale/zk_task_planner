#!/usr/bin/env python3
"""
Script to extract individual task results from trajectory data.
Extracts session_id, task, execution_success, user_success_rating,
and explanations from JSON files and compiles them into a CSV file.
"""

import json
import csv
from pathlib import Path
import argparse


def find_sessions_directories():
    """Find all existing sessions directories by checking multiple possible locations"""
    base_paths = [
        "cognitive_bt_framework/src/task_execution_data",
        "cognitive_bt_framework/src/task_execution_data_v1",
        "task_execution_data",
        "complete_workflow_test",
        "test_task_data"
    ]

    found_directories = []

    for base_path in base_paths:
        sessions_path = Path(base_path) / "sessions"
        if sessions_path.exists() and any(sessions_path.glob("*.json")):
            found_directories.append(sessions_path)
            print(f"Found sessions directory: {sessions_path}")

        # Also check for direct JSON files in the base directory
        base_dir = Path(base_path)
        if base_dir.exists() and any(base_dir.glob("*.json")):
            found_directories.append(base_dir)
            print(f"Found JSON files directory: {base_dir}")

    if found_directories:
        if len(found_directories) > 1:
            print(f"Will combine data from all {len(found_directories)} directories")
        return found_directories

    return []


def extract_task_results(sessions_paths, output_csv_path):
    """
    Extract individual task results from session JSON files and output to CSV

    Args:
        sessions_paths: List of paths to directories containing JSON files
        output_csv_path: Path to output CSV file
    """

    # List to store all task results
    task_results = []

    # Collect all JSON files from all directories
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

            # Extract session ID from filename (remove .json extension)
            session_id = json_file.stem

            # Extract required fields
            natural_language_task = data.get('natural_language_task', '')
            user_success_rating = data.get('user_success_rating', False)
            timing_data = data.get('timing', {})

            # Extract explanation fields (these might have different names)
            explanation_fields = [
                'explanation',
                'failure_explanation',
                'success_explanation',
                'execution_explanation',
                'user_explanation',
                'notes'
            ]

            explanation = ''
            for field in explanation_fields:
                if field in data and data[field]:
                    explanation = str(data[field])
                    break

            # Also check for explanations in nested structures
            if not explanation:
                # Check in execution results or similar nested structures
                if 'execution_result' in data and isinstance(data['execution_result'], dict):
                    for field in explanation_fields:
                        if field in data['execution_result'] and data['execution_result'][field]:
                            explanation = str(data['execution_result'][field])
                            break

            # Additional metadata
            motion_data_available = len(data.get('motion_data', [])) > 0
            timestamp = data.get('timestamp', '')

            # Extract timing information if available
            planning_time = None
            inference_time = None
            execution_time = None
            total_time = None

            if timing_data:
                planning_start = timing_data.get('planning_start')
                planning_end = timing_data.get('planning_end')
                inference_start = timing_data.get('inference_start')
                inference_end = timing_data.get('inference_end')
                execution_start = timing_data.get('execution_start')
                execution_end = timing_data.get('execution_end')

                # Calculate timing durations
                if planning_start is not None and planning_end is not None:
                    calculated_planning_time = planning_end - planning_start
                    if calculated_planning_time >= 0:  # Sanity check
                        planning_time = calculated_planning_time

                if inference_start is not None and inference_end is not None:
                    calculated_inference_time = inference_end - inference_start
                    if calculated_inference_time >= 0:  # Sanity check
                        inference_time = calculated_inference_time

                if execution_start is not None and execution_end is not None:
                    calculated_execution_time = execution_end - execution_start
                    if calculated_execution_time >= 0:  # Sanity check
                        execution_time = calculated_execution_time

                # Calculate total time (from planning start to execution end)
                if planning_start is not None and execution_end is not None:
                    calculated_total_time = execution_end - planning_start
                    if calculated_total_time >= 0:  # Sanity check
                        total_time = calculated_total_time

            # Create result record
            result = {
                'session_id': session_id,
                'task': natural_language_task,
                'user_success_rating': user_success_rating,
                'explanation': explanation,
                'motion_data_available': motion_data_available,
                'timestamp': timestamp,
                'planning_time': planning_time,
                'inference_time': inference_time,
                'execution_time': execution_time,
                'total_time': total_time,
                'source_file': json_file.name
            }

            task_results.append(result)
            processed_count += 1

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {json_file.name}: {e}")
            error_count += 1
            continue

    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")

    # Write to CSV
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'session_id',
            'task',
            'user_success_rating',
            'explanation',
            'motion_data_available',
            'timestamp',
            'planning_time',
            'inference_time',
            'execution_time',
            'total_time',
            'source_file'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by session_id for consistent output
        for result in sorted(task_results, key=lambda x: x['session_id']):
            writer.writerow(result)

    print(f"\nTask results extracted to: {output_path}")

    # Print summary statistics
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total task attempts: {len(task_results)}")

    user_success_count = sum(1 for r in task_results if r['user_success_rating'])
    with_explanation_count = sum(1 for r in task_results if r['explanation'])
    with_motion_data_count = sum(1 for r in task_results if r['motion_data_available'])
    with_timing_data_count = sum(1 for r in task_results if r['total_time'] is not None)

    print(f"User successes: {user_success_count} ({user_success_count/len(task_results)*100:.1f}%)")
    print(f"With explanations: {with_explanation_count} ({with_explanation_count/len(task_results)*100:.1f}%)")
    print(f"With motion data: {with_motion_data_count} ({with_motion_data_count/len(task_results)*100:.1f}%)")
    print(f"With timing data: {with_timing_data_count} ({with_timing_data_count/len(task_results)*100:.1f}%)")

    # Show timing statistics if available
    if with_timing_data_count > 0:
        planning_times = [r['planning_time'] for r in task_results if r['planning_time'] is not None]
        inference_times = [r['inference_time'] for r in task_results if r['inference_time'] is not None]
        execution_times = [r['execution_time'] for r in task_results if r['execution_time'] is not None]
        total_times = [r['total_time'] for r in task_results if r['total_time'] is not None]

        print("\nTiming Statistics:")
        if planning_times:
            avg_planning = sum(planning_times) / len(planning_times)
            print(f"  Average planning time: {avg_planning:.2f}s ({len(planning_times)} samples)")
        if inference_times:
            avg_inference = sum(inference_times) / len(inference_times)
            print(f"  Average inference time: {avg_inference:.3f}s ({len(inference_times)} samples)")
        if execution_times:
            avg_execution = sum(execution_times) / len(execution_times)
            print(f"  Average execution time: {avg_execution:.2f}s ({len(execution_times)} samples)")
        if total_times:
            avg_total = sum(total_times) / len(total_times)
            print(f"  Average total time: {avg_total:.2f}s ({len(total_times)} samples)")

    # Show unique task types
    unique_tasks = set(r['task'] for r in task_results if r['task'])
    print(f"\nUnique tasks found: {len(unique_tasks)}")
    for task in sorted(unique_tasks):
        task_count = sum(1 for r in task_results if r['task'] == task)
        print(f"  {task} ({task_count} attempts)")


def main():
    parser = argparse.ArgumentParser(description="Extract individual task results from trajectory data")
    parser.add_argument(
        "--sessions",
        help="Path to sessions directory containing JSON files (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output",
        default="./task_results.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview extraction without writing CSV file"
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

        print(f"Found {len(all_json_files)} total JSON files across all directories")

        # Preview first few files
        preview_files = all_json_files[:5]
        print("\nPreview of first 5 files:")
        for json_file in preview_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                task = data.get('natural_language_task', 'N/A')
                user_success = data.get('user_success_rating', False)
                print(f"  {json_file.name}: {task[:50]}... | User: {user_success}")
            except Exception as e:
                print(f"  {json_file.name}: Error - {e}")
    else:
        extract_task_results(sessions_paths, args.output)


if __name__ == "__main__":
    main()