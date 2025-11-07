#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and optimized trajectory organization
"""

import time
import subprocess
import sys
from pathlib import Path
import argparse


def run_command_with_timing(cmd, description):
    """Run a command and measure execution time"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ SUCCESS - Elapsed time: {elapsed:.2f} seconds")
        print(f"Output:\n{result.stdout}")
        
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        
        return elapsed, True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"❌ FAILED - Elapsed time: {elapsed:.2f} seconds")
        print(f"Error output:\n{e.stderr}")
        
        return elapsed, False


def main():
    parser = argparse.ArgumentParser(description="Benchmark trajectory organization scripts")
    parser.add_argument(
        "--sessions",
        help="Path to sessions directory"
    )
    parser.add_argument(
        "--motion-images", 
        help="Path to motion images directory"
    )
    parser.add_argument(
        "--output-original",
        default="./benchmark_original",
        help="Output directory for original script"
    )
    parser.add_argument(
        "--output-optimized",
        default="./benchmark_optimized", 
        help="Output directory for optimized script"
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="Only run dry-run benchmarks (no actual file copying)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers for optimized version"
    )
    
    args = parser.parse_args()
    
    # Check if scripts exist
    original_script = Path("organize_trajectory_data.py")
    optimized_script = Path("organize_trajectory_data_optimized.py")
    
    if not original_script.exists():
        print(f"❌ Original script not found: {original_script}")
        return
    
    if not optimized_script.exists():
        print(f"❌ Optimized script not found: {optimized_script}")
        return
    
    print("🚀 Trajectory Organization Benchmark")
    print(f"Original script: {original_script}")
    print(f"Optimized script: {optimized_script}")
    
    # Build base command arguments
    base_args = []
    if args.sessions:
        base_args.extend(["--sessions", args.sessions])
    if args.motion_images:
        base_args.extend(["--motion-images", args.motion_images])
    
    results = {}
    
    # Benchmark 1: Dry run comparison
    print(f"\n📊 BENCHMARK 1: Dry Run Performance")
    
    # Original dry run
    original_dry_cmd = [sys.executable, str(original_script)] + base_args + ["--dry-run"]
    original_dry_time, original_dry_success = run_command_with_timing(
        original_dry_cmd, "Original Script - Dry Run"
    )
    results["original_dry"] = original_dry_time
    
    # Optimized dry run
    optimized_dry_cmd = [sys.executable, str(optimized_script)] + base_args + ["--dry-run"]
    if args.workers:
        optimized_dry_cmd.extend(["--workers", str(args.workers)])
    
    optimized_dry_time, optimized_dry_success = run_command_with_timing(
        optimized_dry_cmd, "Optimized Script - Dry Run"
    )
    results["optimized_dry"] = optimized_dry_time
    
    if not args.dry_run_only:
        # Benchmark 2: Actual file operations
        print(f"\n📊 BENCHMARK 2: Full Processing Performance")
        
        # Original full processing
        original_full_cmd = [sys.executable, str(original_script)] + base_args + ["--output", args.output_original]
        original_full_time, original_full_success = run_command_with_timing(
            original_full_cmd, "Original Script - Full Processing"
        )
        results["original_full"] = original_full_time
        
        # Optimized full processing
        optimized_full_cmd = [sys.executable, str(optimized_script)] + base_args + ["--output", args.output_optimized]
        if args.workers:
            optimized_full_cmd.extend(["--workers", str(args.workers)])
        
        optimized_full_time, optimized_full_success = run_command_with_timing(
            optimized_full_cmd, "Optimized Script - Full Processing"
        )
        results["optimized_full"] = optimized_full_time
        
        # Benchmark 3: Sequential vs Parallel comparison
        print(f"\n📊 BENCHMARK 3: Sequential vs Parallel Comparison")
        
        # Optimized sequential
        optimized_seq_cmd = [sys.executable, str(optimized_script)] + base_args + [
            "--output", f"{args.output_optimized}_sequential", "--sequential"
        ]
        optimized_seq_time, optimized_seq_success = run_command_with_timing(
            optimized_seq_cmd, "Optimized Script - Sequential Mode"
        )
        results["optimized_sequential"] = optimized_seq_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("📈 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    if original_dry_success and optimized_dry_success:
        speedup_dry = original_dry_time / optimized_dry_time
        print(f"Dry Run Performance:")
        print(f"  Original:  {original_dry_time:.2f}s")
        print(f"  Optimized: {optimized_dry_time:.2f}s")
        print(f"  Speedup:   {speedup_dry:.2f}x {'🚀' if speedup_dry > 1 else '🐌'}")
    
    if not args.dry_run_only and "original_full" in results and "optimized_full" in results:
        if original_full_success and optimized_full_success:
            speedup_full = results["original_full"] / results["optimized_full"]
            print(f"\nFull Processing Performance:")
            print(f"  Original:  {results['original_full']:.2f}s")
            print(f"  Optimized: {results['optimized_full']:.2f}s")
            print(f"  Speedup:   {speedup_full:.2f}x {'🚀' if speedup_full > 1 else '🐌'}")
        
        if "optimized_sequential" in results and optimized_seq_success and optimized_full_success:
            speedup_parallel = results["optimized_sequential"] / results["optimized_full"]
            print(f"\nParallel vs Sequential:")
            print(f"  Sequential: {results['optimized_sequential']:.2f}s")
            print(f"  Parallel:   {results['optimized_full']:.2f}s")
            print(f"  Speedup:    {speedup_parallel:.2f}x {'🚀' if speedup_parallel > 1 else '🐌'}")
    
    print(f"\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()