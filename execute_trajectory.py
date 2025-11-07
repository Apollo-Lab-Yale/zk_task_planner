#!/usr/bin/env python3
"""
Execute Trajectory Script

This script reads motion data from a successful session file, moves to the same initial 
position, and then directly executes the recorded trajectory by sending joint commands 
and gripper commands to the robot.
"""

import json
import os
import sys
import logging
import argparse
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner


class TrajectoryExecutor:
    """System for executing trajectories directly from recorded motion data"""
    
    def __init__(self, robot_ip: str = "192.168.1.224", execution_speed: float = 1.0):
        """
        Initialize the trajectory execution system
        
        Args:
            robot_ip: IP address of the robot
            execution_speed: Speed multiplier for trajectory execution (0.1-2.0)
        """
        self.logger = logging.getLogger(__name__)
        self.robot_ip = robot_ip
        self.execution_speed = max(0.1, min(2.0, execution_speed))
        
        # Initialize robot motion planner
        self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        
        # Initialize xArm connection
        if not self.motion_planner.connect():
            raise RuntimeError(f"Failed to connect to robot at {robot_ip}")
            
        self.logger.info(f"Connected to robot at {robot_ip}")
        self.logger.info(f"Execution speed set to {self.execution_speed}x")

    def load_session_data(self, session_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from a JSON file
        
        Args:
            session_file_path: Path to the session JSON file
            
        Returns:
            Loaded session data or None if failed
        """
        try:
            with open(session_file_path, 'r') as f:
                session_data = json.load(f)
            
            self.logger.info(f"Loaded session data: {session_data.get('task_name', 'Unknown')}")
            motion_data_count = len(session_data.get('motion_data', []))
            self.logger.info(f"Found {motion_data_count} motion data points")
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Failed to load session data from {session_file_path}: {e}")
            return None

    def validate_motion_data(self, motion_data: List[Dict[str, Any]]) -> bool:
        """
        Validate that motion data has the required structure
        
        Args:
            motion_data: List of motion data points
            
        Returns:
            True if valid, False otherwise
        """
        if not motion_data:
            self.logger.error("No motion data found")
            return False
            
        # Check first data point for required fields
        first_point = motion_data[0]
        required_fields = ['joint_positions', 'gripper_state', 'timestamp']
        
        for field in required_fields:
            if field not in first_point:
                self.logger.error(f"Missing required field '{field}' in motion data")
                return False
                
        # Check joint positions format
        joint_positions = first_point['joint_positions']
        if not isinstance(joint_positions, list) or len(joint_positions) != 7:
            self.logger.error(f"Invalid joint positions format: expected 7 values, got {len(joint_positions) if isinstance(joint_positions, list) else 'non-list'}")
            return False
            
        self.logger.info(f"Motion data validation passed: {len(motion_data)} points")
        return True

    def move_to_initial_position(self, initial_joint_positions: List[float], initial_gripper_state: int) -> bool:
        """
        Move robot to the initial position from the recorded trajectory
        
        Args:
            initial_joint_positions: Initial joint positions (7 DOF)
            initial_gripper_state: Initial gripper position
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Moving to initial position: {initial_joint_positions}")
            
            # Move to initial joint configuration
            success = self.motion_planner.move_to_joint_positions(
                joint_positions=initial_joint_positions,
                speed=50,  # Slower speed for initial positioning
                wait=True
            )
            
            if not success:
                self.logger.error("Failed to move to initial joint positions")
                return False
                
            # Set initial gripper position
            self.logger.info(f"Setting initial gripper position: {initial_gripper_state}")
            gripper_success = self.motion_planner.set_gripper_position(
                position=initial_gripper_state,
                speed=100,
                wait=True
            )
            
            if not gripper_success:
                self.logger.error("Failed to set initial gripper position")
                return False
                
            self.logger.info("Successfully moved to initial position")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving to initial position: {e}")
            return False

    def calculate_trajectory_timing(self, motion_data: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate timing intervals between trajectory points
        
        Args:
            motion_data: List of motion data points with timestamps
            
        Returns:
            List of time intervals between points
        """
        if len(motion_data) < 2:
            return [0.1]  # Default interval for single point
            
        intervals = []
        for i in range(1, len(motion_data)):
            dt = motion_data[i]['timestamp'] - motion_data[i-1]['timestamp']
            # Apply speed multiplier and ensure reasonable bounds
            dt = dt / self.execution_speed
            dt = max(0.01, min(1.0, dt))  # Clamp between 10ms and 1s
            intervals.append(dt)
            
        return intervals

    def execute_trajectory(self, motion_data: List[Dict[str, Any]]) -> bool:
        """
        Execute the recorded trajectory by sending joint and gripper commands
        
        Args:
            motion_data: List of motion data points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.validate_motion_data(motion_data):
                return False
                
            # Calculate timing between points
            intervals = self.calculate_trajectory_timing(motion_data)
            
            self.logger.info(f"Starting trajectory execution with {len(motion_data)} points")
            
            # Execute each point in the trajectory
            for i, data_point in enumerate(motion_data):
                start_time = time.time()
                
                joint_positions = data_point['joint_positions']
                gripper_state = data_point['gripper_state']
                
                # Send joint command (non-blocking for smooth motion)
                joint_success = self.motion_planner.move_to_joint_positions(
                    joint_positions=joint_positions,
                    speed=100,  # Higher speed for trajectory execution
                    wait=False  # Don't wait to allow smooth continuous motion
                )
                
                if not joint_success:
                    self.logger.warning(f"Joint command failed at point {i+1}/{len(motion_data)}")
                
                # Send gripper command if it changed significantly
                if i == 0 or abs(gripper_state - motion_data[i-1]['gripper_state']) > 10:
                    gripper_success = self.motion_planner.set_gripper_position(
                        position=gripper_state,
                        speed=200,  # Fast gripper motion
                        wait=False  # Don't wait to allow smooth motion
                    )
                    
                    if not gripper_success:
                        self.logger.warning(f"Gripper command failed at point {i+1}/{len(motion_data)}")
                
                # Log progress periodically
                if i % 10 == 0 or i == len(motion_data) - 1:
                    self.logger.info(f"Trajectory progress: {i+1}/{len(motion_data)} points")
                
                # Wait for the next point timing (if not the last point)
                if i < len(intervals):
                    elapsed = time.time() - start_time
                    sleep_time = intervals[i] - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            # Wait for final position to be reached
            self.logger.info("Waiting for final position...")
            time.sleep(2.0)
            
            self.logger.info("Trajectory execution completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trajectory: {e}")
            return False

    def execute_session_trajectory(self, session_file_path: str) -> Dict[str, Any]:
        """
        Execute trajectory from a session file
        
        Args:
            session_file_path: Path to the session JSON file
            
        Returns:
            Execution results dictionary
        """
        results = {
            "session_file": session_file_path,
            "start_time": time.time(),
            "task_description": None,
            "motion_points": 0,
            "success": False,
            "error_message": None
        }
        
        try:
            # Load session data
            session_data = self.load_session_data(session_file_path)
            if not session_data:
                results["error_message"] = "Failed to load session data"
                return results
                
            results["task_description"] = session_data.get("natural_language_task", "Unknown task")
            
            # Extract motion data
            motion_data = session_data.get('motion_data', [])
            if not motion_data:
                results["error_message"] = "No motion data found in session"
                return results
                
            results["motion_points"] = len(motion_data)
            
            # Check if session was successful
            user_success = session_data.get('user_success_rating', False)
            execution_success = session_data.get('execution_success', False)
            
            if not (user_success and execution_success):
                self.logger.warning(f"Session was not marked as successful (user: {user_success}, execution: {execution_success})")
                user_input = input("Continue with trajectory execution anyway? (y/n): ").strip().lower()
                if user_input not in ['y', 'yes', '1', 'true']:
                    results["error_message"] = "User chose not to execute unsuccessful session"
                    return results
            
            # Extract initial position
            initial_joint_positions = motion_data[0]['joint_positions']
            initial_gripper_state = motion_data[0]['gripper_state']
            
            # Move to initial position
            self.logger.info("Moving to initial position...")
            if not self.move_to_initial_position(initial_joint_positions, initial_gripper_state):
                results["error_message"] = "Failed to move to initial position"
                return results
            
            # Ask user for confirmation
            print(f"\nReady to execute trajectory for: {results['task_description']}")
            print(f"Motion points: {results['motion_points']}")
            print(f"Execution speed: {self.execution_speed}x")
            user_input = input("Execute trajectory? (y/n): ").strip().lower()
            
            if user_input not in ['y', 'yes', '1', 'true']:
                results["error_message"] = "User cancelled trajectory execution"
                return results
            
            # Execute the trajectory
            self.logger.info("Starting trajectory execution...")
            success = self.execute_trajectory(motion_data)
            
            results["success"] = success
            if not success:
                results["error_message"] = "Trajectory execution failed"
            
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]
            
        except Exception as e:
            error_message = f"Trajectory execution failed: {str(e)}"
            results["error_message"] = error_message
            results["end_time"] = time.time()
            self.logger.error(error_message)
        
        return results

    def shutdown(self):
        """Clean shutdown of the trajectory execution system"""
        try:
            if hasattr(self, 'motion_planner') and self.motion_planner:
                self.motion_planner.disconnect()
                self.logger.info("Robot disconnected")
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")


def find_successful_sessions(data_directory: str, task_filter: Optional[str] = None, exact_match: bool = False) -> List[str]:
    """
    Find successful session files in the data directory
    
    Args:
        data_directory: Directory containing session files
        task_filter: Optional task name filter (e.g., "open_the_bottle")
        exact_match: If True, require exact task name match; if False, match as substring
        
    Returns:
        List of paths to successful session files
    """
    session_files = []
    sessions_dir = os.path.join(data_directory, "sessions")
    
    if not os.path.exists(sessions_dir):
        print(f"Sessions directory not found: {sessions_dir}")
        return session_files
    
    for filename in os.listdir(sessions_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(sessions_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                # Check if this is a successful session
                user_success_rating = session_data.get('user_success_rating', False)
                execution_success = session_data.get('execution_success', False)
                task_name = session_data.get('task_name', '')
                motion_data = session_data.get('motion_data', [])
                
                # Apply task filter if specified
                if task_filter:
                    if exact_match:
                        if task_name.lower() != task_filter.lower():
                            continue
                    else:
                        import re
                        pattern = r'\b' + re.escape(task_filter) + r'\b'
                        if not re.search(pattern, task_name, re.IGNORECASE):
                            continue
                
                # Only include successful sessions with motion data
                if user_success_rating and execution_success and len(motion_data) > 0:
                    session_files.append(filepath)
                    print(f"Found successful session: {task_name} - {filename} ({len(motion_data)} motion points)")
                
            except Exception as e:
                print(f"Error reading session file {filename}: {e}")
                continue
    
    return sorted(session_files)


def main():
    """Main function for the trajectory execution script"""
    parser = argparse.ArgumentParser(description="Execute robot trajectories from recorded session data")
    parser.add_argument("--data-dir", 
                       default="/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/task_execution_data",
                       help="Directory containing session data")
    parser.add_argument("--session-file", 
                       help="Specific session file to execute trajectory from")
    parser.add_argument("--task-filter",
                       help="Filter sessions by task name (e.g., 'open_the_bottle')")
    parser.add_argument("--robot-ip", 
                       default="192.168.1.224",
                       help="Robot IP address")
    parser.add_argument("--speed", 
                       type=float, 
                       default=1.0,
                       help="Execution speed multiplier (0.1-2.0, default: 1.0)")
    parser.add_argument("--max-sessions", 
                       type=int, 
                       help="Maximum number of sessions to execute")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--exact-match", 
                       action="store_true",
                       help="Require exact task name match for filtering")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Initialize trajectory executor
    executor = None
    try:
        executor = TrajectoryExecutor(
            robot_ip=args.robot_ip,
            execution_speed=args.speed
        )
        
        if args.session_file:
            # Single session execution
            logger.info(f"Executing trajectory from session: {args.session_file}")
            
            # Handle relative paths
            session_path = args.session_file
            if not os.path.isabs(session_path):
                possible_paths = [
                    session_path,
                    os.path.join(args.data_dir, "sessions", session_path),
                    os.path.join("cognitive_bt_framework/src/task_execution_data/sessions", session_path)
                ]
                
                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if not found_path:
                    print(f"Session file not found: {session_path}")
                    print(f"Checked paths: {possible_paths}")
                    return
                
                session_path = found_path
            
            result = executor.execute_session_trajectory(session_path)
            
            # Print results
            print(f"\nTrajectory Execution Results:")
            print(f"Task: {result['task_description']}")
            print(f"Motion points: {result['motion_points']}")
            print(f"Success: {result['success']}")
            print(f"Duration: {result.get('duration', 0):.2f}s")
            if not result['success']:
                print(f"Error: {result['error_message']}")
                
        else:
            # Batch execution
            logger.info(f"Finding successful sessions in: {args.data_dir}")
            session_files = find_successful_sessions(args.data_dir, args.task_filter, args.exact_match)
            
            if not session_files:
                print("No successful sessions found!")
                return
            
            if args.max_sessions:
                session_files = session_files[:args.max_sessions]
            
            print(f"\nFound {len(session_files)} successful sessions to execute")
            
            results = []
            for i, session_file in enumerate(session_files):
                logger.info(f"Executing trajectory {i+1}/{len(session_files)}: {session_file}")
                
                try:
                    result = executor.execute_session_trajectory(session_file)
                    results.append(result)
                    
                    # Log summary
                    if result["success"]:
                        logger.info(f"✓ Trajectory {i+1} executed successfully")
                    else:
                        logger.warning(f"✗ Trajectory {i+1} failed: {result.get('error_message', 'Unknown error')}")
                    
                    # Small delay between executions
                    if i < len(session_files) - 1:
                        print("\nWaiting before next trajectory...")
                        time.sleep(3.0)
                    
                except Exception as e:
                    error_result = {
                        "session_file": session_file,
                        "success": False,
                        "error_message": f"Exception during execution: {str(e)}"
                    }
                    results.append(error_result)
                    logger.error(f"Exception executing trajectory {i+1}: {e}")
            
            # Print batch summary
            successful = sum(1 for r in results if r.get("success", False))
            print(f"\nBatch Execution Summary:")
            print(f"Total trajectories: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            
            # Print individual results
            for i, result in enumerate(results):
                status = "✓" if result.get("success", False) else "✗"
                task = result.get("task_description", "Unknown")
                print(f"{status} Trajectory {i+1}: {task}")
                if not result.get("success", False):
                    print(f"  Error: {result.get('error_message', 'Unknown error')}")
        
    except KeyboardInterrupt:
        logger.info("Trajectory execution interrupted by user")
    except Exception as e:
        logger.error(f"Trajectory execution system error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        if executor:
            executor.shutdown()


if __name__ == "__main__":
    main()