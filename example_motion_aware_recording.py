#!/usr/bin/env python3
"""
Example demonstrating motion-aware data recording that automatically pauses
during planning phases and resumes during robot execution.

This example shows how the system:
1. Only records data when the robot is actually moving
2. Automatically pauses recording when robot is stationary for planning
3. Provides manual control for skill executors to pause/resume recording
"""

import sys
import time
import logging
from pathlib import Path

# Add the cognitive_bt_framework to Python path
sys.path.append(str(Path(__file__).parent / "cognitive_bt_framework" / "src"))

from task_planner import TaskPlanner

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Motion-Aware Data Recording Example ===")
    print("This example demonstrates intelligent data recording that:")
    print("- Only records when robot is actually moving")
    print("- Automatically pauses during planning/perception phases")
    print("- Resumes when robot starts moving again")
    
    try:
        # Initialize task planner with motion-aware recording
        planner = TaskPlanner(
            robot_ip="192.168.1.224",
            use_llm=True,
            enable_data_recording=True,    # Enable data recording
            recording_timestep=0.1,        # Record data every 0.1 seconds  
            collect_openvla_data=True      # Enable OpenVLA demonstration data collection
        )
        
        print(f"\nData recording enabled: {planner.is_data_recording_enabled()}")
        print(f"Recording timestep: {planner.get_recording_timestep()}s")
        
        # Configure motion detection parameters
        if planner.data_recorder:
            planner.data_recorder.motion_threshold = 0.005  # More sensitive motion detection
            planner.data_recorder.stationary_timeout = 1.5  # Pause after 1.5s of no motion
        
        # Example task that will involve planning pauses
        task_description = "open the cabinet and take out the towel"
        
        print(f"\nExecuting task: {task_description}")
        print("\nThe system will automatically:")
        print("- Start recording when robot begins moving")
        print("- Pause recording when robot stops for planning")
        print("- Resume recording when robot starts moving again")
        print("- Skip recording redundant stationary data")
        
        # Execute task with motion-aware recording
        results = planner.execute_task(
            task_description=task_description,
            execute_on_robot=True,
            record_data=True
        )
        
        print(f"\nTask execution completed!")
        print(f"Success: {results['success']}")
        
        if 'session_data_path' in results:
            print(f"Motion data saved to: {results['session_data_path']}")
        
        if 'openvla_dataset_path' in results:
            print(f"OpenVLA dataset saved to: {results['openvla_dataset_path']}")
        
        # Demonstrate manual recording control (for use by skill executors)
        print(f"\nDemonstrating manual recording control:")
        
        # These methods can be called by skill executors during execution
        print("- Skill executor can call planner.pause_data_recording() before planning")
        print("- Skill executor can call planner.resume_data_recording() before motion")
        
        # Example of how a skill executor might use this:
        print(f"\nExample skill execution flow:")
        print("1. Robot starts moving -> Recording automatically starts")
        print("2. Robot stops for perception -> Recording pauses after timeout")
        print("3. Skill executor calls planner.pause_data_recording() -> Explicit pause")
        print("4. Skill generation/planning happens -> No data recorded")
        print("5. Skill executor calls planner.resume_data_recording() -> Recording ready")
        print("6. Robot starts moving -> Recording resumes automatically")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure proper cleanup
        if 'planner' in locals():
            planner.shutdown()

def simulate_skill_executor_with_recording_control():
    """
    Simulate how a skill executor would control recording during execution
    """
    print(f"\n=== Skill Executor Recording Control Simulation ===")
    
    # This would be called within the skill executor during task execution
    planner = TaskPlanner(enable_data_recording=True, collect_openvla_data=True)
    
    try:
        # Start a recording session
        planner.data_recorder.start_recording_session("example task")
        planner.data_recorder.start_motion_recording()
        
        print("1. Robot starts moving - recording begins automatically")
        time.sleep(1)
        
        print("2. Robot stops for object detection - pause recording")
        planner.pause_data_recording()
        time.sleep(2)  # Simulate perception/planning time
        
        print("3. Robot resumes motion - resume recording")
        planner.resume_data_recording()
        time.sleep(1)
        
        print("4. Robot stops for skill generation - pause recording")
        planner.pause_data_recording()
        time.sleep(1)  # Simulate skill generation time
        
        print("5. Robot executes skill - resume recording")
        planner.resume_data_recording()
        time.sleep(1)
        
        print("6. Task complete - stop recording")
        planner.data_recorder.stop_motion_recording()
        
    finally:
        planner.shutdown()

if __name__ == "__main__":
    main()
    simulate_skill_executor_with_recording_control()