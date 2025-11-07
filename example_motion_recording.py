#!/usr/bin/env python3
"""
Example demonstrating motion data recording during task execution.

This example shows how to:
1. Enable data recording with configurable timestep
2. Record joint states, gripper states, and RGBD images during execution  
3. Prompt user to save motion data at the end
"""

import sys
import logging
from pathlib import Path

# Add the cognitive_bt_framework to Python path
sys.path.append(str(Path(__file__).parent / "cognitive_bt_framework" / "src"))

from task_planner import TaskPlanner

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Motion Data Recording Example ===")
    print("This example demonstrates recording robot motion data during task execution")
    
    try:
        # Initialize task planner with motion recording enabled
        planner = TaskPlanner(
            robot_ip="192.168.1.224",
            use_llm=True,
            enable_data_recording=True,    # Enable data recording
            recording_timestep=0.1,        # Record data every 0.1 seconds  
            collect_openvla_data=True      # Enable OpenVLA demonstration data collection
        )
        
        print(f"Data recording enabled: {planner.is_data_recording_enabled()}")
        print(f"Recording timestep: {planner.get_recording_timestep()}s")
        
        # Example task
        task_description = "open the bottle and put the cap in the paper bag"
        
        print(f"\nExecuting task: {task_description}")
        print("During execution, the system will record:")
        print("- Joint positions and commands at 0.1s intervals")
        print("- Gripper state and commands")
        print("- RGB and depth images from RealSense camera")
        print("- Natural language command and task name")
        print("- OpenVLA-compatible demonstration data (224x224 images, delta actions)")
        
        # Execute task with motion recording
        results = planner.execute_task(
            task_description=task_description,
            execute_on_robot=True,  # Set to False for simulation
            record_data=True        # Override to enable recording for this task
        )
        
        print(f"\nTask execution completed!")
        print(f"Success: {results['success']}")
        
        if 'session_data_path' in results:
            print(f"Session data saved to: {results['session_data_path']}")
        
        if 'openvla_dataset_path' in results:
            print(f"OpenVLA dataset saved to: {results['openvla_dataset_path']}")
        
        # Demonstrate changing recording timestep
        print(f"\nChanging recording timestep to 0.05s for higher frequency recording:")
        planner.set_recording_timestep(0.05)
        print(f"New recording timestep: {planner.get_recording_timestep()}s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure proper cleanup
        if 'planner' in locals():
            planner.shutdown()

if __name__ == "__main__":
    main()