#!/usr/bin/env python3
"""
Example script demonstrating how to use the data recording toggle
"""

import sys
import os

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.task_planner import TaskPlanner

def demo_data_recording_toggle():
    """Demonstrate different ways to control data recording"""
    
    print("="*60)
    print("DATA RECORDING TOGGLE DEMONSTRATION")
    print("="*60)
    
    # Method 1: Initialize with recording disabled (default for development)
    print("\n1. Initialize with recording DISABLED (development mode):")
    planner = TaskPlanner(
        robot_ip="192.168.1.224",
        use_llm=False,  # Disable for demo to avoid API calls
        enable_data_recording=False  # Explicitly disabled
    )
    print(f"   Recording enabled: {planner.is_data_recording_enabled()}")
    
    # Method 2: Initialize with recording enabled 
    print("\n2. Initialize with recording ENABLED:")
    planner_with_recording = TaskPlanner(
        robot_ip="192.168.1.224", 
        use_llm=False,
        enable_data_recording=True
    )
    print(f"   Recording enabled: {planner_with_recording.is_data_recording_enabled()}")
    
    # Method 3: Toggle recording at runtime
    print("\n3. Toggle recording at runtime:")
    print(f"   Before toggle: {planner.is_data_recording_enabled()}")
    planner.set_data_recording(True)
    print(f"   After enabling: {planner.is_data_recording_enabled()}")
    planner.set_data_recording(False)
    print(f"   After disabling: {planner.is_data_recording_enabled()}")
    
    # Method 4: Override recording for specific tasks
    print("\n4. Override recording for specific tasks:")
    print("   Task planner has recording disabled by default...")
    print(f"   Default recording state: {planner.is_data_recording_enabled()}")
    
    # Even with recording disabled globally, you can enable it for specific tasks
    print("   Running task with recording explicitly ENABLED:")
    try:
        # This would try to create a data recorder since record_data=True
        # but will fail due to no robot connection - that's expected for demo
        results = planner.execute_task(
            task_description="demo task with recording", 
            execute_on_robot=False,  # Simulation mode
            record_data=True  # Override to enable recording for this task
        )
        print(f"   Task completed. Recording would have been active.")
    except Exception as e:
        print(f"   Expected error (no robot): {type(e).__name__}")
    
    print("   Running task with recording explicitly DISABLED:")
    try:
        results = planner_with_recording.execute_task(
            task_description="demo task without recording",
            execute_on_robot=False,
            record_data=False  # Override to disable recording for this task  
        )
        print(f"   Task completed. Recording was disabled for this task only.")
    except Exception as e:
        print(f"   Expected error (no robot): {type(e).__name__}")
    
    print("\n" + "="*60)
    print("USAGE RECOMMENDATIONS")
    print("="*60)
    print("""
DEVELOPMENT MODE (recommended default):
    planner = TaskPlanner(enable_data_recording=False)
    
PRODUCTION/DATA COLLECTION MODE:
    planner = TaskPlanner(enable_data_recording=True)
    
SELECTIVE RECORDING:
    # Default off, but enable for specific important tasks
    planner = TaskPlanner(enable_data_recording=False)
    results = planner.execute_task("critical task", record_data=True)
    
RUNTIME TOGGLE:
    planner.set_data_recording(True)   # Enable
    planner.set_data_recording(False)  # Disable
    
CHECK STATUS:
    if planner.is_data_recording_enabled():
        print("Recording is active")
    """)

if __name__ == "__main__":
    demo_data_recording_toggle()