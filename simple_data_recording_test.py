#!/usr/bin/env python3
"""
Simple Data Recording Test Script

This script demonstrates basic data recording functionality without complex task execution.
It will record:
- Joint states (if xArm robot is connected)
- RGBD images from RealSense camera  
- Natural language commands
- OpenVLA demonstration data
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
    
    print("=== Simple Data Recording Test ===")
    print("This test will show data recording functionality")
    
    try:
        # Initialize task planner with data recording enabled
        print("\n1. Initializing TaskPlanner with data recording...")
        planner = TaskPlanner(
            robot_ip="192.168.1.224",      # xArm robot IP
            use_llm=True,                  # Enable LLM for task decomposition
            enable_data_recording=True,    # Enable data recording
            recording_timestep=0.2,        # Record every 0.2 seconds (5Hz)
            collect_openvla_data=True      # Enable OpenVLA format
        )
        
        # Show current configuration
        print(f"\n2. Configuration:")
        print(f"   Data Recording: {planner.is_data_recording_enabled()}")
        print(f"   OpenVLA Collection: {planner.collect_openvla_data}")
        print(f"   Recording Timestep: {planner.get_recording_timestep()}s")
        print(f"   Robot IP: {planner.robot_ip}")
        
        # Simple test task
        task = "open the bottle"
        print(f"\n3. Testing with task: '{task}'")
        
        # Execute task with data recording
        print("\n4. Executing task...")
        results = planner.execute_task(
            task_description=task,
            execute_on_robot=True,        # Set to False for simulation mode
            record_data=True              # Force recording for this task
        )
        
        # Show results
        print(f"\n5. Execution Results:")
        print(f"   Task Success: {results['success']}")
        print(f"   Commands Generated: {len(results.get('skill_commands', []))}")
        
        # Show data recording results
        print(f"\n6. Data Recording Results:")
        if 'session_data_path' in results:
            print(f"   ✅ Session data saved: {results['session_data_path']}")
        else:
            print(f"   ❌ No session data recorded")
            
        if 'openvla_dataset_path' in results:
            print(f"   ✅ OpenVLA dataset saved: {results['openvla_dataset_path']}")
        else:
            print(f"   ❌ No OpenVLA data recorded")
        
        # Show what was recorded
        if planner.data_recorder and planner.data_recorder.current_record:
            record = planner.data_recorder.current_record
            print(f"\n7. Data Summary:")
            print(f"   Motion samples: {len(record.motion_data)}")
            print(f"   OpenVLA timesteps: {len(record.demonstration_data)}")
            print(f"   Task name: {record.task_name}")
            print(f"   Recording timestep: {record.recording_timestep}s")
        
        print(f"\n8. Check the 'task_execution_data' directory for saved files")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nTroubleshooting:")
        print(f"- Ensure xArm robot is connected at IP 192.168.1.224")
        print(f"- Ensure RealSense camera is connected")
        print(f"- Check that LLM API keys are configured")
        
    finally:
        # Clean shutdown
        if 'planner' in locals():
            planner.shutdown()
            print(f"\nTaskPlanner shutdown complete")

def test_recording_only():
    """Test just the data recording components without full task execution"""
    
    print("\n" + "="*50)
    print("RECORDING-ONLY TEST")
    print("="*50)
    
    try:
        from data_recorder import DataRecorder
        
        # Test data recorder directly
        recorder = DataRecorder(recording_timestep=0.1)
        
        print("1. Testing DataRecorder initialization...")
        print(f"   Recording timestep: {recorder.recording_timestep}s")
        print(f"   Data directory: {recorder.data_dir}")
        
        # Start a test session
        print("\n2. Starting test recording session...")
        session_id = recorder.start_recording_session(
            natural_language_task="test recording task",
            task_name="test_task"
        )
        print(f"   Session ID: {session_id}")
        
        # Test basic recording functions
        print("\n3. Testing recording functions...")
        recorder.record_planning_complete(["test_command"])
        recorder.record_inference_start()
        recorder.record_inference_complete([], {})
        recorder.record_execution_start()
        recorder.record_execution_complete(success=True)
        
        print("   Basic recording functions: ✅")
        
        # Finalize without user prompts
        print("\n4. Finalizing session...")
        session_path = recorder.finalize_session(prompt_for_save=False)
        print(f"   Session saved to: {session_path}")
        
        print("\n✅ Recording-only test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Recording-only test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    # Run recording-only test as backup
    test_recording_only()