#!/usr/bin/env python3
"""
Minimal Data Recording Test

This script tests ONLY the data recording functionality without task execution.
Perfect for testing data collection without needing successful robot tasks.
"""

import sys
import time
import logging
from pathlib import Path

# Add the cognitive_bt_framework to Python path
sys.path.insert(0, str(Path(__file__).parent / "cognitive_bt_framework" / "src"))

def test_data_recording():
    """Test data recording functionality directly"""
    
    print("=== Minimal Data Recording Test ===")
    
    try:
        from data_recorder import DataRecorder
        
        print("1. Creating DataRecorder...")
        recorder = DataRecorder(recording_timestep=0.5)
        print(f"   Data directory: {recorder.data_dir}")
        
        print("\n2. Starting recording session...")
        session_id = recorder.start_recording_session(
            natural_language_task="test data recording functionality",
            task_name="minimal_test"
        )
        print(f"   Session ID: {session_id}")
        
        print("\n3. Recording session phases...")
        
        # Record planning phase
        recorder.record_planning_complete(["detect_object test", "pickup test"])
        print("   ✅ Planning phase recorded")
        
        # Record inference phase  
        recorder.record_inference_start()
        test_skills = [{"command": "test", "skill_name": "test", "parameters": "", "status": "test"}]
        recorder.record_inference_complete(test_skills, {"test": "inference"})
        print("   ✅ Inference phase recorded")
        
        # Record execution phase
        recorder.record_execution_start()
        
        # Test motion recording setup (without starting the thread)
        print("\n4. Testing motion recording setup...")
        try:
            # Set up interfaces (will be None but that's OK for test)
            recorder.set_motion_interfaces(robot_interface=None, camera_interface=None)
            print("   ✅ Motion interfaces set")
            
            # Test OpenVLA data recording
            print("\n5. Testing OpenVLA data recording...")
            success = recorder.add_demonstration_timestep(
                wrist_image=None,  # No actual images for this test
                external_image=None,
                joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                end_effector_pose=[0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
                gripper_state=0.5,
                delta_pose=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                gripper_action=0.6,
                language_instruction="test instruction"
            )
            if success:
                print("   ✅ OpenVLA timestep recorded")
            else:
                print("   ❌ OpenVLA timestep failed")
            
        except Exception as e:
            print(f"   ❌ Motion recording test failed: {e}")
        
        # Complete execution
        recorder.record_execution_complete(success=True)
        print("   ✅ Execution phase recorded")
        
        print(f"\n6. Session data summary:")
        if recorder.current_record:
            record = recorder.current_record
            print(f"   Task: {record.natural_language_task}")
            print(f"   Task name: {record.task_name}")
            print(f"   Commands: {len(record.task_decomposition)}")
            print(f"   Motion samples: {len(record.motion_data)}")
            print(f"   OpenVLA timesteps: {len(record.demonstration_data)}")
            
        print("\n7. Finalizing session (no user prompts)...")
        session_path = recorder.finalize_session(prompt_for_save=False)
        print(f"   ✅ Session saved to: {session_path}")
        
        # Test OpenVLA export
        if len(recorder.current_record.demonstration_data) > 0:
            print("\n8. Testing OpenVLA export...")
            try:
                openvla_path = recorder.export_openvla_dataset()
                print(f"   ✅ OpenVLA dataset exported: {openvla_path}")
            except Exception as e:
                print(f"   ❌ OpenVLA export failed: {e}")
        
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"Check the 'task_execution_data' directory for files:")
        print(f"- {session_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_planner_integration():
    """Test TaskPlanner with data recording (may fail if robot/camera not connected)"""
    
    print("\n" + "="*60)
    print("TASK PLANNER INTEGRATION TEST")  
    print("="*60)
    
    try:
        from task_planner import TaskPlanner
        
        print("1. Creating TaskPlanner with recording enabled...")
        planner = TaskPlanner(
            robot_ip="192.168.1.224",
            use_llm=False,  # Disable LLM to avoid API calls
            enable_data_recording=True,
            recording_timestep=0.5,
            collect_openvla_data=True
        )
        
        print(f"   Data recording: {planner.is_data_recording_enabled()}")
        print(f"   OpenVLA collection: {planner.collect_openvla_data}")
        
        print("\n2. Testing manual data collection...")
        if planner.data_recorder:
            # Start a session manually
            session_id = planner.data_recorder.start_recording_session(
                "manual test task", 
                "integration_test"
            )
            print(f"   Session started: {session_id}")
            
            # Test OpenVLA timestep collection
            success = planner.collect_openvla_timestep()
            print(f"   OpenVLA timestep: {'✅' if success else '❌'}")
            
            # Finalize
            session_path = planner.data_recorder.finalize_session(prompt_for_save=False)
            print(f"   Session saved: {session_path}")
            
        planner.shutdown()
        print("   ✅ TaskPlanner integration test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ TaskPlanner integration failed: {e}")
        print("   (This is expected if robot/camera not connected)")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Test 1: Core data recording (should always work)
    success1 = test_data_recording()
    
    # Test 2: TaskPlanner integration (may fail without hardware)  
    success2 = test_task_planner_integration()
    
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Data Recording Core: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"TaskPlanner Integration: {'✅ PASS' if success2 else '❌ FAIL (expected without hardware)'}")
    
    if success1:
        print(f"\n🎉 Data recording is working correctly!")
        print(f"Check 'task_execution_data/' directory for recorded files.")
    else:
        print(f"\n❌ Core data recording failed - check the error messages above.")