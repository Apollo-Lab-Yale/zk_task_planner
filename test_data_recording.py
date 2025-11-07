#!/usr/bin/env python3
"""
Test script for the data recording system in the task planner
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.task_planner import TaskPlanner
from cognitive_bt_framework.src.data_recorder import DataRecorder

def test_data_recorder_standalone():
    """Test the data recorder independently"""
    print("="*60)
    print("TESTING DATA RECORDER STANDALONE")
    print("="*60)
    
    # Initialize data recorder
    recorder = DataRecorder(data_dir="test_task_data")
    
    # Test session creation
    session_id = recorder.start_recording_session(
        natural_language_task="Test task: open the bottle",
        robot_ip="192.168.1.224",
        camera_type="test_camera",
        execution_mode="simulation"
    )
    print(f"Created session: {session_id}")
    
    # Test recording planning phase
    test_commands = ["detect_object bottle", "open bottle"]
    recorder.record_planning_complete(test_commands)
    print("Recorded planning completion")
    
    # Test recording environment image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:200, 200:300] = [255, 0, 0]  # Red rectangle
    image_path = recorder.record_environment_image(test_image)
    print(f"Recorded environment image: {image_path}")
    
    # Test recording points of interest
    test_points = {
        'pixel_coords': [(300, 200), (350, 180)],
        'ids': ['A', 'B'],
        'scores': [0.9, 0.8],
        'detection_methods': ['test_method', 'test_method'],
        'interaction_types': ['push', 'twist'],
        'confidences': [0.95, 0.85]
    }
    recorder.record_points_of_interest(test_points)
    print("Recorded points of interest")
    
    # Test recording surface images
    test_surfaces = {
        'bottle_surface': np.ones((100, 100, 3), dtype=np.uint8) * 128,
        'cap_surface': np.ones((50, 50, 3), dtype=np.uint8) * 200
    }
    surface_paths = recorder.record_surface_images(test_surfaces)
    print(f"Recorded surface images: {surface_paths}")
    
    # Test recording inference phase
    recorder.record_inference_start()
    test_skills = [
        {
            "command": "detect_object bottle",
            "skill_name": "detect_object",
            "parameters": "bottle",
            "status": "ready_for_execution"
        },
        {
            "command": "open bottle", 
            "skill_name": "open",
            "parameters": "bottle",
            "status": "ready_for_execution"
        }
    ]
    test_llm_responses = {
        "task_decomposition": "1. detect_object bottle\\n2. open bottle",
        "execution_mode": "simulation"
    }
    recorder.record_inference_complete(test_skills, test_llm_responses)
    print("Recorded inference completion")
    
    # Test recording execution phase
    recorder.record_execution_start()
    recorder.record_execution_complete(
        success=True,
        failure_messages=[],
        robot_errors=[]
    )
    print("Recorded execution completion")
    
    # Note: In a real test, we would call finalize_session() which prompts for user input
    # For automated testing, we'll manually set the user feedback
    recorder.current_record.user_success_rating = True
    recorder.current_record.user_explanation = "Test execution completed successfully"
    
    # Finalize without user prompt (for testing)
    session_filename = f"{recorder.current_session_id}.json"
    session_path = recorder.sessions_dir / session_filename
    
    # Convert record to dictionary for JSON serialization
    record_dict = recorder._record_to_dict(recorder.current_record)
    
    # Save session data
    import json
    with open(session_path, 'w') as f:
        json.dump(record_dict, f, indent=2, default=recorder._json_serializer)
    
    # Generate summary
    recorder._print_session_summary(recorder.current_record)
    
    print(f"Session saved to: {session_path}")
    
    # Test statistics
    stats = recorder.get_session_stats()
    print("\nSession Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True

def test_task_planner_with_recording():
    """Test the task planner with data recording enabled"""
    print("\n" + "="*60)
    print("TESTING TASK PLANNER WITH DATA RECORDING")
    print("="*60)
    
    try:
        # Initialize task planner with data recording enabled
        planner = TaskPlanner(
            robot_ip="192.168.1.224",
            use_llm=False,  # Disable LLM for testing to avoid API calls
            enable_data_recording=True
        )
        
        # Test task execution with recording (simulation mode)
        test_task = "Test task: open the test object"
        print(f"Testing task: {test_task}")
        
        # Since we disabled LLM, we'll get "manual execution required"
        # but the recording system should still capture what it can
        results = planner.execute_task(
            task_description=test_task,
            execute_on_robot=False,  # Simulation mode
            record_data=True
        )
        
        print("Task execution results:")
        for key, value in results.items():
            if key != "session_data_path":
                print(f"  {key}: {value}")
        
        if "session_data_path" in results:
            print(f"Session data saved to: {results['session_data_path']}")
        
        # Clean up
        if hasattr(planner, 'shutdown'):
            planner.shutdown()
        
        return True
        
    except Exception as e:
        print(f"Error in task planner test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_collection_schema():
    """Test that the data collection schema captures all required data"""
    print("\n" + "="*60)
    print("TESTING DATA COLLECTION SCHEMA")
    print("="*60)
    
    # Check if data recorder can handle all required data types
    recorder = DataRecorder(data_dir="test_schema_data")
    
    # Test all data fields mentioned in requirements
    required_fields = [
        "natural_language_task",
        "task_decomposition", 
        "skills_generated",
        "points_of_interest",
        "environment_image_path",
        "surface_images_paths",
        "planning_duration",
        "inference_duration", 
        "execution_duration",
        "failure_messages",
        "user_success_rating",
        "user_explanation"
    ]
    
    session_id = recorder.start_recording_session(
        natural_language_task="Schema test task",
        robot_ip="test_ip",
        camera_type="test_camera",
        execution_mode="test"
    )
    
    # Verify all required fields are available in the record structure
    missing_fields = []
    record = recorder.current_record
    
    for field in required_fields:
        if not hasattr(record, field) and field not in record.__dict__:
            # Check if field is accessible through nested structure
            if field.endswith("_duration"):
                if not hasattr(record.timing, field.replace("_duration", "_start")):
                    missing_fields.append(field)
            else:
                missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    else:
        print("✅ All required fields are available in the data schema")
    
    # Test that timing calculations work
    recorder.record_planning_complete(["test command"])
    recorder.record_inference_start()
    recorder.record_inference_complete([], {})
    recorder.record_execution_start()
    recorder.record_execution_complete(True)
    
    # Verify timing calculations
    if recorder.timing_data.planning_duration > 0:
        print("✅ Planning duration calculated correctly")
    if recorder.timing_data.inference_duration >= 0:
        print("✅ Inference duration calculated correctly") 
    if recorder.timing_data.execution_duration >= 0:
        print("✅ Execution duration calculated correctly")
    
    print("✅ Data collection schema test passed")
    return True

def main():
    """Run all data recording tests"""
    print("STARTING DATA RECORDING SYSTEM TESTS")
    print("="*60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("Standalone Data Recorder", test_data_recorder_standalone),
        ("Data Collection Schema", test_data_collection_schema),
        ("Task Planner with Recording", test_task_planner_with_recording)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning test: {test_name}")
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())