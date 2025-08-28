#!/usr/bin/env python3
"""
Simple test script for the data recording system without robot dependencies
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from cognitive_bt_framework.src.data_recorder import DataRecorder

def test_full_workflow():
    """Test a complete task execution workflow with data recording"""
    print("="*70)
    print("COMPLETE TASK EXECUTION DATA RECORDING WORKFLOW TEST")
    print("="*70)
    
    recorder = DataRecorder(data_dir="complete_workflow_test")
    
    # Simulate a complete task execution
    print("Starting task execution simulation...")
    
    # 1. Start session
    session_id = recorder.start_recording_session(
        natural_language_task="move the bottle to the counter and open it",
        robot_ip="192.168.1.224",
        camera_type="realsense",
        execution_mode="real"
    )
    print(f"✅ Session started: {session_id}")
    
    # 2. Record task decomposition
    task_commands = [
        "detect_object bottle",
        "detect_object counter", 
        "pickup bottle",
        "place bottle,counter",
        "open bottle"
    ]
    recorder.record_planning_complete(task_commands)
    print("✅ Task decomposition recorded")
    
    # 3. Record environment image
    env_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    env_image[100:200, 200:400] = [100, 150, 200]  # Simulate bottle area
    env_image[300:400, 500:600] = [150, 100, 50]   # Simulate counter area
    recorder.record_environment_image(env_image)
    print("✅ Environment image recorded")
    
    # 4. Record inference phase
    recorder.record_inference_start()
    
    # Simulate skill generation
    generated_skills = []
    for i, command in enumerate(task_commands):
        parts = command.split()
        skill_data = {
            "command": command,
            "skill_name": parts[0],
            "parameters": " ".join(parts[1:]) if len(parts) > 1 else "",
            "status": "generated",
            "confidence": 0.9 - i * 0.1,
            "generation_method": "llm_with_vision"
        }
        generated_skills.append(skill_data)
    
    # Simulate LLM responses
    llm_responses = {
        "task_decomposition": """Looking at the environment, I can see a bottle and a counter surface.
Task breakdown:
1. detect_object bottle - to locate the bottle precisely
2. detect_object counter - to identify the target placement surface  
3. pickup bottle - to grasp the bottle
4. place bottle,counter - to move bottle to counter
5. open bottle - to open the bottle once it's positioned""",
        "skill_generation": "Generated 5 skills with vision-based context",
        "execution_mode": "real_robot"
    }
    
    recorder.record_inference_complete(generated_skills, llm_responses)
    print("✅ Inference phase recorded")
    
    # 5. Record points of interest (from skill generation)
    points_data = {
        'pixel_coords': [(320, 150), (350, 140), (580, 350), (550, 370)],
        'ids': ['A', 'B', 'C', 'D'],
        'scores': [0.95, 0.88, 0.92, 0.85],
        'detection_methods': ['fastsam', 'fastsam', 'surface_detection', 'surface_detection'],
        'interaction_types': ['grasp_point', 'twist_point', 'placement_surface', 'edge'],
        'confidences': [0.98, 0.89, 0.94, 0.87]
    }
    recorder.record_points_of_interest(points_data)
    print("✅ Points of interest recorded")
    
    # 6. Record surface images
    surface_images = {}
    # Simulate cropped object images
    for obj in ['bottle', 'counter', 'bottle_cap']:
        surface_img = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
        if obj == 'bottle':
            surface_img[:, :, 0] = 100  # Reddish tint for bottle
        elif obj == 'counter':  
            surface_img[:, :] = [180, 160, 140]  # Wood-like color
        elif obj == 'bottle_cap':
            surface_img[:, :] = [200, 200, 200]  # Silver-like cap
        surface_images[obj] = surface_img
    
    recorder.record_surface_images(surface_images)
    print("✅ Surface images recorded")
    
    # 6.5. Record skill generation files
    skill_gen_files = {
        'skill_paths': ['/path/to/stored/skill1.json', '/path/to/stored/skill2.json'],
        'image_paths': [
            '/path/to/skill/images/surface_open_bottle_123456.png',
            '/path/to/skill/images/points_open_bottle_123456.png',
            '/path/to/skill/images/mask_open_bottle_123456.png',
            '/path/to/skill/images/open_bottle_123456.png'
        ],
        'image_id': 'open_bottle_123456'
    }
    recorder.record_skill_generation_files(skill_gen_files)
    print("✅ Skill generation files recorded")
    
    # 7. Record execution phase
    recorder.record_execution_start()
    
    # Simulate execution with some failures
    import time
    time.sleep(0.1)  # Simulate execution time
    
    # Simulate a failure scenario
    failure_messages = [
        "Robot gripper failed to close completely on bottle",
        "Bottle slipped during pickup attempt #1",
        "Retry successful on attempt #2"
    ]
    
    recorder.record_execution_complete(
        success=True,  # Eventually succeeded
        failure_messages=failure_messages,
        robot_errors=["gripper_sensor_timeout", "force_limit_exceeded"]
    )
    print("✅ Execution phase recorded (with failure recovery)")
    
    # 8. Simulate user feedback collection (normally interactive)
    print("\nSimulating user feedback collection...")
    recorder.current_record.user_success_rating = True
    recorder.current_record.user_explanation = """The task was completed successfully but had some issues:
- Initial gripper failure required retry
- Bottle placement was slightly off-center on counter  
- Bottle opening worked smoothly
- Overall: objective achieved with minor execution issues"""
    
    print("✅ User feedback recorded")
    
    # 9. Finalize session (skip interactive parts)
    session_filename = f"{recorder.current_session_id}.json"
    session_path = recorder.sessions_dir / session_filename
    
    # Convert record to dictionary and save
    record_dict = recorder._record_to_dict(recorder.current_record)
    
    import json
    with open(session_path, 'w') as f:
        json.dump(record_dict, f, indent=2, default=recorder._json_serializer)
    
    recorder._print_session_summary(recorder.current_record)
    print(f"\n✅ Session finalized and saved to: {session_path}")
    
    # 10. Verify data completeness
    print("\n" + "="*50)
    print("DATA COMPLETENESS VERIFICATION")
    print("="*50)
    
    with open(session_path, 'r') as f:
        saved_data = json.load(f)
    
    # Check all required fields are present
    required_fields = [
        'natural_language_task', 'task_decomposition', 'skills_generated',
        'points_of_interest', 'environment_image_path', 'surface_images_paths',
        'skill_generation_image_paths', 'stored_skill_paths', 'skill_image_id',
        'timing', 'failure_messages', 'user_success_rating', 'user_explanation',
        'llm_responses', 'robot_errors'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in saved_data:
            missing_fields.append(field)
        elif saved_data[field] is None:
            missing_fields.append(f"{field} (null)")
    
    if missing_fields:
        print(f"❌ Missing or null fields: {missing_fields}")
        return False
    else:
        print("✅ All required fields present and populated")
    
    # Check timing data
    timing = saved_data['timing']
    planning_duration = timing.get('planning_duration', timing['planning_end'] - timing['planning_start'])
    inference_duration = timing.get('inference_duration', timing['inference_end'] - timing['inference_start'])
    execution_duration = timing.get('execution_duration', timing['execution_end'] - timing['execution_start'])
    
    if (planning_duration >= 0 and 
        inference_duration >= 0 and 
        execution_duration >= 0):
        print("✅ All timing measurements recorded")
    else:
        print("❌ Invalid timing measurements")
        return False
    
    # Check data types and structure
    assert isinstance(saved_data['task_decomposition'], list)
    assert isinstance(saved_data['skills_generated'], list)  
    assert isinstance(saved_data['points_of_interest'], list)
    assert isinstance(saved_data['surface_images_paths'], list)
    assert isinstance(saved_data['skill_generation_image_paths'], list)
    assert isinstance(saved_data['stored_skill_paths'], list)
    assert isinstance(saved_data['failure_messages'], list)
    assert isinstance(saved_data['llm_responses'], dict)
    print("✅ Data types and structures correct")
    
    # Check content
    assert len(saved_data['task_decomposition']) == 5
    assert len(saved_data['skills_generated']) == 5
    assert len(saved_data['points_of_interest']) == 4  
    assert len(saved_data['surface_images_paths']) == 3
    assert len(saved_data['skill_generation_image_paths']) == 4  # 4 skill generation images
    assert len(saved_data['stored_skill_paths']) == 2  # 2 stored skills
    assert saved_data['skill_image_id'] == 'open_bottle_123456'
    print("✅ Content counts correct")
    
    print(f"\n🎉 Complete workflow test PASSED!")
    print(f"📊 Recorded data for task: '{saved_data['natural_language_task']}'")
    total_duration = timing.get('total_duration', timing['execution_end'] - timing['planning_start'])
    print(f"⏱️  Total execution time: {total_duration:.3f}s")
    print(f"💾 Session data: {len(json.dumps(saved_data))} characters")
    
    return True

def main():
    """Run the complete workflow test"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        if test_full_workflow():
            print("\n🎉 ALL TESTS PASSED - Data recording system is working correctly!")
            return 0
        else:
            print("\n❌ TEST FAILED")
            return 1
    except Exception as e:
        print(f"\n💥 TEST CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())