#!/usr/bin/env python3
"""
Simple test for Task Planner with RealSense camera
"""

import sys
import logging

# Add the project root to Python path
sys.path.append('/home/liam/dev/zk_task_planner/cognitive_bt_framework')

from src.task_planner import TaskPlanner


def test_task_planner_basic():
    """Test basic task planner functionality"""
    print("=== Testing Task Planner with RealSense ===")
    
    try:
        # Initialize task planner (this will connect to robot and RealSense)
        planner = TaskPlanner(
            robot_ip="192.168.1.224",
            camera_type="realsense",
            use_llm=False  # Start without LLM for basic testing
        )
        
        print("✓ Task planner initialized successfully")
        print(f"✓ Robot connection: {'Connected' if planner.motion_planner else 'Failed'}")
        print(f"✓ Camera connection: {'Connected' if planner.camera else 'Failed'}")
        print(f"✓ Perception system: {'Ready' if planner.perception_system else 'Failed'}")
        
        # Test task decomposition
        test_task = "put away the dishes"
        print(f"\n--- Testing Task Decomposition: '{test_task}' ---")
        
        task_plan = planner.analyze_task(test_task)
        
        print(f"Generated Task Plan:")
        print(f"  Task: {task_plan.task_description}")
        print(f"  Steps: {len(task_plan.steps)}")
        
        for i, step in enumerate(task_plan.steps):
            print(f"    {i+1}. {step.description}")
            print(f"       Skill: {step.skill_type}")
            print(f"       Target: {step.target_object}")
        
        print(f"  Estimated Duration: {task_plan.estimated_duration:.1f} seconds")
        
        # Test environment capture
        print(f"\n--- Testing Environment Capture ---")
        
        try:
            rgb_image, env_info = planner.capture_environment()
            
            if rgb_image is not None:
                print(f"✓ Captured image: {rgb_image.shape}")
                if env_info:
                    print(f"✓ Environment info: {len(env_info.get('detected_objects', []))} objects detected")
                    print(f"  Image shape: {env_info.get('image_shape')}")
                    print(f"  Depth available: {env_info.get('depth_available')}")
                    print(f"  Timestamp: {env_info.get('timestamp')}")
                else:
                    print("⚠ No environment info (perception may have failed)")
            else:
                print("✗ Failed to capture environment image")
                
        except Exception as e:
            print(f"✗ Environment capture error: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing task planner: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_available_skills():
    """Test skill template system"""
    print("\n=== Testing Available Skills ===")
    
    try:
        planner = TaskPlanner(use_llm=False)
        
        available_skills = planner.get_available_skills()
        print(f"Available Skills ({len(available_skills)}):")
        
        for skill in available_skills:
            skill_info = planner.get_skill_info(skill)
            print(f"  - {skill}: {skill_info['skill_type']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing skills: {e}")
        return False


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Task Planner Test Suite")
    print("=" * 50)
    
    # Run basic test
    success1 = test_task_planner_basic()
    
    # Run skills test
    success2 = test_available_skills()
    
    # Summary
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    print("Test complete.")