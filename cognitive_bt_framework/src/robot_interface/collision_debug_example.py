#!/usr/bin/env python3
"""
Practical example of using collision debugging tools
Shows how to inspect and verify collision objects during robot manipulation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger, create_test_collision_objects


def example_robot_manipulation_with_collision_debugging():
    """
    Example of using collision debugging during robot manipulation
    This shows how to verify collision objects are working correctly
    """
    print("=== Robot Manipulation with Collision Debugging ===")
    
    # Initialize motion planner (connect to robot if available)
    robot_ip = "192.168.1.224"  # Change to your robot's IP
    motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
    
    # Initialize CuRobo
    print("Initializing CuRobo...")
    motion_planner.init_curobo()
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    # Step 1: Check initial state
    print("\n1. Checking initial collision objects:")
    debugger.print_collision_objects()
    
    # Step 2: Add environment collision objects
    print("\n2. Adding environment collision objects:")
    motion_planner.add_collision_object(
        name="table_surface",
        dimensions=[1.2, 0.8, 0.02],  # 1.2m x 0.8m x 2cm
        position=[0.0, 0.0, -0.01],   # Slightly below origin
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    motion_planner.add_collision_object(
        name="back_wall",
        dimensions=[0.1, 0.8, 1.0],   # 10cm thick, 0.8m wide, 1m tall
        position=[-0.5, 0.0, 0.5],    # Behind robot
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    # Step 3: Verify collision objects were added
    print("\n3. Verifying collision objects after addition:")
    debugger.print_collision_objects()
    
    # Step 4: Test collision detection at key positions
    print("\n4. Testing collision detection at key positions:")
    key_positions = [
        [0.0, 0.0, 0.1],   # Above table
        [-0.5, 0.0, 0.5],  # Inside back wall
        [0.0, 0.0, 0.5],   # Free space above table
        [0.6, 0.0, 0.3],   # Far from obstacles
    ]
    
    collision_results = debugger.test_collision_detection(key_positions)
    
    # Step 5: Test motion planning with collisions
    print("\n5. Testing motion planning scenarios:")
    
    # Scenario 1: Simple motion in free space
    print("\n--- Scenario 1: Free space motion ---")
    result1 = debugger.test_motion_planning_with_collisions(
        start_pos=[0.0, 0.0, 0.3],
        target_pos=[0.4, 0.0, 0.3]
    )
    
    # Scenario 2: Motion that should avoid obstacles
    print("\n--- Scenario 2: Motion avoiding obstacles ---")
    result2 = debugger.test_motion_planning_with_collisions(
        start_pos=[0.0, 0.0, 0.3],
        target_pos=[0.4, 0.2, 0.3]
    )
    
    # Step 6: Add dynamic collision objects (simulating detected objects)
    print("\n6. Adding dynamic collision objects (simulating detected objects):")
    
    # Simulate detecting a cup
    motion_planner.add_collision_object(
        name="detected_cup",
        dimensions=[0.08, 0.08, 0.12],  # 8cm x 8cm x 12cm
        position=[0.2, 0.1, 0.06],      # On table
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    # Simulate detecting a book
    motion_planner.add_collision_object(
        name="detected_book",
        dimensions=[0.15, 0.22, 0.03],  # 15cm x 22cm x 3cm
        position=[0.3, -0.1, 0.015],    # On table
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    print("\n7. Updated collision objects:")
    debugger.print_collision_objects()
    
    # Step 7: Test motion planning with dynamic objects
    print("\n8. Testing motion planning with dynamic objects:")
    
    # Test motion that should avoid the cup
    result3 = debugger.test_motion_planning_with_collisions(
        start_pos=[0.0, 0.0, 0.3],
        target_pos=[0.4, 0.0, 0.3]  # This should avoid the cup
    )
    
    # Step 8: Benchmark performance
    print("\n9. Benchmarking collision detection performance:")
    performance = debugger.benchmark_collision_performance(num_tests=100)
    
    # Step 9: Summary
    print("\n" + "="*50)
    print("COLLISION DEBUGGING SUMMARY")
    print("="*50)
    print(f"Total collision objects: {len(debugger.list_collision_objects())}")
    print(f"Motion planning scenarios tested: 3")
    print(f"Collision detection performance: {performance.get('tests_per_second', 0):.1f} tests/sec")
    
    # Check if motion planning was successful
    successful_plans = sum([
        1 if result1.get('success', False) else 0,
        1 if result2.get('success', False) else 0,
        1 if result3.get('success', False) else 0
    ])
    
    print(f"Successful motion plans: {successful_plans}/3")
    
    if successful_plans == 3:
        print("✅ All motion planning scenarios successful!")
    else:
        print("⚠️  Some motion planning scenarios failed")
    
    print("="*50)


def example_integration_with_skill_executor():
    """
    Example of integrating collision debugging with skill execution
    This shows how to verify collision objects during skill execution
    """
    print("\n=== Integration with Skill Executor Example ===")
    
    # This would be integrated into your skill executor
    # Here's how you could add collision debugging to your existing code:
    
    class SkillExecutorWithCollisionDebug:
        def __init__(self, robot_ip="192.168.1.224"):
            self.motion_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
            self.motion_planner.init_curobo()
            self.debugger = CollisionDebugger(self.motion_planner)
            
        def execute_skill_with_collision_verification(self, skill_name, parameters):
            """Execute a skill with collision object verification"""
            
            print(f"\n--- Executing skill: {skill_name} ---")
            
            # Step 1: Verify collision objects before execution
            print("1. Verifying collision objects before skill execution:")
            self.debugger.print_collision_objects()
            
            # Step 2: Check if motion planning is possible
            print("2. Testing motion planning capability:")
            current_pos = [0.0, 0.0, 0.3]  # Example current position
            target_pos = [0.4, 0.0, 0.3]   # Example target position
            
            planning_test = self.debugger.test_motion_planning_with_collisions(
                start_pos=current_pos,
                target_pos=target_pos
            )
            
            if not planning_test.get('success', False):
                print("❌ Motion planning failed - cannot execute skill")
                return False
            
            # Step 3: Execute the skill (your existing skill execution code would go here)
            print("3. Executing skill...")
            # self.execute_skill_implementation(skill_name, parameters)
            
            # Step 4: Verify collision objects after execution
            print("4. Verifying collision objects after skill execution:")
            self.debugger.print_collision_objects()
            
            print("✅ Skill execution completed with collision verification")
            return True
        
        def add_detected_object_as_collision(self, object_name, bbox, position):
            """Add a detected object as a collision object"""
            print(f"\n--- Adding detected object: {object_name} ---")
            
            # Convert bbox to collision object dimensions
            width, height = bbox[2], bbox[3]
            depth = 0.1  # Assume 10cm depth for detected objects
            
            dimensions = [width, height, depth]
            
            # Add collision object
            self.motion_planner.add_collision_object(
                name=f"detected_{object_name}",
                dimensions=dimensions,
                position=position,
                orientation=[1.0, 0.0, 0.0, 0.0]
            )
            
            # Verify it was added
            print(f"✅ Added collision object for {object_name}")
            self.debugger.print_collision_objects()
    
    # Example usage
    executor = SkillExecutorWithCollisionDebug()
    
    # Simulate adding detected objects
    executor.add_detected_object_as_collision(
        object_name="cup",
        bbox=[100, 150, 80, 120],  # [x, y, width, height] in pixels
        position=[0.2, 0.1, 0.06]
    )
    
    # Simulate skill execution
    executor.execute_skill_with_collision_verification("pick", "cup")


if __name__ == "__main__":
    # Run the examples
    example_robot_manipulation_with_collision_debugging()
    example_integration_with_skill_executor() 