#!/usr/bin/env python3
"""
Test script for collision debugging tools
Demonstrates how to inspect and verify collision objects in CuRobo motion planning
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger, create_test_collision_objects, run_collision_debug_tests


def test_basic_collision_debugging():
    """Test basic collision debugging functionality"""
    print("=== Basic Collision Debugging Test ===")
    
    # Initialize motion planner (without connecting to robot)
    motion_planner = CuRoboMotionPlanner(robot_ip=None)
    
    # Initialize CuRobo
    print("Initializing CuRobo...")
    motion_planner.init_curobo()
    
    # Create test collision objects
    create_test_collision_objects(motion_planner)
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    # Test basic functionality
    print("\n1. Listing collision objects:")
    debugger.print_collision_objects()
    
    print("\n2. Verifying world configuration:")
    debugger.verify_world_config()
    
    print("\n3. Testing collision detection:")
    test_positions = [
        [0.0, 0.0, 0.1],   # Above table
        [0.2, 0.1, 0.1],   # Inside obstacle
        [0.5, 0.0, 0.1],   # Far from obstacles
    ]
    debugger.test_collision_detection(test_positions)
    
    print("\n4. Benchmarking performance:")
    debugger.benchmark_collision_performance(num_tests=20)
    
    print("\n✅ Basic collision debugging test completed")


def test_motion_planning_with_collisions():
    """Test motion planning with collision objects"""
    print("\n=== Motion Planning with Collisions Test ===")
    
    # Initialize motion planner
    motion_planner = CuRoboMotionPlanner(robot_ip=None)
    motion_planner.init_curobo()
    
    # Create test collision objects
    create_test_collision_objects(motion_planner)
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    # Test motion planning scenarios
    test_scenarios = [
        {
            'name': 'Free space motion',
            'start': [0.0, 0.0, 0.3],
            'target': [0.4, 0.0, 0.3]
        },
        {
            'name': 'Motion near obstacle',
            'start': [0.0, 0.0, 0.3],
            'target': [0.3, 0.0, 0.3]
        },
        {
            'name': 'Motion avoiding obstacle',
            'start': [0.0, 0.0, 0.3],
            'target': [0.4, 0.2, 0.3]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")
        result = debugger.test_motion_planning_with_collisions(
            start_pos=scenario['start'],
            target_pos=scenario['target']
        )
        
        if result['success']:
            print(f"✅ {scenario['name']}: SUCCESS")
        else:
            print(f"❌ {scenario['name']}: FAILED")
    
    print("\n✅ Motion planning with collisions test completed")


def test_collision_object_management():
    """Test collision object management functions"""
    print("\n=== Collision Object Management Test ===")
    
    # Initialize motion planner
    motion_planner = CuRoboMotionPlanner(robot_ip=None)
    motion_planner.init_curobo()
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    print("1. Initial state:")
    debugger.print_collision_objects()
    
    print("\n2. Adding collision objects:")
    motion_planner.add_collision_object(
        name="test_box",
        dimensions=[0.1, 0.1, 0.1],
        position=[0.2, 0.0, 0.05],
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    motion_planner.add_collision_object(
        name="test_wall",
        dimensions=[0.05, 0.5, 0.5],
        position=[0.0, 0.3, 0.25],
        orientation=[1.0, 0.0, 0.0, 0.0]
    )
    
    print("\n3. After adding objects:")
    debugger.print_collision_objects()
    
    print("\n4. Testing collision detection:")
    test_positions = [
        [0.2, 0.0, 0.05],  # Inside test_box
        [0.0, 0.3, 0.25],  # Inside test_wall
        [0.5, 0.0, 0.1],   # Free space
    ]
    debugger.test_collision_detection(test_positions)
    
    print("\n5. Clearing collision objects:")
    motion_planner.clear_collision_objects()
    
    print("\n6. After clearing:")
    debugger.print_collision_objects()
    
    print("\n✅ Collision object management test completed")


def test_visualization():
    """Test 3D visualization of collision objects"""
    print("\n=== Collision Object Visualization Test ===")
    
    # Initialize motion planner
    motion_planner = CuRoboMotionPlanner(robot_ip=None)
    motion_planner.init_curobo()
    
    # Create test collision objects
    create_test_collision_objects(motion_planner)
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    print("Creating 3D visualization...")
    try:
        debugger.visualize_collision_objects()
        print("✅ Visualization test completed")
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        print("Note: Open3D is required for visualization")


def run_comprehensive_tests():
    """Run all collision debugging tests"""
    print("="*60)
    print("COLLISION DEBUGGING COMPREHENSIVE TESTS")
    print("="*60)
    
    try:
        # Test 1: Basic functionality
        test_basic_collision_debugging()
        
        # Test 2: Motion planning with collisions
        test_motion_planning_with_collisions()
        
        # Test 3: Collision object management
        test_collision_object_management()
        
        # Test 4: Visualization (optional)
        test_visualization()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run comprehensive tests
    run_comprehensive_tests() 