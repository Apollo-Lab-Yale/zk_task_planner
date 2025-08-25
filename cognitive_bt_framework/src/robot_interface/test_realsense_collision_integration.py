#!/usr/bin/env python3
"""
Comprehensive test of RealSense camera integration with dynamic collision object generation
This test demonstrates the full pipeline from RealSense point cloud to collision objects to motion planning
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera


def test_realsense_collision_integration():
    """
    Test the full integration of RealSense camera with dynamic collision object generation
    """
    print("="*70)
    print("REALSENSE + DYNAMIC COLLISION OBJECTS INTEGRATION TEST")
    print("="*70)
    
    # Step 1: Initialize motion planner
    print("\n1. Initializing motion planner...")
    motion_planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")
    motion_planner.init_curobo()
    
    # Step 2: Initialize RealSense camera
    print("\n2. Initializing RealSense camera...")
    try:
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True
        )
        print("✅ RealSense camera initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RealSense camera: {e}")
        print("Make sure RealSense camera is connected and not in use by another application")
        return False
    
    # Step 3: Create collision debugger
    print("\n3. Creating collision debugger...")
    debugger = CollisionDebugger(motion_planner)
    
    # Step 4: Check initial state
    print("\n4. Checking initial collision objects...")
    debugger.print_collision_objects()
    debugger.verify_world_config()
    
    # Step 5: Capture point cloud and generate collision objects
    print("\n5. Capturing point cloud from RealSense...")
    try:
        # Get point cloud from RealSense
        pcd = camera.get_point_cloud()
        if pcd is not None:
            print(f"✅ Captured point cloud with {len(pcd)} points")
            
            # Update dynamic collision objects
            print("Updating dynamic collision objects...")
            motion_planner.update_dynamic_collision_objects(pcd)
            
            # Verify collision objects were added
            print("\n6. Verifying dynamic collision objects...")
            debugger.print_collision_objects()
            
        else:
            print("❌ Failed to capture point cloud from RealSense")
            return False
            
    except Exception as e:
        print(f"❌ Error capturing point cloud: {e}")
        return False
    
    # Step 7: Test collision detection with real objects
    print("\n7. Testing collision detection with real objects...")
    
    # Test positions in the workspace
    test_positions = [
        [0.0, 0.0, 0.3],   # Above table
        [0.2, 0.0, 0.3],   # In front of robot
        [0.4, 0.0, 0.3],   # Further in front
        [0.0, 0.2, 0.3],   # To the right
        [0.0, -0.2, 0.3],  # To the left
    ]
    
    collision_results = debugger.test_collision_detection(test_positions)
    
    # Step 8: Test motion planning with real collision objects
    print("\n8. Testing motion planning with real collision objects...")
    
    # Test different motion planning scenarios
    planning_scenarios = [
        {
            'name': 'Simple forward motion',
            'start': [0.0, 0.0, 0.3],
            'target': [0.3, 0.0, 0.3]
        },
        {
            'name': 'Motion avoiding obstacles',
            'start': [0.0, 0.0, 0.3],
            'target': [0.4, 0.2, 0.3]
        },
        {
            'name': 'Lateral motion',
            'start': [0.0, 0.0, 0.3],
            'target': [0.0, 0.3, 0.3]
        }
    ]
    
    successful_plans = 0
    for scenario in planning_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")
        result = debugger.test_motion_planning_with_collisions(
            start_pos=scenario['start'],
            target_pos=scenario['target']
        )
        
        if result.get('success', False):
            successful_plans += 1
            print(f"✅ {scenario['name']}: SUCCESS")
        else:
            print(f"❌ {scenario['name']}: FAILED")
    
    # Step 9: Benchmark performance
    print("\n9. Benchmarking collision detection performance...")
    performance = debugger.benchmark_collision_performance(num_tests=50)
    
    # Step 10: Create visualization
    print("\n10. Creating 3D visualization...")
    try:
        debugger.visualize_collision_objects()
        print("✅ Visualization completed")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Step 11: Summary
    print("\n" + "="*50)
    print("INTEGRATION TEST SUMMARY")
    print("="*50)
    
    collision_objects = debugger.list_collision_objects()
    print(f"Total collision objects: {len(collision_objects)}")
    
    if collision_objects:
        print("Detected objects:")
        for obj in collision_objects:
            print(f"  - {obj['name']}: {obj['type']} at {obj['position']}")
    
    print(f"Motion planning scenarios: {successful_plans}/{len(planning_scenarios)} successful")
    print(f"Collision detection performance: {performance.get('tests_per_second', 0):.1f} tests/sec")
    
    if successful_plans == len(planning_scenarios):
        print("✅ All motion planning scenarios successful!")
    else:
        print("⚠️  Some motion planning scenarios failed")
    
    # Step 12: Cleanup
    print("\n12. Cleaning up...")
    camera.stop()
    motion_planner.disconnect_robot()
    
    print("="*50)
    print("INTEGRATION TEST COMPLETED")
    print("="*50)
    
    return successful_plans == len(planning_scenarios)


def test_realsense_point_cloud_quality():
    """
    Test the quality of RealSense point cloud data
    """
    print("\n" + "="*50)
    print("REALSENSE POINT CLOUD QUALITY TEST")
    print("="*50)
    
    try:
        # Initialize camera
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True
        )
        
        print("✅ RealSense camera initialized")
        
        # Capture multiple point clouds to test consistency
        point_clouds = []
        for i in range(5):
            print(f"Capturing point cloud {i+1}/5...")
            pcd = camera.get_point_cloud()
            if pcd is not None:
                point_clouds.append(pcd)
                print(f"  Points: {len(pcd)}")
            else:
                print("  Failed to capture point cloud")
        
        if point_clouds:
            # Analyze point cloud quality
            point_counts = [len(pcd) for pcd in point_clouds]
            avg_points = np.mean(point_counts)
            std_points = np.std(point_counts)
            
            print(f"\nPoint cloud quality analysis:")
            print(f"  Average points: {avg_points:.0f}")
            print(f"  Standard deviation: {std_points:.0f}")
            print(f"  Consistency: {std_points/avg_points*100:.1f}% CV")
            
            if std_points/avg_points < 0.2:  # Less than 20% variation
                print("✅ Point cloud quality is consistent")
            else:
                print("⚠️  Point cloud quality is inconsistent")
        
        camera.stop()
        return True
        
    except Exception as e:
        print(f"❌ Point cloud quality test failed: {e}")
        return False


def test_dynamic_collision_object_generation():
    """
    Test the dynamic collision object generation pipeline
    """
    print("\n" + "="*50)
    print("DYNAMIC COLLISION OBJECT GENERATION TEST")
    print("="*50)
    
    # Initialize motion planner
    motion_planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")
    motion_planner.init_curobo()
    
    # Create debugger
    debugger = CollisionDebugger(motion_planner)
    
    # Test with synthetic point cloud data
    print("Testing with synthetic point cloud data...")
    
    # Create synthetic point cloud with some obstacles
    np.random.seed(42)
    num_points = 1000
    
    # Create a table surface
    table_points = np.random.uniform(-0.5, 0.5, (num_points//2, 3))
    table_points[:, 2] = 0.0  # Z = 0 for table surface
    
    # Create some obstacles
    obstacle_points = np.random.uniform(-0.1, 0.1, (num_points//4, 3))
    obstacle_points[:, 0] += 0.2  # Move to x=0.2
    obstacle_points[:, 2] += 0.1  # Raise to z=0.1
    
    # Create another obstacle
    obstacle2_points = np.random.uniform(-0.1, 0.1, (num_points//4, 3))
    obstacle2_points[:, 1] += 0.2  # Move to y=0.2
    obstacle2_points[:, 2] += 0.15  # Raise to z=0.15
    
    # Combine all points
    synthetic_pcd = np.vstack([table_points, obstacle_points, obstacle2_points])
    
    print(f"Created synthetic point cloud with {len(synthetic_pcd)} points")
    
    # Test collision object generation
    try:
        motion_planner.update_dynamic_collision_objects(synthetic_pcd)
        print("✅ Dynamic collision objects generated successfully")
        
        # Verify collision objects
        debugger.print_collision_objects()
        
        # Test collision detection
        test_positions = [
            [0.2, 0.0, 0.1],   # Inside first obstacle
            [0.0, 0.2, 0.15],  # Inside second obstacle
            [0.0, 0.0, 0.3],   # Above table (free)
            [0.5, 0.0, 0.3],   # Far from obstacles (free)
        ]
        
        collision_results = debugger.test_collision_detection(test_positions)
        
        # Test motion planning
        result = debugger.test_motion_planning_with_collisions(
            start_pos=[0.0, 0.0, 0.3],
            target_pos=[0.4, 0.0, 0.3]
        )
        
        if result.get('success', False):
            print("✅ Motion planning with synthetic obstacles successful")
        else:
            print("❌ Motion planning with synthetic obstacles failed")
        
        motion_planner.disconnect_robot()
        return True
        
    except Exception as e:
        print(f"❌ Dynamic collision object generation failed: {e}")
        motion_planner.disconnect_robot()
        return False


def run_comprehensive_integration_test():
    """
    Run all integration tests
    """
    print("="*70)
    print("COMPREHENSIVE REALSENSE + COLLISION INTEGRATION TEST")
    print("="*70)
    
    results = {}
    
    # Test 1: Point cloud quality
    print("\n" + "="*30)
    print("TEST 1: REALSENSE POINT CLOUD QUALITY")
    print("="*30)
    results['point_cloud_quality'] = test_realsense_point_cloud_quality()
    
    # Test 2: Dynamic collision object generation
    print("\n" + "="*30)
    print("TEST 2: DYNAMIC COLLISION OBJECT GENERATION")
    print("="*30)
    results['collision_generation'] = test_dynamic_collision_object_generation()
    
    # Test 3: Full integration
    print("\n" + "="*30)
    print("TEST 3: FULL REALSENSE + COLLISION INTEGRATION")
    print("="*30)
    results['full_integration'] = test_realsense_collision_integration()
    
    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    # Run the comprehensive integration test
    success = run_comprehensive_integration_test()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1) 