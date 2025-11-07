#!/usr/bin/env python3
"""
Simple test of RealSense camera integration with collision objects
This test focuses on the core functionality without multiple camera starts
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm_curobo_interface import CuRoboMotionPlanner
from collision_debug import CollisionDebugger
from cognitive_bt_framework.src.vision.realsense import Camera


def test_realsense_collision_simple():
    """
    Simple test of RealSense + collision object integration
    """
    print("="*60)
    print("SIMPLE REALSENSE + COLLISION INTEGRATION TEST")
    print("="*60)
    
    # Step 1: Initialize motion planner
    print("\n1. Initializing motion planner...")
    motion_planner = CuRoboMotionPlanner(robot_ip="192.168.1.224")
    motion_planner.init_curobo()
    print("✅ Motion planner initialized")
    
    # Step 2: Initialize RealSense camera
    print("\n2. Initializing RealSense camera...")
    try:
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True,
            debug=True
        )
        
        if not camera.start():
            print("❌ Failed to start RealSense camera")
            return False
        print("✅ RealSense camera started")
        
    except Exception as e:
        print(f"❌ Failed to initialize RealSense camera: {e}")
        return False
    
    # Step 3: Create collision debugger
    print("\n3. Creating collision debugger...")
    debugger = CollisionDebugger(motion_planner)
    
    # Step 4: Check initial state
    print("\n4. Checking initial collision objects...")
    debugger.print_collision_objects()
    
    # Step 5: Capture point cloud and generate collision objects
    print("\n5. Capturing point cloud from RealSense...")
    try:
        # Get point cloud from RealSense
        pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
        if pcd is not None and len(pcd) > 0:
            print(f"✅ Captured point cloud with {len(pcd)} points")
            
            # Update dynamic collision objects
            print("Updating dynamic collision objects...")
            motion_planner.update_dynamic_collision_objects(pcd)
            
            # Verify collision objects were added
            print("\n6. Verifying dynamic collision objects...")
            debugger.print_collision_objects()
            
        else:
            print("❌ Failed to capture point cloud from RealSense")
            camera.stop()
            return False
            
    except Exception as e:
        print(f"❌ Error capturing point cloud: {e}")
        camera.stop()
        return False
    
    # Step 7: Test collision detection
    print("\n7. Testing collision detection...")
    test_positions = [
        [0.0, 0.0, 0.3],   # Above table
        [0.2, 0.0, 0.3],   # In front of robot
        [0.4, 0.0, 0.3],   # Further in front
    ]
    
    collision_results = debugger.test_collision_detection(test_positions)
    
    # Step 8: Test motion planning
    print("\n8. Testing motion planning with collision objects...")
    result = debugger.test_motion_planning_with_collisions(
        start_pos=[0.0, 0.0, 0.3],
        target_pos=[0.3, 0.0, 0.3]
    )
    
    if result.get('success', False):
        print("✅ Motion planning with collision objects successful")
    else:
        print("❌ Motion planning with collision objects failed")
    
    # Step 9: Summary
    print("\n" + "="*40)
    print("INTEGRATION TEST SUMMARY")
    print("="*40)
    
    collision_objects = debugger.list_collision_objects()
    print(f"Total collision objects: {len(collision_objects)}")
    
    if collision_objects:
        print("Detected objects:")
        for obj in collision_objects:
            print(f"  - {obj['name']}: {obj['type']} at {obj['position']}")
    
    # Step 10: Cleanup
    print("\n10. Cleaning up...")
    camera.stop()
    motion_planner.disconnect_robot()
    
    print("="*40)
    print("SIMPLE INTEGRATION TEST COMPLETED")
    print("="*40)
    
    return result.get('success', False)


def test_realsense_point_cloud_quality():
    """
    Test RealSense point cloud quality
    """
    print("\n" + "="*50)
    print("REALSENSE POINT CLOUD QUALITY TEST")
    print("="*50)
    
    try:
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True,
            debug=True
        )
        
        if not camera.start():
            print("❌ Failed to start RealSense camera")
            return False
        print("✅ RealSense camera started")
        
        # Capture multiple point clouds
        point_clouds = []
        for i in range(3):
            print(f"Capturing point cloud {i+1}/3...")
            pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
            if pcd is not None and len(pcd) > 0:
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


if __name__ == "__main__":
    print("Simple RealSense + Collision Integration Test")
    print("="*60)
    
    # Run the simple integration test
    success = test_realsense_collision_simple()
    
    if success:
        print("\n✅ Simple integration test completed successfully!")
        print("RealSense camera is working with collision objects!")
    else:
        print("\n❌ Simple integration test failed!")
        print("Check the output above for details.")
    
    # Run point cloud quality test
    quality_success = test_realsense_point_cloud_quality()
    
    if quality_success:
        print("\n✅ Point cloud quality test passed!")
    else:
        print("\n❌ Point cloud quality test failed!")
    
    sys.exit(0 if success and quality_success else 1) 