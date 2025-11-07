#!/usr/bin/env python3
"""
Simple diagnostic test for RealSense camera to check if it's working properly
"""

import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cognitive_bt_framework.src.vision.realsense import Camera


def test_realsense_basic_functionality():
    """
    Test basic RealSense camera functionality
    """
    print("="*50)
    print("REALSENSE CAMERA DIAGNOSTIC TEST")
    print("="*50)
    
    try:
        # Initialize camera
        print("1. Initializing RealSense camera...")
        camera = Camera(
            width=640,
            height=480,
            fps=30,
            use_viewer_defaults=True,
            debug=True
        )
        
        # Start the camera
        print("2. Starting RealSense camera...")
        if not camera.start():
            print("❌ Failed to start RealSense camera")
            return False
        print("✅ RealSense camera started successfully")
        
        # Test basic frame capture
        print("\n3. Testing basic frame capture...")
        frames = camera.get_frames(use_averaging=False)
        if frames is not None:
            color_image, depth_image = frames
            print(f"✅ Captured frames - Color: {color_image.shape}, Depth: {depth_image.shape}")
            
            # Check depth data validity
            valid_depth_pixels = np.sum(depth_image > 0)
            total_pixels = depth_image.size
            valid_percentage = (valid_depth_pixels / total_pixels) * 100
            
            print(f"Depth data analysis:")
            print(f"  Total pixels: {total_pixels}")
            print(f"  Valid depth pixels: {valid_depth_pixels}")
            print(f"  Valid percentage: {valid_percentage:.1f}%")
            
            if valid_percentage > 10:  # At least 10% valid depth data
                print("✅ Depth data looks good")
            else:
                print("⚠️  Low percentage of valid depth data")
                
            # Show depth statistics
            valid_depths = depth_image[depth_image > 0]
            if len(valid_depths) > 0:
                print(f"  Depth range: {valid_depths.min():.3f} - {valid_depths.max():.3f}")
                print(f"  Mean depth: {valid_depths.mean():.3f}")
            else:
                print("  No valid depth data found")
                
        else:
            print("❌ Failed to capture frames")
            return False
        
        # Test point cloud generation
        print("\n4. Testing point cloud generation...")
        pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
        if pcd is not None and len(pcd) > 0:
            print(f"✅ Generated point cloud with {len(pcd)} points")
            
            # Analyze point cloud
            if len(pcd) > 0:
                print(f"Point cloud analysis:")
                print(f"  X range: {pcd[:, 0].min():.3f} - {pcd[:, 0].max():.3f}")
                print(f"  Y range: {pcd[:, 1].min():.3f} - {pcd[:, 1].max():.3f}")
                print(f"  Z range: {pcd[:, 2].min():.3f} - {pcd[:, 2].max():.3f}")
        else:
            print("❌ Failed to generate point cloud")
            return False
        
        # Test camera parameters
        print("\n5. Testing camera parameters...")
        params = camera.get_current_parameters()
        if params:
            print("✅ Camera parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print("⚠️  Could not retrieve camera parameters")
        
        # Test camera matrix
        print("\n6. Testing camera matrix...")
        camera_matrix = camera.get_camera_matrix()
        if camera_matrix is not None:
            print("✅ Camera matrix:")
            if hasattr(camera_matrix, 'fx'):
                print(f"  fx: {camera_matrix.fx:.2f}")
                print(f"  fy: {camera_matrix.fy:.2f}")
                print(f"  ppx: {camera_matrix.ppx:.2f}")
                print(f"  ppy: {camera_matrix.ppy:.2f}")
            else:
                print(f"  Matrix shape: {camera_matrix.shape}")
                print(f"  Matrix type: {type(camera_matrix)}")
        else:
            print("❌ Could not retrieve camera matrix")
            return False
        
        # Test visualization
        print("\n7. Testing visualization...")
        try:
            camera.visualize(use_averaging=False)
            print("✅ Visualization test completed")
        except Exception as e:
            print(f"⚠️  Visualization test failed: {e}")
        
        # Cleanup
        print("\n8. Cleaning up...")
        camera.stop()
        print("✅ Camera stopped")
        
        print("\n" + "="*50)
        print("DIAGNOSTIC TEST COMPLETED SUCCESSFULLY")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_realsense_with_manual_calculation():
    """
    Test RealSense with manual point cloud calculation
    """
    print("\n" + "="*50)
    print("REALSENSE MANUAL POINT CLOUD TEST")
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
        
        # Test manual point cloud calculation
        print("Testing manual point cloud calculation...")
        pcd = camera.get_point_cloud(use_averaging=False, manual_calculation=True)
        
        if pcd is not None and len(pcd) > 0:
            print(f"✅ Manual point cloud: {len(pcd)} points")
            
            # Test with averaging
            print("Testing with depth averaging...")
            pcd_avg = camera.get_point_cloud(use_averaging=True, manual_calculation=True)
            if pcd_avg is not None and len(pcd_avg) > 0:
                print(f"✅ Averaged point cloud: {len(pcd_avg)} points")
            else:
                print("❌ Averaged point cloud failed")
        else:
            print("❌ Manual point cloud calculation failed")
            return False
        
        camera.stop()
        return True
        
    except Exception as e:
        print(f"❌ Manual point cloud test failed: {e}")
        return False


def test_realsense_frame_queue():
    """
    Test RealSense frame queue functionality
    """
    print("\n" + "="*50)
    print("REALSENSE FRAME QUEUE TEST")
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
        
        # Test frame queue
        print("Testing frame queue...")
        for i in range(5):
            try:
                aligned_frames = camera._frame_queue.get(timeout=2.0)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if depth_frame and color_frame:
                    print(f"  Frame {i+1}: ✅ Depth and color frames available")
                else:
                    print(f"  Frame {i+1}: ❌ Missing frames")
                    
            except Exception as e:
                print(f"  Frame {i+1}: ❌ Error: {e}")
        
        camera.stop()
        return True
        
    except Exception as e:
        print(f"❌ Frame queue test failed: {e}")
        return False


if __name__ == "__main__":
    print("RealSense Camera Diagnostic Tests")
    print("="*50)
    
    # Run all diagnostic tests
    tests = [
        ("Basic Functionality", test_realsense_basic_functionality),
        ("Manual Point Cloud", test_realsense_with_manual_calculation),
        ("Frame Queue", test_realsense_frame_queue),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*50)
    print("DIAGNOSTIC TEST SUMMARY")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL DIAGNOSTIC TESTS PASSED!")
        print("RealSense camera is working correctly.")
    else:
        print("\n⚠️  Some diagnostic tests failed.")
        print("Check the output above for details.")
    
    sys.exit(0 if all_passed else 1) 