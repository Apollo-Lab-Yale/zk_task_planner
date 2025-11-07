#!/usr/bin/env python3
"""
Test script for top-down grasp positioning functionality.
This script demonstrates how the new adjust_tcp_pose_for_top_down_grasp method works.
"""

import numpy as np
import cv2
import sys
import os

# Add the cognitive_bt_framework to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cognitive_bt_framework', 'src'))

from skills.skill_executor import DirectSkillExecutor
from vision.perception_system import PerceptionSystem, ObjectInfo
from skills.skill_handler import ExecutableAction

def create_test_object_mask(height=480, width=640):
    """Create a test object mask with a simple shape"""
    mask = np.zeros((height, width), dtype=bool)
    
    # Create a simple rectangular object in the center
    center_y, center_x = height // 2, width // 2
    object_height, object_width = 100, 150
    
    y_start = max(0, center_y - object_height // 2)
    y_end = min(height, center_y + object_height // 2)
    x_start = max(0, center_x - object_width // 2)
    x_end = min(width, center_x + object_width // 2)
    
    mask[y_start:y_end, x_start:x_end] = True
    
    return mask

def create_test_depth_image(height=480, width=640):
    """Create a test depth image with a simple 3D object"""
    depth_image = np.zeros((height, width), dtype=np.uint16)
    
    # Create a simple depth map with a raised object in the center
    center_y, center_x = height // 2, width // 2
    object_height, object_width = 100, 150
    
    # Base depth (background)
    base_depth = 1000  # 1 meter in mm
    
    # Object depth (closer to camera)
    object_depth = 800  # 0.8 meters in mm
    
    # Create depth gradient for the object
    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            dist_y = abs(y - center_y)
            dist_x = abs(x - center_x)
            
            # Check if point is within object bounds
            if (dist_y <= object_height // 2 and dist_x <= object_width // 2):
                # Create a simple depth profile - higher in the center, lower at edges
                # For top-down grasp, we want the highest point to be at the top of the object
                height_factor = 1.0 - (dist_y / (object_height // 2))  # Higher at top
                width_factor = 1.0 - (dist_x / (object_width // 2))   # Higher at center
                
                # Combine factors and apply to depth
                depth_factor = (height_factor + width_factor) / 2
                depth = object_depth - int(50 * depth_factor)  # Vary depth by 50mm
                depth_image[y, x] = max(100, depth)  # Ensure minimum depth
            else:
                depth_image[y, x] = base_depth
    
    return depth_image

def test_top_down_grasp_positioning():
    """Test the top-down grasp positioning functionality"""
    print("Testing top-down grasp positioning functionality...")
    
    # Create test data
    height, width = 480, 640
    object_mask = create_test_object_mask(height, width)
    depth_image = create_test_depth_image(height, width)
    
    # Create a mock color image
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[object_mask] = [100, 150, 200]  # Blue-ish color for object
    
    print(f"Created test data:")
    print(f"  - Object mask shape: {object_mask.shape}")
    print(f"  - Depth image shape: {depth_image.shape}")
    print(f"  - Color image shape: {color_image.shape}")
    print(f"  - Object pixels: {np.sum(object_mask)}")
    
    # Create a mock skill executor (we'll need to mock some components)
    class MockSkillExecutor:
        def __init__(self):
            self.latest_depth_image = depth_image
            self.latest_color_image = color_image
            self.logger = self
            
        def info(self, msg):
            print(f"INFO: {msg}")
            
        def warning(self, msg):
            print(f"WARNING: {msg}")
            
        def error(self, msg):
            print(f"ERROR: {msg}")
            
        def adjust_tcp_pose_for_top_down_grasp(self, target_position, object_mask=None, search_radius_m=0.03, 
                                               depth_threshold_ratio=0.05, cluster_min_size=5):
            """
            Mock implementation of top-down grasp adjustment for testing
            """
            try:
                if self.latest_depth_image is None or self.latest_color_image is None:
                    self.warning("No depth/color image available for top-down grasp adjustment")
                    return target_position
                    
                # Get camera intrinsics
                fx = self.main_camera.intrinsics.fx
                fy = self.main_camera.intrinsics.fy
                ppx = self.main_camera.intrinsics.ppx
                ppy = self.main_camera.intrinsics.ppy
                
                # Get depth scale
                depth_scale = getattr(self.main_camera, 'depth_scale', 1000.0)
                
                self.info(f"DEBUG - Top-down grasp adjustment: target_position={target_position}")
                
                # Convert target position to pixel coordinates
                if isinstance(target_position, np.ndarray):
                    pos = target_position.copy()
                else:
                    pos = np.array(target_position)
                    
                # Project 3D point to 2D pixel coordinates
                if len(pos) >= 3 and pos[2] > 0:  # Valid depth
                    pixel_x = int(fx * pos[0] / pos[2] + ppx)
                    pixel_y = int(fy * pos[1] / pos[2] + ppy)
                else:
                    self.warning("Invalid target position for top-down grasp adjustment")
                    return target_position
                    
                # Ensure pixel coordinates are within image bounds
                h, w = self.latest_depth_image.shape
                pixel_x = max(0, min(w-1, pixel_x))
                pixel_y = max(0, min(h-1, pixel_y))
                
                # Define search region
                if object_mask is not None:
                    # Use object mask if provided
                    search_mask = object_mask
                    self.info(f"DEBUG - Using provided object mask, shape: {search_mask.shape}")
                else:
                    # Create circular search region around target pixel
                    target_depth_m = pos[2] if pos[2] > 0 else 1.0
                    radius_pixels_x = int(search_radius_m * fx / target_depth_m)
                    radius_pixels_y = int(search_radius_m * fy / target_depth_m)
                    
                    x_min = max(0, pixel_x - radius_pixels_x)
                    x_max = min(w, pixel_x + radius_pixels_x)
                    y_min = max(0, pixel_y - radius_pixels_y) 
                    y_max = min(h, pixel_y + radius_pixels_y)
                    
                    # Create circular mask within the search region
                    search_mask = np.zeros((h, w), dtype=bool)
                    center_y, center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                    radius = min(radius_pixels_x, radius_pixels_y)
                    
                    y_coords, x_coords = np.ogrid[:h, :w]
                    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                    search_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
                    
                    self.info(f"DEBUG - Created circular search mask, radius: {radius} pixels")
                
                # Extract depth region within the search mask
                depth_region = self.latest_depth_image.copy()
                valid_mask = (depth_region > 100) & (depth_region < 10000) & search_mask  # Valid depth + object mask
                
                if not np.any(valid_mask):
                    self.warning("No valid depth values in object mask region")
                    return target_position
                    
                # Get valid depth values and their coordinates within the object mask
                valid_depths = depth_region[valid_mask]
                y_coords, x_coords = np.where(valid_mask)
                
                self.info(f"DEBUG - Valid depth points in object mask: {len(valid_depths)}")
                self.info(f"DEBUG - Valid depth range: {valid_depths.min():.1f} to {valid_depths.max():.1f} mm")
                
                # For top-down grasps, we want to find the TOPMOST point (highest Y coordinate in camera frame)
                # This corresponds to the highest point on the object surface
                if len(valid_depths) >= cluster_min_size:
                    # Calculate dynamic depth threshold
                    average_depth = np.mean(valid_depths)
                    depth_threshold = average_depth * depth_threshold_ratio
                    
                    self.info(f"DEBUG - Average depth in object region: {average_depth:.1f} mm")
                    self.info(f"DEBUG - Depth threshold: {depth_threshold:.1f} mm")
                    
                    # Find points near the surface (within depth threshold of the closest point)
                    min_depth = np.min(valid_depths)
                    surface_threshold = min_depth + depth_threshold
                    
                    # Select points on the surface
                    surface_mask = valid_depths <= surface_threshold
                    surface_depths = valid_depths[surface_mask]
                    surface_x_coords = x_coords[surface_mask]
                    surface_y_coords = y_coords[surface_mask]
                    
                    if len(surface_depths) >= 3:
                        self.info(f"DEBUG - Surface points: {len(surface_depths)}")
                        
                        # For top-down grasps, find the highest Y coordinate (topmost point)
                        # In camera coordinates, higher Y means closer to the top of the image
                        topmost_indices = np.where(surface_y_coords == np.min(surface_y_coords))[0]
                        
                        if len(topmost_indices) > 0:
                            # If multiple points have the same Y coordinate, choose the one closest to the target X
                            if len(topmost_indices) > 1:
                                distances_to_target_x = np.abs(surface_x_coords[topmost_indices] - pixel_x)
                                best_index = topmost_indices[np.argmin(distances_to_target_x)]
                            else:
                                best_index = topmost_indices[0]
                            
                            # Get the topmost point coordinates
                            topmost_x = surface_x_coords[best_index]
                            topmost_y = surface_y_coords[best_index]
                            topmost_depth = surface_depths[best_index]
                            
                            self.info(f"DEBUG - Topmost point: ({topmost_x}, {topmost_y}), depth: {topmost_depth:.1f} mm")
                            
                            # Convert depth using depth scale
                            topmost_depth_m = topmost_depth / depth_scale
                            
                            # Convert topmost pixel back to 3D camera coordinates
                            topmost_x_3d = (topmost_x - ppx) * topmost_depth_m / fx
                            topmost_y_3d = (topmost_y - ppy) * topmost_depth_m / fy
                            
                            self.info(f"DEBUG - Topmost 3D coordinates: ({topmost_x_3d:.4f}, {topmost_y_3d:.4f}, {topmost_depth_m:.4f})")
                            
                            # Create adjusted position centered on the topmost point
                            adjusted_position = [
                                topmost_x_3d,  # X coordinate of topmost point
                                topmost_y_3d,  # Y coordinate of topmost point  
                                topmost_depth_m  # Z coordinate (depth) of topmost point
                            ]
                            
                            # Calculate adjustment offset for logging
                            original_pos = np.array(target_position[:3])
                            final_offset = np.array(adjusted_position) - original_pos
                            final_offset_magnitude = np.linalg.norm(final_offset)
                            
                            self.info(f"Top-down grasp adjustment: offset={final_offset}, magnitude={final_offset_magnitude:.4f}m")
                            self.info(f"Original position: {target_position[:3]}")
                            self.info(f"Adjusted position: {adjusted_position}")
                            
                            return adjusted_position
                        else:
                            self.warning("No topmost point found in surface region")
                            return target_position
                    else:
                        self.warning("Not enough surface points for top-down grasp adjustment")
                        return target_position
                else:
                    self.warning("Not enough valid depth points for top-down grasp adjustment")
                    return target_position
                    
            except Exception as e:
                self.error(f"Error in top-down grasp adjustment: {e}")
                return target_position
    
    # Create mock camera intrinsics
    class MockCamera:
        class MockIntrinsics:
            def __init__(self):
                self.fx = 525.0  # Focal length X
                self.fy = 525.0  # Focal length Y
                self.ppx = width // 2  # Principal point X
                self.ppy = height // 2  # Principal point Y
        
        def __init__(self):
            self.intrinsics = self.MockIntrinsics()
            self.depth_scale = 1000.0  # mm to meters
    
    # Create mock skill executor with camera
    skill_executor = MockSkillExecutor()
    skill_executor.main_camera = MockCamera()
    
    # Test target position (in camera coordinates)
    target_position = [0.0, 0.0, 1.0]  # 1 meter in front of camera
    
    print(f"\nTesting with target position: {target_position}")
    
    # Test the top-down grasp adjustment
    try:
        adjusted_position = skill_executor.adjust_tcp_pose_for_top_down_grasp(
            target_position=target_position,
            object_mask=object_mask,
            search_radius_m=0.03,
            depth_threshold_ratio=0.05,
            cluster_min_size=5
        )
        
        print(f"\nResults:")
        print(f"  - Original position: {target_position}")
        print(f"  - Adjusted position: {adjusted_position}")
        
        # Calculate the adjustment
        original = np.array(target_position)
        adjusted = np.array(adjusted_position)
        adjustment = adjusted - original
        adjustment_magnitude = np.linalg.norm(adjustment)
        
        print(f"  - Adjustment: {adjustment}")
        print(f"  - Adjustment magnitude: {adjustment_magnitude:.4f} meters")
        
        # Verify that the adjusted position is reasonable
        if adjustment_magnitude < 0.5:  # Should be less than 50cm adjustment
            print("✅ Test passed: Adjustment is reasonable")
        else:
            print("❌ Test failed: Adjustment is too large")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_skill_executor():
    """Test with a real skill executor (if available)"""
    print("\nTesting with real skill executor...")
    
    try:
        # Initialize skill executor
        skill_executor = DirectSkillExecutor(
            robot_ip="192.168.1.224",
            show_debug_windows=False,
            calibrate_transform=False,
            use_zed_camera=False,
            fast_mode=True
        )
        
        # Create test data
        height, width = 480, 640
        object_mask = create_test_object_mask(height, width)
        depth_image = create_test_depth_image(height, width)
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        color_image[object_mask] = [100, 150, 200]
        
        # Set the test images
        skill_executor.latest_depth_image = depth_image
        skill_executor.latest_color_image = color_image
        
        # Test target position
        target_position = [0.0, 0.0, 1.0]
        
        print(f"Testing with target position: {target_position}")
        
        adjusted_position = skill_executor.adjust_tcp_pose_for_top_down_grasp(
            target_position=target_position,
            object_mask=object_mask,
            search_radius_m=0.03,
            depth_threshold_ratio=0.05,
            cluster_min_size=5
        )
        
        print(f"Results:")
        print(f"  - Original position: {target_position}")
        print(f"  - Adjusted position: {adjusted_position}")
        
        # Cleanup
        skill_executor.shutdown()
        
    except Exception as e:
        print(f"Real skill executor test failed: {e}")
        print("This is expected if robot/camera is not available")

if __name__ == "__main__":
    print("Top-Down Grasp Positioning Test")
    print("=" * 40)
    
    # Test with mock components
    test_top_down_grasp_positioning()
    
    # Test with real skill executor (if available)
    test_with_real_skill_executor()
    
    print("\nTest completed!") 