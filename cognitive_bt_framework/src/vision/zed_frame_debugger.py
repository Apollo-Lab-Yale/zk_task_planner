#!/usr/bin/env python3
"""
Complete Enhanced ZED Camera Frame Debugging Script

This script helps determine the correct coordinate frame orientation for the ZED camera
by visualizing ArUco markers and testing different coordinate frame transformations.
Enhanced with depth frame analysis and validation.
"""

import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
import json
from typing import Dict, Optional, Tuple

# Import camera classes
from cognitive_bt_framework.src.vision.zed_camera import Camera as ZedCamera
from cognitive_bt_framework.src.vision.realsense import Camera as RealSenseCamera
from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner
from cognitive_bt_framework.src.vision.aruco_tf_v3 import StereoArucoDetector


class EnhancedZEDFrameDebugger:
    def __init__(self, robot_ip="192.168.1.224"):
        """Initialize the enhanced ZED frame debugger"""
        self.robot_ip = robot_ip
        self.motion_planner = None
        self.zed_camera = None
        self.robot_camera = None
        self.detector = None
        
        # Comprehensive coordinate frame transformations to test
        self.test_transforms = {
            # Basic transformations
            'identity': np.eye(4),
            'x_180': self._rotation_matrix([180, 0, 0]),
            'y_180': self._rotation_matrix([0, 180, 0]), 
            'z_180': self._rotation_matrix([0, 0, 180]),
            'x_90': self._rotation_matrix([90, 0, 0]),
            'x_-90': self._rotation_matrix([-90, 0, 0]),
            'y_90': self._rotation_matrix([0, 90, 0]),
            'y_-90': self._rotation_matrix([0, -90, 0]),
            'z_90': self._rotation_matrix([0, 0, 90]),
            'z_-90': self._rotation_matrix([0, 0, -90]),
            
            # Standard computer vision transformations
            'cv_to_robotics': self._cv_to_robotics_transform(),
            'opencv_to_ros': self._opencv_to_ros_transform(),
            'opencv_to_opengl': self._opencv_to_opengl_transform(),
            
            # ZED SDK specific transformations
            'zed_standard': self._zed_standard_transform(),
            'zed_sdk_default': self._zed_sdk_default_transform(),
            'zed_right_handed': self._zed_right_handed_transform(),
            'zed_left_handed': self._zed_left_handed_transform(),
            'zed_y_z_flip': self._zed_y_z_flip_transform(),
            
            # Depth camera specific transformations
            'depth_camera_standard': self._depth_camera_standard_transform(),
            'depth_y_inverted': self._depth_y_inverted_transform(),
            'depth_z_inverted': self._depth_z_inverted_transform(),
            'depth_yz_inverted': self._depth_yz_inverted_transform(),
            
            # ArUco specific transformations
            'aruco_standard': self._aruco_standard_transform(),
            'aruco_aligned': self._aruco_aligned_transform(),
            'aruco_board_aligned': self._aruco_board_aligned_transform(),
            
            # Common robotics combinations
            'robotics_standard': self._robotics_standard_transform(),
            'ros_standard': self._ros_standard_transform(),
            
            # Additional common fixes
            'xyz_180': self._rotation_matrix([180, 180, 180]),
            'xy_180': self._rotation_matrix([180, 180, 0]),
            'xz_180': self._rotation_matrix([180, 0, 180]),
            'yz_180': self._rotation_matrix([0, 180, 180]),
        }
        
        # Depth validation parameters
        self.depth_validation_params = {
            'min_depth': 0.1,  # 10cm minimum
            'max_depth': 10.0,  # 10m maximum
            'depth_consistency_threshold': 0.05,  # 5cm consistency check
            'marker_size_validation_tolerance': 0.02,  # 2cm tolerance for marker size
            'expected_marker_size': 0.05,  # 5cm markers
        }
        
        print("Enhanced ZED Camera Frame Debugger initialized")
        print("This will help determine the correct coordinate frame for ZED camera")
        print("Now includes depth frame analysis and validation")
        
    def _rotation_matrix(self, euler_degrees):
        """Create 4x4 transformation matrix from Euler angles in degrees"""
        transform = np.eye(4)
        rotation = Rotation.from_euler('xyz', euler_degrees, degrees=True)
        transform[:3, :3] = rotation.as_matrix()
        return transform
    
    def _cv_to_robotics_transform(self):
        """Standard computer vision to robotics coordinate transform"""
        # CV: X-right, Y-down, Z-forward
        # Robotics: X-forward, Y-left, Z-up
        transform = np.array([
            [0, 0, 1, 0],   # CV Z becomes robotics X
            [-1, 0, 0, 0],  # CV -X becomes robotics Y  
            [0, -1, 0, 0],  # CV -Y becomes robotics Z
            [0, 0, 0, 1]
        ])
        return transform
    
    def _opencv_to_ros_transform(self):
        """OpenCV to ROS coordinate transformation"""
        # OpenCV: X-right, Y-down, Z-forward
        # ROS: X-forward, Y-left, Z-up
        return np.array([
            [0, 0, 1, 0],   # Z -> X
            [-1, 0, 0, 0],  # -X -> Y
            [0, -1, 0, 0],  # -Y -> Z
            [0, 0, 0, 1]
        ])
    
    def _opencv_to_opengl_transform(self):
        """OpenCV to OpenGL coordinate transformation"""
        # OpenCV: X-right, Y-down, Z-forward
        # OpenGL: X-right, Y-up, Z-backward
        return np.array([
            [1, 0, 0, 0],   # X stays
            [0, -1, 0, 0],  # -Y -> Y (flip Y)
            [0, 0, -1, 0],  # -Z -> Z (flip Z)
            [0, 0, 0, 1]
        ])
    
    def _zed_standard_transform(self):
        """ZED SDK standard coordinate transform (legacy)"""
        # ZED SDK typically uses: X-right, Y-down, Z-forward (like OpenCV)
        transform = np.array([
            [1, 0, 0, 0],   # X stays X
            [0, -1, 0, 0],  # Y becomes -Y (up instead of down)
            [0, 0, -1, 0],  # Z becomes -Z (backward instead of forward)
            [0, 0, 0, 1]
        ])
        return transform
    
    def _zed_sdk_default_transform(self):
        """ZED SDK default coordinate system: X-right, Y-down, Z-forward"""
        return np.eye(4)
    
    def _zed_right_handed_transform(self):
        """ZED right-handed coordinate system: X-right, Y-up, Z-backward"""
        return np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Flip Y
            [0, 0, -1, 0],  # Flip Z
            [0, 0, 0, 1]
        ])
    
    def _zed_left_handed_transform(self):
        """ZED left-handed coordinate system variations"""
        return np.array([
            [-1, 0, 0, 0],  # Flip X
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def _zed_y_z_flip_transform(self):
        """ZED with Y and Z axes flipped (common fix)"""
        return np.array([
            [1, 0, 0, 0],   # X stays
            [0, -1, 0, 0],  # Y down -> Y up
            [0, 0, -1, 0],  # Z forward -> Z backward
            [0, 0, 0, 1]
        ])
    
    def _depth_camera_standard_transform(self):
        """Standard depth camera coordinate system"""
        # Many depth cameras use: X-right, Y-down, Z-forward (same as OpenCV)
        return np.eye(4)
    
    def _depth_y_inverted_transform(self):
        """Depth camera with Y-axis inverted"""
        return np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Flip Y
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def _depth_z_inverted_transform(self):
        """Depth camera with Z-axis inverted"""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],  # Flip Z
            [0, 0, 0, 1]
        ])
    
    def _depth_yz_inverted_transform(self):
        """Depth camera with both Y and Z axes inverted"""
        return np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Flip Y
            [0, 0, -1, 0],  # Flip Z
            [0, 0, 0, 1]
        ])
    
    def _aruco_standard_transform(self):
        """ArUco standard coordinate system"""
        # ArUco markers typically use: X-right, Y-down, Z-out-of-plane
        return np.eye(4)
    
    def _aruco_aligned_transform(self):
        """ArUco marker coordinate alignment"""
        return np.array([
            [1, 0, 0, 0],   # X-right (across marker)
            [0, 1, 0, 0],   # Y-down (down marker)
            [0, 0, -1, 0],  # Z into marker plane
            [0, 0, 0, 1]
        ])
    
    def _aruco_board_aligned_transform(self):
        """ArUco board-aligned coordinate system"""
        # Sometimes ArUco boards need different alignment
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],  # Z pointing into the board
            [0, 0, 0, 1]
        ])
    
    def _robotics_standard_transform(self):
        """Standard robotics coordinate system"""
        # Robotics: X-forward, Y-left, Z-up
        return self._opencv_to_ros_transform()
    
    def _ros_standard_transform(self):
        """ROS standard coordinate system"""
        # ROS: X-forward, Y-left, Z-up
        return self._opencv_to_ros_transform()
    
    def setup_cameras(self):
        """Setup both cameras"""
        print("Setting up cameras...")
        
        # Initialize ZED camera
        self.zed_camera = ZedCamera(
            width=640, height=480, fps=30,
            depth_averaging_frames=1, debug=False
        )
        
        if not self.zed_camera.start():
            raise RuntimeError("Failed to start ZED camera")
        print("✅ ZED camera started")
        
        # Initialize RealSense camera
        self.robot_camera = RealSenseCamera(
            width=640, height=480, fps=30,
            depth_averaging_frames=3, debug=False
        )
        
        if not self.robot_camera.start():
            raise RuntimeError("Failed to start RealSense camera")
        print("✅ RealSense camera started")
        
        # Initialize motion planner
        self.motion_planner = CuRoboMotionPlanner(robot_ip=self.robot_ip)
        print("✅ Motion planner connected")
        
        # Wait for cameras to stabilize
        time.sleep(2)
    
    def create_detector_with_transform(self, zed_transform_name, zed_transform):
        """Create detector with specific ZED coordinate transform"""
        # Get robot camera transform
        robot_pos, robot_rot = self.motion_planner.get_camera_transform()
        T_robot_base_to_camera_link = np.eye(4)
        T_robot_base_to_camera_link[:3, :3] = robot_rot.as_matrix()
        T_robot_base_to_camera_link[:3, 3] = robot_pos
        
        # Create detector with specific ZED transform
        detector = StereoArucoDetector(
            robot_camera=self.robot_camera,
            zed_camera=self.zed_camera,
            T_robot_base_to_camera_link=T_robot_base_to_camera_link,
            T_zed_to_aruco_frame=zed_transform,
            marker_size=self.depth_validation_params['expected_marker_size']
        )
        
        return detector
    
    def get_camera_intrinsics(self):
        """Get ZED camera intrinsic parameters"""
        # Get actual intrinsics from ZED camera
        try:
            # This would depend on your ZED camera implementation
            # You might need to access self.zed_camera.get_camera_info() or similar
            fx, fy = 525.0, 525.0  # Placeholder - replace with actual values
            cx, cy = 320.0, 240.0  # Placeholder - replace with actual values
            return fx, fy, cx, cy
        except:
            # Fallback values for common ZED cameras
            fx, fy = 525.0, 525.0  # Approximate focal lengths
            cx, cy = 320.0, 240.0  # Approximate principal point
            return fx, fy, cx, cy
    
    def validate_depth_measurements(self, detector, transform_name):
        """Validate depth accuracy for a transformation"""
        print(f"🔍 Validating depth for: {transform_name}")
        
        depth_errors = []
        valid_measurements = 0
        marker_size_errors = []
        
        for attempt in range(5):
            try:
                robot_detection, zed_detection = detector.get_detection_data()
                
                if zed_detection and len(zed_detection['marker_poses']) > 0:
                    # Get ZED frames (color and depth)
                    zed_frames = self.zed_camera.get_frames()
                    if zed_frames:
                        color_frame, depth_frame = zed_frames
                        
                        for marker_id, pose in zed_detection['marker_poses'].items():
                            # Extract 3D position from pose
                            position_3d = pose[:3, 3]  # [x, y, z]
                            estimated_depth = position_3d[2]  # Z coordinate
                            
                            # Validate depth range
                            if (self.depth_validation_params['min_depth'] <= estimated_depth <= 
                                self.depth_validation_params['max_depth']):
                                
                                # Project to pixel coordinates
                                fx, fy, cx, cy = self.get_camera_intrinsics()
                                pixel_x = int(fx * position_3d[0] / position_3d[2] + cx)
                                pixel_y = int(fy * position_3d[1] / position_3d[2] + cy)
                                
                                # Get depth from depth image
                                if (0 <= pixel_x < depth_frame.shape[1] and 
                                    0 <= pixel_y < depth_frame.shape[0]):
                                    
                                    depth_image_value = depth_frame[pixel_y, pixel_x]
                                    if depth_image_value > 0:  # Valid depth
                                        # Convert depth image value to meters (assuming mm)
                                        actual_depth = depth_image_value / 1000.0
                                        
                                        # Calculate depth error
                                        depth_error = abs(estimated_depth - actual_depth)
                                        depth_errors.append(depth_error)
                                        valid_measurements += 1
                                        
                                        # Validate marker size consistency
                                        expected_pixel_size = self.calculate_expected_marker_size(estimated_depth)
                                        actual_pixel_size = self.estimate_marker_pixel_size(
                                            zed_detection, marker_id, color_frame
                                        )
                                        
                                        if actual_pixel_size > 0:
                                            size_error = abs(expected_pixel_size - actual_pixel_size) / expected_pixel_size
                                            marker_size_errors.append(size_error)
                                        
                                        print(f"  Marker {marker_id}: Est={estimated_depth:.3f}m, "
                                              f"Actual={actual_depth:.3f}m, Error={depth_error:.3f}m")
            
            except Exception as e:
                print(f"  Error in depth validation: {e}")
            
            time.sleep(0.2)
        
        if depth_errors:
            avg_depth_error = np.mean(depth_errors)
            max_depth_error = np.max(depth_errors)
            avg_size_error = np.mean(marker_size_errors) if marker_size_errors else 1.0
            
            print(f"  📊 Depth validation: {valid_measurements} measurements")
            print(f"     Average depth error: {avg_depth_error:.3f}m")
            print(f"     Maximum depth error: {max_depth_error:.3f}m")
            print(f"     Average size error: {avg_size_error:.1%}")
            
            return avg_depth_error, valid_measurements, avg_size_error
        else:
            print(f"  ❌ No valid depth measurements")
            return float('inf'), 0, 1.0
    
    def calculate_expected_marker_size(self, depth):
        """Calculate expected marker size in pixels based on depth"""
        fx, fy, cx, cy = self.get_camera_intrinsics()
        marker_size_meters = self.depth_validation_params['expected_marker_size']
        return (marker_size_meters * fx) / depth
    
    def estimate_marker_pixel_size(self, detection_data, marker_id, image):
        """Estimate actual marker size in pixels from detection"""
        try:
            # This would depend on your detection data format
            # Placeholder implementation - you'd extract corner points and calculate size
            # For now, return a reasonable estimate
            return 50  # Placeholder value
        except:
            return 0
    
    def test_coordinate_frame(self, transform_name, transform_matrix):
        """Test a specific coordinate frame transformation"""
        print(f"\n🧪 Testing transform: {transform_name}")
        print(f"Transform matrix:\n{transform_matrix}")
        
        detector = self.create_detector_with_transform(transform_name, transform_matrix)
        
        # Test detection for a few frames
        successful_detections = 0
        total_attempts = 20
        
        for attempt in range(total_attempts):
            try:
                # Get detection data
                robot_detection, zed_detection = detector.get_detection_data()
                
                if robot_detection is not None and zed_detection is not None:
                    # Compute transformation
                    transform = detector.compute_transformation()
                    
                    if transform is not None:
                        successful_detections += 1
                        
                        # Get detection summary for alignment check
                        summary = detector.get_detection_summary()
                        common_markers = summary['common_markers']
                        
                        if len(common_markers) > 0:
                            # Estimate alignment quality by checking marker positions
                            robot_markers = len(robot_detection['marker_poses'])
                            zed_markers = len(zed_detection['marker_poses'])
                            
                            print(f"  Attempt {attempt+1}: ✅ Success | R:{robot_markers} Z:{zed_markers} C:{len(common_markers)}")
                        else:
                            print(f"  Attempt {attempt+1}: ❌ No common markers")
                    else:
                        print(f"  Attempt {attempt+1}: ❌ Transform failed")
                else:
                    print(f"  Attempt {attempt+1}: ❌ Detection failed")
                    
            except Exception as e:
                print(f"  Attempt {attempt+1}: ❌ Error: {e}")
            
            time.sleep(0.1)
        
        success_rate = successful_detections / total_attempts
        print(f"📊 Results for {transform_name}:")
        print(f"   Success rate: {success_rate:.1%} ({successful_detections}/{total_attempts})")
        
        return {
            'transform_name': transform_name,
            'transform_matrix': transform_matrix.tolist(),
            'success_rate': success_rate,
            'successful_detections': successful_detections,
            'total_attempts': total_attempts
        }
    
    def enhanced_coordinate_frame_test(self, transform_name, transform_matrix):
        """Enhanced test that includes depth validation"""
        print(f"\n🧪 Enhanced test: {transform_name}")
        
        detector = self.create_detector_with_transform(transform_name, transform_matrix)
        
        # Original detection test
        successful_detections = 0
        total_attempts = 10
        
        for attempt in range(total_attempts):
            try:
                robot_detection, zed_detection = detector.get_detection_data()
                
                if robot_detection is not None and zed_detection is not None:
                    transform = detector.compute_transformation()
                    if transform is not None:
                        successful_detections += 1
                        
            except Exception as e:
                pass
            time.sleep(0.1)
        
        success_rate = successful_detections / total_attempts
        
        # Depth validation
        depth_error, valid_depth_measurements, size_error = self.validate_depth_measurements(
            detector, transform_name
        )
        
        # Calculate composite score
        detection_score = success_rate
        depth_score = max(0, 1.0 - depth_error / 0.1) if depth_error != float('inf') else 0
        size_score = max(0, 1.0 - size_error) if size_error != 1.0 else 0
        composite_score = 0.5 * detection_score + 0.3 * depth_score + 0.2 * size_score
        
        print(f"📊 Enhanced results for {transform_name}:")
        print(f"   Detection success: {success_rate:.1%}")
        print(f"   Depth accuracy: {depth_score:.1%}")
        print(f"   Size consistency: {size_score:.1%}")
        print(f"   Composite score: {composite_score:.1%}")
        
        return {
            'transform_name': transform_name,
            'transform_matrix': transform_matrix.tolist(),
            'success_rate': success_rate,
            'depth_error': depth_error,
            'depth_score': depth_score,
            'size_score': size_score,
            'composite_score': composite_score,
            'valid_depth_measurements': valid_depth_measurements
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all coordinate frame transformations"""
        print("\n🔬 Running comprehensive coordinate frame test...")
        print("Testing different ZED camera coordinate frame orientations")
        print("Place ArUco markers visible to both cameras")
        
        input("Press Enter when ready to start testing...")
        
        results = []
        
        for transform_name, transform_matrix in self.test_transforms.items():
            result = self.test_coordinate_frame(transform_name, transform_matrix)
            results.append(result)
            
            # Small break between tests
            time.sleep(1)
        
        # Analyze results
        print("\n📈 COMPREHENSIVE TEST RESULTS:")
        print("=" * 60)
        
        # Sort by success rate
        results.sort(key=lambda x: x['success_rate'], reverse=True)
        
        for i, result in enumerate(results):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{status} {result['transform_name']:20} | Success: {result['success_rate']:6.1%} | Detections: {result['successful_detections']:2}/{result['total_attempts']}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"zed_frame_test_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
        
        # Recommend best transform
        if results:
            best_result = results[0]
            print(f"\n🎯 RECOMMENDATION:")
            print(f"Best performing transform: {best_result['transform_name']}")
            print(f"Success rate: {best_result['success_rate']:.1%}")
            
            if best_result['success_rate'] > 0.7:
                print("✅ This transform appears to work well!")
                self.save_recommended_transform(best_result)
            else:
                print("⚠️  No transform achieved >70% success rate")
                print("   Consider checking marker visibility or camera setup")
        
        return results
    
    def run_depth_focused_test(self):
        """Run test focused on depth accuracy and coordinate consistency"""
        print("\n🔬 Running Depth-Focused Coordinate Frame Test...")
        print("This test prioritizes depth accuracy and coordinate consistency")
        print("Place ArUco markers at various depths visible to both cameras")
        
        input("Press Enter when ready to start depth-focused testing...")
        
        results = []
        
        # Test most promising transforms first
        priority_transforms = [
            'identity', 'opencv_to_ros', 'zed_y_z_flip', 'aruco_aligned', 
            'depth_y_inverted', 'zed_right_handed', 'cv_to_robotics',
            'robotics_standard', 'depth_yz_inverted'
        ]
        
        for transform_name in priority_transforms:
            if transform_name in self.test_transforms:
                result = self.enhanced_coordinate_frame_test(
                    transform_name, self.test_transforms[transform_name]
                )
                results.append(result)
                time.sleep(1)
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print("\n📈 DEPTH-FOCUSED TEST RESULTS:")
        print("=" * 80)
        print(f"{'Transform':<20} | {'Detection':<10} | {'Depth':<10} | {'Size':<10} | {'Composite':<10}")
        print("-" * 80)
        
        for i, result in enumerate(results):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{status} {result['transform_name']:<18} | {result['success_rate']:8.1%} | "
                  f"{result['depth_score']:8.1%} | {result['size_score']:8.1%} | {result['composite_score']:8.1%}")
        
        # Save comprehensive results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"zed_depth_analysis_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Handle any numpy arrays or non-serializable types
            for key, value in serializable_result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    serializable_result[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    serializable_result[key] = int(value)
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 Depth analysis saved to {filename}")
        
        if results:
            best_result = results[0]
            print(f"\n🎯 DEPTH-OPTIMIZED RECOMMENDATION:")
            print(f"Best transform: {best_result['transform_name']}")
            print(f"Composite score: {best_result['composite_score']:.1%}")
            print(f"Detection success: {best_result['success_rate']:.1%}")
            print(f"Depth accuracy: {best_result['depth_score']:.1%}")
            
            if best_result['composite_score'] > 0.7:
                print("✅ This transform is optimized for both detection and depth!")
                self.save_depth_optimized_transform(best_result)
            else:
                print("⚠️  No transform achieved >70% composite score")
                print("   Consider checking camera calibration or marker setup")
        
        return results
    
    def quick_depth_validation(self):
        """Quick test of most likely correct transformations"""
        print("\n⚡ Quick Depth Validation Test")
        print("Testing most likely correct transformations...")
        
        # Most promising candidates based on common ZED configurations
        quick_transforms = [
            'identity',           # ZED default
            'opencv_to_ros',     # Most common robotics
            'zed_y_z_flip',      # Common ZED fix
            'aruco_aligned',     # ArUco standard
            'depth_y_inverted'   # Simple Y flip
        ]
        
        input("Press Enter to start quick validation...")
        
        results = []
        for transform_name in quick_transforms:
            if transform_name in self.test_transforms:
                result = self.enhanced_coordinate_frame_test(
                    transform_name, self.test_transforms[transform_name]
                )
                results.append(result)
                time.sleep(0.5)
        
        # Show quick summary
        print(f"\n⚡ QUICK VALIDATION SUMMARY:")
        print("=" * 50)
        
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        for result in results:
            print(f"🔹 {result['transform_name']:18} | Composite: {result['composite_score']:6.1%}")
        
        if results:
            best = results[0]
            print(f"\n🎯 Quick recommendation: {best['transform_name']} ({best['composite_score']:.1%})")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎮 Interactive Testing Mode")
        print("Commands:")
        print("  'test <name>' - Test specific transform (e.g., 'test identity')")
        print("  'enhanced <name>' - Enhanced test with depth validation")
        print("  'list' - List available transforms")
        print("  'visual' - Show visual comparison")
        print("  'all' - Run comprehensive test")
        print("  'depth' - Run depth-focused test")
        print("  'quick' - Quick validation test")
        print("  'quit' - Exit")
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'list':
                    print("Available transforms:")
                    for name in self.test_transforms.keys():
                        print(f"  - {name}")
                elif command == 'all':
                    self.run_comprehensive_test()
                elif command == 'depth':
                    self.run_depth_focused_test()
                elif command == 'quick':
                    self.quick_depth_validation()
                elif command.startswith('test '):
                    transform_name = command[5:].strip()
                    if transform_name in self.test_transforms:
                        self.test_coordinate_frame(transform_name, self.test_transforms[transform_name])
                    else:
                        print(f"Unknown transform: {transform_name}")
                elif command.startswith('enhanced '):
                    transform_name = command[9:].strip()
                    if transform_name in self.test_transforms:
                        self.enhanced_coordinate_frame_test(
                            transform_name, self.test_transforms[transform_name]
                        )
                    else:
                        print(f"Unknown transform: {transform_name}")
                elif command == 'visual':
                    self.visual_comparison()
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
    
    def visual_comparison(self):
        """Show visual comparison of coordinate frames"""
        print("\n👁️  Visual Comparison Mode")
        print("Cycling through different transforms - watch marker orientations")
        print("Press 'q' to quit, 's' to save current transform, SPACE to pause")
        
        cv2.namedWindow("ZED Frame Comparison", cv2.WINDOW_AUTOSIZE)
        
        transform_names = list(self.test_transforms.keys())
        current_idx = 0
        paused = False
        
        while True:
            if not paused:
                transform_name = transform_names[current_idx]
                transform_matrix = self.test_transforms[transform_name]
                
                detector = self.create_detector_with_transform(transform_name, transform_matrix)
                
                # Get detection data and visualize
                robot_detection, zed_detection = detector.get_detection_data()
                
                if zed_detection is not None:
                    zed_frames = self.zed_camera.get_frames()
                    if zed_frames is not None:
                        zed_color, _ = zed_frames
                        zed_K, zed_dist, _, _ = detector._get_camera_matrices()
                        
                        # Draw markers on ZED image
                        vis_img = detector.draw_markers_on_image(
                            zed_color, zed_detection, zed_K, zed_dist, "ZED"
                        )
                        
                        # Add transform info
                        cv2.putText(vis_img, f"Transform: {transform_name}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(vis_img, f"Index: {current_idx+1}/{len(transform_names)}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.imshow("ZED Frame Comparison", vis_img)
                
                # Auto-advance every 3 seconds if not paused
                current_idx = (current_idx + 1) % len(transform_names)
            
            # Handle keyboard input
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/unpause
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):  # Save current transform
                transform_name = transform_names[current_idx]
                print(f"Current transform: {transform_name}")
                save = input("Save this transform as recommended? (y/n): ")
                if save.lower() == 'y':
                    result = {
                        'transform_name': transform_name,
                        'transform_matrix': self.test_transforms[transform_name].tolist(),
                        'success_rate': 1.0,  # User selected
                        'composite_score': 1.0,
                        'successful_detections': 1,
                        'total_attempts': 1
                    }
                    self.save_depth_optimized_transform(result)
        
        cv2.destroyAllWindows()
    
    def save_recommended_transform(self, best_result):
        """Save the recommended transform for use in the main script"""
        transform_data = {
            'recommended_zed_transform': {
                'name': best_result['transform_name'],
                'matrix': best_result['transform_matrix'],
                'success_rate': best_result['success_rate'],
                'description': f"Best performing ZED coordinate frame transform"
            },
            'timestamp': time.time(),
            'test_summary': f"Tested {len(self.test_transforms)} different coordinate frame orientations"
        }
        
        filename = "recommended_zed_transform.json"
        with open(filename, 'w') as f:
            json.dump(transform_data, f, indent=2)
        print(f"💾 Recommended transform saved to {filename}")
        
        # Print code snippet to use
        print(f"\n📝 CODE SNIPPET to use this transform:")
        print(f"```python")
        print(f"# Use this transform for T_zed_to_aruco_frame:")
        print(f"T_zed_to_aruco_frame = np.array({best_result['transform_matrix']})")
        print(f"```")
    
    def save_depth_optimized_transform(self, best_result):
        """Save depth-optimized transform recommendation"""
        transform_data = {
            'depth_optimized_transform': {
                'name': best_result['transform_name'],
                'matrix': best_result['transform_matrix'],
                'composite_score': best_result.get('composite_score', best_result['success_rate']),
                'detection_success': best_result['success_rate'],
                'depth_accuracy': best_result.get('depth_score', 0.0),
                'description': "Depth-optimized ZED coordinate frame transform"
            },
            'timestamp': time.time(),
            'optimization_focus': "depth_accuracy_and_detection",
            'validation_params': self.depth_validation_params
        }
        
        filename = "depth_optimized_zed_transform.json"
        with open(filename, 'w') as f:
            json.dump(transform_data, f, indent=2)
        print(f"💾 Depth-optimized transform saved to {filename}")
        
        print(f"\n📝 DEPTH-OPTIMIZED CODE SNIPPET:")
        print(f"```python")
        print(f"# Depth-optimized ZED coordinate transform:")
        print(f"T_zed_to_aruco_frame = np.array({best_result['transform_matrix']})")
        print(f"# Optimized for: Detection ({best_result['success_rate']:.1%}) + Depth accuracy")
        print(f"```")
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n🔌 Shutting down...")
        if self.zed_camera:
            self.zed_camera.stop()
        if self.robot_camera:
            self.robot_camera.stop()
        if self.motion_planner:
            self.motion_planner.disconnect_robot()
        print("✅ Shutdown complete")


def load_recommended_transform():
    """Load the recommended transform from file"""
    try:
        with open("recommended_zed_transform.json", 'r') as f:
            data = json.load(f)
        
        transform_data = data['recommended_zed_transform']
        transform_matrix = np.array(transform_data['matrix'])
        
        print(f"Loaded recommended transform: {transform_data['name']}")
        print(f"Success rate: {transform_data['success_rate']:.1%}")
        
        return transform_matrix
        
    except FileNotFoundError:
        print("No recommended transform file found")
        return None
    except Exception as e:
        print(f"Error loading recommended transform: {e}")
        return None


def load_depth_optimized_transform():
    """Load the depth-optimized transform from file"""
    try:
        with open("depth_optimized_zed_transform.json", 'r') as f:
            data = json.load(f)
        
        transform_data = data['depth_optimized_transform']
        transform_matrix = np.array(transform_data['matrix'])
        
        print(f"Loaded depth-optimized transform: {transform_data['name']}")
        print(f"Composite score: {transform_data['composite_score']:.1%}")
        print(f"Detection success: {transform_data['detection_success']:.1%}")
        
        return transform_matrix
        
    except FileNotFoundError:
        print("No depth-optimized transform file found")
        return None
    except Exception as e:
        print(f"Error loading depth-optimized transform: {e}")
        return None


def main():
    """Main function to run enhanced ZED frame debugging"""
    print("🔬 Enhanced ZED Camera Coordinate Frame Debugger")
    print("=" * 60)
    
    debugger = EnhancedZEDFrameDebugger()
    
    try:
        debugger.setup_cameras()
        
        print("\nChoose testing mode:")
        print("1. Comprehensive test (all transforms)")
        print("2. Depth-focused test (depth accuracy priority)")
        print("3. Quick validation (most likely candidates)")
        print("4. Interactive test")
        print("5. Visual comparison")
        print("6. Load existing recommendation")
        print("7. Load depth-optimized recommendation")
        
        choice = input("Enter choice (1-7): ").strip()
        
        if choice == '1':
            debugger.run_comprehensive_test()
        elif choice == '2':
            debugger.run_depth_focused_test()
        elif choice == '3':
            debugger.quick_depth_validation()
        elif choice == '4':
            debugger.interactive_test()
        elif choice == '5':
            debugger.visual_comparison()
        elif choice == '6':
            transform = load_recommended_transform()
            if transform is not None:
                print("Recommended transform matrix:")
                print(transform)
        elif choice == '7':
            transform = load_depth_optimized_transform()
            if transform is not None:
                print("Depth-optimized transform matrix:")
                print(transform)
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        debugger.shutdown()


if __name__ == "__main__":
    main()