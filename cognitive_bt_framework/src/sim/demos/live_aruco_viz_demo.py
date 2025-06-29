#!/usr/bin/env python3
"""
Updated Live ArUco Demo - Compatible with Unified Interface
"""

import numpy as np
import time
import threading
import sys
import cv2
from scipy.spatial.transform import Rotation

# Import the simplified ArUco detector and visualizer
from cognitive_bt_framework.src.vision.aruco_tf_v3 import StereoArucoDetector
from cognitive_bt_framework.src.sim.demos.aruco_frame_viz import ArucoFrameVisualizer


class UpdatedLiveArucoDemo:
    def __init__(self, urdf_path=None, robot_ip='192.168.1.224'):
        self.urdf_path = urdf_path
        self.robot_ip = robot_ip
        self.robot_planner = None
        self.detector = None
        self.visualizer = None
        self.running = False
        
        # Camera instances
        self.robot_camera = None
        self.zed_camera = None
        
        # Debug statistics
        self.detection_stats = {
            'total_frames': 0,
            'robot_detections': 0,
            'zed_detections': 0,
            'common_detections': 0,
            'transform_successes': 0
        }
        
        # Visualization update thread
        self.viz_thread = None
        
    def create_robot_transform_matrix(self, position, rotation):
        """Helper function to create 4x4 transformation matrix"""
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = position
        return transform
    
    def _get_example_robot_transform(self):
        """Get example robot transform"""
        example_transform = np.array([
            [0.0, -1.0, 0.0, 0.32],    # Camera position from your log
            [0.0, 0.0, -1.0, -0.036],    
            [1.0, 0.0, 0.0, 0.306],     
            [0.0, 0.0, 0.0, 1.0]      
        ])
        print("📍 Using example camera transform:")
        print(f"   Translation: [{example_transform[0,3]:.3f}, {example_transform[1,3]:.3f}, {example_transform[2,3]:.3f}]")
        return example_transform
        
    def initialize_robot_interface(self):
        """Initialize robot interface with better error handling"""
        if not self.robot_ip:
            print("🤖 No robot IP provided - using example transform")
            return self._get_example_robot_transform()
            
        try:
            from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner
            
            print(f"🤖 Connecting to robot at {self.robot_ip}...")
            self.robot_planner = CuRoboMotionPlanner(robot_ip=self.robot_ip)
            
            # Test connection by getting transform
            camera_position, camera_rotation = self.robot_planner.get_camera_transform()
            T_robot_base_to_camera_link = self.create_robot_transform_matrix(camera_position, camera_rotation)
            
            print(f"✅ Robot connected successfully!")
            print(f"📍 Camera position: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")
            print(f"🔄 Camera rotation (quat): [{camera_rotation.as_quat()[0]:.3f}, {camera_rotation.as_quat()[1]:.3f}, {camera_rotation.as_quat()[2]:.3f}, {camera_rotation.as_quat()[3]:.3f}]")
            
            return T_robot_base_to_camera_link
            
        except ImportError:
            print("❌ Robot interface module not available")
            return self._get_example_robot_transform()
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            print("🔄 Using example transform instead")
            return self._get_example_robot_transform()
    
    def initialize_cameras_and_detector(self, T_robot_base_to_camera_link):
        """Initialize cameras and simplified detector"""
        try:
            from cognitive_bt_framework.src.vision.zed_camera import Camera as ZedCam
            from cognitive_bt_framework.src.vision.realsense import Camera as RsCam
            
            print("📷 Initializing cameras...")
            
            # Create camera instances
            self.robot_camera = RsCam(width=640, height=480, fps=30, depth_averaging_frames=3, debug=False)
            self.zed_camera = ZedCam(width=640, height=480, fps=30, depth_averaging_frames=3, debug=False)
            
            # Start cameras
            robot_start = self.robot_camera.start()
            zed_start = self.zed_camera.start()
            
            print(f"   Robot camera: {'✅ Started' if robot_start else '❌ Failed'}")
            print(f"   ZED camera: {'✅ Started' if zed_start else '❌ Failed'}")
            
            if not (robot_start and zed_start):
                raise Exception("One or both cameras failed to start")
            
            print("✅ Cameras started successfully!")
            time.sleep(2)  # Stabilization time
            
            # Create simplified stereo ArUco detector
            self.detector = StereoArucoDetector(
                robot_camera=self.robot_camera,
                zed_camera=self.zed_camera,
                T_robot_base_to_camera_link=T_robot_base_to_camera_link,
                marker_size=0.05  # 5cm markers
            )
            
            print("✅ Simplified ArUco detector created!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize cameras/detector: {e}")
            return False
    
    def initialize_visualizer(self):
        """Initialize PyBullet visualizer"""
        try:
            print("🎯 Initializing PyBullet visualizer...")
            self.visualizer = ArucoFrameVisualizer(self.urdf_path, gui=True)
            print("✅ PyBullet visualizer ready!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize visualizer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_current_joint_positions(self):
        """Get current joint positions from robot"""
        if (self.robot_planner and 
            hasattr(self.robot_planner, 'arm') and 
            hasattr(self.robot_planner.arm, 'angles')):
            try:
                joint_angles = self.robot_planner.arm.angles
                if joint_angles is not None:
                    return list(joint_angles) if hasattr(joint_angles, '__iter__') else [joint_angles]
            except Exception:
                pass
        
        # Default joint positions
        return [0.0021, -1.2677, -0.0341, 0.8922, 0.0197, 1.5883, -0.0050]

    def update_robot_transform(self):
        """Update robot transform from current pose"""
        if not self.robot_planner:
            return False
            
        try:
            camera_position, camera_rotation = self.robot_planner.get_camera_transform()
            new_transform = self.create_robot_transform_matrix(camera_position, camera_rotation)
            
            if self.detector:
                self.detector.update_robot_transform(new_transform)
            
            return True
        except Exception as e:
            return False

    def get_frames_with_markers(self):
        """Get frames with marker visualization"""
        if not self.robot_camera or not self.zed_camera:
            return None, None, None
            
        # Get detection data from simplified detector
        robot_detection, zed_detection = self.detector.get_detection_data()
        
        if robot_detection is None or zed_detection is None:
            return None, None, None
        
        # Get raw frames
        robot_frames = self.robot_camera.get_frames()
        zed_frames = self.zed_camera.get_frames()
        
        if robot_frames is None or zed_frames is None:
            return None, None, None
            
        robot_color, _ = robot_frames
        zed_color, _ = zed_frames
        
        # Get camera matrices
        robot_K, robot_dist, zed_K, zed_dist = self.detector._get_camera_matrices()
        
        # Draw markers on images using the detector's method
        robot_output = self.detector.draw_markers_on_image(robot_color, robot_detection, robot_K, robot_dist, "Robot")
        zed_output = self.detector.draw_markers_on_image(zed_color, zed_detection, zed_K, zed_dist, "ZED")
        
        # Create detection summary
        common_markers = set(robot_detection['marker_poses'].keys()).intersection(set(zed_detection['marker_poses'].keys()))
        detection_data = {
            'robot_camera_markers': len(robot_detection['marker_poses']),
            'zed_camera_markers': len(zed_detection['marker_poses']),
            'common_markers': list(common_markers),
            'transform_available': len(common_markers) > 0
        }
        
        # Try to compute transformation
        if len(common_markers) > 0:
            transform = self.detector.compute_transformation()
            detection_data['transform_available'] = transform is not None
            if transform is not None:
                detection_data['transform_confidence'] = 0.8  # Mock confidence
        
        return robot_output, zed_output, detection_data

    def visualization_update_loop(self):
        """Visualization update loop with robot pose updates"""
        print("🎯 Starting visualization update loop...")
        
        last_update = 0
        last_robot_update = 0
        update_interval = 1.0  # Slower updates: 1 Hz to reduce visual confusion
        robot_update_interval = 2.0  # Even slower robot updates: 0.5 Hz
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update robot transform periodically
                if (current_time - last_robot_update >= robot_update_interval and 
                    self.robot_planner):
                    self.update_robot_transform()
                    last_robot_update = current_time
                
                # Update visualization
                if current_time - last_update >= update_interval:
                    if self.detector and self.visualizer:
                        # Get current joint positions
                        current_joints = self.get_current_joint_positions()
                        
                        # Update visualization with simplified detector
                        self.visualizer.visualize_aruco_transforms(self.detector, current_joints)
                    
                    last_update = current_time
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"⚠️  Visualization error: {e}")
                time.sleep(0.2)
    
    def run_updated_demo(self):
        """Run updated demo with simplified interface"""
        print("🚀 Updated Live ArUco Demo - Unified Interface")
        print("=" * 50)
        
        # Initialize robot interface
        T_robot_base_to_camera_link = self.initialize_robot_interface()
        
        # Initialize cameras and detector
        if not self.initialize_cameras_and_detector(T_robot_base_to_camera_link):
            print("❌ Camera/detector initialization failed")
            return False
        
        # Initialize visualizer
        if not self.initialize_visualizer():
            print("❌ Visualizer initialization failed")
            return False
        
        # Start visualization thread
        self.running = True
        self.viz_thread = threading.Thread(target=self.visualization_update_loop)
        self.viz_thread.daemon = True
        self.viz_thread.start()
        
        print("\n✅ All systems initialized successfully!")
        print("\n🎮 Updated Demo Controls:")
        print("Camera windows:")
        print("  'q': Quit demo")
        print("  't': Print current transform")
        print("  'c': Compute transformation")
        print("  'j': Print joint positions")
        print("  'd': Toggle debug")
        print("PyBullet window:")
        print("  Mouse: Rotate/zoom view")
        print("  'q': Quit")
        print("  'c': Compute transformation")
        print()
        print("🎯 The robot base should be at origin (0,0,0) with thick RGB axes")
        print("🤖 Robot pose updates automatically from current joint angles")
        print("📱 This demo uses the unified ArUco detector interface")
        print()
        
        try:
            self.run_detection_loop()
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
        finally:
            self.cleanup()
        
        return True
    
    def run_detection_loop(self):
        """Updated detection loop using simplified interface"""
        cv2.namedWindow("Robot Camera - Updated Demo", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("ZED Camera - Updated Demo", cv2.WINDOW_AUTOSIZE)
        
        print("🔍 Starting detection loop with unified interface...")
        
        frame_count = 0
        last_info = 0
        
        while self.running:
            frame_count += 1
            
            # Get frames with markers (compatibility wrapper)
            robot_img, zed_img, detection_data = self.get_frames_with_markers()
            
            if robot_img is None or zed_img is None:
                time.sleep(0.01)
                continue
            
            # Extract detection info
            robot_markers = detection_data.get('robot_camera_markers', 0)
            zed_markers = detection_data.get('zed_camera_markers', 0)
            common_markers = detection_data.get('common_markers', [])
            transform_available = detection_data.get('transform_available', False)
            
            # Status text
            status = f"Frame: {frame_count} | Robot: {robot_markers} | ZED: {zed_markers} | Common: {len(common_markers)}"
            cv2.putText(robot_img, status, (10, robot_img.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(zed_img, status, (10, zed_img.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Transform status
            transform_text = f"Transform: {'✅ Available' if transform_available else '❌ None'}"
            if transform_available:
                confidence = detection_data.get('transform_confidence', 0)
                transform_text += f" (conf: {confidence:.3f})"
            
            cv2.putText(robot_img, transform_text, (10, robot_img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if transform_available else (0, 0, 255), 2)
            cv2.putText(zed_img, transform_text, (10, zed_img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if transform_available else (0, 0, 255), 2)
            
            # Current joint positions
            joints = self.get_current_joint_positions()
            joint_text = f"Joints: [{joints[0]:.2f}, {joints[1]:.2f}, {joints[2]:.2f}...]"
            cv2.putText(robot_img, joint_text, (10, robot_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(zed_img, joint_text, (10, zed_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Interface indicator
            interface_text = "Interface: Unified ArUco Detector"
            cv2.putText(robot_img, interface_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(zed_img, interface_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Show images
            cv2.imshow("Robot Camera - Updated Demo", robot_img)
            cv2.imshow("ZED Camera - Updated Demo", zed_img)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('t'):
                self.print_current_transform()
            elif key == ord('c'):
                transform = self.detector.compute_transformation()
                if transform is not None:
                    print("✅ Transformation computed successfully")
                    print(f"Transform matrix:\n{transform}")
                    # Force immediate visualization refresh
                    if self.visualizer:
                        current_joints = self.get_current_joint_positions()
                        self.detector.clear_detection_cache()
                        self.visualizer.clear_all_frames()
                        self.visualizer.visualize_aruco_transforms(self.detector, current_joints)
                else:
                    print("❌ Failed to compute transformation")
            elif key == ord('j'):
                current_joints = self.get_current_joint_positions()
                print(f"\n🤖 Current joint positions: [{', '.join([f'{j:.3f}' for j in current_joints[:7]])}]")
            elif key == ord('d'):
                print("Debug mode toggle not available in current interface")
            
            # Periodic info
            current_time = time.time()
            if current_time - last_info >= 5.0:  # Every 5 seconds
                if transform_available:
                    print(f"🎯 Frame {frame_count}: Transform available")
                else:
                    print(f"📊 Frame {frame_count}: No transform | Robot: {robot_markers} | ZED: {zed_markers}")
                last_info = current_time
    
    def print_current_transform(self):
        """Print current transformation matrix"""
        if not self.detector:
            return
            
        transform = self.detector.get_transformation_matrix()
        
        print(f"\n🔧 CURRENT TRANSFORMATION (Unified Interface):")
        print("=" * 50)
        
        if transform is not None:
            pos = transform[:3, 3]
            print(f"✅ T_zed_to_robot_base:")
            print(f"   Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            print("   Full matrix:")
            print(transform)
        else:
            print(f"❌ T_zed_to_robot_base: Not available")
        
        print("=" * 50)
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        
        self.running = False
        
        if self.viz_thread and self.viz_thread.is_alive():
            self.viz_thread.join(timeout=1.0)
        
        if self.robot_camera:
            try:
                self.robot_camera.stop()
            except:
                pass
        
        if self.zed_camera:
            try:
                self.zed_camera.stop()
            except:
                pass
        
        if self.visualizer:
            try:
                self.visualizer.cleanup()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Updated Live ArUco Demo with Unified Interface")
    parser.add_argument("--urdf", help="Path to robot URDF file", 
                       default="/home/liam/installs/curobo/src/curobo/content/assets/robot/xarm7/xarm7.urdf")
    parser.add_argument("--robot-ip", help="Robot IP address", default='192.168.1.224')
    parser.add_argument("--no-robot", action="store_true", help="Run without robot")
    
    args = parser.parse_args()
    
    robot_ip = None if args.no_robot else args.robot_ip
    
    print("🔧 Updated Live ArUco Demo")
    print("Compatible with StereoArucoDetector!")
    print("Robot base frame fixed at origin (0,0,0)")
    print("=" * 50)
    
    demo = UpdatedLiveArucoDemo(urdf_path=args.urdf, robot_ip=robot_ip)
    success = demo.run_updated_demo()
    
    if success:
        print("✅ Demo completed successfully")
    else:
        print("❌ Demo failed")


if __name__ == "__main__":
    main()