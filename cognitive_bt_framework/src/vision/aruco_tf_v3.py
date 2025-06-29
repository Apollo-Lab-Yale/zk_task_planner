import cv2
import numpy as np
import time
from typing import Optional, Dict, Tuple
from scipy.spatial.transform import Rotation


class StereoArucoDetector:
    """
    Simplified ArUco detector for computing transformation between two cameras.
    
    Mathematical Framework:
    Robot side: T_robot_base^aruco_marker = T_robot_base^camera_link * T_camera_link^aruco_frame * T_aruco_frame^aruco_marker
    ZED side:   T_zed^aruco_marker = T_zed^aruco_frame * T_aruco_frame^aruco_marker  
    Final:      T_zed^robot_base = T_zed^aruco_marker * (T_robot_base^aruco_marker)^(-1)
    """
    
    def __init__(self, 
                 robot_camera,
                 zed_camera,
                 T_robot_base_to_camera_link: np.ndarray,
                 marker_size: float = 0.05,
                 T_camera_link_to_aruco_frame: Optional[np.ndarray] = None,
                 T_zed_to_aruco_frame: Optional[np.ndarray] = None):
        """
        Initialize the stereo ArUco detector.
        
        Args:
            robot_camera: Robot-mounted camera instance
            zed_camera: ZED camera instance  
            T_robot_base_to_camera_link: 4x4 transform from robot base to camera link (from forward kinematics)
            marker_size: Physical size of ArUco marker in meters
            T_camera_link_to_aruco_frame: 4x4 transform from camera link to ArUco frame (default: identity)
            T_zed_to_aruco_frame: 4x4 transform from ZED to ArUco frame (default: identity)
        """
        self.robot_camera = robot_camera
        self.zed_camera = zed_camera
        self.marker_size = marker_size
        
        # Core transforms
        self.T_robot_base_to_camera_link = T_robot_base_to_camera_link.copy()
        self.T_camera_link_to_aruco_frame = T_camera_link_to_aruco_frame.copy() if T_camera_link_to_aruco_frame is not None else np.eye(4)
        self.T_zed_to_aruco_frame = T_zed_to_aruco_frame.copy() if T_zed_to_aruco_frame is not None else np.eye(4)
        
        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        
        # Result
        self.T_zed_to_robot_base = None
        
    def _get_camera_matrices(self):
        """Get camera intrinsic matrices from both cameras."""
        # Robot camera
        robot_intrinsics = self.robot_camera.intrinsics
        robot_K = np.array([
            [robot_intrinsics.fx, 0, robot_intrinsics.ppx],
            [0, robot_intrinsics.fy, robot_intrinsics.ppy],
            [0, 0, 1]
        ])
        robot_dist = np.array(robot_intrinsics.coeffs)
        
        # ZED camera  
        zed_intrinsics = self.zed_camera.intrinsics
        zed_K = np.array([
            [zed_intrinsics.fx, 0, zed_intrinsics.ppx],
            [0, zed_intrinsics.fy, zed_intrinsics.ppy],
            [0, 0, 1]
        ])
        zed_dist = np.array(zed_intrinsics.coeffs)
        
        return robot_K, robot_dist, zed_K, zed_dist
    
    def _detect_markers(self, image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Dict:
        """Detect ArUco markers and estimate their poses."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        detection_data = {
            'corners': corners,
            'ids': ids,
            'marker_poses': {},
            'rvecs': [],
            'tvecs': []
        }
        
        if ids is not None and len(ids) > 0:
            # Define 3D object points for marker (in ArUco frame)
            half_size = self.marker_size / 2
            object_points = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0], 
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float32)
            
            rvecs = []
            tvecs = []
            
            for i, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                
                # Solve PnP to get T_aruco_frame^aruco_marker
                success, rvec, tvec = cv2.solvePnP(
                    object_points, corners[i], camera_matrix, dist_coeffs
                )
                
                if success:
                    # Convert to transformation matrix
                    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
                    T_aruco_frame_to_aruco_marker = np.eye(4)
                    T_aruco_frame_to_aruco_marker[:3, :3] = R
                    T_aruco_frame_to_aruco_marker[:3, 3] = tvec.flatten()
                    
                    detection_data['marker_poses'][marker_id] = T_aruco_frame_to_aruco_marker
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                else:
                    rvecs.append(np.zeros((3, 1)))
                    tvecs.append(np.zeros((3, 1)))
            
            detection_data['rvecs'] = rvecs
            detection_data['tvecs'] = tvecs
        
        return detection_data
    
    def compute_transformation(self) -> Optional[np.ndarray]:
        """
        Compute T_zed^robot_base using detected ArUco markers.
        
        Returns:
            4x4 transformation matrix from ZED to robot base, or None if failed
        """
        # Get frames from both cameras
        robot_frames = self.robot_camera.get_frames()
        zed_frames = self.zed_camera.get_frames()
        
        if robot_frames is None or zed_frames is None:
            return None
            
        robot_color, _ = robot_frames
        zed_color, _ = zed_frames
        
        # Get camera matrices
        robot_K, robot_dist, zed_K, zed_dist = self._get_camera_matrices()
        
        # Detect markers in both cameras
        robot_detection = self._detect_markers(robot_color, robot_K, robot_dist)
        zed_detection = self._detect_markers(zed_color, zed_K, zed_dist)
        
        # Find common markers
        common_markers = set(robot_detection['marker_poses'].keys()).intersection(set(zed_detection['marker_poses'].keys()))
        
        if not common_markers:
            return None
            
        # Use first common marker for transformation
        marker_id = next(iter(common_markers))
        
        # Get T_aruco_frame^aruco_marker from both cameras
        T_aruco_frame_to_aruco_marker_robot = robot_detection['marker_poses'][marker_id]
        T_aruco_frame_to_aruco_marker_zed = zed_detection['marker_poses'][marker_id]
        
        # Apply mathematical chain
        
        # Robot side: T_robot_base^aruco_marker = T_robot_base^camera_link * T_camera_link^aruco_frame * T_aruco_frame^aruco_marker
        T_robot_base_to_aruco_marker = (
            self.T_robot_base_to_camera_link @ 
            self.T_camera_link_to_aruco_frame @ 
            T_aruco_frame_to_aruco_marker_robot
        )
        
        # ZED side: T_zed^aruco_marker = T_zed^aruco_frame * T_aruco_frame^aruco_marker
        T_zed_to_aruco_marker = (
            self.T_zed_to_aruco_frame @ 
            T_aruco_frame_to_aruco_marker_zed
        )
        
        # Final: T_zed^robot_base = T_zed^aruco_marker * (T_robot_base^aruco_marker)^(-1)
        try:
            T_aruco_marker_to_robot_base = np.linalg.inv(T_robot_base_to_aruco_marker)
            T_zed_to_robot_base = T_zed_to_aruco_marker @ T_aruco_marker_to_robot_base
            
            # Store result
            self.T_zed_to_robot_base = T_zed_to_robot_base
            return T_zed_to_robot_base
            
        except np.linalg.LinAlgError:
            return None
    
    def get_detection_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Get detection data from both cameras for visualization.
        
        Returns:
            Tuple of (robot_detection_data, zed_detection_data)
        """
        # Get frames from both cameras
        robot_frames = self.robot_camera.get_frames()
        zed_frames = self.zed_camera.get_frames()
        
        if robot_frames is None or zed_frames is None:
            return None, None
            
        robot_color, _ = robot_frames
        zed_color, _ = zed_frames
        
        # Get camera matrices
        robot_K, robot_dist, zed_K, zed_dist = self._get_camera_matrices()
        
        # Detect markers in both cameras
        robot_detection = self._detect_markers(robot_color, robot_K, robot_dist)
        zed_detection = self._detect_markers(zed_color, zed_K, zed_dist)
        
        return robot_detection, zed_detection
    
    def draw_markers_on_image(self, image: np.ndarray, detection_data: Dict, 
                             camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                             camera_name: str = "") -> np.ndarray:
        """
        Draw detected ArUco markers on image with coordinate axes.
        
        Args:
            image: Input image
            detection_data: Detection data from _detect_markers()
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            camera_name: Name to display on image
            
        Returns:
            Image with markers drawn
        """
        output_image = image.copy()
        
        # Add camera name
        if camera_name:
            cv2.putText(output_image, f"{camera_name} Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Check if we have detections
        if (detection_data['ids'] is not None and 
            len(detection_data['ids']) > 0 and 
            len(detection_data['corners']) > 0):
            
            # Draw marker outlines and IDs using OpenCV ArUco function
            cv2.aruco.drawDetectedMarkers(output_image, detection_data['corners'], detection_data['ids'])
            
            # Draw 3D coordinate axes for each marker
            for i, marker_id in enumerate(detection_data['ids'].flatten()):
                if i < len(detection_data['rvecs']) and i < len(detection_data['tvecs']):
                    rvec = detection_data['rvecs'][i]
                    tvec = detection_data['tvecs'][i]
                    
                    # Draw coordinate axes
                    try:
                        cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
                                        rvec, tvec, self.marker_size, 3)
                    except Exception as e:
                        # If axis drawing fails, skip it
                        pass
                    
                    # Add text with marker info
                    if marker_id in detection_data['marker_poses']:
                        transform = detection_data['marker_poses'][marker_id]
                        translation = transform[:3, 3]
                        distance = np.linalg.norm(translation)
                        
                        # Get marker center for text placement
                        if i < len(detection_data['corners']):
                            center = np.mean(detection_data['corners'][i][0], axis=0).astype(int)
                            
                            text1 = f"ID:{marker_id} Dist:{distance:.2f}m"
                            text2 = f"Pos:({translation[0]:.2f},{translation[1]:.2f},{translation[2]:.2f})"
                            
                            cv2.putText(output_image, text1, (center[0], center[1] - 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            cv2.putText(output_image, text2, (center[0], center[1] - 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return output_image
    
    def transform_point(self, point_zed: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a 3D point from ZED coordinates to robot base coordinates.
        
        Args:
            point_zed: 3D point [x, y, z] in ZED frame
            
        Returns:
            3D point [x, y, z] in robot base frame, or None if no transform available
        """
        if self.T_zed_to_robot_base is None:
            return None
            
        # Convert to homogeneous coordinates and transform
        point_homo = np.append(point_zed, 1.0)
        transformed = self.T_zed_to_robot_base @ point_homo
        return transformed[:3]
    
    def get_detection_summary(self) -> Dict:
        """
        Get a summary of current detection status.
        
        Returns:
            Dictionary with detection statistics
        """
        robot_detection, zed_detection = self.get_detection_data()
        
        if robot_detection is None or zed_detection is None:
            return {
                'robot_markers': 0,
                'zed_markers': 0,
                'common_markers': [],
                'transform_available': False
            }
        
        common_markers = set(robot_detection['marker_poses'].keys()).intersection(set(zed_detection['marker_poses'].keys()))
        
        return {
            'robot_markers': len(robot_detection['marker_poses']),
            'zed_markers': len(zed_detection['marker_poses']),
            'common_markers': list(common_markers),
            'transform_available': len(common_markers) > 0
        }
    
    def get_transformation_matrix(self) -> Optional[np.ndarray]:
        """Get the current ZED to robot base transformation matrix."""
        return self.T_zed_to_robot_base.copy() if self.T_zed_to_robot_base is not None else None
    
    def update_robot_transform(self, T_robot_base_to_camera_link: np.ndarray):
        """Update the robot base to camera link transform (e.g., from new robot pose)."""
        self.T_robot_base_to_camera_link = T_robot_base_to_camera_link.copy()
    
    def run_calibration(self, num_samples: int = 10, display: bool = True) -> bool:
        """
        Run calibration process to compute transformation.
        
        Args:
            num_samples: Number of successful detections to average over
            display: Whether to show camera feeds during calibration
            
        Returns:
            True if calibration successful, False otherwise
        """
        print(f"Starting calibration... Need {num_samples} successful detections")
        
        valid_transforms = []
        
        if display:
            cv2.namedWindow("Robot Camera", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ZED Camera", cv2.WINDOW_AUTOSIZE)
        
        try:
            while len(valid_transforms) < num_samples:
                # Compute transformation
                transform = self.compute_transformation()
                
                if transform is not None:
                    valid_transforms.append(transform)
                    print(f"Valid detection {len(valid_transforms)}/{num_samples}")
                
                # Display if requested
                if display:
                    robot_frames = self.robot_camera.get_frames()
                    zed_frames = self.zed_camera.get_frames()
                    
                    if robot_frames is not None and zed_frames is not None:
                        robot_color, _ = robot_frames
                        zed_color, _ = zed_frames
                        
                        # Add status text
                        status = f"Detections: {len(valid_transforms)}/{num_samples}"
                        cv2.putText(robot_color, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(zed_color, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow("Robot Camera", robot_color)
                        cv2.imshow("ZED Camera", zed_color)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # ESC
                            break
                
        except KeyboardInterrupt:
            print("\nCalibration interrupted")
            
        finally:
            if display:
                cv2.destroyAllWindows()
        
        if len(valid_transforms) >= num_samples:
            # Average the transforms (this is a simplification - proper averaging of SE(3) is more complex)
            avg_transform = np.mean(valid_transforms, axis=0)
            self.T_zed_to_robot_base = avg_transform
            
            print(f"Calibration successful! Averaged {len(valid_transforms)} detections")
            print("ZED to Robot Base transformation matrix:")
            print(self.T_zed_to_robot_base)
            return True
        else:
            print(f"Calibration failed. Only got {len(valid_transforms)}/{num_samples} valid detections")
            return False


    def run_live_detection(self, display: bool = True) -> None:
        """
        Run continuous live detection and transformation computation.
        
        Args:
            display: Whether to show camera feeds
        """
        if display:
            cv2.namedWindow("Robot Camera", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ZED Camera", cv2.WINDOW_AUTOSIZE)
            
            print("Live detection controls:")
            print("  'q' or ESC: Quit")
            print("  't': Print current transformation matrix")
            print("  'c': Compute and update transformation")
            print("  SPACE: Print detection status")
        
        try:
            while True:
                # Get current transformation
                transform = self.compute_transformation()
                
                if display:
                    # Get and display frames
                    robot_frames = self.robot_camera.get_frames()
                    zed_frames = self.zed_camera.get_frames()
                    
                    if robot_frames is not None and zed_frames is not None:
                        robot_color, _ = robot_frames
                        zed_color, _ = zed_frames
                        
                        # Add status text
                        status = "Transform: VALID" if transform is not None else "Transform: INVALID"
                        color = (0, 255, 0) if transform is not None else (0, 0, 255)
                        
                        cv2.putText(robot_color, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(zed_color, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        cv2.imshow("Robot Camera", robot_color)
                        cv2.imshow("ZED Camera", zed_color)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # ESC
                            break
                        elif key == ord('t'):
                            if self.T_zed_to_robot_base is not None:
                                print("\n=== ZED TO ROBOT BASE TRANSFORMATION ===")
                                print(self.T_zed_to_robot_base)
                                print("=" * 45)
                            else:
                                print("No transformation matrix available")
                        elif key == ord('c'):
                            if transform is not None:
                                print("Transformation updated successfully")
                            else:
                                print("Failed to compute transformation")
                        elif key == ord(' '):
                            print(f"\nTransform available: {transform is not None}")
                            if transform is not None:
                                print("Current transformation matrix:")
                                print(self.T_zed_to_robot_base)
                else:
                    if transform is not None:
                        print("Transform computed successfully")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nLive detection interrupted")
        finally:
            if display:
                cv2.destroyAllWindows()


# Full example with camera and robot interface imports
if __name__ == "__main__":
    import time
    import json
    from cognitive_bt_framework.src.vision.zed_camera import Camera as ZedCam
    from cognitive_bt_framework.src.vision.realsense import Camera as RsCam
    # Import the robot interface class (assuming it's in the same directory or properly installed)
    from cognitive_bt_framework.src.robot_interface.xarm_curobo_interface import CuRoboMotionPlanner
    
    def create_robot_transform_matrix(position, rotation):
        """
        Helper function to create 4x4 transformation matrix from position and rotation.
        
        Args:
            position: [x, y, z] position vector
            rotation: scipy.spatial.transform.Rotation object
            
        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = position
        return transform
    
    # Robot configuration
    robot_ip = "192.168.1.224"  # Replace with your robot's IP address
    
    print("Initializing robot interface...")
    
    # Initialize robot interface (uncomment when robot interface is available)
    try:
        robot_planner = CuRoboMotionPlanner(robot_ip=robot_ip)
        
        # Get the current robot base to camera link transform
        camera_position, camera_rotation = robot_planner.get_camera_transform()
        T_robot_base_to_camera_link = create_robot_transform_matrix(camera_position, camera_rotation)
        
        print(f"Retrieved robot transform from interface:")
        print(f"Camera position: {camera_position}")
        print(f"Camera rotation (quaternion): {camera_rotation.as_quat()}")
        print(f"Transform matrix:\n{T_robot_base_to_camera_link}")
        
    except Exception as e:
        print(f"Failed to initialize robot interface: {e}")
        print("Using example transform instead")
        T_robot_base_to_camera_link = np.array([
            [0.0, -1.0, 0.0, 0.5],    # Camera link position/orientation on robot
            [0.0, 0.0, -1.0, 0.0],    
            [1.0, 0.0, 0.0, 1.2],     
            [0.0, 0.0, 0.0, 1.0]      
        ])
        robot_planner = None
    
    # Create two camera instances
    robot_camera = RsCam(  # Robot-mounted camera
        width=640,
        height=480,
        fps=30,
        depth_averaging_frames=3,
        debug=True
    )
    
    zed_camera = ZedCam(   # ZED camera
        width=640,
        height=480,
        fps=30,
        depth_averaging_frames=3,
        debug=True
    )
    
    # Start both cameras
    if robot_camera.start() and zed_camera.start():
        print("Both cameras started successfully!")
        
        # Wait for cameras to stabilize
        time.sleep(2)
        
        # Create simplified stereo ArUco detector
        detector = StereoArucoDetector(
            robot_camera=robot_camera,
            zed_camera=zed_camera,
            T_robot_base_to_camera_link=T_robot_base_to_camera_link,
            marker_size=0.05  # 5cm markers
        )
        
        print("Simplified ArUco detector created!")
        print("\nSimplified Detection Controls:")
        print("  Press 't' to see ZED to robot base transformation matrix")
        print("  Press 'c' to compute/update transformation")
        print("  Press 'u' to update robot transform from current robot pose")
        print("  Press 'r' to run calibration mode")
        print("  Press SPACE for status info")
        print("  Press 'q' or ESC to quit")
        print()
        
        # Define a function to update robot transform during runtime
        def update_robot_transform():
            """Update the robot transform from the current robot pose"""
            if robot_planner is not None:
                try:
                    # Get updated transform from robot
                    camera_position, camera_rotation = robot_planner.get_camera_transform()
                    new_transform = create_robot_transform_matrix(camera_position, camera_rotation)
                    
                    # Update the ArUco detector
                    detector.update_robot_transform(new_transform)
                    
                    print(f"Updated robot transform:")
                    print(f"Camera position: {camera_position}")
                    print(f"Camera rotation (quaternion): {camera_rotation.as_quat()}")
                    print("Transform updated successfully!")
                    
                except Exception as e:
                    print(f"Failed to update robot transform: {e}")
            else:
                print("Robot interface not available - cannot update transform")
        
        # Enhanced detection loop
        def run_enhanced_detection():
            """Run detection loop with enhanced controls"""
            cv2.namedWindow("Robot Camera - Simplified ArUco", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ZED Camera - Simplified ArUco", cv2.WINDOW_AUTOSIZE)
            
            try:
                while True:
                    # Get detection data and visualize markers
                    robot_detection, zed_detection = detector.get_detection_data()
                    
                    if robot_detection is not None and zed_detection is not None:
                        # Get raw frames
                        robot_frames = robot_camera.get_frames()
                        zed_frames = zed_camera.get_frames()
                        
                        if robot_frames is not None and zed_frames is not None:
                            robot_color, _ = robot_frames
                            zed_color, _ = zed_frames
                            
                            # Get camera matrices
                            robot_K, robot_dist, zed_K, zed_dist = detector._get_camera_matrices()
                            
                            # Draw markers on images
                            robot_color = detector.draw_markers_on_image(robot_color, robot_detection, robot_K, robot_dist, "Robot")
                            zed_color = detector.draw_markers_on_image(zed_color, zed_detection, zed_K, zed_dist, "ZED")
                            
                            # Try to compute transformation
                            transform = detector.compute_transformation()
                    else:
                        # Get frames without marker visualization
                        robot_frames = robot_camera.get_frames()
                        zed_frames = zed_camera.get_frames()
                        
                        if robot_frames is None or zed_frames is None:
                            continue
                        
                        robot_color, _ = robot_frames
                        zed_color, _ = zed_frames
                        transform = None
                    
                    # Add status display
                    status = "Transform: VALID" if transform is not None else "Transform: INVALID"
                    color = (0, 255, 0) if transform is not None else (0, 0, 255)
                    
                    robot_status = f"Robot Interface: {'Connected' if robot_planner is not None else 'Not Available'}"
                    
                    cv2.putText(robot_color, status, (10, robot_color.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(robot_color, robot_status, (10, robot_color.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    cv2.putText(zed_color, status, (10, zed_color.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(zed_color, robot_status, (10, zed_color.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    
                    cv2.imshow("Robot Camera - Simplified ArUco", robot_color)
                    cv2.imshow("ZED Camera - Simplified ArUco", zed_color)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('u'):  # Update robot transform
                        update_robot_transform()
                    elif key == ord('t'):
                        if detector.T_zed_to_robot_base is not None:
                            print(f"\n=== ZED TO ROBOT BASE TRANSFORMATION MATRIX ===")
                            print("T_zed_to_robot_base:")
                            print(detector.T_zed_to_robot_base)
                            print("=" * 55)
                        else:
                            print("No ZED to robot base transformation matrix available")
                    elif key == ord('c'):
                        if transform is not None:
                            print("Transformation computed and updated successfully")
                            print("Current transform:")
                            print(detector.T_zed_to_robot_base)
                        else:
                            print("Failed to compute transformation - no common markers detected")
                    elif key == ord('r'):
                        print("Starting calibration mode...")
                        success = detector.run_calibration(num_samples=5, display=True)
                        if success:
                            print("Calibration completed successfully!")
                        else:
                            print("Calibration failed!")
                    elif key == ord('v'):
                        test_marker_visualization()
                    elif key == ord('s'):
                        if detector.T_zed_to_robot_base is not None:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"simplified_zed_to_robot_transform_{timestamp}.json"
                            transform_data = {
                                'timestamp': timestamp,
                                'T_zed_to_robot_base': detector.T_zed_to_robot_base.tolist(),
                                'T_robot_base_to_camera_link': detector.T_robot_base_to_camera_link.tolist()
                            }
                            with open(filename, 'w') as f:
                                json.dump(transform_data, f, indent=2)
                            print(f"Saved transformation to {filename}")
                        else:
                            print("No valid transformation to save")
                    elif key == ord(' '):  # Space bar
                        summary = detector.get_detection_summary()
                        print(f"\n=== DETECTION STATUS ===")
                        print(f"Robot markers: {summary['robot_markers']}")
                        print(f"ZED markers: {summary['zed_markers']}")
                        print(f"Common markers: {summary['common_markers']}")
                        print(f"Transform available: {summary['transform_available']}")
                        print(f"Robot interface: {'Connected' if robot_planner is not None else 'Not Available'}")
                        if detector.T_zed_to_robot_base is not None:
                            print("Current transformation matrix:")
                            print(detector.T_zed_to_robot_base)
                        print("=" * 30)
                
            except KeyboardInterrupt:
                print("\nDetection loop interrupted")
            
            finally:
                cv2.destroyAllWindows()
        
        try:
            # Run enhanced detection loop
            run_enhanced_detection()
        finally:
            # Clean up
            robot_camera.stop()
            zed_camera.stop()
            if robot_planner is not None:
                # robot_planner.disconnect_robot()  # Uncomment when using real robot
                pass
    else:
        print("Failed to start one or both cameras")