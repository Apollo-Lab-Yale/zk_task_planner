import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time
import json
from scipy.spatial.transform import Rotation


class StereoArucoDetector:
    """
    A class for detecting ArUco markers using two cameras and computing the transformation
    from ZED camera to robot base frame using ArUco markers as intermediate reference frames.
    
    Process:
    1. Robot side: T_robot_base_to_aruco_marker = T_robot_base_to_camera_link * T_camera_link_to_aruco_frame * T_aruco_frame_to_aruco_marker
    2. ZED side: T_zed_to_aruco_marker = T_zed_to_aruco_frame * T_aruco_frame_to_aruco_marker
    3. Final: T_zed_to_robot_base = T_zed_to_aruco_marker * (T_robot_base_to_aruco_marker)^-1
    """
    
    def __init__(self, 
                 robot_camera_instance,  # Robot-mounted camera instance
                 zed_camera_instance,    # ZED camera instance
                 T_robot_base_to_camera_link: Optional[np.ndarray] = None,  # Forward kinematics transform
                 T_camera_link_to_aruco_frame: Optional[np.ndarray] = None,  # Camera link to ArUco frame alignment
                 T_zed_to_aruco_frame: Optional[np.ndarray] = None,  # ZED to ArUco frame alignment
                 marker_size: float = 0.05,  # marker size in meters
                 dictionary_type: int = cv2.aruco.DICT_6X6_250,
                 min_markers_for_transform: int = 1,  # Now we need at least 1 common marker
                 debug: bool = False,
                 show_axes: bool = True):  # New parameter for axis visualization
        """
        Initialize the stereo ArUco detector following the principled coordinate frame approach.
        
        Args:
            robot_camera_instance: Instance of Camera class (robot-mounted camera)
            zed_camera_instance: Instance of Camera class (ZED camera)
            T_robot_base_to_camera_link: 4x4 transform from robot base to camera link (forward kinematics)
            T_camera_link_to_aruco_frame: 4x4 transform from camera link to ArUco coordinate frame
            T_zed_to_aruco_frame: 4x4 transform from ZED to ArUco coordinate frame
            marker_size: Physical size of the ArUco marker in meters
            dictionary_type: OpenCV ArUco dictionary type
            min_markers_for_transform: Minimum number of common markers needed for transformation
            debug: Enable debug output
            show_axes: Whether to draw 3D coordinate axes on detected markers
        """
        self.robot_camera = robot_camera_instance  # Robot-mounted camera
        self.zed_camera = zed_camera_instance      # ZED camera
        self.marker_size = marker_size
        self.min_markers_for_transform = min_markers_for_transform
        self.debug = debug
        self.show_axes = show_axes  # Control axes visualization
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera intrinsics (will be populated from camera instances)
        self.robot_camera_matrix = None
        self.robot_camera_dist_coeffs = None
        self.zed_camera_matrix = None
        self.zed_camera_dist_coeffs = None
        
        # Coordinate frame transformations
        self.T_robot_base_to_camera_link = T_robot_base_to_camera_link.copy() if T_robot_base_to_camera_link is not None else None
        self.T_camera_link_to_aruco_frame = T_camera_link_to_aruco_frame.copy() if T_camera_link_to_aruco_frame is not None else np.eye(4)
        self.T_zed_to_aruco_frame = T_zed_to_aruco_frame.copy() if T_zed_to_aruco_frame is not None else np.eye(4)
        
        # Computed transformations
        self.T_zed_to_robot_base = None  # Main result: ZED to robot base
        self.last_transform_timestamp = None
        self.transform_confidence = 0.0
        self.last_common_markers = []
        
        # Intermediate results for debugging
        self.T_robot_base_to_aruco_marker = None
        self.T_zed_to_aruco_marker = None
        
        if self.debug:
            print("Debug: Initialized StereoArucoDetector with coordinate frame approach")
            if self.T_robot_base_to_camera_link is not None:
                print("Debug: T_robot_base_to_camera_link provided:")
                print(self.T_robot_base_to_camera_link)
            print("Debug: T_camera_link_to_aruco_frame:")
            print(self.T_camera_link_to_aruco_frame)
            print("Debug: T_zed_to_aruco_frame:")
            print(self.T_zed_to_aruco_frame)
            print(f"Debug: Axes visualization: {'ON' if self.show_axes else 'OFF'}")
    
    def toggle_axes_visualization(self):
        """Toggle the display of 3D coordinate axes on markers."""
        self.show_axes = not self.show_axes
        status = "ON" if self.show_axes else "OFF"
        print(f"Axes visualization: {status}")
        return self.show_axes
    
    def set_robot_base_to_camera_link_transform(self, transform: np.ndarray):
        """
        Set or update the robot base to camera link transformation matrix (from forward kinematics).
        
        Args:
            transform: 4x4 transformation matrix from robot base to camera link
        """
        if transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        
        self.T_robot_base_to_camera_link = transform.copy()
        
        if self.debug:
            print("Debug: T_robot_base_to_camera_link updated:")
            print(self.T_robot_base_to_camera_link)
    
    def set_camera_link_to_aruco_frame_transform(self, transform: np.ndarray):
        """
        Set the camera link to ArUco frame alignment transformation.
        
        Args:
            transform: 4x4 transformation matrix from camera link to ArUco frame
        """
        if transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        
        self.T_camera_link_to_aruco_frame = transform.copy()
        
        if self.debug:
            print("Debug: T_camera_link_to_aruco_frame updated:")
            print(self.T_camera_link_to_aruco_frame)
    
    def set_zed_to_aruco_frame_transform(self, transform: np.ndarray):
        """
        Set the ZED to ArUco frame alignment transformation.
        
        Args:
            transform: 4x4 transformation matrix from ZED to ArUco frame
        """
        if transform.shape != (4, 4):
            raise ValueError("Transform must be a 4x4 matrix")
        
        self.T_zed_to_aruco_frame = transform.copy()
        
        if self.debug:
            print("Debug: T_zed_to_aruco_frame updated:")
            print(self.T_zed_to_aruco_frame)
    
    def get_zed_to_robot_base_transform(self) -> Optional[np.ndarray]:
        """Get the main result: ZED to robot base transformation matrix."""
        return self.T_zed_to_robot_base.copy() if self.T_zed_to_robot_base is not None else None
    
    def get_all_transforms(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get all transformation matrices.
        
        Returns:
            Dictionary with all available transforms
        """
        return {
            'T_robot_base_to_camera_link': self.T_robot_base_to_camera_link.copy() if self.T_robot_base_to_camera_link is not None else None,
            'T_camera_link_to_aruco_frame': self.T_camera_link_to_aruco_frame.copy(),
            'T_zed_to_aruco_frame': self.T_zed_to_aruco_frame.copy(),
            'T_robot_base_to_aruco_marker': self.T_robot_base_to_aruco_marker.copy() if self.T_robot_base_to_aruco_marker is not None else None,
            'T_zed_to_aruco_marker': self.T_zed_to_aruco_marker.copy() if self.T_zed_to_aruco_marker is not None else None,
            'T_zed_to_robot_base': self.T_zed_to_robot_base.copy() if self.T_zed_to_robot_base is not None else None
        }
    
    def transform_point_zed_to_robot_base(self, point_3d_zed: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a 3D point from ZED coordinate system to robot base coordinate system.
        
        Args:
            point_3d_zed: 3D point in ZED coordinates (x, y, z)
            
        Returns:
            3D point in robot base coordinates or None if no transform available
        """
        if self.T_zed_to_robot_base is None:
            return None
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_3d_zed, 1.0)
        
        # Apply transformation
        transformed_point = self.T_zed_to_robot_base @ point_homogeneous
        
        # Return 3D coordinates
        return transformed_point[:3]
    
    def transform_point_robot_camera_to_robot_base(self, point_3d_robot_cam: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a 3D point from robot camera coordinate system to robot base coordinate system.
        
        Args:
            point_3d_robot_cam: 3D point in robot camera coordinates (x, y, z)
            
        Returns:
            3D point in robot base coordinates or None if no transform available
        """
        if self.T_robot_base_to_camera_link is None:
            return None
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_3d_robot_cam, 1.0)
        
        # Apply transformation
        transformed_point = self.T_robot_base_to_camera_link @ point_homogeneous
        
        # Return 3D coordinates
        return transformed_point[:3]

    def _update_camera_parameters(self):
        """Update camera parameters from both camera instances."""
        robot_camera_ready = False
        zed_camera_ready = False
        
        # Update robot camera parameters
        if self.robot_camera.intrinsics is not None:
            intrinsics_robot = self.robot_camera.intrinsics
            self.robot_camera_matrix = np.array([
                [intrinsics_robot.fx, 0, intrinsics_robot.ppx],
                [0, intrinsics_robot.fy, intrinsics_robot.ppy],
                [0, 0, 1]
            ])
            self.robot_camera_dist_coeffs = np.array(intrinsics_robot.coeffs)
            robot_camera_ready = True
            
        # Update ZED camera parameters
        if self.zed_camera.intrinsics is not None:
            intrinsics_zed = self.zed_camera.intrinsics
            self.zed_camera_matrix = np.array([
                [intrinsics_zed.fx, 0, intrinsics_zed.ppx],
                [0, intrinsics_zed.fy, intrinsics_zed.ppy],
                [0, 0, 1]
            ])
            self.zed_camera_dist_coeffs = np.array(intrinsics_zed.coeffs)
            zed_camera_ready = True
            
        return robot_camera_ready and zed_camera_ready
        
    def is_ready(self) -> bool:
        """
        Check if both cameras are ready.
        
        Returns:
            bool: True if both cameras are ready, False otherwise
        """
        return (self.robot_camera._running and self.robot_camera.intrinsics is not None and 
                self.robot_camera.depth_scale is not None and
                self.zed_camera._running and self.zed_camera.intrinsics is not None and 
                self.zed_camera.depth_scale is not None)
    
    def get_frames_both_cameras(self, use_averaging: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                                         Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aligned color and depth frames from both cameras.
        
        Args:
            use_averaging: Whether to use depth averaging
            
        Returns:
            Tuple[robot_color, robot_depth, zed_color, zed_depth]: Color and depth images from both cameras
        """
        if not self.is_ready():
            return None, None, None, None
            
        frames_robot = self.robot_camera.get_frames(use_averaging=use_averaging)
        frames_zed = self.zed_camera.get_frames(use_averaging=use_averaging)
        
        if frames_robot is None or frames_zed is None:
            return None, None, None, None
            
        robot_color, robot_depth = frames_robot
        zed_color, zed_depth = frames_zed
        
        return robot_color, robot_depth, zed_color, zed_depth
    
    def detect_markers_single_camera(self, color_image: np.ndarray, camera_matrix: np.ndarray, 
                            dist_coeffs: np.ndarray, depth_image: np.ndarray, 
                            depth_scale: float) -> Dict:
        # Convert BGR to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detection_data = {
            'corners': corners,
            'ids': ids,
            'rejected': rejected,
            'markers': {},
            'marker_to_camera_transforms': {},  # T_marker_to_camera (from solvePnP)
            'camera_to_marker_transforms': {}   # T_camera_to_marker (inverse of above)
        }
        
        if ids is not None and len(ids) > 0:
            # Estimate poses (these give T_marker_to_camera from cv2.solvePnP)
            rvecs, tvecs = self._estimate_pose_single_camera(corners, camera_matrix, dist_coeffs)
            
            # Get 3D positions using depth (in camera frame)
            positions_3d = self._get_3d_position_single_camera(corners, depth_image, camera_matrix, depth_scale)
            
            # Store marker data
            for i, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                
                # Get rvec and tvec from solvePnP
                rvec = rvecs[i].flatten()
                tvec = tvecs[i].flatten()
                
                if self.debug:
                    print(f"Debug: Marker {marker_id} - rvec: {rvec}, tvec: {tvec}")
                    print(f"Debug: Marker {marker_id} distance from camera: {np.linalg.norm(tvec):.3f}m")
                
                # Convert rotation vector to rotation matrix
                R = Rotation.from_rotvec(rvec).as_matrix()
                
                # cv2.solvePnP gives us T_marker_to_camera (marker coordinates to camera coordinates)
                T_marker_to_camera = np.eye(4)
                T_marker_to_camera[:3, :3] = R
                T_marker_to_camera[:3, 3] = tvec
                
                # For the mathematical chain, we need T_camera_to_marker
                T_camera_to_marker = np.linalg.inv(T_marker_to_camera)
                
                if self.debug:
                    print(f"Debug: Marker {marker_id} T_marker_to_camera (from solvePnP):")
                    print(T_marker_to_camera)
                    print(f"Debug: Marker {marker_id} T_camera_to_marker (inverse for chain):")
                    print(T_camera_to_marker)
                    print(f"Debug: Marker position in camera frame: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f})")
                    
                    # Verify they are actually inverses
                    identity_check = T_marker_to_camera @ T_camera_to_marker
                    print(f"Debug: Verification - T_marker_to_camera @ T_camera_to_marker should be identity:")
                    print(f"Max error from identity: {np.max(np.abs(identity_check - np.eye(4))):.6f}")
                
                detection_data['markers'][marker_id] = {
                    'corners': corners[i][0].tolist(),
                    'position_3d_camera_frame': positions_3d[i],
                    'distance': np.linalg.norm(positions_3d[i]),
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'marker_position_in_camera': tvec.tolist()  # Clear indication
                }
                
                # Store both transforms for clarity
                detection_data['marker_to_camera_transforms'][marker_id] = T_marker_to_camera
                detection_data['camera_to_marker_transforms'][marker_id] = T_camera_to_marker
        
        return detection_data
    
    def _estimate_pose_single_camera(self, corners: List, camera_matrix: np.ndarray, 
                                   dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pose for a single camera (gives T_aruco_frame_to_aruco_marker)."""
        rvecs = []
        tvecs = []
        
        for i, corner in enumerate(corners):
            # Define 3D object points for the marker in ArUco frame
            half_size = self.marker_size / 2
            object_points = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float32)
            
            # Solve PnP to get pose (this gives T_aruco_frame_to_aruco_marker)
            success, rvec, tvec = cv2.solvePnP(
                object_points, corner, camera_matrix, dist_coeffs
            )
            
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
                if self.debug:
                    print(f"Debug: Marker {i} pose estimation successful - rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")
            else:
                rvecs.append(np.zeros((3, 1)))
                tvecs.append(np.zeros((3, 1)))
                if self.debug:
                    print(f"Debug: Marker {i} pose estimation FAILED - using zero pose")
        
        return np.array(rvecs), np.array(tvecs)
    
    def _get_3d_position_single_camera(self, corners: List, depth_image: np.ndarray, 
                                     camera_matrix: np.ndarray, depth_scale: float) -> List[Tuple[float, float, float]]:
        """Get 3D positions in camera frame for a single camera."""
        positions = []
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        ppx, ppy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        for corner in corners:
            # Get marker center
            center = np.mean(corner[0], axis=0).astype(int)
            cx, cy = center
            
            # Ensure coordinates are within image bounds
            if 0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]:
                depth_value = depth_image[cy, cx]
                
                if depth_value > 0:  # Valid depth
                    z = depth_value * depth_scale
                    x = (cx - ppx) * z / fx
                    y = (cy - ppy) * z / fy
                    positions.append((x, y, z))
                else:
                    positions.append((0, 0, 0))
            else:
                positions.append((0, 0, 0))
                
        return positions
    
    def compute_zed_to_robot_base_transform(self, robot_detection: Dict, zed_detection: Dict) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Compute transformation from ZED to robot base using detected ArUco marker poses.
        
        CORRECT Mathematical chain:
        1. T_robot_base_to_camera_link: From robot kinematics (STATIC)
        2. From solvePnP: T_marker_to_camera (marker pose in camera frame)
        3. For chain: T_camera_to_marker = inv(T_marker_to_camera)
        4. T_robot_base_to_marker = T_robot_base_to_camera_link * T_camera_to_marker_robot
        5. T_zed_to_marker = T_camera_to_marker_zed
        6. T_zed_to_robot_base = T_zed_to_marker * inv(T_robot_base_to_marker)
        """
        # Check if we have the required robot base to camera link transform
        if self.T_robot_base_to_camera_link is None:
            if self.debug:
                print("Debug: Cannot compute transform - T_robot_base_to_camera_link not set")
            return None, 0.0, []
        
        # Find common markers
        robot_markers = set(robot_detection['markers'].keys())
        zed_markers = set(zed_detection['markers'].keys())
        common_markers = list(robot_markers.intersection(zed_markers))
        
        if self.debug:
            print(f"Debug: Robot camera has markers: {list(robot_markers)}")
            print(f"Debug: ZED camera has markers: {list(zed_markers)}")
            print(f"Debug: Common markers found: {common_markers}")
        
        if len(common_markers) < self.min_markers_for_transform:
            if self.debug:
                print(f"Debug: Insufficient markers. Need {self.min_markers_for_transform}, have {len(common_markers)}")
            return None, 0.0, common_markers
        
        # Use the first common marker
        marker_id = common_markers[0]
        
        if self.debug:
            print(f"Debug: Using marker {marker_id} for transformation computation")
        
        # Get the transforms we need for the mathematical chain
        T_marker_to_camera_robot = robot_detection['marker_to_camera_transforms'][marker_id]
        T_marker_to_camera_zed = zed_detection['marker_to_camera_transforms'][marker_id]
        
        # For the chain, we need T_camera_to_marker (the inverses)
        T_camera_to_marker_robot = robot_detection['camera_to_marker_transforms'][marker_id]
        T_camera_to_marker_zed = zed_detection['camera_to_marker_transforms'][marker_id]
        
        if self.debug:
            print(f"Debug: Robot camera T_marker_to_camera (from solvePnP):")
            print(T_marker_to_camera_robot)
            print(f"Debug: Robot camera T_camera_to_marker (inverse for chain):")
            print(T_camera_to_marker_robot)
            print(f"Debug: ZED camera T_marker_to_camera (from solvePnP):")
            print(T_marker_to_camera_zed)
            print(f"Debug: ZED camera T_camera_to_marker (inverse for chain):")
            print(T_camera_to_marker_zed)
            
            # Verify the distances make sense from the original solvePnP results
            robot_tvec = T_marker_to_camera_robot[:3, 3]
            zed_tvec = T_marker_to_camera_zed[:3, 3]
            robot_distance = np.linalg.norm(robot_tvec)
            zed_distance = np.linalg.norm(zed_tvec)
            print(f"Debug: Distances from solvePnP - Robot: {robot_distance:.3f}m, ZED: {zed_distance:.3f}m")
        
        # Apply the CORRECT mathematical chain
        # Robot side: T_robot_base_to_marker = T_robot_base_to_camera_link * T_camera_to_marker_robot
        T_robot_base_to_marker = self.T_robot_base_to_camera_link @ T_camera_to_marker_robot
        
        # ZED side: T_zed_to_marker = T_camera_to_marker_zed
        T_zed_to_marker = T_camera_to_marker_zed
        
        # Compute final transform: T_zed_to_robot_base = T_zed_to_marker * inv(T_robot_base_to_marker)
        try:
            T_marker_to_robot_base = np.linalg.inv(T_robot_base_to_marker)
            T_zed_to_robot_base = T_zed_to_marker @ T_marker_to_robot_base
        except np.linalg.LinAlgError as e:
            if self.debug:
                print(f"Debug: Failed to invert T_robot_base_to_marker: {e}")
            return None, 0.0, common_markers
        
        # Store intermediate results for debugging
        self.T_robot_base_to_aruco_marker = T_robot_base_to_marker
        self.T_zed_to_aruco_marker = T_zed_to_marker
        
        if self.debug:
            print(f"Debug: CORRECT Mathematical chain applied:")
            print(f"T_robot_base_to_camera_link (from robot kinematics):")
            print(self.T_robot_base_to_camera_link)
            print(f"T_camera_to_marker_robot (inverse of solvePnP result):")
            print(T_camera_to_marker_robot)
            print(f"T_robot_base_to_marker = T_robot_base_to_camera_link * T_camera_to_marker_robot:")
            print(T_robot_base_to_marker)
            print(f"T_camera_to_marker_zed (inverse of solvePnP result):")
            print(T_camera_to_marker_zed)
            print(f"T_zed_to_marker = T_camera_to_marker_zed:")
            print(T_zed_to_marker)
            print(f"T_zed_to_robot_base (final result):")
            print(T_zed_to_robot_base)
        
        # Compute confidence based on marker detection quality
        robot_distance = robot_detection['markers'][marker_id]['distance']
        zed_distance = zed_detection['markers'][marker_id]['distance']
        
        # Higher confidence for moderate distances (around 1-2m)
        distance_confidence = (np.exp(-abs(robot_distance - 1.5) / 2.0) * 
                            np.exp(-abs(zed_distance - 1.5) / 2.0))
        
        # Consider marker size in image (larger markers generally more reliable)
        robot_corners = np.array(robot_detection['markers'][marker_id]['corners'])
        zed_corners = np.array(zed_detection['markers'][marker_id]['corners'])
        
        robot_marker_size = np.linalg.norm(robot_corners[0] - robot_corners[2])
        zed_marker_size = np.linalg.norm(zed_corners[0] - zed_corners[2])
        
        size_confidence = np.exp(-abs(robot_marker_size - 50) / 50) * np.exp(-abs(zed_marker_size - 50) / 50)
        
        # Check pose validity (non-identity, reasonable translation)
        pose_confidence = 1.0
        robot_translation_norm = np.linalg.norm(T_marker_to_camera_robot[:3, 3])
        zed_translation_norm = np.linalg.norm(T_marker_to_camera_zed[:3, 3])
        
        if robot_translation_norm < 0.01 or zed_translation_norm < 0.01:
            pose_confidence *= 0.1  # Very low confidence for near-zero translations
        elif np.allclose(T_marker_to_camera_robot, np.eye(4)) or np.allclose(T_marker_to_camera_zed, np.eye(4)):
            pose_confidence *= 0.1  # Very low confidence for identity matrices
        
        confidence = distance_confidence * size_confidence * pose_confidence
        confidence = np.clip(confidence, 0.0, 1.0)
        
        if self.debug:
            print(f"Debug: Confidence components:")
            print(f"  Distance: {distance_confidence:.3f}")
            print(f"  Size: {size_confidence:.3f}")
            print(f"  Pose validity: {pose_confidence:.3f}")
            print(f"  Final confidence: {confidence:.3f}")
        
        return T_zed_to_robot_base, confidence, common_markers
    
    def set_debug(self, debug: bool):
        """Toggle debug output on/off."""
        self.debug = debug
        if debug:
            print("Debug mode: ON")
        else:
            print("Debug mode: OFF")
    
    def set_min_markers_for_transform(self, min_markers: int):
        """
        Set the minimum number of markers required for transformation.
        
        Args:
            min_markers: Minimum markers needed (typically 1 for this approach)
        """
        self.min_markers_for_transform = min_markers
        if self.debug:
            print(f"Debug: Minimum markers for transform set to {min_markers}")
        else:
            print(f"Minimum markers for transform set to {min_markers}")
    
    def process_stereo_frame(self, use_averaging: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Process frames from both cameras and compute ZED to robot base transformation.
        
        Args:
            use_averaging: Whether to use depth averaging
            
        Returns:
            Tuple[robot_output_image, zed_output_image, stereo_data]: Processed images and detection data
        """
        robot_color, robot_depth, zed_color, zed_depth = self.get_frames_both_cameras(use_averaging=use_averaging)
        
        if any(img is None for img in [robot_color, robot_depth, zed_color, zed_depth]):
            return None, None, None
        
        if not self._update_camera_parameters():
            return None, None, None
        
        # Detect markers in both cameras
        robot_detection = self.detect_markers_single_camera(
            robot_color, self.robot_camera_matrix, self.robot_camera_dist_coeffs, 
            robot_depth, self.robot_camera.depth_scale
        )
        zed_detection = self.detect_markers_single_camera(
            zed_color, self.zed_camera_matrix, self.zed_camera_dist_coeffs, 
            zed_depth, self.zed_camera.depth_scale
        )
        
        # Compute ZED to robot base transformation
        transform_matrix, confidence, common_markers = self.compute_zed_to_robot_base_transform(
            robot_detection, zed_detection
        )
        
        # Update stored transformation if confidence is reasonable
        transform_updated = False
        if transform_matrix is not None and confidence > 0.1:
            self.T_zed_to_robot_base = transform_matrix
            self.last_transform_timestamp = time.time()
            self.transform_confidence = confidence
            self.last_common_markers = common_markers
            transform_updated = True
            
            if self.debug:
                print(f"Debug: ZED to robot base transform updated with confidence {confidence:.3f}")
        elif transform_matrix is not None and self.debug:
            print(f"Debug: Transform computed but confidence too low: {confidence:.3f}")
        elif transform_matrix is None and self.debug:
            print(f"Debug: Transform computation failed with {len(common_markers)} common markers")
        
        # Create stereo detection data
        stereo_data = {
            'timestamp': time.time(),
            'robot_camera_markers': len(robot_detection['markers']),
            'zed_camera_markers': len(zed_detection['markers']),
            'common_markers': common_markers,
            'transform_available': transform_matrix is not None,
            'transform_confidence': confidence,
            'T_zed_to_robot_base': transform_matrix.tolist() if transform_matrix is not None else None,
            'robot_camera_detections': robot_detection['markers'],
            'zed_camera_detections': zed_detection['markers'],
            'detection_mode': 'direct_marker_poses',
            'coordinate_frame_transforms': {
                'T_robot_base_to_camera_link': self.T_robot_base_to_camera_link.tolist() if self.T_robot_base_to_camera_link is not None else None,
                'T_robot_base_to_aruco_marker': self.T_robot_base_to_aruco_marker.tolist() if self.T_robot_base_to_aruco_marker is not None else None,
                'T_zed_to_aruco_marker': self.T_zed_to_aruco_marker.tolist() if self.T_zed_to_aruco_marker is not None else None
            },
            'detected_marker_poses': {
                'robot_marker_to_camera': {k: v.tolist() for k, v in robot_detection.get('marker_to_camera_transforms', {}).items()},
                'zed_marker_to_camera': {k: v.tolist() for k, v in zed_detection.get('marker_to_camera_transforms', {}).items()}
            }
        }
        
        # Draw markers on both images with axes visualization
        robot_output_image = self._draw_markers_with_info(
            robot_color, robot_detection, "Robot Camera", 
            self.robot_camera_matrix, self.robot_camera_dist_coeffs
        )
        zed_output_image = self._draw_markers_with_info(
            zed_color, zed_detection, "ZED Camera",
            self.zed_camera_matrix, self.zed_camera_dist_coeffs
        )
        
        # Add transform info to images
        if transform_matrix is not None:
            info_text = f"ZED->Robot: {len(common_markers)} markers, conf: {confidence:.3f}"
            cv2.putText(robot_output_image, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(zed_output_image, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add coordinate frame info
        frame_info = f"Mode: Direct 6DoF Detection | Axes: {'ON' if self.show_axes else 'OFF'}"
        cv2.putText(robot_output_image, frame_info, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(zed_output_image, frame_info, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return robot_output_image, zed_output_image, stereo_data
    
    def _draw_markers_with_info(self, image: np.ndarray, detection: Dict, camera_name: str,
                            camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """Draw markers, information, and 3D coordinate axes on image."""
        output_image = image.copy()
        
        # Add camera name
        cv2.putText(output_image, camera_name, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if detection['ids'] is not None and len(detection['ids']) > 0:
            # Draw marker outlines and IDs
            cv2.aruco.drawDetectedMarkers(output_image, detection['corners'], detection['ids'])
            
            # Draw 3D coordinate axes and pose information
            for i, marker_id in enumerate(detection['ids'].flatten()):
                marker_id = int(marker_id)
                if marker_id in detection['markers']:
                    corner = detection['corners'][i]
                    center = np.mean(corner[0], axis=0).astype(int)
                    pos_3d = detection['markers'][marker_id]['position_3d_camera_frame']
                    
                    # Draw 3D coordinate axes if enabled and pose data is available
                    if (self.show_axes and 
                        marker_id in detection.get('marker_to_camera_transforms', {}) and
                        marker_id in detection['markers']):
                        
                        # Get rvec and tvec for axis drawing (from the original solvePnP)
                        rvec = np.array(detection['markers'][marker_id]['rvec'], dtype=np.float32)
                        tvec = np.array(detection['markers'][marker_id]['tvec'], dtype=np.float32)
                        
                        # Draw coordinate axes using OpenCV function
                        # Axis length is set to marker_size (physical size)
                        try:
                            cv2.drawFrameAxes(output_image, camera_matrix, dist_coeffs, 
                                            rvec, tvec, self.marker_size, 3)
                        except Exception as e:
                            if self.debug:
                                print(f"Debug: Failed to draw axes for marker {marker_id}: {e}")
                    
                    # Get pose information for text display using T_marker_to_camera
                    if marker_id in detection.get('marker_to_camera_transforms', {}):
                        T_marker_to_camera = detection['marker_to_camera_transforms'][marker_id]
                        translation = T_marker_to_camera[:3, 3]  # Marker position in camera frame
                        rotation_matrix = T_marker_to_camera[:3, :3]
                        
                        # Convert to Euler angles for display
                        rotation = Rotation.from_matrix(rotation_matrix)
                        euler_angles = rotation.as_euler('xyz', degrees=True)
                        
                        # Draw 3D position and orientation text
                        distance = np.linalg.norm(translation)
                        text1 = f"ID:{marker_id} Pos:({translation[0]:.2f},{translation[1]:.2f},{translation[2]:.2f})"
                        text2 = f"Dist:{distance:.2f}m Rot:({euler_angles[0]:.1f},{euler_angles[1]:.1f},{euler_angles[2]:.1f})"
                        
                        cv2.putText(output_image, text1, (center[0], center[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(output_image, text2, (center[0], center[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    else:
                        # Fallback to basic position
                        text = f"ID:{marker_id} ({pos_3d[0]:.2f},{pos_3d[1]:.2f},{pos_3d[2]:.2f})"
                        cv2.putText(output_image, text, (center[0], center[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return output_image
    
    def run_stereo_detection_loop(self, show_window: bool = True, save_detections: bool = False,
                                use_averaging: bool = True):
        """
        Run continuous stereo detection loop for ZED to robot base transformation.
        
        Args:
            show_window: Whether to display detection windows
            save_detections: Whether to save detection data to file
            use_averaging: Whether to use depth averaging
        """
        if not self.is_ready():
            print("Cameras not ready. Ensure both cameras are started and running.")
            return
        
        detections_log = []
        current_use_averaging = use_averaging
        
        if show_window:
            cv2.namedWindow("Robot Camera - ArUco Detection", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("ZED Camera - ArUco Detection", cv2.WINDOW_AUTOSIZE)
            print(f"\nZED to Robot Base Transformation Detection Controls:")
            print(f"  'q' or ESC: Quit")
            print(f"  's': Save current detection data")
            print(f"  'c': Toggle depth averaging")
            print(f"  'd': Toggle debug output")
            print(f"  'a': Toggle 3D axes visualization")
            print(f"  'm': Set minimum markers to 1")
            print(f"  't': Print ZED to robot base transformation matrix")
            print(f"  'f': Print all coordinate frame transforms")
            print(f"  'i': Print intermediate transforms (ArUco marker poses)")
            print(f"  SPACE: Print current detection info")
            print()
            print(f"Current settings:")
            print(f"  Minimum markers: {self.min_markers_for_transform}")
            print(f"  Debug output: {'ON' if self.debug else 'OFF'}")
            print(f"  Axes visualization: {'ON' if self.show_axes else 'OFF'}")
            print(f"  Robot base to camera link: {'Available' if self.T_robot_base_to_camera_link is not None else 'NOT SET'}")
            print()
        
        try:
            while True:
                robot_output_image, zed_output_image, stereo_data = self.process_stereo_frame(
                    use_averaging=current_use_averaging
                )
                
                if robot_output_image is None or zed_output_image is None:
                    continue
                
                if save_detections and stereo_data['transform_available']:
                    detections_log.append(stereo_data)
                
                if show_window:
                    # Add status overlay
                    status_text = f"Avg:{current_use_averaging} | Min:{self.min_markers_for_transform} | Common:{len(stereo_data['common_markers'])}"
                    cv2.putText(robot_output_image, status_text, (10, robot_output_image.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(zed_output_image, status_text, (10, zed_output_image.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow("Robot Camera - ArUco Detection", robot_output_image)
                    cv2.imshow("ZED Camera - ArUco Detection", zed_output_image)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('a'):  # Toggle axes visualization
                        self.toggle_axes_visualization()
                    elif key == ord('s'):
                        if stereo_data['transform_available']:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"zed_to_robot_transform_{timestamp}.json"
                            with open(filename, 'w') as f:
                                json.dump(stereo_data, f, indent=2)
                            print(f"Saved detection to {filename}")
                        else:
                            print("No valid transformation to save")
                    elif key == ord('c'):
                        current_use_averaging = not current_use_averaging
                        print(f"Depth averaging: {'ON' if current_use_averaging else 'OFF'}")
                    elif key == ord('d'):
                        self.set_debug(not self.debug)
                    elif key == ord('m'):
                        self.set_min_markers_for_transform(1)
                    elif key == ord('t'):
                        if self.T_zed_to_robot_base is not None:
                            print(f"\n=== ZED TO ROBOT BASE TRANSFORMATION MATRIX ===")
                            print(f"Confidence: {self.transform_confidence:.3f}")
                            print(f"Last updated: {time.time() - self.last_transform_timestamp:.1f} seconds ago")
                            print(f"Common markers used: {self.last_common_markers}")
                            print("T_zed_to_robot_base:")
                            print(self.T_zed_to_robot_base)
                            print("=" * 55)
                        else:
                            print("No ZED to robot base transformation matrix available")
                    elif key == ord('f'):
                        print(f"\n=== ALL COORDINATE FRAME TRANSFORMS ===")
                        transforms = self.get_all_transforms()
                        for name, transform in transforms.items():
                            print(f"{name}:")
                            if transform is not None:
                                print(transform)
                            else:
                                print("Not available")
                            print()
                        print("=" * 45)
                    elif key == ord('i'):
                        print(f"\n=== INTERMEDIATE TRANSFORMS (ArUco Marker Poses) ===")
                        if self.T_robot_base_to_aruco_marker is not None:
                            print("T_robot_base_to_aruco_marker:")
                            print(self.T_robot_base_to_aruco_marker)
                        else:
                            print("T_robot_base_to_aruco_marker: Not available")
                        
                        if self.T_zed_to_aruco_marker is not None:
                            print("T_zed_to_aruco_marker:")
                            print(self.T_zed_to_aruco_marker)
                        else:
                            print("T_zed_to_aruco_marker: Not available")
                        print("=" * 55)
                    elif key == ord(' '):  # Space bar
                        print(f"\n=== DETECTION STATUS ===")
                        print(f"Robot camera markers: {stereo_data['robot_camera_markers']}")
                        print(f"ZED camera markers: {stereo_data['zed_camera_markers']}")
                        print(f"Common markers: {stereo_data['common_markers']}")
                        print(f"Transform available: {stereo_data['transform_available']}")
                        print(f"Min markers required: {self.min_markers_for_transform}")
                        print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
                        print(f"Axes visualization: {'ON' if self.show_axes else 'OFF'}")
                        print(f"Robot base to camera link: {'Set' if self.T_robot_base_to_camera_link is not None else 'NOT SET'}")
                        if stereo_data['transform_available']:
                            print(f"Transform confidence: {stereo_data['transform_confidence']:.3f}")
                        print("=" * 30)
                
                else:
                    # Print detection info without window
                    if stereo_data['transform_available']:
                        print(f"ZED to robot base transform available with {len(stereo_data['common_markers'])} markers, "
                              f"confidence: {stereo_data['transform_confidence']:.3f}")
                    
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nDetection loop interrupted")
        
        finally:
            if show_window:
                cv2.destroyAllWindows()
            
            if save_detections and detections_log:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'zed_to_robot_transforms_batch_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(detections_log, f, indent=2)
                print(f"Saved {len(detections_log)} transform records to {filename}")


# Example usage with robot interface integration
if __name__ == "__main__":
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
        
        # Create stereo ArUco detector with axes visualization enabled
        stereo_detector = StereoArucoDetector(
            robot_camera_instance=robot_camera,
            zed_camera_instance=zed_camera,
            T_robot_base_to_camera_link=T_robot_base_to_camera_link,
            marker_size=0.05,  # 5cm markers
            dictionary_type=cv2.aruco.DICT_6X6_250,
            min_markers_for_transform=1,  # Need at least 1 common marker
            debug=False,  # Start with clean output
            show_axes=True  # Enable 3D axes visualization
        )
        
        # Check if detector is ready
        if stereo_detector.is_ready():
            print("Stereo ArUco detector ready!")
            print("\nCoordinate Frame Approach with 3D Axes Visualization - Controls:")
            print("  Press 't' to see ZED to robot base transformation matrix")
            print("  Press 'f' to see all coordinate frame transforms")
            print("  Press 'i' to see intermediate ArUco marker poses")
            print("  Press 'u' to update robot transform from current robot pose")
            print("  Press 'a' to toggle 3D coordinate axes visualization")
            print("  Press 'd' to toggle debug output")
            print("  Press SPACE for status info")
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
                        stereo_detector.set_robot_base_to_camera_link_transform(new_transform)
                        
                        print(f"Updated robot transform:")
                        print(f"Camera position: {camera_position}")
                        print(f"Camera rotation (quaternion): {camera_rotation.as_quat()}")
                        print("Transform updated successfully!")
                        
                    except Exception as e:
                        print(f"Failed to update robot transform: {e}")
                else:
                    print("Robot interface not available - cannot update transform")
            
            # Modified detection loop to handle robot transform updates
            def run_detection_with_robot_updates():
                """Run detection loop with robot transform update capability"""
                if not stereo_detector.is_ready():
                    print("Cameras not ready. Ensure both cameras are started and running.")
                    return
                
                detections_log = []
                current_use_averaging = False
                
                cv2.namedWindow("Robot Camera - ArUco Detection", cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow("ZED Camera - ArUco Detection", cv2.WINDOW_AUTOSIZE)
                
                try:
                    while True:
                        robot_output_image, zed_output_image, stereo_data = stereo_detector.process_stereo_frame(
                            use_averaging=current_use_averaging
                        )
                        
                        if robot_output_image is None or zed_output_image is None:
                            continue
                        
                        if stereo_data['transform_available']:
                            detections_log.append(stereo_data)
                        
                        # Add robot interface status to display
                        robot_status = f"Robot Interface: {'Connected' if robot_planner is not None else 'Not Available'}"
                        cv2.putText(robot_output_image, robot_status, (10, robot_output_image.shape[0] - 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                        cv2.putText(zed_output_image, robot_status, (10, zed_output_image.shape[0] - 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                        
                        cv2.imshow("Robot Camera - ArUco Detection", robot_output_image)
                        cv2.imshow("ZED Camera - ArUco Detection", zed_output_image)
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            break
                        elif key == ord('a'):  # Toggle axes visualization
                            stereo_detector.toggle_axes_visualization()
                        elif key == ord('u'):  # Update robot transform
                            update_robot_transform()
                        elif key == ord('s'):
                            if stereo_data['transform_available']:
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                filename = f"zed_to_robot_transform_{timestamp}.json"
                                with open(filename, 'w') as f:
                                    json.dump(stereo_data, f, indent=2)
                                print(f"Saved detection to {filename}")
                            else:
                                print("No valid transformation to save")
                        elif key == ord('c'):
                            current_use_averaging = not current_use_averaging
                            print(f"Depth averaging: {'ON' if current_use_averaging else 'OFF'}")
                        elif key == ord('d'):
                            stereo_detector.set_debug(not stereo_detector.debug)
                        elif key == ord('t'):
                            if stereo_detector.T_zed_to_robot_base is not None:
                                print(f"\n=== ZED TO ROBOT BASE TRANSFORMATION MATRIX ===")
                                print(f"Confidence: {stereo_detector.transform_confidence:.3f}")
                                print(f"Last updated: {time.time() - stereo_detector.last_transform_timestamp:.1f} seconds ago")
                                print(f"Common markers used: {stereo_detector.last_common_markers}")
                                print("T_zed_to_robot_base:")
                                print(stereo_detector.T_zed_to_robot_base)
                                print("=" * 55)
                            else:
                                print("No ZED to robot base transformation matrix available")
                        elif key == ord('f'):
                            print(f"\n=== ALL COORDINATE FRAME TRANSFORMS ===")
                            transforms = stereo_detector.get_all_transforms()
                            for name, transform in transforms.items():
                                print(f"{name}:")
                                if transform is not None:
                                    print(transform)
                                else:
                                    print("Not available")
                                print()
                            print("=" * 45)
                        elif key == ord(' '):  # Space bar
                            print(f"\n=== DETECTION STATUS ===")
                            print(f"Robot camera markers: {stereo_data['robot_camera_markers']}")
                            print(f"ZED camera markers: {stereo_data['zed_camera_markers']}")
                            print(f"Common markers: {stereo_data['common_markers']}")
                            print(f"Transform available: {stereo_data['transform_available']}")
                            print(f"Robot interface: {'Connected' if robot_planner is not None else 'Not Available'}")
                            print(f"Axes visualization: {'ON' if stereo_detector.show_axes else 'OFF'}")
                            if stereo_data['transform_available']:
                                print(f"Transform confidence: {stereo_data['transform_confidence']:.3f}")
                            print("=" * 30)
                    
                except KeyboardInterrupt:
                    print("\nDetection loop interrupted")
                
                finally:
                    cv2.destroyAllWindows()
            
            try:
                # Run detection loop with robot integration and axes visualization
                run_detection_with_robot_updates()
            finally:
                # Clean up
                robot_camera.stop()
                zed_camera.stop()
                if robot_planner is not None:
                    # robot_planner.disconnect_robot()  # Uncomment when using real robot
                    pass
        else:
            print("Detector not ready - cameras may not be fully initialized")
            robot_camera.stop()
            zed_camera.stop()
    else:
        print("Failed to start one or both cameras")