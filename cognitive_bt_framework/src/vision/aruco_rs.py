import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import time
import json
from scipy.spatial.transform import Rotation


class StereoArucoDetector:
    """
    A class for detecting ArUco markers using two cameras and computing the transformation
    between camera coordinate systems using common markers.
    """
    
    def __init__(self, 
                 camera1_instance,  # First camera instance
                 camera2_instance,  # Second camera instance
                 marker_size: float = 0.05,  # marker size in meters
                 dictionary_type: int = cv2.aruco.DICT_6X6_250,
                 min_markers_for_transform: int = 3):
        """
        Initialize the stereo ArUco detector with two Camera instances.
        
        Args:
            camera1_instance: Instance of Camera class (reference camera)
            camera2_instance: Instance of Camera class (target camera)
            marker_size: Physical size of the ArUco marker in meters
            dictionary_type: OpenCV ArUco dictionary type
            min_markers_for_transform: Minimum number of common markers needed for transformation
        """
        self.camera1 = camera1_instance  # Reference camera
        self.camera2 = camera2_instance  # Target camera
        self.marker_size = marker_size
        self.min_markers_for_transform = min_markers_for_transform
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera intrinsics (will be populated from camera instances)
        self.camera1_matrix = None
        self.camera1_dist_coeffs = None
        self.camera2_matrix = None
        self.camera2_dist_coeffs = None
        
        # Transformation from camera1 to camera2
        self.transform_1to2 = None
        self.last_transform_timestamp = None
        self.transform_confidence = 0.0
        
    def _update_camera_parameters(self):
        """Update camera parameters from both camera instances."""
        camera1_ready = False
        camera2_ready = False
        
        # Update camera 1 parameters
        if self.camera1.intrinsics is not None:
            intrinsics1 = self.camera1.intrinsics
            self.camera1_matrix = np.array([
                [intrinsics1.fx, 0, intrinsics1.ppx],
                [0, intrinsics1.fy, intrinsics1.ppy],
                [0, 0, 1]
            ])
            self.camera1_dist_coeffs = np.array(intrinsics1.coeffs)
            camera1_ready = True
            
        # Update camera 2 parameters
        if self.camera2.intrinsics is not None:
            intrinsics2 = self.camera2.intrinsics
            self.camera2_matrix = np.array([
                [intrinsics2.fx, 0, intrinsics2.ppx],
                [0, intrinsics2.fy, intrinsics2.ppy],
                [0, 0, 1]
            ])
            self.camera2_dist_coeffs = np.array(intrinsics2.coeffs)
            camera2_ready = True
            
        return camera1_ready and camera2_ready
        
    def is_ready(self) -> bool:
        """
        Check if both cameras are ready.
        
        Returns:
            bool: True if both cameras are ready, False otherwise
        """
        return (self.camera1._running and self.camera1.intrinsics is not None and 
                self.camera1.depth_scale is not None and
                self.camera2._running and self.camera2.intrinsics is not None and 
                self.camera2.depth_scale is not None)
    
    def get_frames_both_cameras(self, use_averaging: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                                         Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aligned color and depth frames from both cameras.
        
        Args:
            use_averaging: Whether to use depth averaging
            
        Returns:
            Tuple[color1, depth1, color2, depth2]: Color and depth images from both cameras
        """
        if not self.is_ready():
            return None, None, None, None
            
        frames1 = self.camera1.get_frames(use_averaging=use_averaging)
        frames2 = self.camera2.get_frames(use_averaging=use_averaging)
        
        if frames1 is None or frames2 is None:
            return None, None, None, None
            
        color1, depth1 = frames1
        color2, depth2 = frames2
        
        return color1, depth1, color2, depth2
    
    def detect_markers_single_camera(self, color_image: np.ndarray, camera_matrix: np.ndarray, 
                                   dist_coeffs: np.ndarray, depth_image: np.ndarray, 
                                   depth_scale: float) -> Dict:
        """
        Detect ArUco markers in a single camera frame and compute 3D positions.
        
        Args:
            color_image: Input color image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            depth_image: Aligned depth image
            depth_scale: Depth scaling factor
            
        Returns:
            Dict containing marker detection data
        """
        # Convert BGR to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        detection_data = {
            'corners': corners,
            'ids': ids,
            'rejected': rejected,
            'markers': {},
            'poses': {}
        }
        
        if ids is not None and len(ids) > 0:
            # Estimate poses
            rvecs, tvecs = self._estimate_pose_single_camera(corners, camera_matrix, dist_coeffs)
            
            # Get 3D positions using depth
            positions_3d = self._get_3d_position_single_camera(corners, depth_image, camera_matrix, depth_scale)
            
            # Store marker data
            for i, marker_id in enumerate(ids.flatten()):
                marker_id = int(marker_id)
                detection_data['markers'][marker_id] = {
                    'corners': corners[i][0].tolist(),
                    'position_3d': positions_3d[i],
                    'distance': np.linalg.norm(positions_3d[i])
                }
                detection_data['poses'][marker_id] = {
                    'rvec': rvecs[i].flatten().tolist(),
                    'tvec': tvecs[i].flatten().tolist()
                }
        
        return detection_data
    
    def _estimate_pose_single_camera(self, corners: List, camera_matrix: np.ndarray, 
                                   dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate pose for a single camera."""
        rvecs = []
        tvecs = []
        
        for corner in corners:
            # Define 3D object points for the marker
            half_size = self.marker_size / 2
            object_points = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0]
            ], dtype=np.float32)
            
            # Solve PnP to get pose
            success, rvec, tvec = cv2.solvePnP(
                object_points, corner, camera_matrix, dist_coeffs
            )
            
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(np.zeros((3, 1)))
                tvecs.append(np.zeros((3, 1)))
        
        return np.array(rvecs), np.array(tvecs)
    
    def _get_3d_position_single_camera(self, corners: List, depth_image: np.ndarray, 
                                     camera_matrix: np.ndarray, depth_scale: float) -> List[Tuple[float, float, float]]:
        """Get 3D positions for a single camera."""
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
    
    def compute_camera_transform(self, detection1: Dict, detection2: Dict) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Compute transformation matrix from camera1 to camera2 using common markers.
        
        Assumption: If an ArUco marker with the same ID is detected in both frames,
        it is the same physical marker (common marker).
        
        Args:
            detection1: Detection data from camera 1
            detection2: Detection data from camera 2
            
        Returns:
            Tuple[transform_matrix, confidence, common_marker_ids]: 
            - 4x4 transformation matrix (or None if insufficient data)
            - Confidence score (0-1)
            - List of marker IDs used for computation
        """
        # Find common markers: any marker ID that appears in both cameras
        # We assume these are the same physical markers
        markers1 = set(detection1['markers'].keys())
        markers2 = set(detection2['markers'].keys())
        common_markers = list(markers1.intersection(markers2))
        
        print(f"Debug: Camera 1 has markers: {list(markers1)}")
        print(f"Debug: Camera 2 has markers: {list(markers2)}")
        print(f"Debug: Common markers found: {common_markers}")
        print(f"Debug: Need {self.min_markers_for_transform} markers, have {len(common_markers)}")
        
        if len(common_markers) < self.min_markers_for_transform:
            print(f"Debug: Insufficient markers for transformation. Need {self.min_markers_for_transform}, have {len(common_markers)}")
            return None, 0.0, common_markers
        
        # Extract 3D points for all common markers
        # Since we assume same ID = same physical marker, we use all detections
        points1 = []
        points2 = []
        
        for marker_id in common_markers:
            pos1 = detection1['markers'][marker_id]['position_3d']
            pos2 = detection2['markers'][marker_id]['position_3d']
            
            print(f"Debug: Marker {marker_id} - Cam1: {pos1}, Cam2: {pos2}")
            
            points1.append(pos1)
            points2.append(pos2)
        
        points1 = np.array(points1)
        points2 = np.array(points2)
        
        # Compute transformation using Kabsch algorithm
        transform_matrix, confidence = self._kabsch_algorithm(points1, points2)
        
    def set_min_markers_for_transform(self, min_markers: int):
        """
        Set the minimum number of markers required for transformation.
        
        Args:
            min_markers: Minimum markers needed (1 for translation-only, 3+ for full 6DOF)
        """
        self.min_markers_for_transform = min_markers
        print(f"Debug: Minimum markers for transform set to {min_markers}")
    
    def compute_simple_transform(self, detection1: Dict, detection2: Dict) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Compute a simple transformation even with limited markers.
        - 1 marker: Translation only (assumes no rotation)
        - 2 markers: Limited rotation + translation
        - 3+ markers: Full 6DOF transformation
        
        Args:
            detection1: Detection data from camera 1
            detection2: Detection data from camera 2
            
        Returns:
            Tuple[transform_matrix, confidence, common_marker_ids]
        """
        markers1 = set(detection1['markers'].keys())
        markers2 = set(detection2['markers'].keys())
        common_markers = list(markers1.intersection(markers2))
        
        if len(common_markers) == 0:
            return None, 0.0, common_markers
        
        print(f"Debug Simple: Found {len(common_markers)} common markers: {common_markers}")
        
        if len(common_markers) == 1:
            # Translation-only transformation
            marker_id = common_markers[0]
            pos1 = np.array(detection1['markers'][marker_id]['position_3d'])
            pos2 = np.array(detection2['markers'][marker_id]['position_3d'])
            
            print(f"Debug Simple: Single marker translation - Cam1: {pos1}, Cam2: {pos2}")
            
            # Create translation-only transform
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = pos2 - pos1
            
            # Simple confidence based on distance (closer = more confident)
            distance = np.linalg.norm(pos1)
            confidence = np.exp(-distance / 2.0)  # 2m characteristic distance
            
            print(f"Debug Simple: Translation-only transform, confidence: {confidence:.3f}")
            return transform_matrix, confidence, common_markers
    
    def set_min_markers_for_transform(self, min_markers: int):
        """
        Set the minimum number of markers required for transformation.
        
        Args:
            min_markers: Minimum markers needed (1 for translation-only, 3+ for full 6DOF)
        """
        self.min_markers_for_transform = min_markers
        print(f"Debug: Minimum markers for transform set to {min_markers}")
    
    def compute_single_marker_transform(self, detection1: Dict, detection2: Dict) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Compute full 6DOF transformation using a single ArUco marker.
        
        Since each ArUco marker provides full pose (rotation + translation) relative to each camera,
        we can compute the complete transformation between cameras using just one marker.
        
        Args:
            detection1: Detection data from camera 1
            detection2: Detection data from camera 2
            
        Returns:
            Tuple[transform_matrix, confidence, common_marker_ids]
        """
        markers1 = set(detection1['markers'].keys())
        markers2 = set(detection2['markers'].keys())
        common_markers = list(markers1.intersection(markers2))
        
        if len(common_markers) == 0:
            return None, 0.0, common_markers
        
        print(f"Debug Single Marker: Found {len(common_markers)} common markers: {common_markers}")
        
        # Use the first common marker for transformation
        marker_id = common_markers[0]
        
        # Get pose data (rotation and translation vectors) for the marker in both cameras
        rvec1 = np.array(detection1['poses'][marker_id]['rvec'])
        tvec1 = np.array(detection1['poses'][marker_id]['tvec'])
        rvec2 = np.array(detection2['poses'][marker_id]['rvec'])
        tvec2 = np.array(detection2['poses'][marker_id]['tvec'])
        
        print(f"Debug Single Marker: Marker {marker_id}")
        print(f"  Cam1 - rvec: {rvec1}, tvec: {tvec1}")
        print(f"  Cam2 - rvec: {rvec2}, tvec: {tvec2}")
        
        # Convert rotation vectors to rotation matrices
        from scipy.spatial.transform import Rotation
        R1 = Rotation.from_rotvec(rvec1).as_matrix()
        R2 = Rotation.from_rotvec(rvec2).as_matrix()
        
        # Create transformation matrices from camera to marker
        T_cam1_to_marker = np.eye(4)
        T_cam1_to_marker[:3, :3] = R1
        T_cam1_to_marker[:3, 3] = tvec1
        
        T_cam2_to_marker = np.eye(4)
        T_cam2_to_marker[:3, :3] = R2
        T_cam2_to_marker[:3, 3] = tvec2
        
        # Compute transformation from camera1 to camera2
        # T_cam1_to_cam2 = T_cam2_to_marker * inverse(T_cam1_to_marker)
        T_marker_to_cam1 = np.linalg.inv(T_cam1_to_marker)
        T_cam1_to_cam2 = T_cam2_to_marker @ T_marker_to_cam1
        
        print(f"Debug Single Marker: Computed transformation matrix:")
        print(T_cam1_to_cam2)
        
        # Compute confidence based on the quality of pose estimation
        # Use the norm of rotation vectors as a proxy (smaller rotations are often more stable)
        rotation_magnitude1 = np.linalg.norm(rvec1)
        rotation_magnitude2 = np.linalg.norm(rvec2)
        distance1 = np.linalg.norm(tvec1)
        distance2 = np.linalg.norm(tvec2)
        
        # Higher confidence for: moderate distances, not too extreme rotations
        distance_confidence = np.exp(-abs(distance1 - 1.0) / 2.0) * np.exp(-abs(distance2 - 1.0) / 2.0)  # Optimal around 1m
        rotation_confidence = np.exp(-rotation_magnitude1 / 2.0) * np.exp(-rotation_magnitude2 / 2.0)  # Prefer smaller rotations
        
        confidence = distance_confidence * rotation_confidence
        confidence = np.clip(confidence, 0.0, 1.0)
        
        print(f"Debug Single Marker: Confidence components - distance: {distance_confidence:.3f}, rotation: {rotation_confidence:.3f}")
        print(f"Debug Single Marker: Final confidence: {confidence:.3f}")
        
        return T_cam1_to_cam2, confidence, common_markers
    
    def compute_simple_transform(self, detection1: Dict, detection2: Dict) -> Tuple[Optional[np.ndarray], float, List[int]]:
        """
        Compute transformation with any number of markers.
        - 1 marker: Full 6DOF using marker pose
        - 2+ markers: Kabsch algorithm for robustness
        
        Args:
            detection1: Detection data from camera 1
            detection2: Detection data from camera 2
            
        Returns:
            Tuple[transform_matrix, confidence, common_marker_ids]
        """
        markers1 = set(detection1['markers'].keys())
        markers2 = set(detection2['markers'].keys())
        common_markers = list(markers1.intersection(markers2))
        
        if len(common_markers) == 0:
            return None, 0.0, common_markers
        
        if len(common_markers) == 1:
            # Use single marker method for full 6DOF
            return self.compute_single_marker_transform(detection1, detection2)
        else:
            # Use Kabsch algorithm for multiple markers (more robust)
            points1 = []
            points2 = []
            
            for marker_id in common_markers:
                pos1 = np.array(detection1['markers'][marker_id]['position_3d'])
                pos2 = np.array(detection2['markers'][marker_id]['position_3d'])
                points1.append(pos1)
                points2.append(pos2)
            
            points1 = np.array(points1)
            points2 = np.array(points2)
            
            # Compute transformation using Kabsch algorithm
            transform_matrix, confidence = self._kabsch_algorithm(points1, points2)
            
            return transform_matrix, confidence, common_markers
        
        else:
            # Use full Kabsch algorithm for 2+ markers
            return self.compute_camera_transform(detection1, detection2)
    
    def _kabsch_algorithm(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute optimal transformation using Kabsch algorithm.
        
        Args:
            points1: 3D points in camera1 coordinate system (N x 3)
            points2: 3D points in camera2 coordinate system (N x 3)
            
        Returns:
            Tuple[transform_matrix, confidence]: 4x4 transformation matrix and confidence score
        """
        try:
            print(f"Debug Kabsch: Input points1 shape: {points1.shape}, points2 shape: {points2.shape}")
            print(f"Debug Kabsch: Points1:\n{points1}")
            print(f"Debug Kabsch: Points2:\n{points2}")
            
            # Check for invalid points (all zeros)
            valid1 = np.any(np.abs(points1) > 0.001, axis=1)
            valid2 = np.any(np.abs(points2) > 0.001, axis=1)
            valid_mask = valid1 & valid2
            
            if np.sum(valid_mask) < 3:
                print(f"Debug Kabsch: Not enough valid points ({np.sum(valid_mask)})")
                return np.eye(4), 0.0
            
            # Use only valid points
            points1_valid = points1[valid_mask]
            points2_valid = points2[valid_mask]
            
            print(f"Debug Kabsch: Using {len(points1_valid)} valid points")
            
            # Center the points
            centroid1 = np.mean(points1_valid, axis=0)
            centroid2 = np.mean(points2_valid, axis=0)
            
            print(f"Debug Kabsch: Centroid1: {centroid1}, Centroid2: {centroid2}")
            
            centered1 = points1_valid - centroid1
            centered2 = points2_valid - centroid2
            
            # Compute cross-covariance matrix
            H = centered1.T @ centered2
            print(f"Debug Kabsch: Covariance matrix H:\n{H}")
            
            # Singular Value Decomposition
            U, S, Vt = np.linalg.svd(H)
            print(f"Debug Kabsch: SVD - U shape: {U.shape}, S: {S}, Vt shape: {Vt.shape}")
            
            # Compute rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            det_R = np.linalg.det(R)
            print(f"Debug Kabsch: det(R) = {det_R}")
            if det_R < 0:
                print("Debug Kabsch: Correcting reflection")
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = centroid2 - R @ centroid1
            print(f"Debug Kabsch: Translation: {t}")
            
            # Create 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = t
            
            # Compute confidence based on residual error
            transformed_points1 = (R @ points1_valid.T).T + t
            residuals = np.linalg.norm(transformed_points1 - points2_valid, axis=1)
            mean_error = np.mean(residuals)
            max_error = np.max(residuals)
            
            print(f"Debug Kabsch: Mean error: {mean_error:.4f}m, Max error: {max_error:.4f}m")
            
            # Convert error to confidence (exponential decay)
            confidence = np.exp(-mean_error / 0.05)  # More sensitive to error
            
            print(f"Debug Kabsch: Final confidence: {confidence:.3f}")
            print(f"Debug Kabsch: Transform matrix:\n{transform_matrix}")
            
            return transform_matrix, confidence
            
        except Exception as e:
            print(f"Debug Kabsch: Exception occurred: {e}")
            return np.eye(4), 0.0
    
    def transform_point_1to2(self, point_3d_cam1: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a 3D point from camera1 coordinate system to camera2 coordinate system.
        
        Args:
            point_3d_cam1: 3D point in camera1 coordinates (x, y, z)
            
        Returns:
            3D point in camera2 coordinates or None if no transform available
        """
        if self.transform_1to2 is None:
            return None
        
        # Convert to homogeneous coordinates
        point_homogeneous = np.append(point_3d_cam1, 1.0)
        
        # Apply transformation
        transformed_point = self.transform_1to2 @ point_homogeneous
        
        # Return 3D coordinates
        return transformed_point[:3]
    
    def process_stereo_frame(self, use_averaging: bool = True, allow_simple_transform: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Process frames from both cameras and compute transformation.
        
        Args:
            use_averaging: Whether to use depth averaging
            allow_simple_transform: Whether to allow simple transforms with < 3 markers
            
        Returns:
            Tuple[output_image1, output_image2, stereo_data]: Processed images and stereo detection data
        """
        color1, depth1, color2, depth2 = self.get_frames_both_cameras(use_averaging=use_averaging)
        
        if any(img is None for img in [color1, depth1, color2, depth2]):
            return None, None, None
        
        if not self._update_camera_parameters():
            return None, None, None
        
        # Detect markers in both cameras
        detection1 = self.detect_markers_single_camera(
            color1, self.camera1_matrix, self.camera1_dist_coeffs, depth1, self.camera1.depth_scale
        )
        detection2 = self.detect_markers_single_camera(
            color2, self.camera2_matrix, self.camera2_dist_coeffs, depth2, self.camera2.depth_scale
        )
        
        # Compute transformation
        if allow_simple_transform:
            transform_matrix, confidence, common_markers = self.compute_simple_transform(detection1, detection2)
        else:
            transform_matrix, confidence, common_markers = self.compute_camera_transform(detection1, detection2)
        
        # Debug: Print transformation computation details
        if len(common_markers) > 0:
            print(f"Debug: Found {len(common_markers)} common markers: {common_markers}")
            print(f"Debug: Transform computed: {transform_matrix is not None}, Confidence: {confidence:.3f}")
        
        # Update stored transformation if confidence is reasonable (lowered threshold)
        if transform_matrix is not None and confidence > 0.1:  # Lowered from 0.5 to 0.1
            self.transform_1to2 = transform_matrix
            self.last_transform_timestamp = time.time()
            self.transform_confidence = confidence
            print(f"Debug: Transform updated with confidence {confidence:.3f}")
        elif transform_matrix is not None:
            print(f"Debug: Transform computed but confidence too low: {confidence:.3f}")
        else:
            print(f"Debug: Transform computation failed with {len(common_markers)} common markers")
        
        # Create stereo detection data
        stereo_data = {
            'timestamp': time.time(),
            'camera1_markers': len(detection1['markers']),
            'camera2_markers': len(detection2['markers']),
            'common_markers': common_markers,
            'transform_available': transform_matrix is not None,
            'transform_confidence': confidence,
            'transform_matrix': transform_matrix.tolist() if transform_matrix is not None else None,
            'camera1_detections': detection1['markers'],
            'camera2_detections': detection2['markers'],
            'simple_transform_used': allow_simple_transform
        }
        
        # Draw markers on both images
        output_image1 = self._draw_markers_with_info(color1, detection1, "Camera 1")
        output_image2 = self._draw_markers_with_info(color2, detection2, "Camera 2")
        
        # Add transform info to images
        if transform_matrix is not None:
            mode = "Simple" if allow_simple_transform and len(common_markers) < 3 else "Full"
            info_text = f"Transform ({mode}): {len(common_markers)} markers, conf: {confidence:.3f}"
            cv2.putText(output_image1, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_image2, info_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output_image1, output_image2, stereo_data
    
    def _draw_markers_with_info(self, image: np.ndarray, detection: Dict, camera_name: str) -> np.ndarray:
        """Draw markers and information on image."""
        output_image = image.copy()
        
        # Add camera name
        cv2.putText(output_image, camera_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if detection['ids'] is not None and len(detection['ids']) > 0:
            # Draw marker outlines and IDs
            cv2.aruco.drawDetectedMarkers(output_image, detection['corners'], detection['ids'])
            
            # Draw 3D position information
            for i, marker_id in enumerate(detection['ids'].flatten()):
                marker_id = int(marker_id)
                if marker_id in detection['markers']:
                    corner = detection['corners'][i]
                    center = np.mean(corner[0], axis=0).astype(int)
                    pos_3d = detection['markers'][marker_id]['position_3d']
                    
                    # Draw 3D position text
                    text = f"ID:{marker_id} ({pos_3d[0]:.2f},{pos_3d[1]:.2f},{pos_3d[2]:.2f})"
                    cv2.putText(output_image, text, (center[0], center[1] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return output_image
    
    def run_stereo_detection_loop(self, show_window: bool = True, save_detections: bool = False,
                                use_averaging: bool = True, allow_simple_transform: bool = False):
        """
        Run continuous stereo detection loop.
        
        Args:
            show_window: Whether to display detection windows
            save_detections: Whether to save detection data to file
            use_averaging: Whether to use depth averaging
            allow_simple_transform: Whether to allow simple transforms with < 3 markers
        """
        if not self.is_ready():
            print("Cameras not ready. Ensure both cameras are started and running.")
            return
        
        detections_log = []
        current_use_averaging = use_averaging
        current_allow_simple = allow_simple_transform
        
        if show_window:
            cv2.namedWindow("Camera 1 - ArUco Detection", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Camera 2 - ArUco Detection", cv2.WINDOW_AUTOSIZE)
            print(f"\nStereo ArUco Detection Controls:")
            print(f"  'q' or ESC: Quit")
            print(f"  's': Save current detection data")
            print(f"  'c': Toggle depth averaging")
            print(f"  'm': Set minimum markers to 1 (allows simple transforms)")
            print(f"  'M': Set minimum markers to 3 (requires full transforms)")
            print(f"  'x': Toggle simple transform mode")
            print(f"  't': Print current transformation matrix")
            print(f"  SPACE: Print current detection info")
            print()
            print(f"Current settings:")
            print(f"  Minimum markers: {self.min_markers_for_transform}")
            print(f"  Simple transform: {'ON' if current_allow_simple else 'OFF'}")
            print()
        
        try:
            while True:
                output_image1, output_image2, stereo_data = self.process_stereo_frame(
                    use_averaging=current_use_averaging, 
                    allow_simple_transform=current_allow_simple
                )
                
                if output_image1 is None or output_image2 is None:
                    continue
                
                if save_detections and stereo_data['transform_available']:
                    detections_log.append(stereo_data)
                
                if show_window:
                    # Add status overlay
                    status_text = f"Avg:{current_use_averaging} | Simple:{current_allow_simple} | Min:{self.min_markers_for_transform} | Common:{len(stereo_data['common_markers'])}"
                    cv2.putText(output_image1, status_text, (10, output_image1.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(output_image2, status_text, (10, output_image2.shape[0] - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow("Camera 1 - ArUco Detection", output_image1)
                    cv2.imshow("Camera 2 - ArUco Detection", output_image2)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('s'):
                        if stereo_data['transform_available']:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"stereo_aruco_detection_{timestamp}.json"
                            with open(filename, 'w') as f:
                                json.dump(stereo_data, f, indent=2)
                            print(f"Saved detection to {filename}")
                        else:
                            print("No valid transformation to save")
                    elif key == ord('c'):
                        current_use_averaging = not current_use_averaging
                        print(f"Depth averaging: {'ON' if current_use_averaging else 'OFF'}")
                    elif key == ord('m'):
                        self.set_min_markers_for_transform(1)
                    elif key == ord('M'):
                        self.set_min_markers_for_transform(3)
                    elif key == ord('x'):
                        current_allow_simple = not current_allow_simple
                        print(f"Simple transform mode: {'ON' if current_allow_simple else 'OFF'}")
                    elif key == ord('t'):
                        if self.transform_1to2 is not None:
                            print(f"\nTransformation matrix (Camera 1 to Camera 2):")
                            print(f"Confidence: {self.transform_confidence:.3f}")
                            print(self.transform_1to2)
                        else:
                            print("No transformation matrix available")
                    elif key == ord(' '):  # Space bar
                        print(f"\nStereo Detection Status:")
                        print(f"Camera 1 markers: {stereo_data['camera1_markers']}")
                        print(f"Camera 2 markers: {stereo_data['camera2_markers']}")
                        print(f"Common markers: {stereo_data['common_markers']}")
                        print(f"Transform available: {stereo_data['transform_available']}")
                        print(f"Simple transform mode: {current_allow_simple}")
                        print(f"Min markers required: {self.min_markers_for_transform}")
                        if stereo_data['transform_available']:
                            print(f"Transform confidence: {stereo_data['transform_confidence']:.3f}")
                
                else:
                    # Print detection info without window
                    if stereo_data['transform_available']:
                        print(f"Transform available with {len(stereo_data['common_markers'])} markers, "
                              f"confidence: {stereo_data['transform_confidence']:.3f}")
                    
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStereo detection loop interrupted")
        
        finally:
            if show_window:
                cv2.destroyAllWindows()
            
            if save_detections and detections_log:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'stereo_aruco_detections_batch_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(detections_log, f, indent=2)
                print(f"Saved {len(detections_log)} detection records to {filename}")


# Example usage
if __name__ == "__main__":
    from cognitive_bt_framework.src.vision.zed_camera import Camera as ZedCam
    from cognitive_bt_framework.src.vision.realsense import Camera as RsCam
    
    
    # Create two camera instances
    camera1 = ZedCam(
        width=640,
        height=480,
        fps=30,
        depth_averaging_frames=3,
        debug=True
    )
    
    camera2 = RsCam(
        width=640,
        height=480,
        fps=30,
        depth_averaging_frames=3,
        debug=True
    )
    
    # Start both cameras
    if camera1.start() and camera2.start():
        print("Both cameras started successfully!")
        
        # Wait for cameras to stabilize
        time.sleep(2)
        
        # Create stereo ArUco detector
        stereo_detector = StereoArucoDetector(
            camera1_instance=camera1,
            camera2_instance=camera2,
            marker_size=0.05,  # 5cm markers
            dictionary_type=cv2.aruco.DICT_6X6_250,
            min_markers_for_transform=3
        )
        
        # Check if detector is ready
        if stereo_detector.is_ready():
            print("Stereo ArUco detector ready!")
            print("\nFor testing with limited markers, try:")
            print("  Press 'm' to set minimum markers to 1")
            print("  Press 'x' to enable simple transform mode")
            print("  This allows translation-only transforms with just 1 marker")
            
            try:
                # Run stereo detection loop with simple transform option
                stereo_detector.run_stereo_detection_loop(
                    show_window=True, 
                    save_detections=True,
                    use_averaging=False,
                    allow_simple_transform=True  # Allow simple transforms for testing
                )
            finally:
                # Clean up
                camera1.stop()
                camera2.stop()
        else:
            print("Stereo detector not ready - cameras may not be fully initialized")
            camera1.stop()
            camera2.stop()
    else:
        print("Failed to start one or both cameras")