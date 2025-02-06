from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import cv2
from dataclasses import dataclass
import torch
from PIL import Image

from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP, FastSAMConfig

@dataclass
class ObjectInfo:
    """Data class for storing detected object information"""
    id: int
    name: str
    mask: np.ndarray
    image: np.ndarray
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    pose: Optional[np.ndarray] = None  # 6D pose [x, y, z, roll, pitch, yaw]
    

class PerceptionSystem:
    def __init__(
        self,
        fast_sam_config: FastSAMConfig,
        camera_matrix: Optional[np.ndarray] = None,
        depth_scale: float = 0.001  # Default for most depth cameras (m/unit)
    ):
        """
        Initialize perception system with FastSAM and camera parameters
        
        Args:
            fast_sam_config: Configuration for FastSAM model
            camera_matrix: 3x3 camera intrinsic matrix. If None, uses default parameters
            depth_scale: Scale factor to convert depth values to meters
        """
        self.segmenter = FastSAMWithCLIP(fast_sam_config)
        
        # Initialize camera parameters
        if camera_matrix is None:
            # Default camera matrix for 640x480 resolution
            self.camera_matrix = np.array([
                [525.0, 0.0, 319.5],
                [0.0, 525.0, 239.5],
                [0.0, 0.0, 1.0]
            ])
        else:
            self.camera_matrix = camera_matrix
            
        self.depth_scale = depth_scale
        self._object_cache = {}  # Cache for detected objects
        
    def detect_object(
        self,
        target_object: str,
        image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.85
    ) -> Optional[ObjectInfo]:
        """
        Detect and segment specific object in the scene
        
        Args:
            target_object: Object name/category to detect
            image: RGB image of the scene
            depth_image: Optional depth image aligned with RGB
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            ObjectInfo if object found, None otherwise
        """
        # Check cache first
        cache_key = f"{target_object}_{hash(image.tobytes())}"
        if cache_key in self._object_cache:
            return self._object_cache[cache_key]
            
        # Use FastSAM with CLIP to find the target object
        labeled_masks, metadata = self.segmenter.process_image(
            image,
            query=f"a {target_object}"
        )
        
        # Find best matching mask
        best_mask_id = None
        best_score = 0
        
        for mask_id, meta in metadata.items():
            score = meta.get('clip_score', 0)
            if score > best_score and score >= confidence_threshold:
                best_score = score
                best_mask_id = mask_id
                
        if best_mask_id is None:
            return None
            
        # Get mask and bounding box
        mask = labeled_masks == best_mask_id
        bbox = metadata[best_mask_id]['bbox']  # [x, y, w, h]
        
        # Crop image to object region
        x, y, w, h = bbox
        object_image = image[y:y+h, x:x+w].copy()
        
        # Create object info
        object_info = ObjectInfo(
            id=best_mask_id,
            name=target_object,
            mask=mask,
            image=object_image,
            bbox=bbox,
            confidence=best_score
        )
        
        # Estimate pose if depth is available
        if depth_image is not None:
            pose = self._estimate_object_pose(mask, depth_image)
            object_info.pose = pose
            
        # Cache result
        self._object_cache[cache_key] = object_info
        return object_info
        
    def get_object_pose(
        self,
        mask: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate 6D pose of object given its mask
        
        Args:
            mask: Binary mask of the object
            depth_image: Optional depth image for better estimation
            
        Returns:
            6D pose array [x, y, z, roll, pitch, yaw]
        """
        if depth_image is not None:
            return self._estimate_object_pose(mask, depth_image)
        else:
            # Fallback to basic pose estimation without depth
            return self._estimate_basic_pose(mask)
            
    def get_3d_point(
        self,
        point_2d: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert 2D image point to 3D world coordinates
        
        Args:
            point_2d: 2D point in image coordinates [x, y]
            depth_image: Optional depth image for accurate Z coordinate
            
        Returns:
            3D point in world coordinates [x, y, z]
        """
        x, y = point_2d.astype(int)
        
        if depth_image is not None:
            # Use depth information
            z = depth_image[y, x] * self.depth_scale
        else:
            # Fallback to assumed depth
            z = 0.5  # Assume 0.5m if no depth available
            
        # Back-project to 3D
        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
        
        return np.array([x_3d, y_3d, z])
        
    def _estimate_object_pose(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray
    ) -> np.ndarray:
        """
        Estimate 6D pose using depth information
        
        Args:
            mask: Binary mask of the object
            depth_image: Depth image aligned with RGB
            
        Returns:
            6D pose array [x, y, z, roll, pitch, yaw]
        """
        # Get object points
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return np.zeros(6)
            
        # Get corresponding depth values
        depth_values = depth_image[y_coords, x_coords] * self.depth_scale
        
        # Filter out invalid depth values
        valid = depth_values > 0
        if not valid.any():
            return np.zeros(6)
            
        x_coords = x_coords[valid]
        y_coords = y_coords[valid]
        depth_values = depth_values[valid]
        
        # Convert to 3D points
        points_3d = np.zeros((len(x_coords), 3))
        points_3d[:, 0] = (x_coords - self.camera_matrix[0, 2]) * depth_values / self.camera_matrix[0, 0]
        points_3d[:, 1] = (y_coords - self.camera_matrix[1, 2]) * depth_values / self.camera_matrix[1, 1]
        points_3d[:, 2] = depth_values
        
        # Calculate centroid and orientation
        centroid = np.mean(points_3d, axis=0)
        
        # Estimate orientation using PCA
        centered_points = points_3d - centroid
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Convert to roll, pitch, yaw
        roll = np.arctan2(eigenvectors[2, 1], eigenvectors[2, 2])
        pitch = np.arctan2(-eigenvectors[2, 0], np.sqrt(eigenvectors[2, 1]**2 + eigenvectors[2, 2]**2))
        yaw = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        return np.array([centroid[0], centroid[1], centroid[2], roll, pitch, yaw])
        
    def _estimate_basic_pose(self, mask: np.ndarray) -> np.ndarray:
        """
        Estimate basic pose without depth information
        
        Args:
            mask: Binary mask of the object
            
        Returns:
            6D pose array [x, y, z, roll, pitch, yaw]
        """
        # Get mask centroid
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return np.zeros(6)
            
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Assume fixed depth
        center_z = 0.5  # 0.5m assumed depth
        
        # Convert to 3D coordinates
        x_3d = (center_x - self.camera_matrix[0, 2]) * center_z / self.camera_matrix[0, 0]
        y_3d = (center_y - self.camera_matrix[1, 2]) * center_z / self.camera_matrix[1, 1]
        
        # Estimate orientation from mask shape
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 0:
            # Fit ellipse to largest contour
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                _, (_, _), angle = cv2.fitEllipse(contour)
                yaw = np.deg2rad(angle)
            else:
                yaw = 0.0
        else:
            yaw = 0.0
            
        # Assume zero roll and pitch without depth information
        return np.array([x_3d, y_3d, center_z, 0.0, 0.0, yaw])
        
    def visualize_detection(
        self,
        image: np.ndarray,
        object_info: ObjectInfo,
        show_pose: bool = True
    ) -> np.ndarray:
        """
        Visualize detected object with pose information
        
        Args:
            image: RGB image to visualize on
            object_info: ObjectInfo for detected object
            show_pose: Whether to show pose axes
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Draw mask overlay
        mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
        mask_overlay[object_info.mask] = [0, 255, 0]  # Green overlay
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, 0.5, 0)
        
        # Draw bounding box
        x, y, w, h = object_info.bbox
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f"{object_info.name} ({object_info.confidence:.2f})"
        cv2.putText(
            vis_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        if show_pose and object_info.pose is not None:
            # Draw pose axes
            pose = object_info.pose
            origin = (int(pose[0]), int(pose[1]))
            
            # X-axis (red)
            end_x = (
                int(origin[0] + 50 * np.cos(pose[5])),
                int(origin[1] + 50 * np.sin(pose[5]))
            )
            cv2.arrowedLine(vis_image, origin, end_x, (0, 0, 255), 2)
            
            # Y-axis (green)
            end_y = (
                int(origin[0] - 50 * np.sin(pose[5])),
                int(origin[1] + 50 * np.cos(pose[5]))
            )
            cv2.arrowedLine(vis_image, origin, end_y, (0, 255, 0), 2)
            
        return vis_image
    