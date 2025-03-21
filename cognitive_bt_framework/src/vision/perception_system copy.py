from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import cv2
from dataclasses import dataclass
import torch
from PIL import Image
import time
import functools
from collections import defaultdict
import traceback

from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP, FastSAMConfig
from cognitive_bt_framework.utils.time_profiler import IterationTimeProfiler


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
    pixel_pose: Optional[np.ndarray] = None  # 2D pose corresponding to pixel in image [x, y]
    

class PerceptionSystem:
    def __init__(
        self,
        fast_sam_config: FastSAMConfig,
        camera_matrix: Optional[np.ndarray] = None,
        depth_scale: float = 0.001,  # Default for most depth cameras (m/unit)
        debug: bool = True
    ):
        """
        Initialize perception system with FastSAM and camera parameters
        
        Args:
            fast_sam_config: Configuration for FastSAM model
            camera_matrix: 3x3 camera intrinsic matrix. If None, uses default parameters
            depth_scale: Scale factor to convert depth values to meters
            debug: Enable time profiling and debug output
        """
        self.debug = debug
        self.profiler = IterationTimeProfiler(enabled=debug)
        
        try:
            # Update FastSAM config to inherit our debug setting if available
            # if hasattr(fast_sam_config, 'debug'):
            #     fast_sam_config.debug = debug
                
            # Initialize segmenter
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
                
            # Pre-calculate camera matrix inverse for faster back-projection
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.fx_inv = 1.0 / self.fx
            self.fy_inv = 1.0 / self.fy
            
            self.depth_scale = depth_scale
            self._object_cache = {}  # Cache for detected objects
            
            if self.debug:
                print(f"PerceptionSystem initialized with debug={debug}")
                
        except Exception as e:
            if self.debug:
                print(f"Error in PerceptionSystem initialization: {str(e)}")
                traceback.print_exc()
            raise  # Re-raise the exception to not silence initialization errors
        
    def detect_object(
        self,
        target_object: str,
        image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.5
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
        # Start a new timing iteration
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_object")
        
        try:
            # Check cache first
            cache_key = f"{target_object}_{hash(image.tobytes())}"
            if cache_key in self._object_cache:
                if self.debug:
                    print(f"Cache hit for '{target_object}'")
                    self.profiler.stop("detect_object")
                    # End iteration but preserve the current timing data
                    self.profiler.end_iteration(preserve_current=True)
                return self._object_cache[cache_key]
                
            if self.debug:
                print(f"Detecting '{target_object}' in image {image.shape}")
                
            # Store original image for later use
            if self.debug:
                self.profiler.start("preprocess_image")
                
            original_image = image.copy()
            original_height, original_width = image.shape[:2]
            
            # Resize only for FastSAM processing
            max_image_size = self.segmenter.config.max_image_size
            scale_factor = max_image_size / max(original_height, original_width)
            resized_height = int(original_height * scale_factor)
            resized_width = int(original_width * scale_factor)
            
            resized_image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Pad to square if needed
            if resized_height != resized_width:
                square_image = np.zeros((max_image_size, max_image_size, 3), dtype=np.uint8)
                y_offset = (max_image_size - resized_height) // 2
                x_offset = (max_image_size - resized_width) // 2
                square_image[y_offset:y_offset+resized_height, 
                            x_offset:x_offset+resized_width] = resized_image
                resized_image = square_image
            
            if self.debug:
                self.profiler.stop("preprocess_image")
            
            # Use FastSAM with CLIP to find the target object
            if self.debug:
                self.profiler.start("segmentation")
                print(f"Running FastSAM segmentation for '{target_object}'")
                
            labeled_masks, metadata = self.segmenter.process_image(
                resized_image,
                query=f"a {target_object}"
            )
            
            if self.debug:
                self.profiler.stop("segmentation")
                print(f"FastSAM returned {len(metadata)} masks for '{target_object}'")
            
            # Find best matching mask
            if self.debug:
                self.profiler.start("find_best_mask")
                
            best_mask_id = None
            best_score = 0
            
            for mask_id, meta in metadata.items():
                score = meta.get('clip_score', 0)
                if self.debug:
                    print(f"  Mask {mask_id} score: {score:.3f}")
                if score > best_score and score >= confidence_threshold:
                    best_score = score
                    best_mask_id = mask_id
                    
            if best_mask_id is None:
                if self.debug:
                    print(f"No mask found for '{target_object}' with confidence >= {confidence_threshold}")
                    self.profiler.stop("find_best_mask")
                    self.profiler.stop("detect_object")
                    # End iteration but preserve the current timing data
                    self.profiler.end_iteration(preserve_current=True)
                return None
            
            if self.debug:
                print(f"Best mask for '{target_object}': {best_mask_id} with score {best_score:.3f}")
                self.profiler.stop("find_best_mask")
                
            # Get mask from segmentation output
            if self.debug:
                self.profiler.start("process_mask")
                
            mask = labeled_masks == best_mask_id
            
            # If we padded the image, remove padding from mask
            if resized_height != resized_width:
                mask = mask[y_offset:y_offset+resized_height, 
                        x_offset:x_offset+resized_width]
            
            # Resize mask back to original dimensions
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            if self.debug:
                self.profiler.stop("process_mask")
            
            # Get bounding box in original image coordinates
            if self.debug:
                self.profiler.start("compute_bbox")
                
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                if self.debug:
                    print(f"Empty mask for '{target_object}'")
                    self.profiler.stop("compute_bbox")
                    self.profiler.stop("detect_object")
                    # End iteration but preserve the current timing data
                    self.profiler.end_iteration(preserve_current=True)
                return None
                
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            if self.debug:
                self.profiler.stop("compute_bbox")
            
            # Create object info
            if self.debug:
                self.profiler.start("create_object_info")
                
            object_info = ObjectInfo(
                id=best_mask_id,
                name=target_object,
                mask=mask,
                image=original_image,
                bbox=bbox,
                confidence=best_score
            )
            
            if self.debug:
                self.profiler.stop("create_object_info")
            
            # Estimate pose if depth is available
            if depth_image is not None:
                if self.debug:
                    self.profiler.start("estimate_pose")
                    
                # Ensure depth image matches RGB dimensions
                if depth_image.shape[:2] != original_image.shape[:2]:
                    depth_image = cv2.resize(
                        depth_image,
                        (original_width, original_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                pose, pixel_pose = self._estimate_object_pose(mask, depth_image)
                object_info.pose = pose
                object_info.pixel_pose = pixel_pose
                
                if self.debug:
                    self.profiler.stop("estimate_pose")
                
            # Cache result
            self._object_cache[cache_key] = object_info
            
            if self.debug:
                print(f"Successfully detected '{target_object}' with confidence {best_score:.3f}")
            
            return object_info
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_object for '{target_object}': {str(e)}")
                traceback.print_exc()
            # Return None on error to maintain original behavior
            return None
            
        finally:
            if self.debug:
                self.profiler.stop("detect_object")
                # End iteration but preserve the current timing data for reporting
                self.profiler.end_iteration(preserve_current=True)
                
                # Print basic timing info after each detection
                current_time = self.profiler.get_current_iteration_total("detect_object")
                avg_time = self.profiler.get_avg_time("detect_object")
                min_time = self.profiler.get_min_time("detect_object")
                max_time = self.profiler.get_max_time("detect_object")
                
                if self.profiler.iteration_count > 1:
                    print(f"\nDetection timing: Current={current_time:.3f}s | Avg={avg_time:.3f}s | "
                          f"Min={min_time:.3f}s | Max={max_time:.3f}s")
        
    def get_object_pose(
        self,
        mask: np.ndarray,
        depth_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Estimate 6D pose of object given its mask
        
        Args:
            mask: Binary mask of the object
            depth_image: Optional depth image for better estimation
            
        Returns:
            Tuple containing:
            - 6D pose array [x, y, z, roll, pitch, yaw]
            - 2D pixel coordinates (x, y) in the camera frame
        """
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("get_object_pose")
        
        try:
            if depth_image is not None:
                pose, pixel_pose = self._estimate_object_pose(mask, depth_image)
                return pose, pixel_pose
            else:
                # Fallback to basic pose estimation without depth
                pose = self._estimate_basic_pose(mask)
                # Get mask centroid for pixel position
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    return np.zeros(6), (0, 0)
                    
                pixel_x = int(np.mean(x_coords))
                pixel_y = int(np.mean(y_coords))
                return pose, (pixel_x, pixel_y)
                
        except Exception as e:
            if self.debug:
                print(f"Error in get_object_pose: {str(e)}")
                traceback.print_exc()
            return np.zeros(6), (0, 0)
        
        finally:
            if self.debug:
                self.profiler.stop("get_object_pose")
                self.profiler.end_iteration()
            
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
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("get_3d_point")
        
        try:
            x, y = point_2d.astype(int)
            
            if depth_image is not None:
                # Use depth information
                z = depth_image[y, x] * self.depth_scale
            else:
                # Fallback to assumed depth
                z = 0.5  # Assume 0.5m if no depth available
                
            # Back-project to 3D using pre-computed values
            x_3d = (x - self.cx) * z / self.fx
            y_3d = (y - self.cy) * z / self.fy
            
            return np.array([x_3d, y_3d, z])
            
        except Exception as e:
            if self.debug:
                print(f"Error in get_3d_point: {str(e)}")
            return np.zeros(3)
            
        finally:
            if self.debug:
                self.profiler.stop("get_3d_point")
                self.profiler.end_iteration()
        
    def _estimate_object_pose(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Estimate 6D pose and 2D pixel location using depth information
        
        Args:
            mask: Binary mask of the object
            depth_image: Depth image aligned with RGB
            
        Returns:
            Tuple containing:
            - 6D pose array [x, y, z, roll, pitch, yaw]
            - 2D pixel coordinates (x, y) in the camera frame
        """
        if self.debug:
            self.profiler.start("_estimate_object_pose")
        
        try:
            # Ensure mask and depth image have same dimensions
            if mask.shape != depth_image.shape:
                # Resize mask to match depth image dimensions
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_image.shape[1], depth_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # Get object points
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                return np.zeros(6), (0, 0)
                
            # Get corresponding depth values
            try:
                depth_values = depth_image[y_coords, x_coords] * self.depth_scale
            except IndexError as e:
                if self.debug:
                    print(f"IndexError in _estimate_object_pose: mask shape {mask.shape}, "
                        f"depth shape {depth_image.shape}, coords max ({np.max(y_coords)}, {np.max(x_coords)})")
                return np.zeros(6), (0, 0)
            
            # Filter out invalid depth values
            valid = depth_values > 0
            if not valid.any():
                return np.zeros(6), (0, 0)
                
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            depth_values = depth_values[valid]
            
            # If we have too many points, sample a subset for efficiency
            MAX_POINTS = 1000
            if len(depth_values) > MAX_POINTS:
                indices = np.random.choice(len(depth_values), MAX_POINTS, replace=False)
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]
                depth_values = depth_values[indices]
            
            # Convert to 3D points
            points_3d = np.zeros((len(x_coords), 3))
            points_3d[:, 0] = (x_coords - self.cx) * depth_values / self.fx
            points_3d[:, 1] = (y_coords - self.cy) * depth_values / self.fy
            points_3d[:, 2] = depth_values
            
            # Calculate centroid
            centroid = np.mean(points_3d, axis=0)
            
            # Project 3D centroid back to 2D
            pixel_x = int((centroid[0] * self.fx) / centroid[2] + self.cx)
            pixel_y = int((centroid[1] * self.fy) / centroid[2] + self.cy)
            
            # Ensure pixels are within image bounds
            pixel_x = max(0, min(pixel_x, depth_image.shape[1] - 1))
            pixel_y = max(0, min(pixel_y, depth_image.shape[0] - 1))
            
            # Handle case where we don't have enough points for PCA
            if len(points_3d) < 3:
                # Return position only with zero rotation
                return np.array([centroid[0], centroid[1], centroid[2], 0.0, 0.0, 0.0]), (pixel_x, pixel_y)
            
            # Calculate orientation using PCA
            centered_points = points_3d - centroid
            try:
                # Compute covariance directly instead of using np.cov (more efficient)
                covariance_matrix = np.dot(centered_points.T, centered_points) / centered_points.shape[0]
                eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
                
                # Sort eigenvectors by eigenvalues (eigh returns them in ascending order)
                sort_indices = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, sort_indices]
                
                # Convert to roll, pitch, yaw
                roll = np.arctan2(eigenvectors[2, 1], eigenvectors[2, 2])
                pitch = np.arctan2(-eigenvectors[2, 0], np.sqrt(eigenvectors[2, 1]**2 + eigenvectors[2, 2]**2))
                yaw = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            except np.linalg.LinAlgError:
                # Fallback to zero rotation if PCA fails
                roll, pitch, yaw = 0.0, 0.0, 0.0
            
            pose_6d = np.array([centroid[0], centroid[1], centroid[2], roll, pitch, yaw])
            return pose_6d, (pixel_x, pixel_y)
            
        except Exception as e:
            if self.debug:
                print(f"Error in _estimate_object_pose: {str(e)}")
                traceback.print_exc()
            return np.zeros(6), (0, 0)
            
        finally:
            if self.debug:
                self.profiler.stop("_estimate_object_pose")
            
    def _estimate_basic_pose(self, mask: np.ndarray) -> np.ndarray:
        """
        Estimate basic pose without depth information
        
        Args:
            mask: Binary mask of the object
            
        Returns:
            6D pose array [x, y, z, roll, pitch, yaw]
        """
        if self.debug:
            self.profiler.start("_estimate_basic_pose")
        
        try:
            # Get mask centroid
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                return np.zeros(6)
                
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Assume fixed depth
            center_z = 0.5  # 0.5m assumed depth
            
            # Convert to 3D coordinates using precomputed values
            x_3d = (center_x - self.cx) * center_z / self.fx
            y_3d = (center_y - self.cy) * center_z / self.fy
            
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
            
        except Exception as e:
            if self.debug:
                print(f"Error in _estimate_basic_pose: {str(e)}")
            return np.zeros(6)
            
        finally:
            if self.debug:
                self.profiler.stop("_estimate_basic_pose")
        
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
        if self.debug:
            self.profiler.start("visualize_detection")
        
        try:
            if object_info is None:
                if self.debug:
                    print("Cannot visualize None object_info")
                return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                
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
                if object_info.pixel_pose is not None:
                    # Use pixel_pose for origin if available
                    origin = object_info.pixel_pose
                else:
                    # Project 3D pose to image
                    pose = object_info.pose
                    depth = pose[2]
                    if depth > 0:
                        pixel_x = int((pose[0] * self.fx) / depth + self.cx)
                        pixel_y = int((pose[1] * self.fy) / depth + self.cy)
                        origin = (pixel_x, pixel_y)
                    else:
                        # Fallback to bounding box center
                        origin = (x + w // 2, y + h // 2)
                
                # Get pose angles
                pose = object_info.pose
                
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
                
            # Add profiling info if debug is enabled
            if self.debug:
                # Get performance metrics without clearing them
                # Current iteration times
                detect_time = self.profiler.get_current_iteration_total("detect_object")
                segmentation_time = self.profiler.get_current_iteration_total("segmentation")
                pose_time = self.profiler.get_current_iteration_total("estimate_pose") 
                
                # Average times (historical)
                avg_detection = self.profiler.get_avg_time("detect_object")
                
                # Create timing summary
                info_text = f"Detection: {detect_time:.3f}s | Segmentation: {segmentation_time:.3f}s"
                if pose_time > 0:
                    info_text += f" | Pose: {pose_time:.3f}s"
                
                # Add second line with averages
                avg_text = f"Avg Detection: {avg_detection:.3f}s (over {self.profiler.iteration_count} iterations)"
                
                # Add timing overlay at bottom of image
                cv2.rectangle(vis_image, (0, vis_image.shape[0]-50), (vis_image.shape[1], vis_image.shape[0]), (0,0,0), -1)
                
                # Add current timing
                cv2.putText(
                    vis_image,
                    info_text,
                    (10, vis_image.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1
                )
                
                # Add average timing
                cv2.putText(
                    vis_image,
                    avg_text,
                    (10, vis_image.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1
                )
                
            return vis_image
            
        except Exception as e:
            if self.debug:
                print(f"Error in visualize_detection: {str(e)}")
            # Return original image on error
            return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
        finally:
            if self.debug:
                self.profiler.stop("visualize_detection")
    
    def print_performance_summary(self, sort_by="total", top_n=None, compact=True):
        """
        Print summary of performance timing
        
        Args:
            sort_by: Field to sort by (total, average, current_total, hierarchy)
            top_n: Limit to top N sections (by sort criteria)
            compact: Use compact display format (less columns)
        """
        if self.debug:
            self.profiler.print_summary(sort_by=sort_by, top_n=top_n, compact=compact)
        else:
            print("Performance profiling is disabled. Create PerceptionSystem with debug=True to enable.")
            
    def reset_profiler(self):
        """Reset the profiler timings"""
        if self.debug:
            self.profiler.reset()
            print("Performance profiler has been reset")

    def get_performance_stats(self):
        """Get performance statistics as a dictionary"""
        if not self.debug:
            return {"debug_enabled": False}
            
        # Get complete summary
        summary = self.profiler.get_summary(sort_by="total")
        
        # Extract key metrics
        stats = {
            "debug_enabled": True,
            "iterations": self.profiler.iteration_count - 1,  # Don't count the current iteration
            
            # Current iteration stats
            "current": {},
            
            # Historical stats
            "average": {},
            "min": {},
            "max": {}
        }
        
        # Fill in data for each section
        for item in summary:
            section = item['section']
            stats["current"][section] = item['current_total']
            stats["average"][section] = item['average']
            stats["min"][section] = item['min']
            stats["max"][section] = item['max']
        
        return stats

    def get_profiler_hotspots(self, top_n=5):
        """
        Get the top N hotspots in performance
        
        Returns a dictionary with the top N slowest sections and their times
        """
        if not self.debug:
            return {"debug_enabled": False}
            
        summary = self.profiler.get_summary(sort_by="current_total", top_n=top_n)
        
        hotspots = {
            "current_iteration": self.profiler.iteration_count,
            "sections": []
        }
        
        for item in summary:
            hotspots["sections"].append({
                "name": item['section'],
                "current_time": item['current_total'],
                "avg_time": item['average'],
                "calls": item['current_calls']
            })
            
        return hotspots