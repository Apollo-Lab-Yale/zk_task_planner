from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import cv2
from dataclasses import dataclass, field
import torch
from PIL import Image
import time
import functools
from collections import defaultdict
import traceback

# Import the FastSAMMaskGenerator and FastSAMConfig instead of FastSAMWithCLIP
from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMMaskGenerator, FastSAMConfig
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.utils.time_profiler import IterationTimeProfiler
from ultralytics import YOLOWorld, YOLOE


@dataclass
class ObjectInfo:
    """Data class for storing detected object information"""
    id: int
    name: str
    bbox: List[int]  # [x, y, w, h] format
    confidence: float
    image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # 6D pose [x, y, z, roll, pitch, yaw]
    pixel_pose: Optional[np.ndarray] = None  # 2D pose corresponding to pixel in image [x, y]
    components: Dict[str, 'ObjectInfo'] = None  # Store detected components within this object
    depth_image: Optional[np.ndarray] = None
    alpha_id: Optional[str] = None  # Alphabetical ID for user-friendly identification
    points: Optional[np.ndarray] = None
    surface_masks: Dict[str, np.ndarray] = None,
    camera_intrinsics: Dict[str, float] = None
    

import random
import string

def get_alpha_id(num):
    """
    Generate a random sequence of three lowercase letters.
    Each call returns a new random ID regardless of the number passed.
    """
    # Generate 3 random lowercase letters
    letters = ''.join(random.choices(string.ascii_lowercase, k=3))
    return letters

class PerceptionSystem:
    def __init__(
        self,
        fast_sam_config: Optional[FastSAMConfig],
        yolo_model_path: str = 'yoloe-v8l-seg.pt',
        camera_matrix: Optional[np.ndarray] = None,
        depth_scale: float = 0.001,  # Default for most depth cameras (m/unit)
        default_conf: float = 0.5,
        default_classes: List[str] = None,
        debug: bool = True
    ):
        """
        Initialize perception system with YOLO-World and FastSAM models
        
        Args:
            yolo_model_path: Path to YOLO-World model weights
            fast_sam_config: Configuration for FastSAM model (optional)
            camera_matrix: 3x3 camera intrinsic matrix. If None, uses default parameters
            depth_scale: Scale factor to convert depth values to meters
            default_conf: Default confidence threshold for YOLO-World detections
            default_classes: Default class list for YOLO-World
            debug: Enable time profiling and debug output
        """
        self.debug = debug
        self.profiler = IterationTimeProfiler(enabled=debug)
        
        try:
            # Initialize YOLO-World detector
            if self.debug:
                print(f"Initializing YOLO-World detector with model {yolo_model_path}")
                
            self.detector = YOLOE()
            self.default_conf = default_conf
            
            # Set default classes if provided
            if default_classes:
                self.detector.set_classes(default_classes)
                if self.debug:
                    print(f"Set default classes: {default_classes}")
            
            # Initialize FastSAM for segmentation (if config is provided)
            self.segmenter = None
            if fast_sam_config is not None:
                # Initialize segmenter
                self.segmenter = FastSAMMaskGenerator(fast_sam_config)
                if self.debug:
                    print("FastSAM segmenter initialized")
            
            # Initialize camera parameters
            if camera_matrix is None:
                self.camera_matrix = np.array([
                    [429.92523193359375, 0.0, 431.7160339355469],
                    [0.0, 429.92523193359375, 233.39739990234375],
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
            self._next_mask_id = 1  # Counter for assigning IDs to masks
            
            if self.debug:
                print(f"PerceptionSystem initialized with debug={debug}")
                
        except Exception as e:
            if self.debug:
                print(f"Error in PerceptionSystem initialization: {str(e)}")
                traceback.print_exc()
            raise  # Re-raise the exception to not silence initialization errors
    
    def detect_objects(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = 0.01,
        segment: bool = True,
        depth_image: Optional[np.ndarray] = None,
    ) -> List[ObjectInfo]:
        """
        Detect objects in the scene using YOLO-World
        
        Args:
            image: RGB image of the scene
            classes: List of classes to detect (None = use default classes)
            conf: Confidence threshold (None = use default)
            segment: Whether to generate segmentation masks for detected objects
            depth_image: Optional depth image aligned with RGB
            
        Returns:
            List of ObjectInfo for detected objects
        """
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_objects")
        
        try:
            # Use provided parameters or defaults
            if classes is not None:
                self.detector.set_classes(classes, self.detector.get_text_pe(classes))
                if self.debug:
                    print(f"Set detection classes: {classes}")
            
            conf_threshold = conf if conf is not None else self.default_conf
            
            # Run YOLO-World detection
            if self.debug:
                self.profiler.start("yolo_detection")
                print(f"Running YOLO-World detection with confidence threshold {conf_threshold}")
            
            results = self.detector.predict(image, verbose=False)
            
            if self.debug:
                self.profiler.stop("yolo_detection")
                print(f"YOLO-World returned {len(results[0])} detections")
            
            # Convert results to ObjectInfo objects
            object_infos = []
            
            if self.debug:
                self.profiler.start("process_detections")
            
            # Process each detection
            for i, detection in enumerate(results[0]):
                # Extract bounding box in [x, y, w, h] format
                xyxy = detection.boxes.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, xyxy)
                bbox = [x1, y1, x2-x1, y2-y1]
                
                # Extract class and confidence
                cls_id = int(detection.boxes.cls.cpu().numpy()[0])
                confidence = float(detection.boxes.conf.cpu().numpy()[0])
                class_name = detection.names[cls_id]
                
                # Create alphabetical ID
                alpha_id = get_alpha_id(self._next_mask_id)
                self._next_mask_id += 1
                mask_coords = detection.masks.xy[0]
                bool_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                num_masks = len(detection.masks)
    
                # Process each mask and merge them
                for i in range(num_masks):
                    try:
                        # Get mask coordinates for current mask
                        mask_coords = detection.masks.xy[i]
                        
                         # Convert to numpy array of integers, make sure it's properly formatted
                        points = np.array(mask_coords, dtype=np.int32)
                        
                        # Create a temporary mask to hold this contour
                        temp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        
                        # Draw filled contour on the temporary mask
                        cv2.fillPoly(temp_mask, [points], 1)
                        
                        # Convert to boolean and merge with main mask
                        bool_mask = np.logical_or(bool_mask, temp_mask.astype(bool))
                        
                        if self.debug and i > 0:
                            print(f"Merged mask {i+1}/{num_masks} for object {class_name}")
                            
                    except Exception as e:
                        if self.debug:
                            print(f"Error processing mask {i}: {str(e)}")
                            continue
                # Set these pixels to True in the boolean mask
                # Create ObjectInfo
                obj_info = ObjectInfo(
                    id=i,
                    name=class_name,
                    bbox=bbox,
                    mask=bool_mask,
                    confidence=confidence,
                    image=image,
                    depth_image=depth_image,
                    alpha_id=alpha_id,
                )
                obj_info.points = self.detect_regions_of_interest(image, obj_info, max_points=10)
                # Initialize surface_masks
                obj_info.surface_masks = {}

                # Generate surface segmentation if requested and object has a valid mask
                if segment and self.segmenter is not None and obj_info.mask is not None:
                    if self.debug:
                        self.profiler.start(f"segment_surfaces_{i}")
                        print(f"Generating surface segmentation for object {i} ({class_name})")
                    
                    obj_info.surface_masks = self.segment_surfaces_by_plane_fitting(
                        image=image,
                        obj_info=obj_info,
                        depth_image=depth_image,
                    )
                    
                    if self.debug:
                        self.profiler.stop(f"segment_surfaces_{i}")
                        print(f"Detected {len(obj_info.surface_masks)} surfaces for object {i}")
                # Generate segmentation mask if requested
                # if segment and self.segmenter is not None:
                #     if self.debug:
                #         self.profiler.start(f"segment_obj_{i}")
                #         print(f"Generating segmentation for object {i} ({class_name})")
                    
                #     # mask = self._generate_segmentation(image, obj_info)
                #     mask = None
                #     # Fallback to bbox mask if segmentation failed
                #     if mask is None:
                #         if self.debug:
                #             print(f"Segmentation failed for object {i}, falling back to bbox mask")
                        
                #         # Create a simple binary mask from bbox
                #         mask = np.zeros(image.shape[:2], dtype=bool)
                #         x, y, w, h = bbox
                #         mask[y:y+h, x:x+w] = True
                    
                #     obj_info.mask = mask
                    
                #     obj_info.points = self.detect_regions_of_interest(image, obj_info, max_points=10)
                
                # Estimate pose if depth image is provided
                if depth_image is not None:
                    if self.debug:
                        self.profiler.start(f"estimate_pose_{i}")
                    
                    # If we have a mask, use it for better pose estimation
                    if obj_info.mask is not None:
                        pose, pixel_pose = self._estimate_object_pose(obj_info.mask, depth_image)
                    else:
                        # Create a simple mask from bbox for pose estimation
                        temp_mask = np.zeros(image.shape[:2], dtype=bool)
                        x, y, w, h = bbox
                        temp_mask[y:y+h, x:x+w] = True
                        pose, pixel_pose = self._estimate_object_pose(temp_mask, depth_image)
                    
                    obj_info.pose = pose
                    obj_info.pixel_pose = pixel_pose
                    
                    if self.debug:
                        self.profiler.stop(f"estimate_pose_{i}")
                
                object_infos.append(obj_info)
            
            if self.debug:
                self.profiler.stop("process_detections")
                print(f"Processed {len(object_infos)} detections")
            
            return object_infos
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_objects: {str(e)}")
                traceback.print_exc()
            return []
            
        finally:
            if self.debug:
                self.profiler.stop("detect_objects")
                self.profiler.end_iteration(preserve_current=True)
    
    
    
    
    
    def detect_object(
        self,
        target_object: str,
        image: np.ndarray,
        conf: Optional[float] = 0.01,
        segment: bool = True,
        depth_image: Optional[np.ndarray] = None,
    ) -> Optional[ObjectInfo]:
        """
        Detect a specific object in the scene
        
        Args:
            target_object: Object name/category to detect
            image: RGB image of the scene
            conf: Confidence threshold (None = use default)
            segment: Whether to generate segmentation mask
            depth_image: Optional depth image aligned with RGB
            
        Returns:
            ObjectInfo if object found, None otherwise
        """
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_object")
        
        try:
            # Set single class for detection
            objects = self.detect_objects(
                image=image,
                classes=[target_object],
                conf=conf,
                segment=segment,
                depth_image=depth_image
            )
            
            # Return highest confidence detection or None
            if not objects:
                if self.debug:
                    print(f"No '{target_object}' detected")
                return None
            
            # Get best match (highest confidence)
            best_object = max(objects, key=lambda obj: obj.confidence)
            
            # Cache the result
            
            if self.debug:
                print(f"Found '{target_object}' with confidence {best_object.confidence:.3f}")
            
            return best_object
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_object: {str(e)}")
                traceback.print_exc()
            return None
            
        finally:
            if self.debug:
                self.profiler.stop("detect_object")
                self.profiler.end_iteration(preserve_current=True)

    def detect_object_parts(
        self,
        image: np.ndarray,
        bbox: List[int],
        conf_threshold: float = 0.01,
        depth_image: Optional[np.ndarray] = None
    ) -> Dict[int, ObjectInfo]:
        """
        Detect parts within a region of interest using FastSAM
        
        Args:
            image: Input image
            bbox: Bounding box of the region of interest [x, y, w, h]
            conf_threshold: Confidence threshold for part detection
            depth_image: Optional depth image
            
        Returns:
            Dictionary mapping mask IDs to ObjectInfo objects
        """
        if self.segmenter is None:
            if self.debug:
                print("Cannot detect parts: FastSAM segmenter not initialized")
            return {}
            
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_parts")
        
        try:
            # Extract region of interest with padding
            x, y, w, h = bbox
            pad = int(max(w, h) * 0.05)  # 5% padding
            
            # Ensure coordinates are within image bounds
            x_roi = max(0, x - pad)
            y_roi = max(0, y - pad)
            w_roi = min(image.shape[1] - x_roi, w + 2*pad)
            h_roi = min(image.shape[0] - y_roi, h + 2*pad)
            
            # Store ROI coordinates for later use
            roi_coords = (x_roi, y_roi, w_roi, h_roi)
            
            # Extract ROI
            roi = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            
            if roi.size == 0:
                if self.debug:
                    print(f"Invalid ROI with dimensions {roi.shape}")
                return {}
            
            # Generate masks for everything in the ROI
            if self.debug:
                self.profiler.start("generate_masks")
                
            try:
                # Use "everything" prompt to segment all parts
                labeled_masks, metadata = self.segmenter.generate_masks(
                    roi,
                    prompt_type="everything"
                )
                
                if self.debug:
                    print(f"FastSAM returned {len(metadata)} potential parts in object region")
                    self.profiler.stop("generate_masks")
            except Exception as e:
                if self.debug:
                    print(f"Error generating masks: {e}")
                    self.profiler.stop("generate_masks")
                return {}
            
            # Process the masks to create ObjectInfo objects
            if self.debug:
                self.profiler.start("process_parts")
                
            parts = {}
            
            # Implement filtering to remove very small masks and masks that are too large 
            # (likely the whole object rather than a part)
            total_roi_area = w_roi * h_roi
            min_part_area = total_roi_area * 0.01  # Parts should be at least 1% of ROI
            max_part_area = total_roi_area * 0.8   # Parts shouldn't be more than 80% of ROI
                
            for mask_id, meta in metadata.items():
                # Skip parts that are too small or too large
                area = meta.get('area', 0)
                if area < min_part_area or area > max_part_area:
                    if self.debug:
                        print(f"Skipping mask {mask_id}: area={area:.0f} (outside range [{min_part_area:.0f}, {max_part_area:.0f}])")
                    continue

                # Create a full image mask
                full_mask = np.zeros(image.shape[:2], dtype=bool)
                
                # Get the mask from labeled masks
                roi_mask = labeled_masks == mask_id
                
                # Try to place ROI mask in the correct position in the full image
                try:
                    full_mask[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = roi_mask
                except ValueError as e:
                    # Handle potential size mismatch
                    if self.debug:
                        print(f"Error placing mask in full image: {e}")
                        
                    try:
                        # Resize mask to match ROI dimensions
                        resized_mask = cv2.resize(
                            roi_mask.astype(np.uint8), 
                            (w_roi, h_roi), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                        
                        full_mask[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = resized_mask
                    except Exception as resize_err:
                        if self.debug:
                            print(f"Failed to resize mask: {resize_err}")
                        continue  # Skip this mask if we can't resize it properly
                
                # Convert bounding box to image coordinates
                part_bbox = meta['bbox']
                img_bbox = [
                    part_bbox[0] + x_roi,
                    part_bbox[1] + y_roi,
                    part_bbox[2],
                    part_bbox[3]
                ]
                
                # Create a new ID and alpha ID for this part
                part_id = self._next_mask_id
                self._next_mask_id += 1
                alpha_id = get_alpha_id(part_id)
                
                # Create ObjectInfo for this part
                part_info = ObjectInfo(
                    id=part_id,
                    name=f"part_{alpha_id}",
                    bbox=img_bbox,
                    confidence=conf_threshold,  # Default confidence
                    image=image,
                    mask=full_mask,
                    depth_image=depth_image,
                    alpha_id=alpha_id
                )
                
                # Estimate pose if depth image is provided
                if depth_image is not None:
                    pose, pixel_pose = self._estimate_object_pose(full_mask, depth_image)
                    part_info.pose = pose
                    part_info.pixel_pose = pixel_pose
                
                parts[part_id] = part_info
            
            if self.debug:
                self.profiler.stop("process_parts")
                print(f"Processed {len(parts)} parts")
            
            return parts
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_object_parts: {str(e)}")
                traceback.print_exc()
            return {}
            
        finally:
            if self.debug:
                self.profiler.stop("detect_parts")
                self.profiler.end_iteration(preserve_current=True)
            
    def _generate_segmentation(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo
    ) -> np.ndarray:
        """
        Generate segmentation mask for detected object using FastSAM, focusing only within
        the object's bounding box
        
        Args:
            image: Original image
            obj_info: Object information with bounding box
            
        Returns:
            Binary mask array for the full image with mask only in the detected region
        """
        if self.segmenter is None:
            if self.debug:
                print("Cannot generate segmentation: FastSAM segmenter not initialized")
            return None
            
        try:
            # Extract the region of interest from the bounding box
            x, y, w, h = obj_info.bbox
            
            if self.debug:
                print(f"Generating segmentation for '{obj_info.name}' with bbox: x={x}, y={y}, w={w}, h={h}")
            
            # Check for invalid bounding box
            if w <= 0 or h <= 0:
                if self.debug:
                    print(f"Invalid bounding box dimensions: w={w}, h={h}")
                return None
                    
            # Check if bbox is outside image bounds
            if x >= image.shape[1] or y >= image.shape[0]:
                if self.debug:
                    print(f"Bounding box outside image bounds: x={x}, y={y}, image={image.shape}")
                return None
            
            # Expand the region slightly to ensure we capture the full object
            pad = int(max(w, h) * 0.1)  # Add 10% padding around the object
            x_expand = max(0, x - pad)
            y_expand = max(0, y - pad)
            w_expand = min(image.shape[1] - x_expand, w + 2*pad)
            h_expand = min(image.shape[0] - y_expand, h + 2*pad)
            
            if self.debug:
                print(f"Expanded ROI: x={x_expand}, y={y_expand}, w={w_expand}, h={h_expand}")
            
            # Extract expanded ROI
            try:
                roi = image[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand].copy()
                
                if roi.size == 0:
                    if self.debug:
                        print("Empty ROI after extraction")
                    return None
                    
                if self.debug:
                    print(f"ROI shape: {roi.shape}")
            except Exception as e:
                if self.debug:
                    print(f"Error extracting ROI: {e}")
                return None
            
            # Ensure ROI is at least 16x16 pixels for FastSAM (minimum size requirement)
            min_size = 16
            if roi.shape[0] < min_size or roi.shape[1] < min_size:
                if self.debug:
                    print(f"ROI too small ({roi.shape}), resizing to minimum size")
                roi = cv2.resize(roi, (max(min_size, roi.shape[1]), max(min_size, roi.shape[0])))
            
            # Create a box prompt that focuses on the central part of the ROI
            roi_h, roi_w = roi.shape[:2]
            center_box = [0, 0, roi.shape[1], roi.shape[0]]
            
            # Run FastSAM on the ROI with box prompt
            if self.debug:
                print(f"Running FastSAM with box prompt")
                
            try:
                # Use the box prompt to focus segmentation within ROI
                labeled_masks, metadata = self.segmenter.generate_masks(
                    roi,
                    prompt_type="boxes",
                    boxes=[center_box]
                )
                
                if self.debug:
                    print(f"FastSAM returned {len(metadata)} masks")
            except Exception as e:
                if self.debug:
                    print(f"Error in FastSAM processing: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Fallback to "everything" prompt if box prompt fails
                try:
                    if self.debug:
                        print("Falling back to 'everything' prompt")
                    labeled_masks, metadata = self.segmenter.generate_masks(
                        roi,
                        prompt_type="everything"
                    )
                    if self.debug:
                        print(f"FastSAM fallback returned {len(metadata)} masks")
                except Exception as e2:
                    if self.debug:
                        print(f"Fallback also failed: {e2}")
                    return None
            
            # Find largest mask or mask with highest area overlapping the central box
            best_mask_id = None
            largest_area = 0
            
            # Calculate the central box region as a numpy mask
            central_mask = np.zeros((roi_h, roi_w), dtype=bool)
            x1, y1, x2, y2 = [int(v) for v in center_box]
            central_mask[y1:y2, x1:x2] = True
            
            for mask_id, meta in metadata.items():
                # Get current mask
                mask = labeled_masks == mask_id
                
                # Calculate overlap with central box region
                overlap = np.logical_and(mask, central_mask).sum()
                
                # Use a scoring function that favors masks with good central overlap
                area = meta.get('area', 0)
                central_score = overlap / max(1, central_mask.sum())  # What percentage of center is covered
                score = area * (0.5 + 0.5 * central_score)  # Weighted score
                
                if self.debug:
                    print(f"  Mask {mask_id} ({meta.get('alpha_id', '')}): area={area:.0f}, overlap={central_score:.2f}, score={score:.0f}")
                
                if score > largest_area:
                    largest_area = score
                    best_mask_id = mask_id
                        
            if best_mask_id is None:
                if self.debug:
                    print(f"No suitable mask found for '{obj_info.name}' in ROI")
                    
                # Create a fallback mask from the bounding box
                if self.debug:
                    print("Creating fallback mask from bounding box")
                full_mask = np.zeros(image.shape[:2], dtype=bool)
                full_mask[y:y+h, x:x+w] = True
                return full_mask
                    
            # Get ROI mask
            roi_mask = labeled_masks == best_mask_id
            
            # Check if mask is empty
            if not np.any(roi_mask):
                if self.debug:
                    print(f"Empty mask for best match (ID: {best_mask_id})")
                    
                # Create a fallback mask from the bounding box
                if self.debug:
                    print("Creating fallback mask from bounding box")
                full_mask = np.zeros(image.shape[:2], dtype=bool)
                full_mask[y:y+h, x:x+w] = True
                return full_mask
            
            # Create a full image mask (initialized to all False)
            full_mask = np.zeros(image.shape[:2], dtype=bool)
            
            # Place ROI mask in the correct position in the full image
            try:
                full_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = roi_mask
                
                if self.debug:
                    print(f"Created full mask with {np.sum(full_mask)} pixels")
            except ValueError as e:
                if self.debug:
                    print(f"Error placing mask in full image: {e}")
                    print(f"ROI mask shape: {roi_mask.shape}, Expected: {(h_expand, w_expand)}")
                    print(f"Full image shape: {full_mask.shape}")
                    
                # Try to resize the mask to fit
                try:
                    resized_mask = cv2.resize(
                        roi_mask.astype(np.uint8), 
                        (w_expand, h_expand), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    full_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = resized_mask
                    
                    if self.debug:
                        print(f"Resized mask to fit ROI: {resized_mask.shape}")
                except Exception as resize_err:
                    if self.debug:
                        print(f"Failed to resize mask: {resize_err}")
                    # Fallback to bounding box mask
                    full_mask[y:y+h, x:x+w] = True
            
            return full_mask
            
        except Exception as e:
            if self.debug:
                print(f"Error in _generate_segmentation: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Create a fallback mask from the bounding box as a last resort
            try:
                if self.debug:
                    print("Creating emergency fallback mask from bounding box")
                full_mask = np.zeros(image.shape[:2], dtype=bool)
                full_mask[y:y+h, x:x+w] = True
                return full_mask
            except:
                return None
    
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
            # Apply statistical outlier filtering
            MAX_DEPTH = 2.0  # Maximum reasonable depth in meters
            MIN_DEPTH = 0.05  # Minimum reasonable depth in meters
            
            # Filter out invalid readings and obvious outliers
            valid = (depth_values > MIN_DEPTH) & (depth_values < MAX_DEPTH)
            
            # If we have enough valid points, apply statistical filtering
            if np.sum(valid) > 10:
                # Get valid depth values
                valid_depths = depth_values[valid]
                
                # Calculate statistics
                depth_mean = np.mean(valid_depths)
                depth_std = np.std(valid_depths)
                
                # Filter values within reasonable standard deviation range
                valid_range = (depth_values > depth_mean - 2.0 * depth_std) & \
                            (depth_values < depth_mean + 2.0 * depth_std) & \
                            valid  # Keep previous valid filter
            else:
                valid_range = valid
            
            # Check if we have enough valid points after filtering
            if np.sum(valid_range) < 3:
                if self.debug:
                    print(f"Too few valid depth points: {np.sum(valid_range)}/{len(depth_values)}")
                return np.zeros(6), (0, 0)
            
            # Use filtered values for further processing
            x_coords = x_coords[valid_range]
            y_coords = y_coords[valid_range]
            depth_values = depth_values[valid_range]
                    
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
                # Compute covariance directly
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
                
                
    def _estimate_point_pose(
        self,
        point: Tuple[int, int],  # (x, y) pixel coordinates
        depth_image: np.ndarray,
        region_size: int = 5  # Size of region around point to sample (half-width)
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate 3D position of a specific point using depth information
        
        Args:
            point: (x, y) pixel coordinates of the point
            depth_image: Depth image aligned with RGB
            region_size: Half-width of the square region to sample around the point
            
        Returns:
            Tuple containing:
            - 3D position array [x, y, z]
            - Confidence value (0-1) based on depth validity
        """
        if self.debug:
            self.profiler.start("_estimate_point_pose")
        
        try:
            # Extract pixel coordinates
            pixel_x, pixel_y = point
            
            # Check if point is within image bounds
            if (pixel_x < 0 or pixel_x >= depth_image.shape[1] or
                pixel_y < 0 or pixel_y >= depth_image.shape[0]):
                if self.debug:
                    print(f"Point ({pixel_x}, {pixel_y}) is outside image bounds")
                return np.zeros(3), 0.0
            
            # Define region around point to sample (clipped to image boundaries)
            x_min = max(0, pixel_x - region_size)
            x_max = min(depth_image.shape[1] - 1, pixel_x + region_size)
            y_min = max(0, pixel_y - region_size)
            y_max = min(depth_image.shape[0] - 1, pixel_y + region_size)
            
            # Extract depth values in the region
            region_depths = depth_image[y_min:y_max+1, x_min:x_max+1] * self.depth_scale
            
            # Filter valid depth values (non-zero and within reasonable range)
            MAX_DEPTH = 2.0  # Maximum reasonable depth in meters
            MIN_DEPTH = 0.05  # Minimum reasonable depth in meters
            
            valid_depths = region_depths[(region_depths > MIN_DEPTH) & (region_depths < MAX_DEPTH)]
            
            # Check if we have enough valid depth values
            if len(valid_depths) < 3:
                # If exact point has valid depth, use it despite few valid neighbors
                center_depth = depth_image[pixel_y, pixel_x] * self.depth_scale
                if MIN_DEPTH < center_depth < MAX_DEPTH:
                    valid_depths = np.array([center_depth])
                else:
                    if self.debug:
                        print(f"Too few valid depth points for ({pixel_x}, {pixel_y}): {len(valid_depths)}")
                    return np.zeros(3), 0.0
            
            # Calculate robust depth estimate using median (more robust to outliers than mean)
            depth = np.median(valid_depths)
            
            # Calculate confidence based on percentage of valid points and depth variance
            confidence_valid_ratio = len(valid_depths) / region_depths.size
            
            # Add variance component to confidence if we have enough points
            if len(valid_depths) >= 3:
                # Lower variance = higher confidence
                depth_std = np.std(valid_depths)
                # Normalize std dev into [0, 1] range (higher is better)
                # 0.1m std dev is considered high, 0.001m is excellent
                confidence_variance = max(0, 1.0 - (depth_std / 0.1))
                # Combine both confidence metrics
                confidence = 0.7 * confidence_valid_ratio + 0.3 * confidence_variance
            else:
                confidence = confidence_valid_ratio
            
            # Convert pixel coordinates to 3D using pinhole camera model
            x_3d = (pixel_x - self.cx) * depth / self.fx
            y_3d = (pixel_y - self.cy) * depth / self.fy
            z_3d = depth
            
            # Create 3D position array
            position_3d = np.array([x_3d, y_3d, z_3d])
            
            return position_3d, confidence
            
        except Exception as e:
            if self.debug:
                print(f"Error in _estimate_point_pose: {str(e)}")
                traceback.print_exc()
            return np.zeros(3), 0.0
            
        finally:
            if self.debug:
                self.profiler.stop("_estimate_point_pose")
    
    
    def visualize_interest_points(
        self,
        image: np.ndarray,
        obj_info: 'ObjectInfo',
        keypoints: List[cv2.KeyPoint],
        scores: List[float] = None,
        method: str = "Unknown",
        max_points: int = 100
    ) -> np.ndarray:
        """
        Create a visualization of interest points on an object with alphabetical labels.
        
        Args:
            image: Original RGB image
            obj_info: ObjectInfo with object details including mask
            keypoints: List of cv2.KeyPoint objects
            scores: Optional list of scores for each keypoint
            method: Point detection method used
            max_points: Maximum points parameter used
            
        Returns:
            Visualization image showing points of interest with alphabetical labels
        """
        # Create a copy of the image
        vis_img = image.copy()
        
        # If no scores provided, use response from keypoints or default to 1.0
        if scores is None:
            scores = [kp.response if hasattr(kp, 'response') and kp.response is not None else 1.0 
                    for kp in keypoints]
        
        # Ensure we have at least one keypoint
        if not keypoints:
            # Add a "no points detected" message
            cv2.putText(
                vis_img,
                f"No interest points detected with {method}",
                (int(image.shape[1]/2 - 200), int(image.shape[0]/2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return vis_img
        
        # Draw object bounding box and mask
        if obj_info.bbox is not None:
            x, y, w, h = obj_info.bbox
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw semi-transparent mask if available
        if obj_info.mask is not None:
            mask_overlay = np.zeros_like(vis_img, dtype=np.uint8)
            mask_overlay[obj_info.mask] = [0, 100, 0]  # Light green
            vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.3, 0)
        
        # Calculate score range for coloring
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        score_range = max_score - min_score if max_score > min_score else 1
        
        # Create a list of keypoints with their scores for sorting
        keypoints_with_scores = [(kp, score) for kp, score in zip(keypoints, scores)]
        
        # Sort by score (highest first)
        keypoints_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to the maximum number of points
        keypoints_with_scores = keypoints_with_scores[:max_points]
        
        # Draw keypoints with colors based on score and alphabetical labels
        for i, (kp, score) in enumerate(keypoints_with_scores):
            # Normalize score to [0, 1]
            norm_score = (score - min_score) / score_range if score_range > 0 else 0.5
            
            # Map to color (blue to red based on score)
            color = (
                int(255 * (1 - norm_score)),  # B
                0,                           # G
                int(255 * norm_score)         # R
            )
            
            # Draw circle for keypoint
            cv2.circle(
                vis_img, 
                (int(kp.pt[0]), int(kp.pt[1])), 
                radius=3, 
                color=color, 
                thickness=-1
            )
            
            # Generate alphabetical label (a-z, then aa, ab, etc.)
            alpha_id = get_alpha_id(i + 1)
            
            # Draw label
            cv2.putText(
                vis_img,
                alpha_id,
                (int(kp.pt[0]) + 5, int(kp.pt[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        # Add title with method and point count
        cv2.putText(
            vis_img,
            f"{method.upper()} Interest Points: {len(keypoints_with_scores)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # Add object name
        alpha_id = obj_info.alpha_id if obj_info.alpha_id else ""
        obj_text = f"Object: {alpha_id} {obj_info.name}"
        cv2.putText(
            vis_img,
            obj_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # Add method and max points info at the bottom
        method_text = f"Method: {method.upper()}, Max Points: {max_points}"
        cv2.putText(
            vis_img,
            method_text,
            (10, vis_img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        return vis_img
    
    def detect_regions_of_interest(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo,
        method: str = 'shi_tomasi',
        max_points: int = 20,
        quality_level: float = 0.01,
        min_distance: int = 50,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Detect points of interest within a specific object mask.
        
        Args:
            image: RGB input image
            obj_info: ObjectInfo containing the object mask
            method: Feature detection method ('harris', 'shi_tomasi', 'sift', 'orb', 'fast')
            max_points: Maximum number of points to detect
            quality_level: Quality level parameter for some detectors (0.0-1.0)
            min_distance: Minimum distance between detected points
            visualize: Whether to generate visualization
            
        Returns:
            Dictionary containing:
                - 'keypoints': List of cv2.KeyPoint objects
                - 'descriptors': Descriptors if available (None for some methods)
                - 'pixel_coords': List of (x,y) pixel coordinates
                - 'scores': Confidence scores for each point (if available)
                - 'visualization': Visualization image (if visualize=True)
        """
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_regions_of_interest")
        
        try:
            # Check if the object has a mask
            if obj_info.mask is None:
                if self.debug:
                    print(f"No mask available for object '{obj_info.name}', generating one...")
                
                # Generate a mask if not available
                mask = self._generate_segmentation(image, obj_info)
                if mask is None:
                    # Fallback to bounding box mask
                    if self.debug:
                        print("Using bounding box as fallback mask")
                    x, y, w, h = obj_info.bbox
                    mask = np.zeros(image.shape[:2], dtype=bool)
                    mask[y:y+h, x:x+w] = True
            else:
                mask = obj_info.mask
            
            # Convert image to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply mask to gray image (mask everything outside object)
            masked_gray = np.zeros_like(gray)
            masked_gray[mask] = gray[mask]
            
            # Select feature detection method
            keypoints = []
            descriptors = None
            
            if self.debug:
                self.profiler.start(f"feature_detection_{method}")
            
            if method.lower() == 'harris':
                # Harris corner detector
                # Convert mask to uint8 for cornerHarris
                mask_uint8 = mask.astype(np.uint8) * 255
                
                # Detect corners
                corner_response = cv2.cornerHarris(masked_gray, blockSize=3, ksize=3, k=0.04)
                
                # Normalize response
                cv2.normalize(corner_response, corner_response, 0, 255, cv2.NORM_MINMAX)
                corner_response = np.uint8(corner_response)
                
                # Threshold and find centroids
                threshold = 0.01 * corner_response.max()
                corner_mask = corner_response > threshold
                
                # Additional filtering with mask
                corner_mask = np.logical_and(corner_mask, mask)
                
                # Get corner coordinates
                corners = np.where(corner_mask)
                
                # Create KeyPoint objects
                for y, x in zip(corners[0], corners[1]):
                    response = corner_response[y, x]
                    keypoints.append(cv2.KeyPoint(float(x), float(y), size=3, response=float(response)))
                
                # Sort by response and limit to max_points
                keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_points]
                
            elif method.lower() == 'shi_tomasi':
                # Shi-Tomasi corner detector (goodFeaturesToTrack)
                mask_uint8 = mask.astype(np.uint8) * 255
                corners = cv2.goodFeaturesToTrack(
                    masked_gray, 
                    maxCorners=max_points,
                    qualityLevel=quality_level,
                    minDistance=min_distance,
                    mask=mask_uint8
                )
                
                if corners is not None:
                    for corner in corners:
                        x, y = corner.ravel()
                        keypoints.append(cv2.KeyPoint(float(x), float(y), size=3))
                        
            elif method.lower() == 'sift':
                # SIFT detector
                try:
                    sift = cv2.SIFT_create(nfeatures=max_points)
                    
                    # Apply mask to limit detection region
                    mask_uint8 = mask.astype(np.uint8) * 255
                    keypoints, descriptors = sift.detectAndCompute(gray, mask=mask_uint8)
                    
                    # Filter out keypoints outside the mask (just to be sure)
                    valid_keypoints = []
                    valid_descriptors = []
                    
                    for i, kp in enumerate(keypoints):
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
                            valid_keypoints.append(kp)
                            if descriptors is not None:
                                valid_descriptors.append(descriptors[i])
                    
                    keypoints = valid_keypoints
                    if descriptors is not None and len(valid_descriptors) > 0:
                        descriptors = np.array(valid_descriptors)
                    else:
                        descriptors = None
                        
                except Exception as e:
                    if self.debug:
                        print(f"SIFT detection error: {e}")
                    # Fallback to Shi-Tomasi
                    return self.detect_regions_of_interest(
                        image, obj_info, method='shi_tomasi', 
                        max_points=max_points, quality_level=quality_level, 
                        min_distance=min_distance, visualize=visualize
                    )
                    
            elif method.lower() == 'orb':
                # ORB detector
                try:
                    orb = cv2.ORB_create(nfeatures=max_points, edgeThreshold=1, patchSize=1)
                    
                    # Apply mask to limit detection region
                    mask_uint8 = mask.astype(np.uint8) * 255
                    keypoints, descriptors = orb.detectAndCompute(gray, mask=mask_uint8)
                    
                    # Filter out keypoints outside the mask
                    valid_keypoints = []
                    valid_descriptors = []
                    valid_indices = []
                    
                    for i, kp in enumerate(keypoints):
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
                            valid_keypoints.append(kp)
                            valid_indices.append(i)
                            if descriptors is not None:
                                valid_descriptors.append(descriptors[i])
                    
                    # Apply minimum distance filtering to ensure spatial distribution
                    if min_distance > 0 and len(valid_keypoints) > 1:
                        # Sort keypoints by response strength (strongest first)
                        sorted_keypoints = sorted(
                            [(i, kp) for i, kp in enumerate(valid_keypoints)], 
                            key=lambda x: x[1].response, 
                            reverse=True
                        )
                        
                        filtered_keypoints = []
                        filtered_descriptors = []
                        filtered_indices = []
                        
                        # Always keep the strongest keypoint
                        strongest_idx, strongest_kp = sorted_keypoints[0]
                        filtered_keypoints.append(strongest_kp)
                        filtered_indices.append(valid_indices[strongest_idx])
                        if descriptors is not None:
                            filtered_descriptors.append(valid_descriptors[strongest_idx])
                        
                        # For each remaining keypoint, check if it's far enough from already-selected keypoints
                        for idx, kp in sorted_keypoints[1:]:
                            # Check minimum distance to all previously filtered keypoints
                            too_close = False
                            kp_x, kp_y = kp.pt
                            
                            for filtered_kp in filtered_keypoints:
                                f_x, f_y = filtered_kp.pt
                                distance = np.sqrt((kp_x - f_x)**2 + (kp_y - f_y)**2)
                                if distance < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                filtered_keypoints.append(kp)
                                filtered_indices.append(valid_indices[idx])
                                if descriptors is not None:
                                    filtered_descriptors.append(valid_descriptors[idx])
                        
                        # Update keypoints and descriptors with filtered ones
                        keypoints = filtered_keypoints
                        if descriptors is not None and len(filtered_descriptors) > 0:
                            descriptors = np.array(filtered_descriptors)
                        else:
                            descriptors = None
                    else:
                        # No min_distance filtering needed
                        keypoints = valid_keypoints
                        if descriptors is not None and len(valid_descriptors) > 0:
                            descriptors = np.array(valid_descriptors)
                        else:
                            descriptors = None
                            
                    if self.debug and len(keypoints) == 0:
                        print("ORB detection found no keypoints after filtering")
                        
                except Exception as e:
                    if self.debug:
                        print(f"ORB detection error: {e}")
                    # Fallback to Shi-Tomasi
                    return self.detect_regions_of_interest(
                        image, obj_info, method='shi_tomasi', 
                        max_points=max_points, quality_level=quality_level, 
                        min_distance=min_distance, visualize=visualize
                    )
                    
            elif method.lower() == 'fast':
                # FAST corner detector
                try:
                    fast = cv2.FastFeatureDetector_create(threshold=10)
                    
                    # Detect points in masked image
                    keypoints = fast.detect(masked_gray, None)
                    
                    # Keep only points within mask (redundant but helps ensure correctness)
                    valid_keypoints = []
                    for kp in keypoints:
                        x, y = int(kp.pt[0]), int(kp.pt[1])
                        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
                            valid_keypoints.append(kp)
                    
                    keypoints = valid_keypoints[:max_points]
                    
                    # Compute descriptors using ORB if needed
                    if len(keypoints) > 0:
                        orb = cv2.ORB_create()
                        _, descriptors = orb.compute(gray, keypoints)
                        
                except Exception as e:
                    if self.debug:
                        print(f"FAST detection error: {e}")
                    # Fallback to Shi-Tomasi
                    return self.detect_regions_of_interest(
                        image, obj_info, method='shi_tomasi', 
                        max_points=max_points, quality_level=quality_level, 
                        min_distance=min_distance, visualize=visualize
                    )
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if self.debug:
                self.profiler.stop(f"feature_detection_{method}")
                print(f"Detected {len(keypoints)} keypoints using {method} method")
            
            # Extract pixel coordinates
            pixel_coords = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
            
            # Extract scores if available
            scores = [kp.response if hasattr(kp, 'response') else 1.0 for kp in keypoints]
            
            # Create visualization if requested
            visualization = None
            if visualize:
                if self.debug:
                    self.profiler.start("roi_visualization")
                
                # Use the dedicated visualization function
                visualization = self.visualize_interest_points(
                    image=image,
                    obj_info=obj_info,
                    keypoints=keypoints,
                    scores=scores,
                    method=method,
                    max_points=max_points
                )
                
                if self.debug:
                    self.profiler.stop("roi_visualization")
            
            # Create results dictionary
            results = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'pixel_coords': pixel_coords,
                'scores': scores,
                'object_name': obj_info.name,
                'method': method,
                'mask_area': np.sum(mask),
                'ids': [get_alpha_id(i) for i in range(len(keypoints))]
            }
            
            if visualize:
                results['visualization'] = visualization
                
            return results
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_regions_of_interest: {str(e)}")
                traceback.print_exc()
            return {
                'keypoints': [],
                'descriptors': None,
                'pixel_coords': [],
                'scores': [],
                'error': str(e)
            }
            
        finally:
            if self.debug:
                self.profiler.stop("detect_regions_of_interest")
                self.profiler.end_iteration(preserve_current=True)
    
    def visualize_detections(
        self,
        image: np.ndarray,
        objects: List[ObjectInfo],
        show_masks: bool = True,
        show_parts: bool = True,
        show_poses: bool = True
    ) -> np.ndarray:
        """
        Visualize detected objects with masks, parts, and poses
        
        Args:
            image: Original RGB image
            objects: List of ObjectInfo objects to visualize
            show_masks: Whether to show segmentation masks
            show_parts: Whether to show object parts
            show_poses: Whether to show object poses
            
        Returns:
            Visualization image
        """
        if self.debug:
            self.profiler.start("visualize_detections")
        
        try:
            if image is None or not objects:
                if self.debug:
                    print("Cannot visualize: image is None or objects is empty")
                return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                
            vis_image = image.copy()
            
            # Process objects in reverse order (to draw smaller objects on top)
            for obj in objects:
                # Draw bounding box
                x, y, w, h = obj.bbox
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add label with confidence and alpha_id
                alpha_id = obj.alpha_id if obj.alpha_id else ""
                label = f"{alpha_id}: {obj.name} ({obj.confidence:.2f})"
                cv2.putText(
                    vis_image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Draw mask overlay if available and requested
                if show_masks and obj.mask is not None:
                    mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
                    mask_overlay[obj.mask] = [0, 255, 0]  # Green overlay with alpha
                    vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, 0.3, 0)
                
                # Draw pose if available and requested
                if show_poses and obj.pose is not None and obj.pixel_pose is not None:
                    # Use pixel_pose as the origin for visualization
                    origin = obj.pixel_pose
                    
                    # Get pose angles
                    pose = obj.pose
                    
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
                
                # Draw components if available and requested
                if show_parts and obj.components is not None and len(obj.components) > 0:
                    # Process each component
                    for comp_name, comp_info in obj.components.items():
                        # Draw component bounding box in a different color (blue)
                        cx, cy, cw, ch = comp_info.bbox
                        cv2.rectangle(vis_image, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)
                        
                        # Add component label with alpha_id
                        alpha_id = comp_info.alpha_id if comp_info.alpha_id else ""
                        comp_label = f"{alpha_id}: {comp_name} ({comp_info.confidence:.2f})"
                        cv2.putText(
                            vis_image,
                            comp_label,
                            (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 0),
                            1
                        )
                        
                        # Draw component mask if available
                        if comp_info.mask is not None:
                            comp_overlay = np.zeros_like(vis_image, dtype=np.uint8)
                            comp_overlay[comp_info.mask] = [255, 0, 0]  # Blue overlay
                            vis_image = cv2.addWeighted(vis_image, 1.0, comp_overlay, 0.2, 0)
            
            # Add profiling info if debug is enabled
            if self.debug:
                # Get performance metrics
                detect_time = self.profiler.get_current_iteration_total("detect_objects") or 0
                yolo_time = self.profiler.get_current_iteration_total("yolo_detection") or 0
                
                # Create timing summary
                info_text = f"Detection: {detect_time:.3f}s | YOLO: {yolo_time:.3f}s"
                
                # Add second line with object count
                count_text = f"Objects: {len(objects)}"
                
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
                
                # Add object count
                cv2.putText(
                    vis_image,
                    count_text,
                    (10, vis_image.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1
                )
                
            return vis_image
            
        except Exception as e:
            if self.debug:
                print(f"Error in visualize_detections: {str(e)}")
                traceback.print_exc()
            # Return original image on error
            return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
        finally:
            if self.debug:
                self.profiler.stop("visualize_detections")
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """
        Visualize segmentation masks using the FastSAMMaskGenerator visualization
        
        Args:
            image: Original RGB image
            masks: Labeled mask array from FastSAM
            metadata: Metadata dictionary from FastSAM
            
        Returns:
            Visualization image
        """
        if self.segmenter is None:
            return image.copy()
            
        try:
            # Use the segmenter's visualization method
            return self.segmenter.visualize_masks(image, masks, metadata, alpha=0.5)
        except Exception as e:
            if self.debug:
                print(f"Error in visualize_segmentation: {str(e)}")
            return image.copy()
    
    def set_detector_classes(self, classes: List[str]):
        """
        Update YOLO-World detector classes
        
        Args:
            classes: List of classes to detect
        """
        self.detector.set_classes(classes)
        if self.debug:
            print(f"Updated detector classes: {classes}")
    
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
        
    def clear_cache(self):
        """Clear the object detection cache"""
        self._object_cache = {}
        if self.debug:
            print("Object detection cache cleared")
            
    def release(self):
        """Release resources used by the perception system"""
        # Free any resources if needed
        if self.debug:
            print("PerceptionSystem resources released")
            
            # Print final performance summary
            print("\nFinal performance summary:")
            self.print_performance_summary(sort_by="total", top_n=10)
            

    def detect_object_surfaces(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo,
        conf_threshold: float = 0.1,
        depth_image: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect surfaces in the scene by segmenting everything except the object mask
        
        Args:
            image: Input image
            obj_info: Object information with mask
            conf_threshold: Confidence threshold for surface detection
            depth_image: Optional depth image
            
        Returns:
            Dictionary mapping surface IDs to masks
        """
        if self.segmenter is None:
            if self.debug:
                print("Cannot detect surfaces: FastSAM segmenter not initialized")
            return {}
            
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_surfaces")
        
        try:
            # Create inverted mask to focus on everything except the object
            if obj_info.mask is None:
                if self.debug:
                    print("Cannot detect surfaces: Object mask is None")
                return {}
                
            inverted_mask = ~obj_info.mask
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Create a masked image where only non-object regions are visible
            masked_image = image.copy()
            # for c in range(3):  # For each color channel
            #     masked_image[:,:,c] = masked_image[:,:,c] * inverted_mask
            
            # Generate masks for everything in the scene except the object
            if self.debug:
                self.profiler.start("generate_surface_masks")
                
            try:
                # Use "everything" prompt to segment all surfaces
                labeled_masks, metadata = self.segmenter.generate_masks(
                    masked_image,
                    prompt_type="everything"
                )
                
                if self.debug:
                    print(f"FastSAM returned {len(metadata)} potential surfaces in the scene")
                    self.profiler.stop("generate_surface_masks")
            except Exception as e:
                if self.debug:
                    print(f"Error generating surface masks: {e}")
                    self.profiler.stop("generate_surface_masks")
                return {}
            
            # Process the masks to create surface dictionary
            if self.debug:
                self.profiler.start("process_surfaces")
                
            surfaces = {}
            
            # Filter masks based on area and other criteria
            total_image_area = h * w
            min_surface_area = total_image_area * 0.01  # Surfaces should be at least 1% of image
            max_surface_area = total_image_area * 0.9   # Surfaces shouldn't be more than 90% of image
                
            for mask_id, meta in metadata.items():
                # Skip surfaces that are too small or too large
                area = meta.get('area', 0)
                if area < min_surface_area or area > max_surface_area:
                    if self.debug:
                        print(f"Skipping surface {mask_id}: area={area:.0f} (outside range [{min_surface_area:.0f}, {max_surface_area:.0f}])")
                    continue

                # Get the mask for this surface
                surface_mask = labeled_masks == mask_id
                
                # Generate a unique name for this surface
                alpha_id = get_alpha_id(mask_id)
                surface_name = f"surface_{alpha_id}"
                
                # Store the mask
                surfaces[surface_name] = surface_mask
            
            if self.debug:
                self.profiler.stop("process_surfaces")
                print(f"Processed {len(surfaces)} surfaces")
            
            return surfaces
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_object_surfaces: {str(e)}")
                traceback.print_exc()
            return {}
            
        finally:
            if self.debug:
                self.profiler.stop("detect_surfaces")
                self.profiler.end_iteration(preserve_current=True)
                           
    def segment_surfaces_by_plane_fitting(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo,
        depth_image: np.ndarray,
        max_plane_distance: float = 0.01,
        min_points_per_plane: int = 100,
        max_planes_per_surface: int = 3,
        ransac_iterations: int = 300,  # Reduced from 1000
        normal_similarity_threshold: float = 0.95,
        downsample_factor: int = 4  # Added downsampling parameter
    ) -> Dict[str, np.ndarray]:
        """
        Segment surfaces by fitting planes to depth data - optimized for speed
        """
        if depth_image is None:
            if self.debug:
                print("Cannot segment surfaces: Depth image is required")
            return {}
            
        if self.debug:
            self.profiler.start("segment_surfaces_by_plane")
        
        try:
            # First detect initial surface masks
            initial_surfaces = self.detect_object_surfaces(image, obj_info, depth_image=depth_image)
            return initial_surfaces
            if not initial_surfaces:
                if self.debug:
                    print("No initial surfaces detected")
                return {}
            
            # Store surface normals in the obj_info
            if not hasattr(obj_info, 'surface_normals'):
                obj_info.surface_normals = {}
                
            # Get camera intrinsics
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
            
            refined_surfaces = {}
            plane_count = 0
            
            # Process each detected surface
            for surface_id, surface_mask in initial_surfaces.items():
                if self.debug:
                    print(f"Processing surface {surface_id}")
                    
                # Get points corresponding to this surface
                y_coords, x_coords = np.where(surface_mask)
                
                if len(y_coords) < min_points_per_plane:
                    if self.debug:
                        print(f"Surface {surface_id} has too few points: {len(y_coords)}")
                    refined_surfaces[surface_id] = surface_mask  # Keep the original mask
                    continue
                
                # Downsample points for speed
                if downsample_factor > 1:
                    # Use systematic sampling instead of random
                    indices = np.arange(0, len(y_coords), downsample_factor)
                    y_coords = y_coords[indices]
                    x_coords = x_coords[indices]
                    
                    if len(y_coords) < min_points_per_plane:
                        # If downsampling leaves too few points, adjust the factor
                        downsample_factor = max(1, len(y_coords) // min_points_per_plane)
                        indices = np.arange(0, len(y_coords), downsample_factor)
                        y_coords = y_coords[indices]
                        x_coords = x_coords[indices]
                
                # Pre-allocate arrays for speed
                surface_points = np.zeros((len(y_coords), 3), dtype=np.float32)
                valid_mask = np.zeros(len(y_coords), dtype=bool)
                
                # Vectorized depth to point cloud conversion
                depth_values = depth_image[y_coords, x_coords] * self.depth_scale
                
                # Filter valid depths
                valid_mask = (depth_values > 0.001) & (depth_values < 2.0)
                
                if np.sum(valid_mask) < min_points_per_plane:
                    if self.debug:
                        print(f"Surface {surface_id} has too few valid points: {np.sum(valid_mask)}")
                    refined_surfaces[surface_id] = surface_mask
                    continue
                
                # Only process valid points
                valid_y = y_coords[valid_mask]
                valid_x = x_coords[valid_mask]
                valid_depths = depth_values[valid_mask]
                
                # Vectorized conversion to 3D
                surface_points = np.column_stack([
                    (valid_x - cx) * valid_depths / fx,
                    (valid_y - cy) * valid_depths / fy,
                    valid_depths
                ])
                
                # Create mapping from point indices to pixel coordinates
                pixel_indices = np.arange(len(valid_y))
                pixel_map = list(zip(valid_y, valid_x))
                
                # Use Open3D for faster plane segmentation if available
                try:
                    import open3d as o3d
                    
                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(surface_points)
                    
                    # Use Open3D's built-in plane segmentation
                    remaining_points = surface_points
                    remaining_indices = pixel_indices
                    remaining_pixel_map = pixel_map
                    
                    plane_masks = []
                    plane_normals = []
                    plane_centroids = []
                    
                    for _ in range(max_planes_per_surface):
                        if len(remaining_points) < min_points_per_plane:
                            break
                            
                        # Create Open3D point cloud from remaining points
                        current_pcd = o3d.geometry.PointCloud()
                        current_pcd.points = o3d.utility.Vector3dVector(remaining_points)
                        
                        # Use Open3D's plane segmentation (much faster than manual RANSAC)
                        plane_model, inliers = current_pcd.segment_plane(
                            distance_threshold=max_plane_distance,
                            ransac_n=3,
                            num_iterations=ransac_iterations
                        )
                        
                        if len(inliers) < min_points_per_plane:
                            break
                            
                        # Extract plane parameters and inlier points
                        a, b, c, d = plane_model
                        normal = np.array([a, b, c])
                        
                        # Ensure normal points towards camera
                        if normal[2] > 0:
                            normal = -normal
                            
                        inlier_points = np.asarray(current_pcd.points)[inliers]
                        centroid = np.mean(inlier_points, axis=0)
                        
                        # Create mask for these inlier points
                        plane_mask = np.zeros_like(surface_mask, dtype=bool)
                        inlier_coords = [remaining_pixel_map[i] for i in inliers]
                        for y, x in inlier_coords:
                            plane_mask[y, x] = True
                        
                        # Check for similar existing planes
                        should_merge = False
                        merge_idx = -1
                        
                        for i, existing_normal in enumerate(plane_normals):
                            # Calculate cosine similarity
                            similarity = np.abs(np.dot(normal, existing_normal))
                            
                            if similarity > normal_similarity_threshold:
                                # Planes have similar orientation
                                should_merge = True
                                merge_idx = i
                                break
                        
                        if should_merge:
                            # Merge with existing plane
                            plane_masks[merge_idx] = plane_masks[merge_idx] | plane_mask
                            
                            # Update centroid
                            existing_count = np.sum(plane_masks[merge_idx])
                            new_count = np.sum(plane_mask)
                            total_count = existing_count + new_count
                            
                            plane_centroids[merge_idx] = (
                                (existing_count * plane_centroids[merge_idx] + 
                                new_count * centroid) / total_count
                            )
                        else:
                            # Add as a new plane
                            plane_masks.append(plane_mask)
                            plane_normals.append(normal)
                            plane_centroids.append(centroid)
                        
                        # Remove inliers from remaining points
                        non_inliers = np.ones(len(remaining_points), dtype=bool)
                        non_inliers[inliers] = False
                        remaining_points = remaining_points[non_inliers]
                        remaining_indices = remaining_indices[non_inliers]
                        remaining_pixel_map = [remaining_pixel_map[i] for i in range(len(remaining_pixel_map)) if non_inliers[i]]
                    
                except ImportError:
                    # Fallback to optimized numpy-based RANSAC if Open3D is not available
                    remaining_points = np.ones(len(surface_points), dtype=bool)
                    plane_masks = []
                    plane_normals = []
                    plane_centroids = []
                    
                    for plane_idx in range(max_planes_per_surface):
                        # Stop if too few points remain
                        if np.sum(remaining_points) < min_points_per_plane:
                            break
                            
                        # Get current points
                        current_points = surface_points[remaining_points]
                        current_indices = pixel_indices[remaining_points]
                        
                        # Fast RANSAC implementation with early stopping
                        best_inliers = None
                        best_normal = None
                        most_inliers = min_points_per_plane  # Set threshold for early stopping
                        min_samples = 3
                        
                        # Pre-select random samples for speed
                        if len(current_points) > min_samples:
                            sample_indices = np.random.choice(
                                len(current_points), 
                                min_samples * ransac_iterations, 
                                replace=True
                            ).reshape(ransac_iterations, min_samples)
                        else:
                            break
                        
                        for i in range(ransac_iterations):
                            # Get sample points
                            idx = sample_indices[i]
                            p1, p2, p3 = current_points[idx]
                            
                            # Calculate plane normal
                            v1 = p2 - p1
                            v2 = p3 - p1
                            normal = np.cross(v1, v2)
                            norm = np.linalg.norm(normal)
                            
                            # Skip if points are collinear
                            if norm < 1e-6:
                                continue
                                
                            normal = normal / norm
                            d = -np.dot(normal, p1)
                            
                            # Calculate distances (vectorized)
                            distances = np.abs(np.dot(current_points, normal) + d)
                            inliers = distances < max_plane_distance
                            num_inliers = np.sum(inliers)
                            
                            if num_inliers > most_inliers:
                                most_inliers = num_inliers
                                best_inliers = inliers
                                best_normal = normal
                                
                                # Early stopping if we found a good plane
                                if num_inliers > len(current_points) * 0.8:
                                    break
                        
                        # If no good plane found, stop
                        if best_inliers is None:
                            break
                        
                        # Calculate centroid and refine normal
                        inlier_points = current_points[best_inliers]
                        centroid = np.mean(inlier_points, axis=0)
                        
                        # Create mask for these inlier points
                        inlier_indices = current_indices[best_inliers]
                        plane_mask = np.zeros_like(surface_mask, dtype=bool)
                        for idx in inlier_indices:
                            y, x = pixel_map[idx]
                            plane_mask[y, x] = True
                        
                        # Check for similar existing planes
                        should_merge = False
                        merge_idx = -1
                        
                        for i, existing_normal in enumerate(plane_normals):
                            similarity = np.abs(np.dot(best_normal, existing_normal))
                            if similarity > normal_similarity_threshold:
                                should_merge = True
                                merge_idx = i
                                break
                        
                        if should_merge:
                            # Merge with existing plane
                            plane_masks[merge_idx] = plane_masks[merge_idx] | plane_mask
                            
                            # Update centroid
                            existing_count = np.sum(plane_masks[merge_idx])
                            new_count = np.sum(plane_mask)
                            total_count = existing_count + new_count
                            
                            plane_centroids[merge_idx] = (
                                (existing_count * plane_centroids[merge_idx] + 
                                new_count * centroid) / total_count
                            )
                        else:
                            # Add as a new plane
                            plane_masks.append(plane_mask)
                            plane_normals.append(best_normal)
                            plane_centroids.append(centroid)
                        
                        # Mark these points as processed
                        current_remaining = np.where(remaining_points)[0]
                        remaining_points[current_remaining[best_inliers]] = False
                
                # Add planes to output dictionary
                if len(plane_masks) > 0:
                    # If we found multiple planes, create new entries
                    if len(plane_masks) > 1:
                        for i, (mask, normal, centroid) in enumerate(zip(plane_masks, plane_normals, plane_centroids)):
                            plane_count += 1
                            alpha_id = get_alpha_id(plane_count)
                            plane_id = f"surface_{alpha_id}"
                            
                            # Add to refined surfaces
                            refined_surfaces[plane_id] = mask
                            
                            # Store normal information
                            obj_info.surface_normals[plane_id] = {
                                'normal': normal,
                                'centroid': centroid,
                                'parent_surface': surface_id
                            }
                    else:
                        # Single plane - keep original ID
                        refined_surfaces[surface_id] = plane_masks[0]
                        
                        # Store normal information
                        obj_info.surface_normals[surface_id] = {
                            'normal': plane_normals[0],
                            'centroid': plane_centroids[0],
                            'parent_surface': None
                        }
                else:
                    # No planes found - keep original surface
                    refined_surfaces[surface_id] = surface_mask
            
            return refined_surfaces
            
        except Exception as e:
            if self.debug:
                print(f"Error in segment_surfaces_by_plane_fitting: {str(e)}")
                traceback.print_exc()
            return {}
            
        finally:
            if self.debug:
                self.profiler.stop("segment_surfaces_by_plane")