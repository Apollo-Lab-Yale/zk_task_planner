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
from ultralytics import YOLOWorld


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
    

class PerceptionSystem:
    def __init__(
        self,
        fast_sam_config: Optional[FastSAMConfig],
        yolo_model_path: str = 'yolov8x-worldv2.pt',
        camera_matrix: Optional[np.ndarray] = None,
        depth_scale: float = 0.001,  # Default for most depth cameras (m/unit)
        default_conf: float = 0.5,
        default_classes: List[str] = None,
        debug: bool = True
    ):
        """
        Initialize hybrid perception system with YOLO-World and FastSAM models
        
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
                
            self.detector = YOLOWorld(yolo_model_path)
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
                self.segmenter = FastSAMWithCLIP(fast_sam_config)
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
            
            if self.debug:
                print(f"HybridPerceptionSystem initialized with debug={debug}")
                
        except Exception as e:
            if self.debug:
                print(f"Error in HybridPerceptionSystem initialization: {str(e)}")
                traceback.print_exc()
            raise  # Re-raise the exception to not silence initialization errors
    
    def detect_objects(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
        conf: Optional[float] = 0.5,
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
                self.detector.set_classes(classes)
                if self.debug:
                    print(f"Set detection classes: {classes}")
            
            conf_threshold = conf if conf is not None else self.default_conf
            
            # Run YOLO-World detection
            if self.debug:
                self.profiler.start("yolo_detection")
                print(f"Running YOLO-World detection with confidence threshold {conf_threshold}")
            
            results = self.detector.predict(image, conf=conf_threshold, verbose=False)
            
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
                
                # Create ObjectInfo
                obj_info = ObjectInfo(
                    id=i,
                    name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    image=image,
                    depth_image=depth_image
                )
                
                # Generate segmentation mask if requested
                if self.debug:
                    self.profiler.start(f"segment_obj_{i}")
                    print(f"Generating segmentation for object {i} ({class_name})")
                
                mask = self._generate_segmentation(image, obj_info, class_name)
                if mask is None:
                    raise Exception("Failed to generate mask for object.")
                obj_info.mask = mask
                
                if self.debug:
                    self.profiler.stop(f"segment_obj_{i}")
                
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
        conf: Optional[float] = 0.5,
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
            # Check cache first
            cache_key = f"{target_object}_{hash(image.tobytes())}"
            if cache_key in self._object_cache:
                if self.debug:
                    print(f"Cache hit for '{target_object}'")
                    self.profiler.stop("detect_object")
                    self.profiler.end_iteration(preserve_current=True)
                return self._object_cache[cache_key]
            
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
            self._object_cache[cache_key] = best_object
            
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
    
    def detect_object_component(
        self,
        object_info: ObjectInfo,
        component_name: str,
        conf_threshold: float = 0.5,
        depth_image: Optional[np.ndarray] = None
    ) -> Optional[ObjectInfo]:
        """
        Detect a specific component within an object using FastSAM-CLIP
        
        Args:
            object_info: Parent object information
            component_name: Name of the component to detect
            conf_threshold: Confidence threshold for component detection
            depth_image: Optional depth image aligned with RGB
            
        Returns:
            ObjectInfo for the component if found, None otherwise
        """
        if self.segmenter is None:
            if self.debug:
                print("Cannot detect components: FastSAM segmenter not initialized")
            return None
        
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_component")
            print(f"Detecting component '{component_name}' within object '{object_info.name}'")
        
        try:
            # Extract object region from image
            x, y, w, h = object_info.bbox
            if object_info.image is None:
                if self.debug:
                    print("Cannot detect component: object image is None")
                return None
            
            # Extract region of interest
            if self.debug:
                self.profiler.start("extract_roi")
                
            # Add a small padding around the object for better component detection
            pad = int(max(w, h) * 0.05)  # 5% padding
            x_roi = max(0, x - pad)
            y_roi = max(0, y - pad)
            w_roi = min(object_info.image.shape[1] - x_roi, w + 2*pad)
            h_roi = min(object_info.image.shape[0] - y_roi, h + 2*pad)
            
            # Store ROI coordinates
            roi_coords = (x_roi, y_roi, w_roi, h_roi)
            
            roi = object_info.image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            if roi.size == 0:
                if self.debug:
                    print(f"Invalid ROI with dimensions {roi.shape}")
                return None
                
            if self.debug:
                self.profiler.stop("extract_roi")
                
            # Run FastSAM-CLIP on the ROI to find the component
            if self.debug:
                self.profiler.start("component_segmentation")
                
            query = f"a {component_name}"
            labeled_masks, metadata = self.segmenter.process_image(
                roi, 
                query=query,
                roi_coords=roi_coords,
                store_results=True
            )
            
            if self.debug:
                self.profiler.stop("component_segmentation")
                print(f"FastSAM returned {len(metadata)} potential component masks")
            
            # Find best matching mask
            if self.debug:
                self.profiler.start("find_best_component")
                
            best_mask_id = None
            best_score = 0
            
            for mask_id, meta in metadata.items():
                score = meta.get('clip_score', 0)
                if self.debug:
                    print(f"  Component mask {mask_id} score: {score:.3f}")
                if score > best_score and score >= conf_threshold:
                    best_score = score
                    best_mask_id = mask_id
                    
            if best_mask_id is None:
                if self.debug:
                    print(f"No component '{component_name}' found with confidence >= {conf_threshold}")
                    self.profiler.stop("find_best_component")
                    self.profiler.stop("detect_component")
                    self.profiler.end_iteration(preserve_current=True)
                return None
            
            if self.debug:
                print(f"Best component mask for '{component_name}': {best_mask_id} with score {best_score:.3f}")
                self.profiler.stop("find_best_component")
                
            # Extract mask and convert to full image coordinates
            if self.debug:
                self.profiler.start("process_component_mask")
                
            # Get component mask from ROI segmentation
            component_roi_mask = labeled_masks == best_mask_id
            
            # Create a full image mask (initialized to all False)
            component_full_mask = np.zeros(object_info.image.shape[:2], dtype=bool)
            
            # Place ROI mask in the correct position in the full image
            component_full_mask[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = component_roi_mask
            
            if self.debug:
                self.profiler.stop("process_component_mask")
                
            # Get component bounding box in full image coordinates
            if self.debug:
                self.profiler.start("compute_component_bbox")
                
            y_coords, x_coords = np.where(component_full_mask)
            if len(y_coords) == 0:
                if self.debug:
                    print(f"Empty component mask for '{component_name}'")
                    self.profiler.stop("compute_component_bbox")
                    self.profiler.stop("detect_component")
                    self.profiler.end_iteration(preserve_current=True)
                return None
                
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            component_bbox = [x1, y1, x2 - x1, y2 - y1]
            
            if self.debug:
                self.profiler.stop("compute_component_bbox")
                
            # Create component ObjectInfo
            if self.debug:
                self.profiler.start("create_component_info")
                
            # Use a different ID scheme for components to avoid confusion with parent objects
            component_id = hash(f"{object_info.id}_{component_name}") % 10000
            
            component_info = ObjectInfo(
                id=component_id,
                name=component_name,
                mask=component_full_mask,
                image=object_info.image,
                bbox=component_bbox,
                confidence=best_score
            )
            
            if self.debug:
                self.profiler.stop("create_component_info")
                
            # Estimate component pose if depth image is available
            if depth_image is not None:
                if self.debug:
                    self.profiler.start("estimate_component_pose")
                    
                # Ensure depth image matches RGB dimensions
                if depth_image.shape[:2] != object_info.image.shape[:2]:
                    if self.debug:
                        print("Resizing depth image to match RGB dimensions")
                    depth_image = cv2.resize(
                        depth_image,
                        (object_info.image.shape[1], object_info.image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                pose, pixel_pose = self._estimate_object_pose(component_full_mask, depth_image)
                component_info.pose = pose
                component_info.pixel_pose = pixel_pose
                
                if self.debug:
                    self.profiler.stop("estimate_component_pose")
                    
            # Store this component in the parent object if components dict exists
            if object_info.components is None:
                object_info.components = {}
            object_info.components[component_name] = component_info
            
            if self.debug:
                print(f"Successfully detected component '{component_name}' with confidence {best_score:.3f}")
                
            return component_info
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_object_component: {str(e)}")
                traceback.print_exc()
            return None
            
        finally:
            if self.debug:
                self.profiler.stop("detect_component")
                self.profiler.end_iteration(preserve_current=True)

    def detect_multiple_components(
        self,
        object_info: ObjectInfo,
        component_names: List[str],
        conf_threshold: float = 0.5,
        depth_image: Optional[np.ndarray] = None
    ) -> Dict[str, ObjectInfo]:
        """
        Detect multiple components within an object
        
        Args:
            object_info: Parent object information
            component_names: List of component names to detect
            conf_threshold: Confidence threshold for component detection
            depth_image: Optional depth image aligned with RGB
            
        Returns:
            Dictionary mapping component names to ObjectInfo objects
        """
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_multiple_components")
            print(f"Detecting {len(component_names)} components within object '{object_info.name}'")
        
        try:
            # Initialize results dictionary
            component_results = {}
            
            # Detect each component
            for component_name in component_names:
                if self.debug:
                    self.profiler.start(f"detect_{component_name}")
                    
                component = self.detect_object_component(
                    object_info=object_info,
                    component_name=component_name,
                    conf_threshold=conf_threshold,
                    depth_image=depth_image
                )
                
                if component is not None:
                    component_results[component_name] = component
                    
                if self.debug:
                    self.profiler.stop(f"detect_{component_name}")
                    
            if self.debug:
                print(f"Successfully detected {len(component_results)}/{len(component_names)} components")
                
            return component_results
            
        except Exception as e:
            if self.debug:
                print(f"Error in detect_multiple_components: {str(e)}")
                traceback.print_exc()
            return {}
            
        finally:
            if self.debug:
                self.profiler.stop("detect_multiple_components")
                self.profiler.end_iteration(preserve_current=True)
    
    def _generate_segmentation(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo,
        prompt: str
    ) -> np.ndarray:
        """
        Generate segmentation mask for detected object using FastSAM
        
        Args:
            image: Original image
            obj_info: Object information with bounding box
            prompt: Text prompt describing the object
            
        Returns:
            Binary mask array
        """
        if self.segmenter is None:
            if self.debug:
                print("Cannot generate segmentation: FastSAM segmenter not initialized")
            return None
            
        try:
            # Extract the region of interest
            x, y, w, h = obj_info.bbox
            
            if self.debug:
                print(f"Generating segmentation for '{prompt}' with bbox: x={x}, y={y}, w={w}, h={h}")
            
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
            x_expand = max(0, x - int(w * 0.1))
            y_expand = max(0, y - int(h * 0.1))
            w_expand = min(image.shape[1] - x_expand, int(w * 1.2))
            h_expand = min(image.shape[0] - y_expand, int(h * 1.2))
            
            # Store ROI coordinates for later use
            roi_coords = (x_expand, y_expand, w_expand, h_expand)
            
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
            
            # Run FastSAM on the ROI
            query = f"a {prompt}"
            if self.debug:
                print(f"Running FastSAM with query: '{query}'")
                
            # Use try-except to catch potential errors in the segmentation process
            try:
                labeled_masks, metadata = self.segmenter.process_image(
                    roi, 
                    query=query,
                    roi_coords=roi_coords,
                    store_results=True
                )
                
                if self.debug:
                    print(f"FastSAM returned {len(metadata)} masks")
            except Exception as e:
                if self.debug:
                    print(f"Error in FastSAM processing: {e}")
                    import traceback
                    traceback.print_exc()
                return None
            
            # Find best matching mask
            best_mask_id = None
            best_score = 0
            
            for mask_id, meta in metadata.items():
                score = meta.get('clip_score', 0)
                if self.debug:
                    print(f"  Mask {mask_id} score: {score:.3f}, area: {meta.get('area', 0):.0f}")
                if score > best_score:
                    best_score = score
                    best_mask_id = mask_id
                    
            if best_mask_id is None:
                if self.debug:
                    print(f"No mask found for '{prompt}' in ROI")
                    
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
            
            # Create a full image mask
            full_mask = np.zeros(image.shape[:2], dtype=bool)
            
            # Place ROI mask in the correct position in the full image
            try:
                full_mask[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand] = roi_mask
                
                if self.debug:
                    print(f"Created full mask with {np.sum(full_mask)} pixels")
            except ValueError as e:
                if self.debug:
                    print(f"Error placing mask in full image: {e}")
                    print(f"ROI shape: {roi_mask.shape}, Expected: {(h_expand, w_expand)}")
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
                x, y, w, h = obj_info.bbox
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
    
    def visualize_detections(
        self,
        image: np.ndarray,
        objects: List[ObjectInfo],
        show_masks: bool = True,
        show_components: bool = True,
        show_poses: bool = True
    ) -> np.ndarray:
        """
        Visualize detected objects with masks, components, and poses
        
        Args:
            image: Original RGB image
            objects: List of ObjectInfo objects to visualize
            show_masks: Whether to show segmentation masks
            show_components: Whether to show object components
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
                
                # Add label with confidence
                label = f"{obj.name} ({obj.confidence:.2f})"
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
                if show_components and obj.components is not None and len(obj.components) > 0:
                    # Process each component
                    for comp_name, comp_info in obj.components.items():
                        # Draw component bounding box in a different color (blue)
                        cx, cy, cw, ch = comp_info.bbox
                        cv2.rectangle(vis_image, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)
                        
                        # Add component label
                        comp_label = f"{comp_name} ({comp_info.confidence:.2f})"
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
            print("Performance profiling is disabled. Create HybridPerceptionSystem with debug=True to enable.")
            
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
            print("HybridPerceptionSystem resources released")
            
            # Print final performance summary
            print("\nFinal performance summary:")
            self.print_performance_summary(sort_by="total", top_n=10)