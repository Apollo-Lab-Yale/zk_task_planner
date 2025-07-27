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
from cognitive_bt_framework.src.vision.contour_shape_detector import ContourShapeDetector, ContourInfo
from cognitive_bt_framework.utils.time_profiler import IterationTimeProfiler
from cognitive_bt_framework.src.vision.interaction_point_detector import RobustInteractionDetector
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
            self.interaction_detector = RobustInteractionDetector(debug=self.debug)
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
            self.camera_matrix = camera_matrix
                
            # Pre-calculate camera matrix inverse for faster back-projection
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            self.camera_intrinsics = {"fx": self.fx, 'fy':self.fy, 'cx':self.cx, 'cy': self.cy}
            self.fx_inv = 1.0 / self.fx
            self.fy_inv = 1.0 / self.fy
            self.shape_detector = ContourShapeDetector(
                min_area=100,
                max_area=100000,
                min_circularity=0.5,  # Fairly circular objects
                min_vertices=3,
                max_vertices=20,
                debug=False
            )
            self.depth_scale = depth_scale
            self._object_cache = {}  # Cache for detected objects
            self._next_mask_id = 1  # Counter for assigning IDs to masks
            
            if self.debug:
                print(f"PerceptionSystem initialized with debug={debug}")
            self.__pixel_to_3d = self._estimate_point_pose
        except Exception as e:
            if self.debug:
                print(f"Error in PerceptionSystem initialization: {str(e)}")
                traceback.print_exc()
            raise  # Re-raise the exception to not silence initialization errors
    
    def detect_objects(self, image: np.ndarray, classes: Optional[List[str]] = None,
                    conf: Optional[float] = 0.01, segment: bool = True,
                    depth_image: Optional[np.ndarray] = None) -> List[ObjectInfo]:
        
        if self.debug:
            self.profiler.start_iteration()
            self.profiler.start("detect_objects")
        
        try:
            import copy
            
            # Store original image dimensions for coordinate transformation
            original_height, original_width = image.shape[:2]
            
            # DEBUG: Verify camera intrinsics match image size
            print(f"DEBUG - Camera setup verification:")
            print(f"  Original image dimensions: {original_width}x{original_height}")
            print(f"  Camera intrinsics:")
            print(f"    fx: {self.fx:.2f}, fy: {self.fy:.2f}")
            print(f"    cx: {self.cx:.2f}, cy: {self.cy:.2f}")
            print(f"  Principal point relative to image center:")
            print(f"    cx_offset: {self.cx - original_width/2:.2f}")
            print(f"    cy_offset: {self.cy - original_height/2:.2f}")
            
            # TEMPORARY: Test corrected principal point
            print(f"  Testing corrected principal point at image center:")
            self.cx_corrected = original_width / 2.0   # True center X
            self.cy_corrected = original_height / 2.0  # True center Y
            print(f"    cx_corrected: {self.cx_corrected:.2f}")
            print(f"    cy_corrected: {self.cy_corrected:.2f}")
            
            if self.debug:
                print(f"Original image dimensions: {original_width}x{original_height}")
            
            # Set classes if provided
            if classes is not None:
                self.detector.set_classes(classes, self.detector.get_text_pe(classes))
                if self.debug:
                    print(f"Set detection classes: {classes}")
            
            temp_img = copy.deepcopy(image)
            conf_threshold = conf if conf is not None else self.default_conf
            
            # Run YOLO detection
            if self.debug:
                self.profiler.start("yolo_detection")
                print(f"Running YOLO detection with confidence threshold {conf_threshold}")
            
            results = self.detector.predict(temp_img, verbose=False)
            
            # DEBUG: Verify YOLO's actual input size
            if len(results) > 0:
                print(f"DEBUG - YOLO results shape info:")
                print(f"  results[0].orig_shape: {getattr(results[0], 'orig_shape', 'Not available')}")
                print(f"  results[0].shape: {getattr(results[0], 'shape', 'Not available')}")
                if hasattr(results[0], 'imgs'):
                    print(f"  results[0].imgs.shape: {results[0].imgs.shape}")
                if hasattr(results[0], 'orig_img'):
                    print(f"  results[0].orig_img.shape: {results[0].orig_img.shape}")
                if hasattr(results[0], 'path'):
                    print(f"  results[0].path: {results[0].path}")
                # Check for other shape attributes
                for attr in ['img_shape', 'input_shape', 'preprocess_shape']:
                    if hasattr(results[0], attr):
                        print(f"  results[0].{attr}: {getattr(results[0], attr)}")
            
            # Check what size YOLO actually used
            yolo_input_size = None
            if len(results) > 0 and hasattr(results[0], 'orig_shape'):
                # YOLO sometimes stores the original shape
                yolo_input_size = results[0].orig_shape
            else:
                # Estimate YOLO's input size (it will use square images)
                max_dim = max(original_height, original_width)
                yolo_input_size = max_dim
                # Make it multiple of 32
                yolo_input_size = int(np.ceil(yolo_input_size / 32) * 32)
            
            if self.debug:
                print(f"YOLO input size: {yolo_input_size}")
                self.profiler.stop("yolo_detection")
                print(f"YOLO returned {len(results[0])} detections")
            
            # Calculate coordinate transformation factors
            if isinstance(yolo_input_size, (tuple, list)):
                yolo_h, yolo_w = yolo_input_size[:2]
            else:
                yolo_h = yolo_w = yolo_input_size  # Square input
            
            scale_x = original_width / yolo_w
            scale_y = original_height / yolo_h
            
            # DEBUG: Always show coordinate transformation info
            print(f"DEBUG - Image shape assumptions:")
            print(f"  Original camera image: {original_width}x{original_height}")
            print(f"  YOLO estimated input: {yolo_w}x{yolo_h}")
            print(f"  Max dimension: {max(original_height, original_width)}")
            print(f"  Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
            print(f"  Scale factor difference: x={abs(scale_x - 1.0):.6f}, y={abs(scale_y - 1.0):.6f}")
            
            if self.debug and (abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01):
                print(f"Coordinate transformation needed:")
                print(f"  YOLO: {yolo_w}x{yolo_h} -> Camera: {original_width}x{original_height}")
                print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            
            # Convert results to ObjectInfo objects
            object_infos = []
            
            if self.debug:
                self.profiler.start("process_detections")
            
            # Process each detection with coordinate transformation
            for i, detection in enumerate(results[0]):
                # Extract bounding box and transform coordinates
                xyxy = detection.boxes.xyxy.cpu().numpy()[0]
                x1_yolo, y1_yolo, x2_yolo, y2_yolo = xyxy
                
                # Transform to camera coordinates
                x1 = int(x1_yolo * scale_x)
                y1 = int(y1_yolo * scale_y)
                x2 = int(x2_yolo * scale_x)
                y2 = int(y2_yolo * scale_y)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                
                # DEBUG: Validate coordinate transformation
                center_x_yolo = (x1_yolo + x2_yolo) / 2
                center_y_yolo = (y1_yolo + y2_yolo) / 2
                center_x_cam = (x1 + x2) / 2
                center_y_cam = (y1 + y2) / 2
                expected_x = center_x_yolo * scale_x
                expected_y = center_y_yolo * scale_y
                error_x = abs(center_x_cam - expected_x)
                error_y = abs(center_y_cam - expected_y)
                
                print(f"DEBUG - Detection {i} coordinate validation:")
                print(f"  YOLO bbox: [{x1_yolo:.1f},{y1_yolo:.1f},{x2_yolo:.1f},{y2_yolo:.1f}]")
                print(f"  Camera bbox: {bbox}")
                print(f"  YOLO center: ({center_x_yolo:.1f}, {center_y_yolo:.1f})")
                print(f"  Camera center: ({center_x_cam:.1f}, {center_y_cam:.1f})")
                print(f"  Expected center: ({expected_x:.1f}, {expected_y:.1f})")
                print(f"  Transform error: X={error_x:.2f}px, Y={error_y:.2f}px")
                
                if self.debug:
                    print(f"Detection {i}: YOLO [{x1_yolo:.1f},{y1_yolo:.1f},{x2_yolo:.1f},{y2_yolo:.1f}] -> Camera {bbox}")
                
                # Extract class and confidence
                cls_id = int(detection.boxes.cls.cpu().numpy()[0])
                confidence = float(detection.boxes.conf.cpu().numpy()[0])
                class_name = detection.names[cls_id]
                
                # Create alphabetical ID
                alpha_id = get_alpha_id(self._next_mask_id)
                self._next_mask_id += 1
                
                # Process mask coordinates if available
                bool_mask = np.zeros((original_height, original_width), dtype=bool)
                
                if detection.masks is not None:
                    num_masks = len(detection.masks)
                    
                    for mask_idx in range(num_masks):
                        try:
                            # Get mask coordinates and transform them
                            mask_coords = detection.masks.xy[mask_idx]
                            
                            # Transform mask coordinates to camera space
                            transformed_coords = []
                            for coord in mask_coords:
                                x_cam = int(coord[0] * scale_x)
                                y_cam = int(coord[1] * scale_y)
                                # Ensure coordinates are within bounds
                                x_cam = max(0, min(x_cam, original_width - 1))
                                y_cam = max(0, min(y_cam, original_height - 1))
                                transformed_coords.append([x_cam, y_cam])
                            
                            # Convert to numpy array and create mask
                            points = np.array(transformed_coords, dtype=np.int32)
                            temp_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                            cv2.fillPoly(temp_mask, [points], 1)
                            bool_mask = np.logical_or(bool_mask, temp_mask.astype(bool))
                            
                            if self.debug and mask_idx > 0:
                                print(f"Merged mask {mask_idx+1}/{num_masks} for object {class_name}")
                                
                        except Exception as e:
                            if self.debug:
                                print(f"Error processing mask {mask_idx}: {str(e)}")
                            continue
                print("Got masks")
                # Create ObjectInfo with transformed coordinates
                obj_info = ObjectInfo(
                    id=i,
                    name=class_name,
                    bbox=bbox,  # Already transformed
                    mask=bool_mask,  # Already transformed
                    confidence=confidence,
                    image=image,  # Original camera image
                    depth_image=depth_image,  # Original camera depth
                    alpha_id=alpha_id,
                    camera_intrinsics=self.camera_intrinsics
                )
                
                # Continue with rest of processing...
                obj_info.points = self.detect_regions_of_interest(image, obj_info, max_points=12, min_distance=45, apply_center_shift=True)
                print("Detected regions of interest")
                # DEBUG: Validate points of interest coordinates
                if obj_info.points and 'pixel_coords' in obj_info.points:
                    print(f"DEBUG - Points of interest for detection {i} ({class_name}):")
                    print(f"  Number of points: {len(obj_info.points['pixel_coords'])}")
                    for j, (px, py) in enumerate(obj_info.points['pixel_coords']):
                        print(f"  Point {j}: ({px}, {py})")
                        # Validate point is within image bounds
                        if not (0 <= px < original_width and 0 <= py < original_height):
                            print(f"  WARNING: Point {j} is outside image bounds!")
                        # Validate point is within object bbox
                        if not (bbox[0] <= px <= bbox[0] + bbox[2] and bbox[1] <= py <= bbox[1] + bbox[3]):
                            print(f"  WARNING: Point {j} is outside object bbox!")
                
                obj_info.surface_masks = {}

                # Generate surface segmentation if requested
                if segment and self.segmenter is not None and obj_info.mask is not None:
                    if self.debug:
                        self.profiler.start(f"segment_surfaces_{i}")
                        print(f"Generating surface segmentation for object {i} ({class_name})")
                    
                    obj_info.surface_masks = self.segment_surfaces_by_plane_fitting(
                        image=temp_img,
                        obj_info=obj_info,
                        depth_image=depth_image,
                    )
                    
                    if self.debug:
                        self.profiler.stop(f"segment_surfaces_{i}")
                        print(f"Detected {len(obj_info.surface_masks)} surfaces for object {i}")
                
                # Estimate pose if depth image is provided
                if depth_image is not None:
                    if self.debug:
                        self.profiler.start(f"estimate_pose_{i}")
                    
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
                print(f"Processed {len(object_infos)} detections with coordinate transformation")
            
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


    def detect_regions_of_interest(
        self,
        image: np.ndarray,
        obj_info: ObjectInfo,
        method: str = 'robust',  # Default is still 'robust'
            max_points: int = 5,
            quality_level: float = 0.01,
            min_distance: int = 50,
            visualize: bool = False,
            orb_params: Dict[str, Any] = None,  # Now used for contour parameters
            apply_center_shift: bool = False  # Control center-shift behavior
        ) -> Dict[str, Any]:
            """
            Detect points of interest within a specific object mask.
            
            Args:
                image: RGB input image
                obj_info: ObjectInfo containing the object mask
                method: Feature detection method ('robust', 'contour', 'legacy')
                max_points: Maximum number of points to detect
                quality_level: Quality level parameter (not used for robust method)
                min_distance: Minimum distance between detected points
                visualize: Whether to generate visualization
                orb_params: Optional dictionary of contour parameters:
                    - min_contour_area: Minimum area for a contour to be considered (default: 50)
                    - perimeter_threshold: Area threshold for adding perimeter points (default: 500)
                    - perimeter_points_ratio: Ratio of perimeter to determine number of points (default: 100)
                    - max_perimeter_points: Maximum perimeter points per contour (default: 8)
                    - contour_approximation: Epsilon factor for contour approximation (default: 0.02)
                
            Returns:
                Dictionary containing:
                    - 'keypoints': List of cv2.KeyPoint objects
                    - 'descriptors': Descriptors if available (None for contour method)
                    - 'pixel_coords': List of (x,y) pixel coordinates
                    - 'scores': Confidence scores for each point
                    - 'interaction_points': List of InteractionPoint objects (if robust method)
                    - 'visualization': Visualization image (if visualize=True)
            """
            if self.debug:
                self.profiler.start_iteration()
                self.profiler.start("detect_regions_of_interest")
            
            try:
                # Check if the object has a mask
                if obj_info.mask is None:
                    if self.debug:
                        print(f"No mask available for object '{obj_info.name}', using bounding box...")
                    
                    # Use bounding box directly instead of expensive segmentation for speed
                    x, y, w, h = obj_info.bbox
                    mask = np.zeros(image.shape[:2], dtype=bool)
                    mask[y:y+h, x:x+w] = True
                else:
                    mask = obj_info.mask
                
                # Convert boolean mask to uint8 for OpenCV operations
                mask_uint8 = mask.astype(np.uint8) * 255
                
                # Use the new robust detection method
                if method == 'robust':
                    if self.debug:
                        self.profiler.start("robust_detection")
                    
                    # Get depth data if available
                    depth_data = obj_info.depth_image if hasattr(obj_info, 'depth_image') else None
                    
                    # Skip depth data processing if not available to save time
                    depth_data = None
                    if hasattr(obj_info, 'depth_image') and obj_info.depth_image is not None:
                        depth_data = obj_info.depth_image
                    
                    # Detect interaction points using the robust method with optimizations
                    interaction_points = self.interaction_detector.detect_interaction_points(
                        image=image,
                        mask=mask,
                        min_distance=min_distance,
                        obj_info=obj_info,
                        max_points=max_points,
                        depth_data=depth_data,
                        apply_center_shift=apply_center_shift,
                        edge_threshold=15.0,
                        shift_factor=0.4,
                        fast_mode=True  # Enable fast mode for better performance
                    )
                    
                    if self.debug:
                        self.profiler.stop("robust_detection")
                        print(f"Detected {len(interaction_points)} interaction points using robust method")
                    
                    # Convert InteractionPoint objects to cv2.KeyPoint format for backward compatibility
                    final_keypoints = []
                    pixel_coords = []
                    scores = []
                    point_types = []
                    
                    for point in interaction_points:
                        # Create cv2.KeyPoint object
                        kp = cv2.KeyPoint(
                            float(point.x), float(point.y), 
                            size=5.0, 
                            response=point.score
                        )
                        final_keypoints.append(kp)
                        pixel_coords.append((point.x, point.y))
                        scores.append(point.score)
                        point_types.append(point.interaction_type.value)
                    
                    # Create results dictionary
                    results = {
                        'keypoints': final_keypoints,
                        'descriptors': None,  # No descriptors for robust method
                        'pixel_coords': pixel_coords,
                        'scores': scores,
                        'object_name': obj_info.name,
                        'method': "robust",
                        'mask_area': np.sum(mask),
                        'ids': [get_alpha_id(i) for i in range(len(final_keypoints))],
                        'point_types': point_types,
                        'interaction_points': interaction_points  # Rich interaction data
                    }
                
                # New contour-based method (replaces ORB)
                elif method == 'contour':
                    if self.debug:
                        self.profiler.start("contour_detection")
                    
                    # Set default contour parameters if none provided
                    if orb_params is None:
                        orb_params = {}
                    
                    # Configure contour detection parameters
                    min_contour_area = orb_params.get('min_contour_area', 50)
                    perimeter_threshold = orb_params.get('perimeter_threshold', 500)
                    perimeter_points_ratio = orb_params.get('perimeter_points_ratio', 100)
                    max_perimeter_points = orb_params.get('max_perimeter_points', 8)
                    contour_approximation = orb_params.get('contour_approximation', 0.02)
                    
                    # Find contours in the mask
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    all_points = []
                    all_scores = []
                    point_types = []
                    
                    # Process each contour
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Skip very small contours
                        if area < min_contour_area:
                            continue
                            
                        # Calculate center point using moments
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Ensure center point is within image bounds
                            if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
                                # Add center point
                                all_points.append((cx, cy))
                                all_scores.append(area)  # Use area as score
                                point_types.append('CONTOUR_CENTER')
                                
                                # For large contours, add perimeter points
                                if area > perimeter_threshold:
                                    # Calculate number of perimeter points based on contour size
                                    perimeter = cv2.arcLength(contour, True)
                                    num_perimeter_points = min(max_perimeter_points, 
                                                            max(3, int(perimeter / perimeter_points_ratio)))
                                    
                                    # Approximate contour to get key points
                                    epsilon = contour_approximation * perimeter
                                    approx = cv2.approxPolyDP(contour, epsilon, True)
                                    
                                    # Add points from the approximated contour
                                    for point in approx[:num_perimeter_points]:
                                        px, py = point[0]
                                        # Ensure point is within image bounds
                                        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                                            all_points.append((px, py))
                                            all_scores.append(area * 0.7)  # Slightly lower score for perimeter points
                                            point_types.append('CONTOUR_PERIMETER')
                    
                    # Apply distance filtering and limit to max_points
                    if min_distance > 0 and len(all_points) > 1:
                        # Sort by score (highest first)
                        scored_data = list(zip(all_points, all_scores, point_types))
                        scored_data.sort(key=lambda x: x[1], reverse=True)
                        
                        filtered_points = []
                        filtered_scores = []
                        filtered_types = []
                        
                        for (px, py), score, ptype in scored_data:
                            # Check distance to already filtered points
                            too_close = False
                            for existing_x, existing_y in filtered_points:
                                dist = np.sqrt((px - existing_x)**2 + (py - existing_y)**2)
                                if dist < min_distance:
                                    too_close = True
                                    break
                            
                            if not too_close:
                                filtered_points.append((px, py))
                                filtered_scores.append(score)
                                filtered_types.append(ptype)
                                
                                if len(filtered_points) >= max_points:
                                    break
                        
                        all_points = filtered_points
                        all_scores = filtered_scores
                        point_types = filtered_types
                    else:
                        # Just limit by max_points if no distance filtering
                        if len(all_points) > max_points:
                            scored_data = list(zip(all_points, all_scores, point_types))
                            scored_data.sort(key=lambda x: x[1], reverse=True)
                            all_points = [p for p, s, t in scored_data[:max_points]]
                            all_scores = [s for p, s, t in scored_data[:max_points]]
                            point_types = [t for p, s, t in scored_data[:max_points]]
                    
                    # Convert to cv2.KeyPoint objects
                    keypoints = []
                    for (x, y), score in zip(all_points, all_scores):
                        kp = cv2.KeyPoint(
                            float(x), float(y),
                            size=5.0,
                            response=float(score)
                        )
                        keypoints.append(kp)
                    
                    pixel_coords = all_points
                    scores = all_scores
                    
                    if self.debug:
                        self.profiler.stop("contour_detection")
                        print(f"Detected {len(keypoints)} contour-based keypoints")
                    
                    # Create results dictionary
                    results = {
                        'keypoints': keypoints,
                        'descriptors': None,  # No descriptors for contour method
                        'pixel_coords': pixel_coords,
                        'scores': scores,
                        'object_name': obj_info.name,
                        'method': "contour",
                        'mask_area': np.sum(mask),
                        'ids': [get_alpha_id(i) for i in range(len(keypoints))],
                        'point_types': point_types
                    }
                    
                    # Create InteractionPoint objects for compatibility
                    interaction_points = []
                    for i, ((x, y), score, ptype) in enumerate(zip(all_points, all_scores, point_types)):
                        try:
                            interaction_type = InteractionPointType.GENERIC
                        except:
                            # Fallback if enum not defined
                            class DummyEnum(Enum):
                                GENERIC = "generic"
                            interaction_type = DummyEnum.GENERIC
                        
                        # Create an InteractionPoint-like object with required attributes
                        interaction_point = type('InteractionPoint', (), {
                            'x': int(x),
                            'y': int(y),
                            'score': score,
                            'interaction_type': interaction_type,
                            'size': 5.0,
                            'angle': 0.0,
                            'id': get_alpha_id(i),
                            'point_subtype': ptype
                        })
                        
                        interaction_points.append(interaction_point)
                    
                    results['interaction_points'] = interaction_points
                    
                elif method == 'legacy':
                    # Fall back to the original legacy method
                    results = self._detect_regions_legacy(
                        image, obj_info, mask, max_points, min_distance
                    )
                    
                else:
                    raise ValueError(f"Unknown detection method: {method}. Available methods: 'robust', 'contour', 'legacy'")
                
                # Create visualization if requested
                if visualize:
                    if self.debug:
                        self.profiler.start("roi_visualization")
                    
                    visualization = self.visualize_interest_points(
                        image=image,
                        obj_info=obj_info,
                        keypoints=results['keypoints'],
                        scores=results['scores'],
                        method=method,
                        max_points=max_points,
                        descriptors=results.get('descriptors')
                    )
                    
                    results['visualization'] = visualization
                    
                    if self.debug:
                        self.profiler.stop("roi_visualization")
                
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
        region_size: int = 1,  # Size of region around point to sample (half-width)
        temporal_smoothing: bool = False,  # Enable temporal smoothing
        smoothing_factor: float = 0.7,  # Weight for current measurement in temporal smoothing
        edge_preservation: bool = False,  # Preserve depth edges during filtering
        label: str = None
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate 3D position of a specific point using depth information with enhanced filtering
        
        Args:
            point: (x, y) pixel coordinates of the point
            depth_image: Depth image aligned with RGB
            region_size: Half-width of the square region to sample around the point
            temporal_smoothing: Whether to apply temporal smoothing with previous estimates
            smoothing_factor: Weight for current measurement (0-1), lower values = more smoothing
            edge_preservation: Whether to apply edge-aware filtering
            
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
            print(f"converting point to 3d {label}, ({pixel_x}, {pixel_y})")
            # Filter valid depth values (non-zero and within reasonable range)
            MAX_DEPTH = 2.0  # Maximum reasonable depth in meters
            MIN_DEPTH = 0.05  # Minimum reasonable depth in meters
            print(f"Region depths {region_depths}, {len(region_depths)}")
            # Create mask of valid depths
            valid_mask = (region_depths > MIN_DEPTH) & (region_depths < MAX_DEPTH)
            valid_depths = region_depths[valid_mask]
            print(f"Valid depths: {valid_depths}, {len(valid_depths)}")
            # Check if we have enough valid depth values
            if len(valid_depths) < 1:
                # If exact point has valid depth, use it despite few valid neighbors
                center_depth = depth_image[pixel_y, pixel_x] * self.depth_scale
                if MIN_DEPTH < center_depth < MAX_DEPTH:
                    valid_depths = np.array([center_depth])
                else:
                    if self.debug:
                        print(f"Too few valid depth points for ({pixel_x}, {pixel_y}): {len(valid_depths)}")
                    return np.zeros(3), 0.0
            
            # Create distance weights (pixels closer to target point have higher weights)
            if len(valid_depths) >= 3:
                # Get coordinates of valid points
                y_indices, x_indices = np.where(valid_mask)
                
                # Calculate distance from center point
                center_y = pixel_y - y_min
                center_x = pixel_x - x_min
                distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                
                # Convert distances to weights (closer points get higher weights)
                weights = 1.0 / (distances + 0.1)  # Adding 0.1 to avoid division by zero
                weights = weights / np.sum(weights)  # Normalize weights to sum to 1
                
                # Apply edge-aware filtering if enabled
                if edge_preservation:
                    # Calculate depth differences from center point
                    center_depth = region_depths[center_y, center_x] if 0 <= center_y < region_depths.shape[0] and 0 <= center_x < region_depths.shape[1] else np.median(valid_depths)
                    if center_depth < MIN_DEPTH or center_depth > MAX_DEPTH:
                        center_depth = np.median(valid_depths)
                    
                    depth_diffs = np.abs(region_depths[valid_mask] - center_depth)
                    
                    # Calculate edge weights (smaller differences get higher weights)
                    edge_sigma = 0.1  # Controls sensitivity to depth discontinuities
                    edge_weights = np.exp(-depth_diffs**2 / (2 * edge_sigma**2))
                    
                    # Combine distance and edge weights
                    weights = weights * edge_weights
                    weights = weights / np.sum(weights)  # Re-normalize
                
                # Apply weighted median or mean
                # Option 1: Weighted mean (faster but less robust to outliers)
                depth = np.sum(valid_depths * weights)
                
                # Option 2: Weighted median (more robust but slower)
                # Uncomment to use weighted median instead of mean
                # sorted_indices = np.argsort(valid_depths)
                # cumsum = np.cumsum(weights[sorted_indices])
                # idx = np.searchsorted(cumsum, 0.5)
                # depth = valid_depths[sorted_indices[idx]]
                
                # Enhanced outlier rejection using MAD (Median Absolute Deviation)
                median_depth = np.median(valid_depths)
                mad = np.median(np.abs(valid_depths - median_depth))
                inlier_mask = np.abs(valid_depths - median_depth) < (3.0 * mad)  # 3.0 is a common threshold
                
                if np.sum(inlier_mask) >= 3:
                    # Recalculate with outliers removed
                    valid_depths = valid_depths[inlier_mask]
                    if edge_preservation:
                        weights = weights[inlier_mask]
                        weights = weights / np.sum(weights)
                        depth = np.sum(valid_depths * weights)
                    else:
                        depth = np.median(valid_depths)
            else:
                # Fall back to simple median for few points
                depth = np.median(valid_depths)
            
            # Apply temporal smoothing if enabled and we have previous estimates
            if temporal_smoothing and hasattr(self, '_prev_depth') and hasattr(self, '_prev_position'):
                if self._prev_depth > 0:
                    # Blend current and previous depth
                    smoothed_depth = smoothing_factor * depth + (1 - smoothing_factor) * self._prev_depth
                    depth = smoothed_depth
            
            # Store current depth for next frame's temporal smoothing
            self._prev_depth = depth
            
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
            
            # Store position for temporal smoothing
            if temporal_smoothing:
                # Apply smoothing to 3D position if we have previous position
                if hasattr(self, '_prev_position') and np.any(self._prev_position):
                    position_3d = smoothing_factor * position_3d + (1 - smoothing_factor) * self._prev_position
                
                self._prev_position = position_3d.copy()
            print(f"{label} pose: {position_3d}")
            
            # DEBUG: Check for depth-dependent scaling issues
            expected_pixel_x = (position_3d[0] * self.fx / position_3d[2]) + self.cx
            expected_pixel_y = (position_3d[1] * self.fy / position_3d[2]) + self.cy
            pixel_error_x = abs(expected_pixel_x - pixel_x)
            pixel_error_y = abs(expected_pixel_y - pixel_y)
            print(f"  Back-projection check: expected_pixel=({expected_pixel_x:.1f}, {expected_pixel_y:.1f}), actual=({pixel_x}, {pixel_y})")
            print(f"  Back-projection error: ({pixel_error_x:.1f}, {pixel_error_y:.1f}) pixels")
            
            return position_3d, confidence
            
        except Exception as e:
            if self.debug:
                print(f"Error in _estimate_point_pose: {str(e)}")
                traceback.print_exc()
            return np.zeros(3), 0.0
            
        finally:
            if self.debug:
                self.profiler.stop("_estimate_point_pose")
    
    
    
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


    def calculate_surface_normal(
        self,
        depth_image: np.ndarray,
        mask: np.ndarray,
        method: str = 'ransac',
        max_plane_distance: float = 0.01,
        ransac_iterations: int = 100,
        sample_count: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate the surface normal for a given mask using the depth image
        
        Args:
            depth_image: Depth image aligned with RGB
            mask: Binary mask of the surface area
            method: Method to use for normal calculation ('ransac' or 'pca')
            max_plane_distance: Maximum distance for a point to be considered inlier (meters)
            ransac_iterations: Number of RANSAC iterations
            sample_count: Maximum number of points to sample (for efficiency)
            
        Returns:
            Tuple containing:
            - Normal vector as a 3D unit vector [nx, ny, nz]
            - Confidence value (0-1) based on inlier ratio
        """
        if self.debug:
            self.profiler.start("calculate_surface_normal")
        
        try:
            # Ensure mask and depth image have same dimensions
            if mask.shape != depth_image.shape:
                # Resize mask to match depth image dimensions
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_image.shape[1], depth_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # Get masked points
            y_coords, x_coords = np.where(mask)
            if len(y_coords) < 10:  # Need at least 10 points for reliable estimation
                if self.debug:
                    print(f"Too few points in mask: {len(y_coords)}")
                return np.array([0, 0, 1]), 0.0  # Default to upward normal with zero confidence
            
            # Get corresponding depth values
            depth_values = depth_image[y_coords, x_coords] * self.depth_scale
            
            # Filter out invalid depth values
            valid = (depth_values > 0.05) & (depth_values < 10.0)  # 5cm to 10m range
            if np.sum(valid) < 10:
                if self.debug:
                    print(f"Too few valid depth points: {np.sum(valid)}/{len(depth_values)}")
                return np.array([0, 0, 1]), 0.0
            
            # Use filtered values
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            depth_values = depth_values[valid]
            
            # If we have too many points, sample a subset for efficiency
            if len(depth_values) > sample_count:
                indices = np.random.choice(len(depth_values), sample_count, replace=False)
                x_coords = x_coords[indices]
                y_coords = y_coords[indices]
                depth_values = depth_values[indices]
            
            # Convert to 3D points
            points_3d = np.zeros((len(x_coords), 3))
            points_3d[:, 0] = (x_coords - self.cx) * depth_values / self.fx
            points_3d[:, 1] = (y_coords - self.cy) * depth_values / self.fy
            points_3d[:, 2] = depth_values
            
            # Use method selected to calculate normal
            if method == 'ransac':
                # RANSAC plane fitting
                best_inliers = 0
                best_normal = np.array([0, 0, 1])  # Default to upward normal
                best_d = 0
                
                # Run RANSAC iterations
                for _ in range(ransac_iterations):
                    # Sample 3 random points
                    sample_indices = np.random.choice(len(points_3d), 3, replace=False)
                    p1, p2, p3 = points_3d[sample_indices]
                    
                    # Calculate normal from cross product of two vectors in the plane
                    v1 = p2 - p1
                    v2 = p3 - p1
                    normal = np.cross(v1, v2)
                    
                    # Skip if normal is too small (collinear points)
                    normal_length = np.linalg.norm(normal)
                    if normal_length < 1e-6:
                        continue
                    
                    # Normalize the normal vector
                    normal = normal / normal_length
                    
                    # Calculate d in the plane equation ax + by + cz + d = 0
                    d = -np.dot(normal, p1)
                    
                    # Count inliers
                    distances = np.abs(np.dot(points_3d, normal) + d)
                    inliers = np.sum(distances < max_plane_distance)
                    
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_normal = normal
                        best_d = d
                
                # Calculate confidence as inlier ratio
                confidence = best_inliers / len(points_3d)
                
                # Ensure normal points toward camera (positive z)
                if best_normal[2] < 0:
                    best_normal = -best_normal
                
                return best_normal, confidence
                
            elif method == 'pca':
                # PCA-based normal estimation
                # Compute the centroid
                centroid = np.mean(points_3d, axis=0)
                
                # Center the points
                centered_points = points_3d - centroid
                
                # Compute covariance matrix
                cov = np.dot(centered_points.T, centered_points) / centered_points.shape[0]
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # The normal is the eigenvector corresponding to the smallest eigenvalue
                normal = eigenvectors[:, 0]
                
                # Ensure normal points toward camera (positive z)
                if normal[2] < 0:
                    normal = -normal
                
                # Normalize the vector
                normal = normal / np.linalg.norm(normal)
                
                # Calculate confidence based on eigenvalue ratio
                # If smallest eigenvalue is much smaller than others, the surface is more planar
                if eigenvalues[1] > 0:
                    planarity = 1.0 - (eigenvalues[0] / eigenvalues[1])
                    confidence = min(1.0, max(0.0, planarity))
                else:
                    confidence = 0.0
                
                return normal, confidence
            
            else:
                if self.debug:
                    print(f"Unknown normal calculation method: {method}")
                return np.array([0, 0, 1]), 0.0
                
        except Exception as e:
            if self.debug:
                print(f"Error in calculate_surface_normal: {str(e)}")
                traceback.print_exc()
            return np.array([0, 0, 1]), 0.0
            
        finally:
            if self.debug:
                self.profiler.stop("calculate_surface_normal")
                           
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
            
        except Exception as e:
            if self.debug:
                print(f"Error in segment_surfaces_by_plane_fitting: {str(e)}")
                traceback.print_exc()
            return {}
            
        finally:
            if self.debug:
                self.profiler.stop("segment_surfaces_by_plane")