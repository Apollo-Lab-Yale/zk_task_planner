import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union

@dataclass
class ContourInfo:
    """Data class for storing detected contour information"""
    id: int
    contour: np.ndarray
    area: float
    perimeter: float
    approx_poly: np.ndarray
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    circularity: float
    is_convex: bool
    min_area_rect: Optional[Any] = None
    shape_type: str = "unknown"
    confidence: float = 0.0
    hierarchy_level: int = 0

class ContourShapeDetector:
    """Class for detecting contours that represent complete shapes"""
    
    def __init__(
        self,
        min_area: float = 100.0,
        max_area: float = 100000.0,
        min_circularity: float = 0.0,
        max_circularity: float = 1.0,
        min_vertices: int = 3,
        max_vertices: int = 50,
        approx_poly_epsilon: float = 0.02,
        debug: bool = False
    ):
        """
        Initialize the contour shape detector
        
        Args:
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            min_circularity: Minimum circularity (0-1) to consider (1 = perfect circle)
            max_circularity: Maximum circularity to consider
            min_vertices: Minimum vertices in the approximated polygon
            max_vertices: Maximum vertices in the approximated polygon
            approx_poly_epsilon: Epsilon value for polygon approximation
            debug: Enable debug output
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.approx_poly_epsilon = approx_poly_epsilon
        self.debug = debug
    
    def preprocess_image(
        self, 
        image: np.ndarray,
        blur_size: int = 5,
        threshold_method: str = "adaptive"
    ) -> np.ndarray:
        """
        Preprocess image for contour detection
        
        Args:
            image: Input image (grayscale or RGB)
            blur_size: Size of Gaussian blur kernel
            threshold_method: Thresholding method ("adaptive", "otsu", or "simple")
            
        Returns:
            Binary image ready for contour detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Apply thresholding
        if threshold_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )
        elif threshold_method == "otsu":
            _, binary = cv2.threshold(
                blurred, 
                0, 
                255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:  # simple threshold
            _, binary = cv2.threshold(
                blurred, 
                127, 
                255, 
                cv2.THRESH_BINARY_INV
            )
            
        # Apply morphological operations to close small gaps and remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        if self.debug:
            print(f"Preprocessed image: {binary.shape}, method: {threshold_method}")
            
        return binary
        
    def detect_contours(
        self, 
        image: np.ndarray,
        retrieval_mode: int = cv2.RETR_EXTERNAL,
        approximation_method: int = cv2.CHAIN_APPROX_SIMPLE,
        preprocess: bool = True,
        threshold_method: str = "adaptive"
    ) -> List[ContourInfo]:
        """
        Detect contours in the image and filter by shape properties
        
        Args:
            image: Input image (grayscale or RGB)
            retrieval_mode: Contour retrieval mode
            approximation_method: Contour approximation method
            preprocess: Whether to preprocess the image
            threshold_method: Thresholding method if preprocessing
            
        Returns:
            List of ContourInfo objects for valid contours
        """
        # Preprocess the image if requested
        if preprocess:
            binary = self.preprocess_image(image, threshold_method=threshold_method)
        else:
            # Use the image as-is but ensure it's binary
            if len(image.shape) == 3:
                binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = image.copy()
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, 
            retrieval_mode, 
            approximation_method
        )
        
        if self.debug:
            print(f"Found {len(contours)} contours")
        
        # Process each contour
        contour_infos = []
        
        for i, contour in enumerate(contours):
            # Calculate basic properties
            area = cv2.contourArea(contour)
            
            # Skip contours that are too small or too large
            if area < self.min_area or area > self.max_area:
                if self.debug:
                    print(f"Skipping contour {i}: area={area:.2f} outside range [{self.min_area}, {self.max_area}]")
                continue
                
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity (1 = perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Skip contours with circularity outside range
            if circularity < self.min_circularity or circularity > self.max_circularity:
                if self.debug:
                    print(f"Skipping contour {i}: circularity={circularity:.2f} outside range " 
                          f"[{self.min_circularity}, {self.max_circularity}]")
                continue
            
            # Calculate approximated polygon
            epsilon = self.approx_poly_epsilon * perimeter
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            
            # Skip if polygon has too few or too many vertices
            num_vertices = len(approx_poly)
            if num_vertices < self.min_vertices or num_vertices > self.max_vertices:
                if self.debug:
                    print(f"Skipping contour {i}: vertices={num_vertices} outside range "
                          f"[{self.min_vertices}, {self.max_vertices}]")
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)
            else:
                center = (x + w // 2, y + h // 2)
            
            # Check convexity
            is_convex = cv2.isContourConvex(approx_poly)
            
            # Calculate minimum area rectangle
            min_area_rect = cv2.minAreaRect(contour)
            
            # Determine shape type and confidence
            shape_type, confidence = self._classify_shape(
                approx_poly, 
                is_convex, 
                circularity, 
                w, 
                h,
                area
            )
            
            # Get hierarchy level
            hierarchy_level = 0
            if hierarchy is not None:
                parent_idx = hierarchy[0][i][3]
                level = 0
                while parent_idx != -1:
                    level += 1
                    parent_idx = hierarchy[0][parent_idx][3]
                hierarchy_level = level
            
            # Create ContourInfo object
            contour_info = ContourInfo(
                id=i,
                contour=contour,
                area=area,
                perimeter=perimeter,
                approx_poly=approx_poly,
                bounding_rect=(x, y, w, h),
                center=center,
                circularity=circularity,
                is_convex=is_convex,
                min_area_rect=min_area_rect,
                shape_type=shape_type,
                confidence=confidence,
                hierarchy_level=hierarchy_level
            )
            
            contour_infos.append(contour_info)
        
        if self.debug:
            print(f"Kept {len(contour_infos)} contours after filtering")
            
        return contour_infos
        
    def _classify_shape(
        self, 
        approx_poly: np.ndarray, 
        is_convex: bool, 
        circularity: float,
        width: int,
        height: int,
        area: float
    ) -> Tuple[str, float]:
        """
        Classify the shape type and assign a confidence
        
        Args:
            approx_poly: Approximated polygon
            is_convex: Whether the contour is convex
            circularity: Circularity measure (1 = perfect circle)
            width: Width of bounding rectangle
            height: Height of bounding rectangle
            area: Area of the contour
            
        Returns:
            Tuple of (shape_type, confidence)
        """
        num_vertices = len(approx_poly)
        confidence = 0.0
        
        # Calculate aspect ratio
        aspect_ratio = float(width) / height if height > 0 else 0
        
        # Check if shape is close to circle
        if circularity > 0.85:
            shape_type = "circle"
            confidence = min(1.0, circularity)
            
        # Check for triangle
        elif num_vertices == 3 and is_convex:
            shape_type = "triangle"
            confidence = 0.9
            
        # Check for square or rectangle
        elif num_vertices == 4 and is_convex:
            # Square has aspect ratio close to 1
            if 0.8 < aspect_ratio < 1.2:
                shape_type = "square"
                confidence = 0.9
            else:
                shape_type = "rectangle"
                confidence = 0.9
                
        # Check for regular pentagon
        elif num_vertices == 5 and is_convex:
            shape_type = "pentagon"
            confidence = 0.8
            
        # Check for regular hexagon
        elif num_vertices == 6 and is_convex:
            shape_type = "hexagon"
            confidence = 0.8
            
        # Check for ellipse
        elif circularity > 0.7 and not (0.8 < aspect_ratio < 1.2):
            shape_type = "ellipse"
            confidence = min(1.0, circularity)
            
        # Other polygons with many sides
        elif num_vertices > 6 and is_convex:
            shape_type = f"polygon_{num_vertices}"
            confidence = 0.7
            
        # Non-convex shapes
        elif not is_convex:
            shape_type = "irregular"
            confidence = 0.6
            
        else:
            shape_type = "unknown"
            confidence = 0.5
            
        return shape_type, confidence

    def generate_shape_masks(
        self,
        image: np.ndarray,
        contour_infos: List[ContourInfo]
    ) -> Dict[str, np.ndarray]:
        """
        Generate binary masks for each detected shape
        
        Args:
            image: Original image
            contour_infos: List of ContourInfo objects
            
        Returns:
            Dictionary mapping shape types to binary masks
        """
        # Initialize empty dictionary for masks
        shape_masks = {}
        def get_alpha_id(num):
            import random
            import string
            """
            Generate a random sequence of three lowercase letters.
            Each call returns a new random ID regardless of the number passed.
            """
            # Generate 3 random lowercase letters
            letters = ''.join(random.choices(string.ascii_lowercase, k=3))
            return letters
        # Create empty mask with same dimensions as input image
        h, w = image.shape[:2]
        
        # Process each contour
        for info in contour_infos:
            # Generate a unique key for this shape
            key = get_alpha_id(1)
            
            # Create empty mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Draw filled contour on mask
            cv2.drawContours(mask, [info.contour], 0, 255, -1)
            
            # Convert to boolean mask
            bool_mask = mask.astype(bool)
            
            # Add to dictionary
            shape_masks[key] = bool_mask
            
        return shape_masks
    
    def visualize_shapes(
        self,
        image: np.ndarray,
        contour_infos: List[ContourInfo],
        draw_contours: bool = True,
        draw_centers: bool = True,
        draw_bounding_boxes: bool = True,
        draw_labels: bool = True,
        color_by_type: bool = True
    ) -> np.ndarray:
        """
        Create a visualization of detected shapes
        
        Args:
            image: Original image
            contour_infos: List of ContourInfo objects
            draw_contours: Whether to draw contour outlines
            draw_centers: Whether to draw center points
            draw_bounding_boxes: Whether to draw bounding boxes
            draw_labels: Whether to draw shape labels
            color_by_type: Whether to use different colors for different shape types
            
        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = image.copy()
            
        # Define colors for different shape types
        shape_colors = {
            "circle": (0, 255, 0),      # Green
            "square": (255, 0, 0),      # Blue
            "rectangle": (0, 0, 255),   # Red
            "triangle": (255, 255, 0),  # Cyan
            "pentagon": (255, 0, 255),  # Magenta
            "hexagon": (0, 255, 255),   # Yellow
            "ellipse": (128, 0, 128),   # Purple
            "polygon": (0, 128, 128),   # Brown
            "irregular": (128, 128, 0), # Olive
            "unknown": (128, 128, 128)  # Gray
        }
        
        # Draw each contour
        for info in contour_infos:
            # Determine color based on shape type
            if color_by_type:
                base_type = info.shape_type.split('_')[0]  # Handle polygon_N
                color = shape_colors.get(base_type, shape_colors["unknown"])
            else:
                color = (0, 255, 0)  # Default to green
                
            # Draw contour
            if draw_contours:
                cv2.drawContours(vis_img, [info.contour], 0, color, 2)
                
            # Draw center point
            if draw_centers:
                cv2.circle(vis_img, info.center, 4, color, -1)
                
            # Draw bounding box
            if draw_bounding_boxes:
                x, y, w, h = info.bounding_rect
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)
                
            # Draw label
            if draw_labels:
                label = f"{info.shape_type} ({info.confidence:.2f})"
                cv2.putText(
                    vis_img,
                    label,
                    (info.center[0] - 20, info.center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
                
        return vis_img
    
    def filter_shapes_by_type(
        self,
        contour_infos: List[ContourInfo],
        shape_types: List[str],
        min_confidence: float = 0.0
    ) -> List[ContourInfo]:
        """
        Filter contours by shape type
        
        Args:
            contour_infos: List of ContourInfo objects
            shape_types: List of shape types to keep
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of ContourInfo objects
        """
        filtered = []
        
        for info in contour_infos:
            # Check if shape type is in the list (handle partial matches like polygon_N)
            type_match = False
            for shape_type in shape_types:
                if shape_type in info.shape_type:
                    type_match = True
                    break
                    
            # Check confidence threshold
            if type_match and info.confidence >= min_confidence:
                filtered.append(info)
                
        return filtered
        
    def detect_complete_shapes(
        self,
        image: np.ndarray,
        shape_types: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        preprocess: bool = True,
        hierarchy_mode: bool = True,
        threshold_methods: Optional[List[str]] = None,
        fit_shapes_to_contours: bool = True,
        shape_fitting_tolerance: float = 0.2
    ) -> Tuple[List[ContourInfo], Dict[str, np.ndarray]]:
        """
        Detect complete shapes in the image, with option to fit best shape to imperfect contours
        
        Args:
            image: Input image
            shape_types: List of shape types to detect (None = all types)
            min_confidence: Minimum confidence threshold
            preprocess: Whether to preprocess the image
            hierarchy_mode: Whether to use hierarchy information to detect nested contours
            threshold_methods: List of threshold methods to try (None = use adaptive)
            fit_shapes_to_contours: Whether to fit best shape to imperfect contours
            shape_fitting_tolerance: Tolerance for shape fitting (0-1, higher = more lenient)
            
        Returns:
            Tuple of (filtered contour infos, shape masks)
        """
        # Set default threshold methods
        if threshold_methods is None:
            threshold_methods = ["adaptive"]
            
        # Try different preprocessing methods
        all_contours = []
        
        for method in threshold_methods:
            # Choose retrieval mode based on hierarchy flag
            retrieval_mode = cv2.RETR_CCOMP if hierarchy_mode else cv2.RETR_EXTERNAL
            
            # Detect contours
            contours = self.detect_contours(
                image,
                retrieval_mode=retrieval_mode,
                preprocess=preprocess,
                threshold_method=method
            )
            
            all_contours.extend(contours)
            
            if self.debug:
                print(f"Method {method} found {len(contours)} contours")
        
        # Apply shape fitting for imperfect contours if requested
        if fit_shapes_to_contours:
            all_contours = self._fit_shapes_to_contours(all_contours, image.shape[:2], shape_fitting_tolerance)
        
        # Filter by shape type if requested
        if shape_types is not None:
            filtered_contours = self.filter_shapes_by_type(
                all_contours,
                shape_types,
                min_confidence
            )
        else:
            # Filter by confidence only
            filtered_contours = [c for c in all_contours if c.confidence >= min_confidence]
            
        # Generate masks
        masks = self.generate_shape_masks(image, filtered_contours)
        
        return filtered_contours, masks

    def _fit_shapes_to_contours(
        self,
        contour_infos: List[ContourInfo],
        image_shape: Tuple[int, int],
        tolerance: float = 0.2
    ) -> List[ContourInfo]:
        """
        Fit the largest possible standard shape to each contour
        
        Args:
            contour_infos: List of ContourInfo objects
            image_shape: Shape of the image (height, width)
            tolerance: Tolerance for shape fitting (0-1, higher = more lenient)
            
        Returns:
            Updated list of ContourInfo objects
        """
        processed_contours = []
        
        for info in contour_infos:
            contour = info.contour
            
            # Skip if contour already has high confidence
            if info.confidence > 0.9:
                processed_contours.append(info)
                continue
            
            # Try to fit different shape types
            fitted_shape = None
            best_fit_score = 0.0
            best_contour = contour
            
            # 1. Try to fit a circle
            if info.circularity > 0.7 - tolerance:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Create a mask for the contour
                mask = np.zeros(image_shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                # Create a mask for the fitted circle
                circle_mask = np.zeros(image_shape, dtype=np.uint8)
                cv2.circle(circle_mask, center, radius, 255, -1)
                
                # Calculate IoU (Intersection over Union)
                intersection = cv2.bitwise_and(mask, circle_mask)
                union = cv2.bitwise_or(mask, circle_mask)
                
                intersection_area = cv2.countNonZero(intersection)
                union_area = cv2.countNonZero(union)
                
                iou = intersection_area / union_area if union_area > 0 else 0
                
                # Calculate area ratio (contour area / circle area)
                circle_area = np.pi * radius * radius
                area_ratio = info.area / circle_area if circle_area > 0 else 0
                
                # Combine metrics for final fit score
                fit_score = (iou * 0.7) + (area_ratio * 0.3)
                
                if fit_score > best_fit_score:
                    best_fit_score = fit_score
                    fitted_shape = "circle"
                    
                    # Generate a proper circle contour
                    circle_contour = []
                    for angle in range(0, 360, 5):  # 5-degree steps
                        x = center[0] + int(radius * np.cos(angle * np.pi / 180))
                        y = center[1] + int(radius * np.sin(angle * np.pi / 180))
                        circle_contour.append([[x, y]])
                    
                    best_contour = np.array(circle_contour, dtype=np.int32)
            
            # 2. Try to fit a rectangle or square
            if len(info.approx_poly) >= 4 - int(tolerance * 2) and len(info.approx_poly) <= 4 + int(tolerance * 2):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Calculate width and height of the rectangle
                width = np.linalg.norm(box[0] - box[1])
                height = np.linalg.norm(box[1] - box[2])
                
                # Check if it's close to a square
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
                is_square = 1.0 <= aspect_ratio <= 1.25
                
                # Create masks for comparison
                mask = np.zeros(image_shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                rect_mask = np.zeros(image_shape, dtype=np.uint8)
                cv2.drawContours(rect_mask, [box], 0, 255, -1)
                
                # Calculate IoU
                intersection = cv2.bitwise_and(mask, rect_mask)
                union = cv2.bitwise_or(mask, rect_mask)
                
                intersection_area = cv2.countNonZero(intersection)
                union_area = cv2.countNonZero(union)
                
                iou = intersection_area / union_area if union_area > 0 else 0
                
                # Calculate area ratio
                rect_area = width * height
                area_ratio = info.area / rect_area if rect_area > 0 else 0
                
                # Combine metrics for final fit score
                fit_score = (iou * 0.7) + (area_ratio * 0.3)
                
                if fit_score > best_fit_score:
                    best_fit_score = fit_score
                    fitted_shape = "square" if is_square else "rectangle"
                    best_contour = box
            
            # 3. Try to fit a triangle
            if len(info.approx_poly) == 3 or (3 - int(tolerance * 2) <= len(info.approx_poly) <= 3 + int(tolerance * 2)):
                triangle = cv2.minEnclosingTriangle(contour)[1]
                if triangle is not None:
                    triangle = np.int0(triangle)
                    
                    # Create masks for comparison
                    mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    tri_mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.drawContours(tri_mask, [triangle], 0, 255, -1)
                    
                    # Calculate IoU
                    intersection = cv2.bitwise_and(mask, tri_mask)
                    union = cv2.bitwise_or(mask, tri_mask)
                    
                    intersection_area = cv2.countNonZero(intersection)
                    union_area = cv2.countNonZero(union)
                    
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    # Calculate area ratio
                    tri_area = cv2.contourArea(triangle)
                    area_ratio = info.area / tri_area if tri_area > 0 else 0
                    
                    # Combine metrics for final fit score
                    fit_score = (iou * 0.7) + (area_ratio * 0.3)
                    
                    if fit_score > best_fit_score:
                        best_fit_score = fit_score
                        fitted_shape = "triangle"
                        best_contour = triangle
            
            # 4. Try to fit an ellipse (if not already a good circle)
            if fitted_shape != "circle" and info.circularity > 0.6 - tolerance:
                ellipse = cv2.fitEllipse(contour) if len(contour) >= 5 else None
                if ellipse is not None:
                    # Create a contour from the ellipse
                    ellipse_contour = []
                    for angle in range(0, 360, 5):  # 5-degree steps
                        center, axes, angle_deg = ellipse
                        a, b = axes[0] / 2, axes[1] / 2
                        
                        # Convert angle to radians and adjust for ellipse rotation
                        angle_rad = angle * np.pi / 180
                        ellipse_angle_rad = angle_deg * np.pi / 180
                        
                        # Parametric equations for ellipse
                        x = center[0] + a * np.cos(angle_rad) * np.cos(ellipse_angle_rad) - b * np.sin(angle_rad) * np.sin(ellipse_angle_rad)
                        y = center[1] + a * np.cos(angle_rad) * np.sin(ellipse_angle_rad) + b * np.sin(angle_rad) * np.cos(ellipse_angle_rad)
                        
                        ellipse_contour.append([[int(x), int(y)]])
                    
                    ellipse_contour = np.array(ellipse_contour, dtype=np.int32)
                    
                    # Create masks for comparison
                    mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    ellipse_mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.drawContours(ellipse_mask, [ellipse_contour], 0, 255, -1)
                    
                    # Calculate IoU
                    intersection = cv2.bitwise_and(mask, ellipse_mask)
                    union = cv2.bitwise_or(mask, ellipse_mask)
                    
                    intersection_area = cv2.countNonZero(intersection)
                    union_area = cv2.countNonZero(union)
                    
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    # Calculate area ratio
                    ellipse_area = np.pi * axes[0] * axes[1] / 4  # pi * a * b
                    area_ratio = info.area / ellipse_area if ellipse_area > 0 else 0
                    
                    # Combine metrics for final fit score
                    fit_score = (iou * 0.7) + (area_ratio * 0.3)
                    
                    if fit_score > best_fit_score:
                        best_fit_score = fit_score
                        fitted_shape = "ellipse"
                        best_contour = ellipse_contour
                        
            # 5. Attempt to fit a polygon (pentagon, hexagon, etc.)
            if fitted_shape is None and 5 <= len(info.approx_poly) <= 8:
                # Use the approximated polygon as the best fit
                fit_score = 0.7  # Default fit score for polygons
                
                # Identify the likely polygon type based on vertex count
                if len(info.approx_poly) == 5:
                    fitted_shape = "pentagon"
                elif len(info.approx_poly) == 6:
                    fitted_shape = "hexagon"
                elif len(info.approx_poly) == 7:
                    fitted_shape = "heptagon"
                else:  # 8 vertices
                    fitted_shape = "octagon"
                    
                # Use the approximated polygon as the best fit
                best_contour = info.approx_poly
                best_fit_score = fit_score
            
            # Create updated ContourInfo if a better shape was found
            if fitted_shape is not None and best_fit_score >= 0.5:
                # Calculate new properties for the fitted contour
                area = cv2.contourArea(best_contour)
                perimeter = cv2.arcLength(best_contour, True)
                approx_poly = cv2.approxPolyDP(best_contour, self.approx_poly_epsilon * perimeter, True)
                
                # Calculate circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(best_contour)
                
                # Calculate center
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                else:
                    center = (x + w // 2, y + h // 2)
                
                # Check convexity
                is_convex = cv2.isContourConvex(best_contour)
                
                # Calculate minimum area rectangle
                min_area_rect = cv2.minAreaRect(best_contour)
                
                # Create updated ContourInfo object
                updated_info = ContourInfo(
                    id=info.id,
                    contour=best_contour,
                    area=area,
                    perimeter=perimeter,
                    approx_poly=approx_poly,
                    bounding_rect=(x, y, w, h),
                    center=center,
                    circularity=circularity,
                    is_convex=is_convex,
                    min_area_rect=min_area_rect,
                    shape_type=fitted_shape,
                    confidence=best_fit_score,
                    hierarchy_level=info.hierarchy_level
                )
                
                processed_contours.append(updated_info)
            else:
                # If no better shape was found, keep the original contour
                processed_contours.append(info)
        
        return processed_contours