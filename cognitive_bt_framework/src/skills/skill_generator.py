from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import time
import pprint

from cognitive_bt_framework.src.llm_interface import LLMInterfaceOpenAI
from cognitive_bt_framework.src.vision.perception_system import ObjectInfo, get_alpha_id

@dataclass
class PointOfInterest:
    """Data class to store labeled point of interest"""
    label: str  # Alphabetical label (a, b, c, etc.)
    position: Tuple[float, float]  # Normalized (x, y) coordinates
    description: str = ""  # Optional description of the point
    pixel_coords: Tuple[Tuple[float, float]]=()
    detection_method: str = "unknown"  # Method used to detect this point
    interaction_type: str = "unknown"  # Type of interaction at this point
    confidence: float = 1.0  # Confidence score for this point
    score: float = 1.0  # Overall score for this point
    stability: float = 0.5  # Stability metric
    accessibility: float = 0.5  # Accessibility metric
    
    
@dataclass
class Skill:
    """Data class to store skill information"""
    name: str
    abstract_action: str
    target_object: str
    primitive_sequence: List[str]
    parameters: Dict[str, Any]
    prerequisites: List[str]
    constraints: List[str]
    image_id: str
    points_of_interest: Dict[str, PointOfInterest]  # Dictionary of labeled points
    explanations: List[str]

class SkillGenerator:
    def __init__(self, llm_interface, skills_dir: str = "stored_skills"):
        """
        Initialize the skill generator with an LLM interface and storage directory
        
        Args:
            llm_interface: LLM interface for generating skills
            skills_dir: Directory to store generated skills
        """
        self.llm = llm_interface
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.image_dir = self.skills_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.skills_cache: Dict[str, Skill] = {}
        self.debug = False
        # Track generated files for data recording
        self.last_generation_files = {
            'skill_paths': [],
            'image_paths': [],
            'image_id': None
        }
        # self._load_stored_skills()

    def _load_stored_skills(self):
        """Load previously stored skills from disk"""
        for skill_file in self.skills_dir.glob("*.json"):
            try:
                with open(skill_file, 'r') as f:
                    skill_data = json.load(f)
                    
                    # Convert points_of_interest dict to PointOfInterest objects
                    if "points_of_interest" in skill_data:
                        points = {}
                        for label, point_data in skill_data["points_of_interest"].items():
                            points[label] = PointOfInterest(
                                label=point_data["label"],
                                position=tuple(point_data["position"]),
                                description=point_data.get("description", "")
                            )
                        skill_data["points_of_interest"] = points
                    else:
                        skill_data["points_of_interest"] = {}
                        
                    skill = Skill(**skill_data)
                    self.skills_cache[skill.name] = skill
                    
            except Exception as e:
                print(f"Error loading skill from {skill_file}: {e}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_image(self, image: np.ndarray, image_id: str) -> str:
        """Save image to disk and return the path"""
        image_path = self.image_dir / f"{image_id}.png"
        cv2.imwrite(str(image_path), image)
        return str(image_path)

    def _load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image from disk"""
        image_path = self.image_dir / f"{image_id}.png"
        if image_path.exists():
            return cv2.imread(str(image_path))
        return None

    def _create_mask_debug_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Create a debug visualization of the object mask overlay on the original image
        
        Args:
            image: Original input image
            mask: Binary mask of the detected object
            
        Returns:
            Debug image with mask overlay
        """
        debug_img = image.copy()
        h, w = image.shape[:2]
        
        # Ensure mask has same dimensions as image
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # Create colored overlay for the mask
        mask_overlay = np.zeros_like(debug_img, dtype=np.uint8)
        mask_overlay[mask] = [0, 255, 0]  # Green overlay for detected object
        
        # Add overlay with transparency
        debug_img = cv2.addWeighted(debug_img, 0.7, mask_overlay, 0.3, 0)
        
        # Add title
        cv2.putText(
            debug_img,
            "Object Mask Debug",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add mask statistics
        mask_area = np.sum(mask)
        total_area = h * w
        coverage_percent = (mask_area / total_area) * 100
        
        cv2.putText(
            debug_img,
            f"Mask Coverage: {coverage_percent:.1f}%",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        cv2.putText(
            debug_img,
            f"Mask Area: {mask_area} pixels",
            (10, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return debug_img



    def _project_normal_to_image(self, normal: Tuple[float, float, float], 
                            centroid_x: int, centroid_y: int,
                            arrow_length: int = 30) -> Tuple[int, int]:
        """
        Project a 3D normal vector onto the 2D image plane.
        
        Args:
            normal: (nx, ny, nz) normal vector
            centroid_x, centroid_y: Origin point of the normal in image coordinates
            arrow_length: Length of the arrow in pixels
            
        Returns:
            (end_x, end_y) coordinates for the arrow
        """
        nx, ny, nz = normal
        
        # Scale factor to adjust for the projection
        scale = arrow_length / max(0.1, abs(nz))
        
        # Project the normal onto the image plane
        # Flip the y-coordinate due to image coordinate system
        dx = int(nx * scale)
        dy = int(-ny * scale)  # Negate to align with image y-axis direction
        
        end_x = centroid_x + dx
        end_y = centroid_y + dy
        
        return (end_x, end_y)

    def _calculate_surface_normals(self, 
                            surface_mask: np.ndarray, 
                            depth_image: np.ndarray,
                            camera_intrinsics: Optional[Dict[str, float]] = None) -> Tuple[float, float, float]:
        """
        Calculate the surface normal vector using the depth image and mask.
        
        Args:
            surface_mask: Binary mask of the surface
            depth_image: Depth image with values in meters
            camera_intrinsics: Dictionary with camera parameters (fx, fy, cx, cy)
                If None, uses default parameters
        
        Returns:
            Tuple of (nx, ny, nz) representing the normal vector
        """
        # Find coordinates of points in the mask
        y_coords, x_coords = np.where(surface_mask)
        
        if len(y_coords) < 10:  # Not enough points for a reliable normal
            return (0, 0, 1)  # Default to pointing outward from the camera
        
        # Default camera intrinsics if not provided (approximate values)
        if camera_intrinsics is None:
            # These are reasonable defaults for a typical RGB-D camera
            fx = 525.0  # focal length x
            fy = 525.0  # focal length y
            cx = depth_image.shape[1] / 2  # principal point x
            cy = depth_image.shape[0] / 2  # principal point y
        else:
            fx = camera_intrinsics.get('fx', 525.0)
            fy = camera_intrinsics.get('fy', 525.0)
            cx = camera_intrinsics.get('cx', depth_image.shape[1] / 2)
            cy = camera_intrinsics.get('cy', depth_image.shape[0] / 2)
        
        # Sample points from the mask (limit to 100 points for efficiency)
        if len(x_coords) > 100:
            indices = np.random.choice(len(x_coords), 100, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # Create 3D points from depth and camera parameters
        points_3d = []
        
        for x, y in zip(x_coords, y_coords):
            # Skip if depth value is invalid (0, NaN, or infinity)
            depth = depth_image[y, x]
            if depth <= 0 or np.isnan(depth) or np.isinf(depth):
                continue
            
            # Convert from pixel coordinates to 3D coordinates
            # Formula: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth
            z = depth
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            
            points_3d.append([x_3d, y_3d, z])
        
        # If we don't have enough valid 3D points, return a default normal
        if len(points_3d) < 10:
            return (0, 0, 1)
        
        # Convert to numpy array
        points_3d = np.array(points_3d)
        
        # Use PCA to find the principal components
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        pca.fit(points_3d)
        
        # The normal is perpendicular to the first two principal components
        # Get the third principal component (the least significant direction)
        normal = pca.components_[2]
        
        # Ensure the normal points toward the camera (negative z direction)
        # In camera coordinates, the camera looks along the positive z axis,
        # so a normal facing the camera would have a negative z component
        if normal[2] > 0:
            normal = -normal
        
        # Normalize the vector
        normal = normal / np.linalg.norm(normal)
        
        return tuple(normal)


     # Helper function to find optimal label position touching the point
    def find_optimal_label_position(self, px, py, alpha_id, all_points, placed_labels, img_width, img_height, object_center=None):
        """
        Find the best corner position for a label that touches the point and minimizes overlaps.
        The label will be positioned so one of its corners touches the point.
        
        Args:
            px, py: Point coordinates
            alpha_id: Label text
            all_points: All point coordinates for overlap checking
            placed_labels: Already placed label rectangles
            img_width, img_height: Image dimensions
            object_center: Optional tuple (cx, cy) representing object center
        """
        
        # Get label dimensions - using COMPLEX font for better readability (Times-like)
        label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
        label_width, label_height = label_bg_size[0] + 8, label_bg_size[1] + 6  # Extra width for letter spacing
        
        # Define 8 corner positions where the label can be placed relative to the point
        # Each position is defined as (corner_name, label_x_calc, label_y_calc, offset_x, offset_y)
        corner_positions = [
            # Top-right corner of label touches point (label extends down and left from point)
            ("top-right", px - label_width, py, -label_width, 0),
            # Top-left corner of label touches point (label extends down and right from point)  
            ("top-left", px, py, 0, 0),
            # Bottom-right corner of label touches point (label extends up and left from point)
            ("bottom-right", px - label_width, py - label_height, -label_width, -label_height),
            # Bottom-left corner of label touches point (label extends up and right from point)
            ("bottom-left", px, py - label_height, 0, -label_height),
            # Center-right edge of label touches point (label extends left from point)
            ("center-right", px - label_width, py - label_height // 2, -label_width, -label_height // 2),
            # Center-left edge of label touches point (label extends right from point)
            ("center-left", px, py - label_height // 2, 0, -label_height // 2),
            # Center-top edge of label touches point (label extends down from point)
            ("center-top", px - label_width // 2, py, -label_width // 2, 0),
            # Center-bottom edge of label touches point (label extends up from point)
            ("center-bottom", px - label_width // 2, py - label_height, -label_width // 2, -label_height),
        ]
        
        # Score each position based on overlap avoidance and image bounds
        best_position = None
        best_score = -1
        
        for corner_name, label_x, label_y, offset_x, offset_y in corner_positions:
            # Check if label is within image bounds with margin
            margin = 5
            if (label_x < margin or label_y < margin or 
                label_x + label_width > img_width - margin or 
                label_y + label_height > img_height - margin):
                continue
            
            # STRICT: Check for point overlaps first and skip this position if any overlap exists
            has_point_overlap = False
            for other_px, other_py in all_points:
                if abs(other_px - px) < 2 and abs(other_py - py) < 2:
                    continue  # Skip the current point itself
                
                # Calculate minimum distance from point to label rectangle with buffer
                closest_x = max(label_x, min(other_px, label_x + label_width))
                closest_y = max(label_y, min(other_py, label_y + label_height))
                distance = np.sqrt((closest_x - other_px)**2 + (closest_y - other_py)**2)
                
                # Require minimum clearance from other points (increased from 15 to 25)
                if distance < 25:
                    has_point_overlap = True
                    break
            
            if has_point_overlap:
                continue  # Skip this position entirely
            
            # STRICT: Check for label overlaps and skip this position if any overlap exists
            has_label_overlap = False
            for placed_x, placed_y, placed_w, placed_h in placed_labels:
                # Add buffer zone around existing labels (increased buffer)
                buffer = 15  # Increased from 8 to 15 for better separation
                if not (label_x + label_width + buffer < placed_x or 
                        placed_x + placed_w + buffer < label_x or
                        label_y + label_height + buffer < placed_y or 
                        placed_y + placed_h + buffer < label_y):
                    has_label_overlap = True
                    break
            
            if has_label_overlap:
                continue  # Skip this position entirely
            
            # Calculate score for valid positions (higher is better)
            score = 100
            
            # Preference bonus based on object center and corner position
            if object_center is not None:
                cx, cy = object_center
                # Prefer positions that place labels away from object center
                label_center_x = label_x + label_width // 2
                label_center_y = label_y + label_height // 2
                
                # Distance from label center to object center
                dist_from_center = np.sqrt((label_center_x - cx)**2 + (label_center_y - cy)**2)
                score += min(dist_from_center / 10, 20)  # Bonus for being away from center
            
            # Slight preference for certain corner positions (top-right, bottom-left, etc.)
            preferred_corners = ["top-right", "bottom-left", "center-right", "center-left"]
            if corner_name in preferred_corners:
                score += 5
            
            if score > best_score:
                best_score = score
                best_position = (label_x, label_y, offset_x, offset_y)
        
        # If no good corner position found, do a systematic grid search for any valid position
        if best_position is None:
            search_radius = 100  # Increased search radius
            step_size = 8  # Smaller steps for more thorough search
            
            for distance in range(30, search_radius, step_size):  # Start at minimum distance
                for angle in range(0, 360, 15):  # Check every 15 degrees
                    rad = np.radians(angle)
                    test_x = int(px + distance * np.cos(rad))
                    test_y = int(py + distance * np.sin(rad))
                    
                    # Check bounds
                    if (test_x < 10 or test_y < 10 or 
                        test_x + label_width > img_width - 10 or 
                        test_y + label_height > img_height - 10):
                        continue
                    
                    # Check for point overlaps
                    valid_position = True
                    for other_px, other_py in all_points:
                        if abs(other_px - px) < 2 and abs(other_py - py) < 2:
                            continue  # Skip current point
                        
                        closest_x = max(test_x, min(other_px, test_x + label_width))
                        closest_y = max(test_y, min(other_py, test_y + label_height))
                        distance_to_point = np.sqrt((closest_x - other_px)**2 + (closest_y - other_py)**2)
                        
                        if distance_to_point < 25:  # Same clearance as before
                            valid_position = False
                            break
                    
                    if not valid_position:
                        continue
                    
                    # Check for label overlaps
                    for placed_x, placed_y, placed_w, placed_h in placed_labels:
                        buffer = 15  # Same buffer as before
                        if not (test_x + label_width + buffer < placed_x or 
                                placed_x + placed_w + buffer < test_x or
                                test_y + label_height + buffer < placed_y or 
                                placed_y + placed_h + buffer < test_y):
                            valid_position = False
                            break
                    
                    if valid_position:
                        offset_x = test_x - px
                        offset_y = test_y - py
                        best_position = (test_x, test_y, offset_x, offset_y)
                        break
                
                if best_position is not None:
                    break
            
            # Final fallback: if still no position found, place in image corner (may overlap)
            if best_position is None:
                if px < img_width // 2:
                    fallback_x = img_width - label_width - 10
                else:
                    fallback_x = 10
                    
                if py < img_height // 2:
                    fallback_y = img_height - label_height - 10
                else:
                    fallback_y = 10
                
                offset_x = fallback_x - px
                offset_y = fallback_y - py
                best_position = (fallback_x, fallback_y, offset_x, offset_y)
        
        return best_position


    def _visualize_points_of_interest(
            self, 
            image: np.ndarray, 
            obj_info: ObjectInfo,
            points: Dict[str, Any] = None,
            top_n_surfaces: int = 5,  # Number of most confident surface masks to visualize
            image_id = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create three separate visualizations:
        1. Image with surface masks and their normals as arrows
        2. Points of interest visualization
        3. Depth visualization with small point markers (saved locally)
        
        Args:
            image: Input image
            obj_info: Object information with surface masks and depth image
            points: Dictionary of points of interest (optional)
            top_n_surfaces: Number of most confident surface masks to visualize
            
        Returns:
            Tuple of (surface_img, points_img)
        """
        h, w = image.shape[:2]
        # Debug: Check for dimension mismatches
        print(f"Visualization image shape: {image.shape}")
        if obj_info.mask is not None:
            print(f"Object mask shape: {obj_info.mask.shape}")
        if hasattr(obj_info, 'depth_image') and obj_info.depth_image is not None:
            print(f"Depth image shape: {obj_info.depth_image.shape}")
        if obj_info.camera_intrinsics:
            print(f"Camera intrinsics: {obj_info.camera_intrinsics}")
        
        # Check if resizing is happening
        if obj_info.mask is not None and obj_info.mask.shape[:2] != (h, w):
            print(f"WARNING: Resizing mask from {obj_info.mask.shape[:2]} to {(h, w)}")
        # Define a consistent color palette for points (BGR format for OpenCV)
        point_colors = [
            (0, 0, 255),      # Red
            (255, 0, 0),      # Blue  
            (0, 255, 0),      # Green
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 255, 0),    # Cyan
            (0, 128, 255),    # Orange
            (128, 0, 128),    # Purple
            (0, 128, 0),      # Dark Green
            (128, 128, 0),    # Olive
            (255, 128, 0),    # Light Blue
            (0, 0, 128),      # Dark Red
        ]

        # Ensure mask has same dimensions as visualization image
        if obj_info.mask is not None and obj_info.mask.shape[:2] != (h, w):
            obj_info.mask = cv2.resize(
                obj_info.mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        
        # Ensure all surface masks have same dimensions as visualization image
        if obj_info.surface_masks is not None:
            for surface_name in list(obj_info.surface_masks.keys()):
                surface_mask = obj_info.surface_masks[surface_name]
                if surface_mask.shape[:2] != (h, w):
                    obj_info.surface_masks[surface_name] = cv2.resize(
                        surface_mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
        
        # Create the first image: Surface masks overlay with normals
        surface_img = image.copy()
        
        # Check if depth image is available in obj_info
        has_depth = hasattr(obj_info, 'depth_image') and obj_info.depth_image is not None
        
        # Draw surface masks with different colors and normals if available
        if obj_info.surface_masks is not None and len(obj_info.surface_masks) > 0:
            # Use different colors for each surface
            color_map = [
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (128, 0, 0),    # Maroon
                (0, 128, 0),    # Green
                (0, 0, 128),    # Navy
                (128, 128, 0),  # Olive
                (128, 0, 128)   # Purple
            ]
            
            # Sort surface masks by size (as a confidence proxy) and take top N
            surface_items = list(obj_info.surface_masks.items())
            surface_areas = [np.sum(mask) for _, mask in surface_items]
            surface_indices = np.argsort(surface_areas)[::-1][:top_n_surfaces]  # Take top N largest
            
            # Create a separate overlay for all surfaces
            surface_overlay = np.zeros_like(surface_img, dtype=np.uint8)
            
            # Add each surface to the overlay with a different color
            for idx, i in enumerate(surface_indices):
                if i >= len(surface_items):
                    continue
                    
                surface_name, surface_mask = surface_items[i]
                color_idx = idx % len(color_map)
                surface_overlay[surface_mask] = color_map[color_idx]
                
                # Find centroid of the surface for label placement and normal calculation
                y_coords, x_coords = np.where(surface_mask)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    
                    # Extract alphabetical ID if it's in the format "surface_abc"
                    if "_" in surface_name:
                        alpha_id = surface_name.split("_")[1]
                    else:
                        alpha_id = get_alpha_id(idx + 1)
                    cv2.putText(
                        surface_img,
                        alpha_id,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (0, 0, 0),  # Black outline
                        4
                    )
                    
                    # Add surface ID at centroid
                    cv2.putText(
                        surface_img,
                        alpha_id,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),  # White text
                        2
                    )
                    
            # Add surface overlay with transparency
            surface_img = cv2.addWeighted(surface_img, 1.0, surface_overlay, 0.3, 0)
            
        
        # Add title to the surface image
        cv2.putText(
            surface_img,
            "Surface Segmentation with Normals",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add legend for normals
        cv2.putText(
            surface_img,
            "Arrows show surface normals",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add information about depth usage
        cv2.putText(
            surface_img,
            f"Using {'depth data' if has_depth else 'PCA estimation'} for normals",
            (10, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Create the second image: Points of interest with object segmentation mask
        points_img = image.copy()
        # Add edge labels with arrows to the points image to indicate hinge locations
        if obj_info.mask is not None:
            # Get object bounding box from mask
            mask_coords = np.where(obj_info.mask > 0)
            if len(mask_coords[0]) > 0:
                min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
                min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
                
                # Calculate edge positions
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                
                # Arrow and label properties
                arrow_length = 30
                arrow_color = (255, 255, 255)  # White arrows
                label_color = (255, 255, 255)  # White text
                outline_color = (0, 0, 0)     # Black outline
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                outline_thickness = 4
                
                # Top edge
                top_arrow_start = (center_x, min_y - arrow_length - 10)
                top_arrow_end = (center_x, min_y - 5)
                cv2.arrowedLine(points_img, top_arrow_start, top_arrow_end, arrow_color, thickness)
                
                # Top label
                top_text = "top"
                text_size = cv2.getTextSize(top_text, font, font_scale, thickness)[0]
                top_text_pos = (center_x - text_size[0] // 2, min_y - arrow_length - 15)
                cv2.putText(points_img, top_text, top_text_pos, font, font_scale, outline_color, outline_thickness)
                cv2.putText(points_img, top_text, top_text_pos, font, font_scale, label_color, thickness)
                
                # Bottom edge
                bottom_arrow_start = (center_x, max_y + arrow_length + 10)
                bottom_arrow_end = (center_x, max_y + 5)
                cv2.arrowedLine(points_img, bottom_arrow_start, bottom_arrow_end, arrow_color, thickness)
                
                # Bottom label
                bottom_text = "bottom"
                text_size = cv2.getTextSize(bottom_text, font, font_scale, thickness)[0]
                bottom_text_pos = (center_x - text_size[0] // 2, max_y + arrow_length + 25)
                cv2.putText(points_img, bottom_text, bottom_text_pos, font, font_scale, outline_color, outline_thickness)
                cv2.putText(points_img, bottom_text, bottom_text_pos, font, font_scale, label_color, thickness)
                
                # Left edge
                left_arrow_start = (min_x - arrow_length - 10, center_y)
                left_arrow_end = (min_x - 5, center_y)
                cv2.arrowedLine(points_img, left_arrow_start, left_arrow_end, arrow_color, thickness)
                
                # Left label
                left_text = "left"
                text_size = cv2.getTextSize(left_text, font, font_scale, thickness)[0]
                left_text_pos = (min_x - arrow_length - 10 - text_size[0], center_y + text_size[1] // 2)
                cv2.putText(points_img, left_text, left_text_pos, font, font_scale, outline_color, outline_thickness)
                cv2.putText(points_img, left_text, left_text_pos, font, font_scale, label_color, thickness)
                
                # Right edge
                right_arrow_start = (max_x + arrow_length + 10, center_y)
                right_arrow_end = (max_x + 5, center_y)
                cv2.arrowedLine(points_img, right_arrow_start, right_arrow_end, arrow_color, thickness)
                
                # Right label
                right_text = "right"
                text_size = cv2.getTextSize(right_text, font, font_scale, thickness)[0]
                right_text_pos = (max_x + arrow_length + 15, center_y + text_size[1] // 2)
                cv2.putText(points_img, right_text, right_text_pos, font, font_scale, outline_color, outline_thickness)
                cv2.putText(points_img, right_text, right_text_pos, font, font_scale, label_color, thickness)
                
                # Add center label for spatial reference
                center_text = "center"
                center_font_scale = 0.5
                center_thickness = 2
                center_outline_thickness = 3
                text_size = cv2.getTextSize(center_text, font, center_font_scale, center_thickness)[0]
                center_text_pos = (center_x - text_size[0] // 2, center_y + text_size[1] // 2)
                cv2.putText(points_img, center_text, center_text_pos, font, center_font_scale, outline_color, center_outline_thickness)
                cv2.putText(points_img, center_text, center_text_pos, font, center_font_scale, label_color, center_thickness)
        # Add object segmentation mask outline if available
        if obj_info.mask is not None:
            # Add object mask outline for better visibility
            mask_contours, _ = cv2.findContours(
                obj_info.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(points_img, mask_contours, -1, (0, 255, 255), 1)  # Yellow outline, slightly thicker
        
        cv2.putText(points_img, "Points of Interest + Object Segmentation", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        if points is not None and len(points) > 0:
            # Calculate object center for label positioning
            object_center = None
            if obj_info.bbox is not None:
                # Use bounding box center
                bbox_x, bbox_y, bbox_w, bbox_h = obj_info.bbox
                object_center = (bbox_x + bbox_w // 2, bbox_y + bbox_h // 2)
            elif obj_info.mask is not None:
                # Use mask centroid
                y_coords, x_coords = np.where(obj_info.mask)
                if len(y_coords) > 0:
                    object_center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            else:
                # Use image center as fallback
                object_center = (w // 2, h // 2)
            
            # Collect all point coordinates for overlap checking
            all_point_coords = []
            placed_labels = []  # Track placed label positions
            
            # Sort points by some criteria to prioritize important ones
            # This helps ensure important points get good label positions first
            if 'pixel_coords' in points and points['pixel_coords']:
                pixel_coords = points['pixel_coords']
                scores = points.get('scores', [1.0] * len(pixel_coords))
                ids = points.get('ids', [f"p{i}" for i in range(len(pixel_coords))])
                
                # Sort by score (highest first) to prioritize better detection results
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                
                # Collect all point coordinates
                all_point_coords = [(pixel_coords[i][0], pixel_coords[i][1]) for i in sorted_indices]
                
                # STEP 1: Draw all points first (underneath labels)
                point_data = []  # Store data for later label drawing
                for idx, i in enumerate(sorted_indices):
                    px, py = pixel_coords[i]
                    _ = scores[i]
                    point_id = ids[i]
                    
                    # Get color for this point (cycle through available colors)
                    color = point_colors[idx % len(point_colors)]
                    
                    # Generate alphabetical ID if not already alphabetical
                    if not point_id.isalpha():
                        alpha_id = get_alpha_id(idx + 1)
                    else:
                        alpha_id = point_id
                    
                    # Draw small colored circle as the interaction point with black and white outlines
                    cv2.circle(points_img, (px, py), radius=8, color=(0, 0, 0), thickness=-1)  # Black outer outline
                    cv2.circle(points_img, (px, py), radius=6, color=(255, 255, 255), thickness=-1)  # White outline
                    cv2.circle(points_img, (px, py), radius=4, color=color, thickness=-1)  # Colored center
                    
                    # Store data for label drawing (but don't draw labels yet)
                    point_data.append({
                        'px': px, 'py': py, 'color': color, 'alpha_id': alpha_id, 'idx': idx
                    })
                
                # STEP 2: Now draw all labels on top of points (labels have priority)
                for data in point_data:
                    px, py = data['px'], data['py']
                    color = data['color']
                    alpha_id = data['alpha_id']
                    
                    # Find optimal label position using improved algorithm
                    label_x, label_y, offset_x, offset_y = self.find_optimal_label_position(
                        px, py, alpha_id, all_point_coords, placed_labels, w, h, object_center
                    )
                    
                    # Get label dimensions for tracking - using COMPLEX font for better readability (Times-like)
                    label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
                    label_width, label_height = label_bg_size[0] + 8, label_bg_size[1] + 6  # Extra width for letter spacing
                    
                    # Draw connecting line if label is far from point
                    if abs(offset_x) > 20 or abs(offset_y) > 20:
                        # Draw a thin line from center of small circle to label
                        line_end_x = label_x + label_width // 2
                        line_end_y = label_y + label_height // 2
                        
                        cv2.line(points_img, (px, py), 
                                (line_end_x, line_end_y), color, 2)
                    
                    # Draw label background rectangle with matching color and slight transparency effect
                    label_rect = (label_x - 3, label_y - label_bg_size[1] - 3,
                                label_x + label_bg_size[0] + 3, label_y + 3)
                    
                    # Create a small overlay for transparency effect
                    overlay = points_img.copy()
                    cv2.rectangle(overlay, (label_rect[0], label_rect[1]), 
                                (label_rect[2], label_rect[3]), color, -1)
                    cv2.addWeighted(overlay, 0.8, points_img, 0.2, 0, points_img)
                    
                    # Draw label border
                    cv2.rectangle(points_img, (label_rect[0], label_rect[1]), 
                                (label_rect[2], label_rect[3]), (0, 0, 0), 2)
                    
                    # Draw label text with black outline for better visibility - using COMPLEX font for Times-like appearance
                    # Black outline
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
                    # White text
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Track this label's position for future overlap checking
                    placed_labels.append((label_rect[0], label_rect[1], 
                                        label_width + 6, label_height + 6))
                    
            else:
                # Handle dictionary of named points (original format)
                point_items = list(points.items())
                
                # Collect all point coordinates
                for label, point in point_items:
                    if hasattr(point, 'position'):
                        if max(point.position) <= 1.0:
                            px, py = int(point.position[0] * w), int(point.position[1] * h)
                        else:
                            px, py = int(point.position[0]), int(point.position[1])
                    else:
                        px, py = int(point[0]), int(point[1])
                    all_point_coords.append((px, py))
                
                # STEP 1: Draw all points first (underneath labels)
                dict_point_data = []  # Store data for later label drawing
                for i, (label, point) in enumerate(point_items):
                    if hasattr(point, 'position'):
                        # Handle PointOfInterest objects
                        if max(point.position) <= 1.0:
                            px, py = int(point.position[0] * w), int(point.position[1] * h)
                        else:
                            px, py = int(point.position[0]), int(point.position[1])
                    else:
                        # Handle direct (x,y) tuples
                        px, py = int(point[0]), int(point[1])
                    
                    # Get color for this point (cycle through available colors)
                    color = point_colors[i % len(point_colors)]
                    
                    # Generate alphabetical ID if the label is not already alphabetical
                    if not label.isalpha():
                        alpha_id = get_alpha_id(i + 1)
                    else:
                        alpha_id = label
                    
                    # Draw small colored circle as the interaction point with black and white outlines
                    cv2.circle(points_img, (px, py), radius=8, color=(0, 0, 0), thickness=-1)  # Black outer outline
                    cv2.circle(points_img, (px, py), radius=6, color=(255, 255, 255), thickness=-1)  # White outline
                    cv2.circle(points_img, (px, py), radius=4, color=color, thickness=-1)  # Colored center
                    
                    # Store data for label drawing (but don't draw labels yet)
                    dict_point_data.append({
                        'px': px, 'py': py, 'color': color, 'alpha_id': alpha_id, 'label': label
                    })
                
                # STEP 2: Now draw all labels on top of points (labels have priority)
                for data in dict_point_data:
                    px, py = data['px'], data['py']
                    color = data['color']
                    alpha_id = data['alpha_id']
                    
                    # Find optimal label position using improved algorithm
                    label_x, label_y, offset_x, offset_y = self.find_optimal_label_position(
                        px, py, alpha_id, all_point_coords, placed_labels, w, h, object_center
                    )
                    
                    # Get label dimensions for tracking - using COMPLEX font for better readability (Times-like)
                    label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_COMPLEX, 0.7, 2)[0]
                    label_width, label_height = label_bg_size[0] + 8, label_bg_size[1] + 6  # Extra width for letter spacing
                    
                    # Draw connecting line if label is far from point
                    if abs(offset_x) > 20 or abs(offset_y) > 20:
                        # Draw a thin line from center of small circle to label
                        line_end_x = label_x + label_width // 2
                        line_end_y = label_y + label_height // 2
                        
                        cv2.line(points_img, (px, py), 
                                (line_end_x, line_end_y), color, 2)
                    
                    # Draw label background rectangle with matching color and slight transparency effect
                    label_rect = (label_x - 3, label_y - label_bg_size[1] - 3,
                                label_x + label_bg_size[0] + 3, label_y + 3)
                    
                    # Create a small overlay for transparency effect
                    overlay = points_img.copy()
                    cv2.rectangle(overlay, (label_rect[0], label_rect[1]), 
                                (label_rect[2], label_rect[3]), color, -1)
                    cv2.addWeighted(overlay, 0.8, points_img, 0.2, 0, points_img)
                    
                    # Draw label border
                    cv2.rectangle(points_img, (label_rect[0], label_rect[1]), 
                                (label_rect[2], label_rect[3]), (0, 0, 0), 2)
                    
                    # Draw label text with black outline for better visibility - using COMPLEX font for Times-like appearance
                    # Black outline
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
                    # White text
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Track this label's position for future overlap checking
                    placed_labels.append((label_rect[0], label_rect[1], 
                                        label_width + 6, label_height + 6))

        # Add title to the points image
        
        
        # Create the third image: Depth visualization with RealSense-quality processing
        depth_img = None
        if hasattr(obj_info, 'depth_image') and obj_info.depth_image is not None:
            # Get the processed depth image
            depth = obj_info.depth_image.copy()
            
            # Handle potential NaN or inf values
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply the same visualization approach as RealSense viewer
            valid_mask = depth > 0
            if np.any(valid_mask):
                # Get depth statistics for proper normalization
                valid_depths = depth[valid_mask]
                min_val = np.min(valid_depths)
                max_val = np.max(valid_depths)
                
                print(f"Depth range: {min_val:.3f} to {max_val:.3f} meters")
                print(f"Valid pixels: {np.sum(valid_mask)}/{depth.size} ({100*np.sum(valid_mask)/depth.size:.1f}%)")
                
                if max_val > min_val:
                    # Method 1: Direct normalization (like RealSense viewer)
                    # This creates a clean, consistent depth visualization
                    depth_normalized = np.zeros_like(depth, dtype=np.float32)
                    depth_normalized[valid_mask] = (valid_depths - min_val) / (max_val - min_val)
                    
                    # Convert to 8-bit for colormap
                    depth_8bit = (depth_normalized * 255).astype(np.uint8)
                    
                    # Apply colormap (use JET to match RealSense default, or PLASMA for better perception)
                    colorized_depth = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                    
                    # Alternative: Use the same colorization as RealSense visualization
                    # depth_vis = np.clip(depth.astype(np.float32) * 50, 0, 255).astype(np.uint8)
                    # colorized_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    
                    depth_img = colorized_depth.copy()
                    
                    # Add depth scale information
                    if hasattr(obj_info, 'depth_scale') and obj_info.depth_scale:
                        # Convert to same units as RealSense (usually millimeters for display)
                        _ = depth * 1000  # Convert meters to mm
                        min_mm = min_val * 1000
                        max_mm = max_val * 1000
                        
                        cv2.putText(
                            depth_img,
                            f"Depth: {min_mm:.0f}-{max_mm:.0f}mm",
                            (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                    else:
                        cv2.putText(
                            depth_img,
                            f"Depth: {min_val:.3f}-{max_val:.3f}m",
                            (10, 60),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                    
                    # Add title
                    cv2.putText(
                        depth_img,
                        "Processed Depth Visualization",
                        (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    # Draw points of interest with same color coding as points image
                    if points is not None and len(points) > 0:
                        point_colors = [
                            (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255),
                            (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 128)
                        ]
                        
                        if 'pixel_coords' in points and points['pixel_coords']:
                            pixel_coords = points['pixel_coords']
                            for i, (px, py) in enumerate(pixel_coords):
                                color = point_colors[i % len(point_colors)]
                                cv2.circle(depth_img, (px, py), 5, color, 2)
                                cv2.circle(depth_img, (px, py), 3, (255, 255, 255), -1)
                        else:
                            for i, (label, point) in enumerate(points.items()):
                                if hasattr(point, 'position'):
                                    if max(point.position) <= 1.0:
                                        px, py = int(point.position[0] * w), int(point.position[1] * h)
                                    else:
                                        px, py = int(point.position[0]), int(point.position[1])
                                else:
                                    px, py = int(point[0]), int(point[1])
                                
                                color = point_colors[i % len(point_colors)]
                                cv2.circle(depth_img, (px, py), 5, color, 2)
                                cv2.circle(depth_img, (px, py), 3, (255, 255, 255), -1)

        # If depth image is not available, create a placeholder
        if depth_img is None:
            depth_img = np.zeros_like(image)
            cv2.putText(
                depth_img,
                "Depth data not available",
                (int(w/2) - 100, int(h/2)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Save the depth image
        depth_img_path = self._save_image(depth_img, f"depth_image_{image_id}")
        self.last_generation_files['image_paths'].append(depth_img_path)
        
        
        
        return surface_img, points_img
    
    def _categorize_points_from_point_objects(self,
                                        image: np.ndarray,
                                        point_objects: Dict[str, Any],
                                        object_info: Optional[ObjectInfo] = None) -> List[str]:
        """
        Categorize points from PointOfInterest objects based on their position within the object's bounding box.
        
        Args:
            image: Input image of the object
            point_objects: Dictionary of label->PointOfInterest objects
            object_info: ObjectInfo containing additional object data including bounding box
            
        Returns:
            List of point descriptions with spatial categorization
        """
        h, w = image.shape[:2]
        
        # Get object bounding box from object_info
        if object_info is not None and object_info.bbox is not None:
            # Note: ObjectInfo.bbox is in [x, y, w, h] format, convert to [x_min, y_min, x_max, y_max]
            bbox_x, bbox_y, bbox_w, bbox_h = object_info.bbox
            x_min, y_min = bbox_x, bbox_y
            x_max, y_max = bbox_x + bbox_w, bbox_y + bbox_h
        else:
            # If no bounding box provided, calculate from point pixel coordinates
            pixel_coords = []
            for point in point_objects.values():
                # Always use pixel_coords if available
                if hasattr(point, 'pixel_coords') and point.pixel_coords:
                    pixel_coords.append(point.pixel_coords)
                else:
                    # Convert normalized position to pixel coordinates as fallback
                    norm_x, norm_y = point.position
                    pixel_x = int(norm_x * w)
                    pixel_y = int(norm_y * h)
                    pixel_coords.append((pixel_x, pixel_y))
            
            if pixel_coords:
                # Calculate bounding box from pixel coordinates
                x_points = [p[0] for p in pixel_coords]
                y_points = [p[1] for p in pixel_coords]
                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)
                
                # Add padding
                padding_x = int(0.05 * (x_max - x_min))  # 5% padding
                padding_y = int(0.05 * (y_max - y_min))
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
            else:
                # Fallback to full image if no points
                x_min, y_min = 0, 0
                x_max, y_max = w, h
        
        # Define regions of the bounding box in pixel coordinates
        width = x_max - x_min
        height = y_max - y_min
        
        # Horizontal divisions
        left_bound = x_min + width * 0.25
        right_bound = x_max - width * 0.25
        
        # Vertical divisions
        top_bound = y_min + height * 0.25
        bottom_bound = y_max - height * 0.25
        
        # Edge margin (for detecting points close to edges)
        edge_margin = min(width, height) * 0.1
        
        point_descriptions = []
        for label, point in point_objects.items():
            # Always get pixel coordinates for positioning
            if hasattr(point, 'pixel_coords') and point.pixel_coords:
                pixel_x, pixel_y = point.pixel_coords
            else:
                # Convert normalized position to pixel coordinates as fallback
                norm_x, norm_y = point.position
                pixel_x = int(norm_x * w)
                pixel_y = int(norm_y * h)
            
            # Get normalized coordinates for description
            norm_x, norm_y = point.position
            
            # Determine horizontal position using pixel coordinates
            if pixel_x < left_bound:
                h_pos = "left"
            elif pixel_x > right_bound:
                h_pos = "right"
            else:
                h_pos = "center"
            
            # Determine vertical position using pixel coordinates
            if pixel_y < top_bound:
                v_pos = "top"
            elif pixel_y > bottom_bound:
                v_pos = "bottom"
            else:
                v_pos = "middle"
            
            # Check if point is near an edge
            is_edge = (pixel_x < x_min + edge_margin or 
                    pixel_x > x_max - edge_margin or 
                    pixel_y < y_min + edge_margin or 
                    pixel_y > y_max - edge_margin)
            
            position_desc = f"{v_pos} {h_pos}"
            if is_edge:
                position_desc += " (edge)"
            
            # Use description if available, otherwise generate one
            if hasattr(point, 'description') and point.description:
                desc = f"Point {label}: {point.description}, position: {position_desc}"
            else:
                desc = f"Point {label}, position: {position_desc}"
            
            point_descriptions.append(desc)
        
        return point_descriptions

    def _format_executed_skills_context(self, executed_skills: List) -> str:
        """Format executed skills list into context string for the LLM"""
        if not executed_skills:
            return "No skills have been executed yet. This is the first skill in the sequence."
        
        context_lines = [
            f"Previously executed skills in this sequence ({len(executed_skills)} skills):"
        ]
        
        for i, skill in enumerate(executed_skills, 1):
            skill_command = skill.get('skill_command', f"{skill.get('skill_name', 'unknown')} {skill.get('parameters', '')}")
            target_obj = skill.get('target_object', 'unknown')
            context_lines.append(f"  {i}. {skill_command} (target: {target_obj})")
        
        context_lines.append("")
        context_lines.append("Consider this execution history when planning the current skill:")
        context_lines.append("- Avoid interfering with objects that have already been manipulated")
        context_lines.append("- Build upon the current state established by previous skills")
        context_lines.append("- Ensure compatibility with the sequence of actions performed")
        
        return "\n".join(context_lines)

    def generate_skill(self, 
                    image: np.ndarray,
                    points_of_interest: Dict[str, Any],
                    abstract_action: str,
                    target_object: str,
                    object_info: Optional[ObjectInfo] = None,
                    executed_skills: List = None,
                    task_context: str = None,
                    ) -> Optional[Skill]:
        """
        Generate a new skill using the LLM interface based on image input with labeled points and surfaces
        
        Args:
            image: Input image of the object
            points_of_interest: Dictionary of points information (from detect_regions_of_interest)
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
            object_info: ObjectInfo containing additional object data including surface_masks
            executed_skills: List of previously executed skills for context
        
        Returns:
            Generated skill if successful, None otherwise
        """
        
        # Generate unique ID for this image
        timestamp = int(time.time())
        
        # Create a hash from the points data
        if 'pixel_coords' in points_of_interest:
            # New format from detect_regions_of_interest
            point_hash = hash(str(points_of_interest['pixel_coords'])) & 0xFFFFFF
        else:
            # Original format with PointOfInterest objects
            point_hash = hash(str([(p.label, p.position) for p in points_of_interest.values()])) & 0xFFFFFF
        
        image_id = f"{abstract_action}_{target_object}_{timestamp}_{point_hash:06x}"
        
        # Reset tracking for this generation
        self.last_generation_files = {
            'skill_paths': [],
            'image_paths': [],
            'image_id': image_id
        }
        
        if object_info is None:
            if self.debug:
                self.get_logger().warn("No object_info provided for skill generation, visualizations will be limited")
            # Create dummy object_info for visualization
            object_info = ObjectInfo(
                id=0,
                name=target_object,
                mask=None,
                bbox=None,
                surface_masks=None
            )
            # Create visualization of just points of interest (legacy mode)
            surface_img, points_img = self._visualize_points_of_interest(image, object_info, points_of_interest, image_id=image_id)
        else:
            # Create the separate visualizations for surfaces with normals and points
            surface_img, points_img = self._visualize_points_of_interest(image, object_info, points_of_interest,image_id=image_id)
        # Save each visualization separately
        surface_img_path = self._save_image(surface_img, f"surface_{image_id}")
        points_img_path = self._save_image(points_img, f"points_{image_id}")
        self.last_generation_files['image_paths'].extend([surface_img_path, points_img_path])
        
        # Save object mask as debug image (not provided to LLM)
        if object_info is not None and object_info.mask is not None:
            mask_debug_img = self._create_mask_debug_image(image, object_info.mask)
            mask_img_path = self._save_image(mask_debug_img, f"mask_{image_id}")
            self.last_generation_files['image_paths'].append(mask_img_path)
            if self.debug:
                print(f"Saved mask debug image: mask_{image_id}.png")
        
        if self.debug:
            self.get_logger().info(f"Saved surface image to {surface_img_path}")
            self.get_logger().info(f"Saved points image to {points_img_path}")
        
        # Also save the original image for reference
        original_img_path = self._save_image(image, image_id)
        self.last_generation_files['image_paths'].append(original_img_path)
        
        # Create descriptions for prompt
        point_descriptions = []
        surface_descriptions = []
        object_segmentation_info = ""

        # Generate object segmentation description if mask is available
        if object_info is not None and object_info.mask is not None:
            mask_area = np.sum(object_info.mask)
            total_area = image.shape[0] * image.shape[1]
            coverage_percent = (mask_area / total_area) * 100
            
            # Find bounding box of the mask
            mask_contours, _ = cv2.findContours(
                object_info.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if mask_contours:
                # Get bounding rect of the largest contour
                largest_contour = max(mask_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                object_segmentation_info = f"""
--- TARGET OBJECT SEGMENTATION ---
The target object "{target_object}" has been precisely segmented and is outlined with a YELLOW BOUNDARY in the points image.

Object Segmentation Details:
• Coverage: {coverage_percent:.1f}% of the image
• Bounding box: x={x}, y={y}, width={w}, height={h}
• Mask area: {mask_area} pixels

EDGE LABELS FOR HINGE LOCATION:
The points image includes WHITE ARROWS and LABELS pointing to each edge of the target object:
• "top" - points to the top edge of the object
• "bottom" - points to the bottom edge of the object  
• "left" - points to the left edge of the object
• "right" - points to the right edge of the object

These labels indicate where hinges would be located for rotational motion. When selecting a hinge_location, 
you MUST identify these labels in the image and reference them in your justification.

IMPORTANT: Only select interaction points that fall WITHIN the yellow outlined region. 
Any points outside this segmented area are NOT part of the target object and should be rejected.
The segmentation provides precise boundaries of the target object to avoid confusion with adjacent objects.
"""
            else:
                object_segmentation_info = f"""
--- TARGET OBJECT SEGMENTATION ---
The target object "{target_object}" has been segmented covering {coverage_percent:.1f}% of the image.

EDGE LABELS FOR HINGE LOCATION:
The points image includes WHITE ARROWS and LABELS pointing to each edge of the target object:
• "top" - points to the top edge of the object
• "bottom" - points to the bottom edge of the object  
• "left" - points to the left edge of the object
• "right" - points to the right edge of the object

These labels indicate where hinges would be located for rotational motion. When selecting a hinge_location, 
you MUST identify these labels in the image and reference them in your justification.

Only select interaction points within the segmented region.
"""

        # Handle points
        # Original format with PointOfInterest objects
        # Use the new function to categorize points
        point_descriptions = self._categorize_points_from_point_objects(
            image=image,
            point_objects=points_of_interest,
            object_info=object_info
        )
        # Handle surfaces with normals if available
        if object_info is not None and object_info.surface_masks is not None:
            # Sort surface masks by size (as a confidence proxy) and take top N
            surface_items = list(object_info.surface_masks.items())
            surface_areas = [np.sum(mask) for _, mask in surface_items]
            top_n_surfaces = 5  # Number of most confident surface masks to describe
            surface_indices = np.argsort(surface_areas)[::-1][:top_n_surfaces]
            
            for idx, i in enumerate(surface_indices):
                if i >= len(surface_items):
                    continue
                    
                surface_name, surface_mask = surface_items[i]
                
                # Extract alphabetical ID
                if "_" in surface_name:
                    alpha_id = surface_name.split("_")[1]
                else:
                    alpha_id = get_alpha_id(idx + 1)
                
                # Calculate surface area as percentage of total image
                _ = (np.sum(surface_mask) / (image.shape[0] * image.shape[1])) * 100
                
                # Find centroid of the surface
                y_coords, x_coords = np.where(surface_mask)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    
                    # Calculate normal for this surface
                    _ = self._calculate_surface_normals(surface_mask, object_info.depth_image)
                    
                    # Normalize coordinates for description
                    _, _ = centroid_x / image.shape[1], centroid_y / image.shape[0]
                    
                    desc = f"Surface {alpha_id}"
                    surface_descriptions.append(desc)
        
        # Create prompt for skill definition
        combined_prompt = [
            {
                "role": "system",
                "content": f"""You are a robotic task planner with semantic-geometric understanding for automated assistance.

                CONTEXT: This is for robotic task automation to assist with everyday objects and activities.
                {f"FULL TASK CONTEXT: {task_context}" if task_context else ""}

                Your task is to generate a complete, structured skill definition for performing **{abstract_action}** on a **{target_object}**, using only the provided visual inputs.
                {f"This skill is part of the larger task: '{task_context}'" if task_context else ""}
                
                --- EXECUTED SKILLS CONTEXT ---
                {self._format_executed_skills_context(executed_skills or [])}

                --- VISUAL INPUT UNDERSTANDING ---
                You will receive TWO images:
                
                IMAGE 1 - SURFACE SEGMENTS:
                • Shows labeled surfaces (aaa, aab, aac...)
                • Each surface has a NORMAL VECTOR shown as an arrow
                • Normal vectors point OUTWARD from the surface
                • Use these for determining push/pull directions
                
                IMAGE 2 - INTERACTION POINTS:
                • Shows COLORED CIRCLES marking exact interaction locations
                • Each circle has a label (aaa, aab, aac...)
                • Labels are connected to circles by colored lines
                • CRITICAL: The colored CIRCLES are the actual points - NOT where lines intersect objects OR the label locations
                • Lines are ONLY for matching labels to circles
                • Labels are ONLY for identifying circles 

                --- POINT SELECTION METHODOLOGY ---
                
                STEP 1: Identify the Target Object
                • Look for the object matching: {target_object}
                • Use visual cues: shape, color, texture, hardware
                • The target object is outlined with a YELLOW BOUNDARY in the points image
                • CRITICAL: Only consider points within the yellow outlined region
                • Distinguish from similar adjacent objects using the segmentation boundaries
                
                STEP 2: Validate Points
                For each potential point:
                • Follow the colored line from label to find the COLORED CIRCLE
                • Check if the CIRCLE (not line) is on the target object
                • Verify the circle is on a functional component (handle, button, etc.)
                • Reject circles on adjacent objects or decorative features
                
                STEP 3: Select Optimal Points
                • Choose circles that PHYSICALLY OVERLAP functional features
                • For handles: ONLY select circles where the circle COVERS part of the handle
                • For buttons: ONLY select circles that OVERLAP the pushable surface
                • REJECT circles that are merely "near", "aligned with", or "offset from" features
                • The colored circle must INTERSECT with the actual functional hardware
                • For hinged objects: identify which edge the hinge is on (top/bottom/left/right)
                • CONSIDER ROBOT REACH: Select points that are within the robot's comfortable working range
                • AVOID points that would require overextension or create collision risks with the robot's body
                • EDGE GRASPING: If no accessible functional features are available, edges are usually good grasp points - choose points along edges that maximize the available grip for the robot gripper

                {point_descriptions}
                
                {surface_descriptions}
                
                {object_segmentation_info}
                
                --- ACTION PRIMITIVES ---
                Available primitives and their parameters:
                
                • move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)
                - point_label: Use labels from IMAGE 2 colored circles only
                - is_top_down_grasp: True for vertical movements (lifting, pressing down)
                - is_side_grasp: True for horizontal movements (drawers, sliding)
                
                • push('surface_label', 'force_direction', is_button, has_pivot, 'hinge_location')
                • pull('surface_label', 'force_direction', is_button, has_pivot, 'hinge_location')
                - surface_label: Use labels from IMAGE 1 only
                - force_direction: 'perpendicular' (along normal) or 'parallel' (along surface)
                - is_button: True for short activation pushes, False for sustained contact
                - has_pivot: True for rotational motion, False for linear motion
                - hinge_location: 'top', 'bottom', 'left', or 'right' indicating which edge the hinge is on, '' if no hinge
                  CRITICAL: You MUST identify the white edge labels and center label in the image and reference them when selecting hinge_location
                
                • close_gripper() / open_gripper() - Control gripper state
                • retract_gripper() - Move gripper away
                • twist('direction') - 'clockwise' or 'counterclockwise'
                
                --- TWIST ROTATION MECHANICS ---
                CRITICAL: Twist rotation is ALWAYS about the gripper's Z-axis (finger-to-finger axis).
                
                GRASP ORIENTATION DETERMINES GLOBAL ROTATION AXIS:
                
                TOP-DOWN GRASP (is_top_down_grasp=True):
                • Gripper Z-axis aligns with GLOBAL Z-axis (vertical)
                • twist('clockwise') → Rotates about GLOBAL Z-axis (vertical rotation)
               
                SIDE GRASP (is_side_grasp=True):
                • Gripper Z-axis aligns with GLOBAL X-axis (horizontal, left-right)
                • twist('clockwise') → Rotates about GLOBAL X-axis (roll rotation)
            
                GRASP PLANNING FOR TWIST OPERATIONS:
                • IDENTIFY the intended rotation axis of the object (what axis should it spin around?)
                • CHOOSE the grasp type that aligns the gripper Z-axis with that rotation axis:
                  - Object rotates vertically → use TOP-DOWN grasp
                  - Object rotates horizontally  → use SIDE grasp
                • ENSURE the twist direction matches the desired object motion
                • CONSIDER the object's physical constraints and threading direction

                --- OPENING MECHANISMS ---
                
                BUTTON-BASED OPENING:
                • Some objects may open via button press rather than traditional handles
                • For objects like microwaves, coffee makers, printers, electronic devices
                • Look for buttons, touch panels, or pressure-sensitive areas
                • Use push() with is_button=True for button activation
                • Common button locations: front panel, side panel, top panel
                • After button press, object may automatically open or unlock for manual opening
            

                --- CRITICAL RULES (IN ORDER OF IMPORTANCE) ---
                
                ⚠️ CRITICAL MISTAKES TO AVOID:
                ✗ Circles "near" or "aligned with" interaction points (must PHYSICALLY OVERLAP)
                ✗ Using line intersections or label locations instead of colored circles
                ✗ Selecting points on adjacent objects instead of target object

                1. POINT SELECTION REQUIREMENTS:
                ✓ Use ONLY colored circles with black/white outlines as interaction points
                ✓ Circle must DIRECTLY COVER part of the handle/button/feature
                ✓ If no circle overlaps the functional feature, reject all circles for that feature
                ✓ Follow colored lines to match labels to circles (same color = same point)
                
                2. TARGET OBJECT FOCUS:
                ✓ Select points ONLY on the specified target object
                ✗ NEVER select points on adjacent objects
                ✗ NEVER select points separated by vizual markers of the end of an object like cracks gaps or color changes
                
                3. FUNCTIONAL FEATURES:
                ✓ Choose handles, buttons, knobs with clear function
                ✗ AVOID decorative circles or mounting hardware

                4. ROBOT CONSTRAINTS:
                ✓ Select points within comfortable reach (avoid overextension/collision)
                ✓ Prefer natural, ergonomic robot poses
                ✓ Top-down grasp for vertical movements, side grasp for horizontal movements

                ⚠️⚠️⚠️ CRITICAL GRASP TYPE SELECTION ⚠️⚠️⚠️
                ABSOLUTELY ESSENTIAL - FOLLOW THIS GUIDANCE STRICTLY:
                
                🔴 SIDE GRASP (is_side_grasp=True) MANDATORY when objects are:
                  ✓ HIGH UP or ELEVATED (above robot's comfortable reach height)
                  ✓ FAR AWAY from robot's base (at extended reach distances)  
                  ✓ Positioned at AWKWARD ANGLES for top-down access
                  ✓ Require horizontal approach for accessibility
                
                🔵 TOP-DOWN GRASP (is_top_down_grasp=True) MANDATORY when objects are:
                  ✓ At COMFORTABLE robot working height
                  ✓ CLOSE to robot's base position
                  ✓ EASILY ACCESSIBLE from above
                  ✓ Vertical approach is natural and ergonomic
                
                ❌ WRONG GRASP SELECTION WILL CAUSE TASK FAILURE
                ❌ Always consider object height and distance from robot base
                ❌ Choose grasp type based on robot reach limitations and ergonomics

                5. LID/CAP REMOVAL COMPLETION:
                ✓ When opening containers with lids or caps, the skill must fully remove the lid/cap
                ✓ Use retract_gripper() to move the lid/cap completely away from the container opening
                ✓ Ensure the container opening is fully accessible after lid/cap removal
                ✓ For twist-off lids/caps, combine twist() and retract_gripper() actions
                ✓ The removal should be complete, not just partial loosening


                !!! CRITICAL POINT AND PIVOT POINT LOCATION RULE !!!
                ====================================

                COLORED CIRCLES WITH BLACK AND WHITE OUTLINES = EXACT INTERACTION POINTS
                ✓ Look for colored circles with black outer ring and white inner ring - these are the ONLY interaction points
                ✓ ONLY the pixel location of the COLORED CIRCLE matters
                ✓ The CIRCLE itself is the EXACT interaction point
                ✓ IGNORE labels, connecting lines, and edge arrows when determining spatial location

                COLOR-MATCHING SYSTEM FOR POINT IDENTIFICATION:
                🎨 CRITICAL: Points, labels, and connecting lines share the SAME COLOR to show association
                ✓ RED circle → connected to RED label text → via RED connecting line (if present)
                ✓ BLUE circle → connected to BLUE label text → via BLUE connecting line (if present)
                ✓ GREEN circle → connected to GREEN label text → via GREEN connecting line (if present)
                ✓ Use color matching to identify which label belongs to which interaction point

                LABELS, LINES, AND ARROWS = IDENTIFICATION ONLY, NOT INTERACTION POINTS
                ✗ Labels (text like "jxh", "mbb") are ONLY for naming points - use COLOR to match them to circles
                ✗ Lines connecting labels to circles are ONLY for visual association - same COLOR as both
                ✗ Edge arrows (labeled "top", "bottom", "left", "right") are WHITE and for hinge identification only
                ✗ The spatial position of labels has NO MEANING for task planning
                ✗ The path or direction of connecting lines has NO MEANING for task planning
                ✗ Edge arrows are NOT interaction points - they just indicate hinge locations

                STEP-BY-STEP POINT PROCESSING:
                1. For each point label in the image:
                a. Identify the COLOR of the label text (e.g., red, blue, green)
                b. Find the COLORED CIRCLE that matches that exact same color
                c. Follow the connecting line (if present) to verify - it should be the same color
                d. Record the EXACT PIXEL COORDINATES of that matching colored circle
                e. Determine if that specific pixel location is on the target object
                f. ONLY consider the circle's location, NEVER the label's or line's location

                EXAMPLES:
                - RED label "abc" → RED line → RED CIRCLE: Use circle's pixel location as interaction point
                - BLUE CIRCLE on handle + BLUE label "xyz": Circle location is the interaction point  
                - White arrows ("left", "right", "top", "bottom"): Hinge indicators ONLY, NOT interaction points

                VALIDATION TEST:
                For each point you select, explicitly state:
                "Point [LABEL] is located at the [COLOR] CIRCLE which is positioned at [OBJECT FEATURE]"

                --- HINGE LOCATION IDENTIFICATION FOR HINGED OBJECTS ---

                CRITICAL HINGE IDENTIFICATION RULES:
                1. VISUAL HINGE INDICATORS:
                ✓ Look for actual visible hinges (metallic, cylindrical hardware)
                ✓ Identify the edge where the object rotates (stationary edge)
                ✓ Hinges are typically at 'left', 'right', 'top', or 'bottom' edges

                2. HINGE LOCATION DETERMINATION:
                • Examine the target object boundaries
                • Identify which edge remains stationary during operation (THE HINGE EDGE)
                • Specify hinge location as: 'left', 'right', 'top', 'bottom'
                • Use '' (empty) for non-hinged objects 
                

                --- EXPLANATION REQUIREMENTS ---

                    Your decision_rationale MUST include:

                    1. TARGET OBJECT IDENTIFICATION:
                    • What visual features confirmed this is the target object?
                    • What boundaries separate it from other objects?
                    • How confident are you in this identification?

                    2. POINT SELECTION ANALYSIS:
                    • List EVERY colored circle you can see and where it is, considering only the circle location
                    • For EACH circle, state whether it OVERLAPS or is just "near/aligned with" features
                    • For EACH rejected point, explain WHY (wrong object? decorative? poor position? no overlap?)
                    • For EACH selected point, confirm it PHYSICALLY OVERLAPS the functional feature
                    • Explicitly verify visual continuity between selected points
                    • CRITICAL: Distinguish between "overlapping" vs "near" or "aligned with"
                    
                    3. PIVOT SELECTION ANALYSIS (if applicable)
                    • List EVERY colored circle you can see and where it is, considering only the circle location
                    • For EACH rejected point, explain WHY (wrong object? wrong distance? poor position?)
                    • For THE selected point, explain WHY it's optimal
                    • Explicitly verify visual continuity between selected points

                    3. PRIMITIVE SEQUENCE REASONING:
                    For EACH action in your sequence, explain:
                    • WHY this specific point/surface was chosen
                    • WHY each parameter has its value
                    • WHAT alternatives you considered and rejected
                    • HOW robot reach constraints influenced your selection
                    • WHY the selected points are within comfortable working range

                    4. HINGE LOCATION IDENTIFICATION (for rotational objects):
                    • Explicitly identify and list the white edge labels and center label visible in the image
                    • SPATIAL ANALYSIS: Describe the interaction point location RELATIVE TO THE OBJECT BOUNDARIES:
                      ⚠️ CRITICAL: Analyze position relative to the OBJECT'S edges, NOT relative to other points
                      ⚠️ USE THE SPATIAL LABELS: Reference the visible "top", "bottom", "left", "right", "center" labels
                      - Where on the object is the handle/interaction point? (e.g., "near the LEFT label", "between CENTER and RIGHT labels")
                      - Which object edge(s) is it closest to? (e.g., "closest to the LEFT edge where I see the 'left' label")
                      - Which object edge(s) is it farthest from? (e.g., "farthest from the RIGHT edge marked by 'right' label")
                    
                      

                    5. MECHANISM UNDERSTANDING:
                    • What type of mechanism is this?
                    • How does it move?
                    • Why did you choose this opening strategy?
                    • How confident are you?

                --- FORCED SPATIAL ANALYSIS (REQUIRED) ---

                STEP 1: Draw a complete mental map of the scene:
                • Label each distinct object in the scene 
                • Mark visible boundaries between objects (seams, color changes)
                • Identify which labeled points belong to which objects

                STEP 2: For hinged objects, explicitly:
                • Measure approximate distance between points (in pixels or relative terms)
                • Calculate whether this distance matches expected door width
                • Identify which edge of the door remains fixed during opening
                • Select points only if they satisfy basic physical constraints
                
                --- PARAMETER QUICK REFERENCE ---
                
                FORCE DIRECTION:
                • 'perpendicular': Force along surface normal (into/out of surface)
                • 'parallel': Force parallel to surface (sliding motion)
            

                --- BEFORE FINALIZING YOUR SELECTION ---
                Complete this checklist:

                □ I identified the colored circles (not line intersections)
                □ I verified each circle PHYSICALLY OVERLAPS (not just "near") functional features
                □ I confirmed each circle is on the target object (not adjacent objects)
                □ I confirmed the features are functional (not decorative)
                □ I described the precise spatial location of the interaction point (e.g., "lower-left", "center-right")
                □ I referenced the white edge labels ('top', 'bottom', 'left', 'right') and center label visible in the image
                □ The selected points create mechanically sound motion
                □ I considered robot reach constraints and selected points within comfortable working range
                □ I avoided points that would cause robot overextension or collision risks
                □ For lids/caps: I included retract_gripper() to fully remove the lid/cap from the container
                □ For doors/ hinged covers: I included open_gripper() before retract_gripper() to avoid breaking the object.

                !!! CRITICAL OUTPUT REQUIREMENT !!!
                ==================================

                YOUR RESPONSE MUST BE VALID JSON ONLY - NO EXPLANATORY TEXT
                ANY NON-JSON OUTPUT WILL BE REJECTED BY THE SYSTEM

                RESPONSE FORMAT:
                {{
                    "name": "generic_manipulation_skill",
                    "abstract_action": "perform task",
                    "target_object": "target object",
                    "primitive_sequence": [
                        "move_gripper_to_pose('point_label', false, true)",
                        "close_gripper()",
                        ...
                    ],
                    "parameters": {{
                        "force_threshold": "medium",
                        "precision_required": "medium",
                        "speed_requirement": "slow"
                    }},
                    "prerequisites": [
                        "target object is in initial state",
                        "interaction points are accessible"
                    ],
                    "constraints": [
                        "avoid collision with nearby objects",
                        "limit applied force to prevent damage"
                    ],
                    "explanations": [
                        "Detailed explanation for point selection and mechanism understanding",
                        "Explanation for primitive sequence choice",
                        "Explanation for parameter values"
                    ]
                }}

                DO NOT INCLUDE ANY TEXT BEFORE OR AFTER THE JSON OBJECT
                DO NOT USE MARKDOWN CODE BLOCKS OR FORMATTING
                RETURN RAW JSON ONLY"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Task: {abstract_action} on {target_object}

                        Remember: Use ONLY colored circles as points, NOT line intersections!
                        
                        The images below show:
                        - FIRST IMAGE: Surface segments with normal vectors (for push/pull directions)
                        - SECOND IMAGE: Points of interest marked as colored circles (for grasping/manipulation)"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(surface_img)}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(points_img)}"
                        }
                    }
                ]
            }
        ]


        pprint.pp(point_descriptions)
        # Get response
        skill_response = self.llm.query_llm_sync(combined_prompt)
        try:
            skill_response = skill_response.replace('```json', '').replace('```', '')
            skill_data = json.loads(skill_response)
        except json.JSONDecodeError:
            print("Error parsing skill definition")
            return None

        # Create skill using the skill_data
        skill = Skill(
            name=skill_data.get('skill_name', f"{abstract_action}_{target_object}_generic_{hash(image_id) & 0xFFFFFF:06x}"),
            abstract_action=abstract_action,
            target_object=target_object,
            primitive_sequence=skill_data.get('primitive_sequence', []),
            parameters=skill_data.get('parameters', {}),
            prerequisites=skill_data.get('prerequisites', []),
            constraints=skill_data.get('constraints', []),
            explanations=skill_data.get('explanations', []),
            image_id=image_id,
            points_of_interest=points_of_interest
        )
        
        # Store the processed object_info with the skill for future reference
        if object_info is not None:
            # We can add a reference to the object_info if needed
            skill.object_info = object_info
        
        skill_path = self._store_skill(skill)
        self.last_generation_files['skill_paths'].append(skill_path)
        return skill

    def find_similar_skill(self, 
                          image: np.ndarray,
                          points_of_interest: Dict[str, PointOfInterest],
                          abstract_action: str,
                          target_object: str) -> Optional[Skill]:
        """
        Find a similar existing skill based on points of interest by comparing against all eligible skills at once
        
        Args:
            image: Input image of the object
            points_of_interest: Dictionary of labeled points of interest
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
        
        Returns:
            Most similar existing skill if found, None otherwise
        """
        if not self.skills_cache:
            return None
            
        # Create visualization for comparison
        vis_img = self._visualize_points_of_interest(image, points_of_interest)
        
        # Filter relevant skills by action and object type
        relevant_skills = []
        for skill in self.skills_cache.values():
            if skill.abstract_action == abstract_action and skill.target_object == target_object:
                existing_image = self._load_image(skill.image_id)
                if existing_image is not None:
                    relevant_skills.append((skill, existing_image))
        
        # If no relevant skills, return early
        if not relevant_skills:
            return None
        
        # Prepare skill images for batch comparison
        skill_images = []
        for _, img in relevant_skills:
            skill_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(img)}"
                }
            })
        
        # Create comparison prompt with all skills
        comparison_prompt = [
            {"role": "system", "content": f"""Compare the first image (new object) with the {len(relevant_skills)} following images (existing skills).
            For each comparison, provide a similarity score between 0 and 1, where 1 means identical and 0 means completely different.
            Consider visual similarity of the mechanism, interaction points, and their spatial relationships.
            Focus particularly on the positions of the labeled points relative to each other.
            The action to perform is "{abstract_action}" on a "{target_object}".
            
            Return ONLY a raw JSON object with scores in this format with NO additional formatting:
            {{
                "scores": [0.75, 0.42, 0.91, ...],
                "best_match_index": 2,  // Index of the image with highest score (0-based)
                "best_match_score": 0.91  // The highest score value
            }}"""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Compare this new object with the {len(relevant_skills)} existing skills below.
                Return the similarity scores for each, and identify the best match if any are similar enough.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(vis_img)}"
                }}
            ]}
        ]
        
        # Add all skill images to the prompt
        for img_data in skill_images:
            comparison_prompt[1]["content"].append(img_data)
        
        # Query LLM for all comparisons at once
        comparison_response = self.llm.query_llm_sync(comparison_prompt)
        
        try:
            # Parse the JSON response
            result = json.loads(comparison_response)
            
            # Validate response format
            if "scores" not in result or "best_match_index" not in result or "best_match_score" not in result:
                print("Invalid comparison response format")
                return None
            
            # Get best score and corresponding skill
            best_score = result["best_match_score"]
            best_index = result["best_match_index"]
            
            # Ensure index is valid
            if best_index < 0 or best_index >= len(relevant_skills):
                print(f"Invalid best match index: {best_index}")
                return None
            
            # Return the best skill if score is high enough
            if best_score > 0.8:
                return relevant_skills[best_index][0]
            else:
                return None
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing comparison response: {e}")
            print(f"Response: {comparison_response}")
            return None

    def adapt_skill(self, 
                         base_skill: Skill,
                         new_image: np.ndarray,
                         new_points: Dict[str, PointOfInterest]) -> Optional[Skill]:
        """
        Adapt an existing skill based on new image with different points of interest
        
        Args:
            base_skill: The skill to adapt
            new_image: Image of the new object
            new_points: Dictionary of labeled points of interest for the new object
        
        Returns:
            Adapted skill if successful, None otherwise
        """
        # Create visualization for the new object
        new_vis_img = self._visualize_points_of_interest(new_image, new_points)
        base_image = self._load_image(base_skill.image_id)
        
        if base_image is None:
            return None
            
        # Extract point descriptions for prompt
        new_point_descriptions = []
        for label, point in new_points.items():
            x, y = point.position
            desc = f"Point {label}: {point.description}" if point.description else f"Point {label}: Located at normalized coordinates ({x:.2f}, {y:.2f})"
            new_point_descriptions.append(desc)
            
        adaptation_prompt = [
            {"role": "system", "content": f"""Adapt the existing skill's primitive sequence and parameters for the new object based on the new labeled points.
            
            The new image contains an object with labeled points of interest (red circles with alphabetical labels).
            
            New points of interest:
            {chr(10).join(new_point_descriptions)}
            
            Consider visual differences in the mechanism and interaction points.
            Make sure to update the point labels in the primitive sequence to match the new object's labeled points.
            
            Return ONLY a JSON object with 'primitive_sequence' and 'parameters' fields. Return nothing else."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Original Skill:
                {json.dumps(asdict(base_skill), indent=2)}
                
                Adapt the skill based on visual differences and the new labeled points.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(base_image)}"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(new_vis_img)}"
                }}
            ]}
        ]
        
        adaptation_response = self.llm.query_llm_sync(adaptation_prompt)
        try:
            adapted_data = json.loads(adaptation_response)
            
            # Generate new image ID and save
            point_hash = hash(str([(p.label, p.position) for p in new_points.values()])) & 0xFFFFFF
            new_image_id = f"{base_skill.abstract_action}_{base_skill.target_object}_{point_hash:06x}"
            adapted_img_path = self._save_image(new_vis_img, new_image_id)
            self.last_generation_files['image_paths'].append(adapted_img_path)
            
            adapted_skill = Skill(
                name=f"{base_skill.name}_adapted_{hash(new_image_id) & 0xFFFFFF:06x}",
                abstract_action=base_skill.abstract_action,
                target_object=base_skill.target_object,
                primitive_sequence=adapted_data['primitive_sequence'],
                parameters=adapted_data['parameters'],
                prerequisites=base_skill.prerequisites,
                constraints=base_skill.constraints,
                image_id=new_image_id,
                points_of_interest=new_points
            )
            
            adapted_skill_path = self._store_skill(adapted_skill)
            self.last_generation_files['skill_paths'].append(adapted_skill_path)
            return adapted_skill
            
        except json.JSONDecodeError:
            print("Error parsing adaptation response")
            return None

    def _store_skill(self, skill: Skill) -> str:
        """Store a skill both in cache and on disk, return the path"""
        self.skills_cache[skill.name] = skill
        skill_path = self.skills_dir / f"{skill.name}.json"
        with open(skill_path, 'w') as f:
            json.dump(asdict(skill), f, indent=2)
        return str(skill_path)
            
    def get_last_generation_files(self) -> Dict[str, Any]:
        """
        Get information about files generated in the last skill generation
        
        Returns:
            Dictionary containing skill paths, image paths, and image ID
        """
        return self.last_generation_files.copy()
    
    def list_skills(self) -> List[str]:
        """Return a list of all available skill names"""
        return list(self.skills_cache.keys())

    def get_skill_details(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific skill"""
        skill = self.get_skill(skill_name)
        if skill:
            details = asdict(skill)
            image = self._load_image(skill.image_id)
            if image is not None:
                details['image'] = self._encode_image(image)
            return details
        return None

    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """Retrieve a skill by name"""
        return self.skills_cache.get(skill_name)