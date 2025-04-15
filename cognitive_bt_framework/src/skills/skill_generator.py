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
from cognitive_bt_framework.src.vision.perception_system import ObjectInfo

@dataclass
class PointOfInterest:
    """Data class to store labeled point of interest"""
    label: str  # Alphabetical label (a, b, c, etc.)
    position: Tuple[float, float]  # Normalized (x, y) coordinates
    description: str = ""  # Optional description of the point
    pixel_coords: List[Tuple[float, float]]
    
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
        self._load_stored_skills()

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

    def _save_image(self, image: np.ndarray, image_id: str):
        """Save image to disk"""
        cv2.imwrite(str(self.image_dir / f"{image_id}.png"), image)

    def _load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image from disk"""
        image_path = self.image_dir / f"{image_id}.png"
        if image_path.exists():
            return cv2.imread(str(image_path))
        return None



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

    def _visualize_points_of_interest(
            self, 
            image: np.ndarray, 
            obj_info: ObjectInfo,
            points: Dict[str, Any] = None,
            top_n_surfaces: int = 5  # Number of most confident surface masks to visualize
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create two separate visualizations:
        1. Image with surface masks and their normals as arrows
        2. Points of interest visualization
        
        Args:
            image: Input image
            obj_info: Object information with surface masks and depth image
            points: Dictionary of points of interest (optional)
            top_n_surfaces: Number of most confident surface masks to visualize
            
        Returns:
            Tuple of (surface_img, points_img)
        """
        h, w = image.shape[:2]
        
        # Create the first image: Surface masks overlay with normals
        surface_img = image.copy()
        
        # Draw object mask as a semi-transparent overlay
        if obj_info.mask is not None:
            mask_overlay = np.zeros_like(surface_img, dtype=np.uint8)
            mask_overlay[obj_info.mask] = [0, 255, 0]  # Green for object mask
            surface_img = cv2.addWeighted(surface_img, 1.0, mask_overlay, 0.2, 0)
            
            # Draw object label with alphabetical ID
            alpha_id = obj_info.alpha_id if obj_info.alpha_id else get_alpha_id(obj_info.id)
            # Find centroid of the object mask for label placement
            y_coords, x_coords = np.where(obj_info.mask)
            if len(y_coords) > 0:
                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))
                
                # Add object ID and name at centroid
                cv2.putText(
                    surface_img,
                    f"{alpha_id}: {obj_info.name}",
                    (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),  # Black text
                    2
                )
        
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
                    
                    # Calculate normal for this surface using depth if available
                    if has_depth:
                        normal = self._calculate_surface_normals(
                            surface_mask, 
                            obj_info.depth_image,
                            getattr(obj_info, 'camera_intrinsics', None)
                        )
                    else:
                        # Fallback to PCA-based estimation without depth
                        from sklearn.decomposition import PCA
                        
                        # Find coordinates of points in the mask
                        surf_y, surf_x = np.where(surface_mask)
                        
                        if len(surf_y) >= 10:  # Enough points for PCA
                            # Create points array with z-coordinate estimated as constant
                            points = np.column_stack((surf_x, surf_y, np.ones_like(surf_x)))
                            
                            # Fit PCA to find the principal components
                            pca = PCA(n_components=3)
                            pca.fit(points)
                            
                            # The normal is perpendicular to the first two principal components
                            normal = pca.components_[2]
                            
                            # Ensure the normal points toward the camera (positive z direction)
                            if normal[2] < 0:
                                normal = -normal
                                
                            # Normalize the vector
                            normal = normal / np.linalg.norm(normal)
                            normal = tuple(normal)
                        else:
                            normal = (0, 0, 1)  # Default to pointing out from camera
                    
                    # Project normal to image plane
                    arrow_end = self._project_normal_to_image(
                        normal, centroid_x, centroid_y, arrow_length=40
                    )
                    
                    # Draw normal as an arrow
                    cv2.arrowedLine(
                        surface_img,
                        (centroid_x, centroid_y),
                        arrow_end,
                        color_map[color_idx],
                        2,
                        tipLength=0.3
                    )
                    
                    # Add surface ID at centroid
                    cv2.putText(
                        surface_img,
                        alpha_id,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2
                    )
                    
                    # Add normal vector information
                    normal_text = f"n: ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
                    cv2.putText(
                        surface_img,
                        normal_text,
                        (centroid_x + 5, centroid_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),  # White text
                        1
                    )
            
            # Add surface overlay with transparency
            surface_img = cv2.addWeighted(surface_img, 1.0, surface_overlay, 0.3, 0)
            
        
        # Add title to the surface image
        cv2.putText(
            surface_img,
            "Surface Segmentation with Normals",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add legend for normals
        cv2.putText(
            surface_img,
            "Arrows show surface normals",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Add information about depth usage
        cv2.putText(
            surface_img,
            f"Using {'depth data' if has_depth else 'PCA estimation'} for normals",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Create the second image: Points of interest
        points_img = image.copy()
        
        
        # Draw points of interest if provided
        if points is not None and len(points) > 0:
            # If 'pixel_coords' is in points, use that
            if 'pixel_coords' in points and points['pixel_coords']:
                pixel_coords = points['pixel_coords']
                scores = points.get('scores', [1.0] * len(pixel_coords))
                ids = points.get('ids', [f"p{i}" for i in range(len(pixel_coords))])
                
                # Calculate score range for coloring
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 1
                score_range = max_score - min_score if max_score > min_score else 1
                
                # Draw each point
                for i, ((px, py), score, point_id) in enumerate(zip(pixel_coords, scores, ids)):
                    # Normalize score to [0, 1]
                    norm_score = (score - min_score) / score_range if score_range > 0 else 0.5
                    
                    # Map to color (blue to red based on score)
                    color = (
                        int(255 * (1 - norm_score)),  # B
                        0,                           # G
                        int(255 * norm_score)         # R
                    )
                    
                    # Draw circle for point
                    cv2.circle(
                        points_img, 
                        (px, py), 
                        radius=5, 
                        color=color, 
                        thickness=2
                    )
                    
                    # Draw label (use alphabetical ID)
                    if not point_id.isalpha():
                        # If ID is not already alphabetical, generate one
                        point_id = get_alpha_id(i + 1)
                        
                    cv2.putText(
                        points_img,
                        point_id,
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        1
                    )
                    
                    
            else:
                # Handle dictionary of named points (original format)
                for i, (label, point) in enumerate(points.items()):
                    if hasattr(point, 'position'):
                        # Handle PointOfInterest objects
                        # Convert normalized coordinates to pixel coordinates if needed
                        if max(point.position) <= 1.0:
                            px, py = int(point.position[0] * w), int(point.position[1] * h)
                        else:
                            px, py = int(point.position[0]), int(point.position[1])
                    else:
                        # Handle direct (x,y) tuples
                        px, py = int(point[0]), int(point[1])
                    
                    # Generate alphabetical ID if the label is not already alphabetical
                    if not label.isalpha():
                        alpha_id = get_alpha_id(i + 1)
                    else:
                        alpha_id = label
                    
                    # Draw circle for point
                    cv2.circle(points_img, (px, py), 5, (0, 0, 255), -1)
                    
                    # Draw label
                    cv2.putText(
                        points_img,
                        alpha_id,
                        (px, py ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )
        
        # Add title to the points image
        cv2.putText(
            points_img,
            "Points of Interest",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add object name and ID
        alpha_id = obj_info.alpha_id if obj_info.alpha_id else get_alpha_id(obj_info.id)
        obj_text = f"Object: {alpha_id} {obj_info.name}"
        cv2.putText(
            points_img,
            obj_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # If depth is available, add this information
        if has_depth:
            cv2.putText(
                points_img,
                "Depth data available",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return surface_img, points_img
    
    def _categorize_points_in_bounding_box(self, 
                                      image: np.ndarray,
                                      pixel_coords: List[Tuple[float, float]],
                                      object_info: Optional[ObjectInfo] = None,
                                      ids: Optional[List[str]] = None,
                                      scores: Optional[List[float]] = None) -> List[str]:
        """
        Categorize points based on their position within the object's bounding box.
        
        Args:
            image: Input image of the object
            pixel_coords: List of (x, y) coordinates for points of interest
            object_info: ObjectInfo containing additional object data including bounding box
            ids: Optional list of point IDs
            scores: Optional list of confidence scores for each point
            
        Returns:
            List of point descriptions with spatial categorization
        """
        # Use provided IDs or generate default alphabetical IDs
        if ids is None:
            ids = [f"{chr(97 + i)}" for i in range(len(pixel_coords))]
            
        # Use provided scores or default to 1.0
        if scores is None:
            scores = [1.0] * len(pixel_coords)
        
        # Get object bounding box - either from object_info or calculate from points
        if object_info is not None and object_info.bbox is not None:
            x_min, y_min, x_max, y_max = object_info.bbox
        else:
            # Calculate approximate bounding box from points
            if len(pixel_coords) > 0:
                x_points = [p[0] for p in pixel_coords]
                y_points = [p[1] for p in pixel_coords]
                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)
                # Add padding to ensure points on the edge are covered
                padding = 0.05  # 5% padding
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - padding * width)
                y_min = max(0, y_min - padding * height)
                x_max = min(image.shape[1], x_max + padding * width)
                y_max = min(image.shape[0], y_max + padding * height)
            else:
                # Fallback to full image if no points
                x_min, y_min = 0, 0
                x_max, y_max = image.shape[1], image.shape[0]
        
        # Define regions of the bounding box
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
        for i, ((x, y), score, point_id) in enumerate(zip(pixel_coords, scores, ids)):
            # Normalize coordinates for description
            norm_x, norm_y = x / image.shape[1], y / image.shape[0]
            
            # Determine horizontal position
            if x < left_bound:
                h_pos = "left"
            elif x > right_bound:
                h_pos = "right"
            else:
                h_pos = "center"
            
            # Determine vertical position
            if y < top_bound:
                v_pos = "top"
            elif y > bottom_bound:
                v_pos = "bottom"
            else:
                v_pos = "middle"
            
            # Check if point is near an edge
            is_edge = (x < x_min + edge_margin or 
                    x > x_max - edge_margin or 
                    y < y_min + edge_margin or 
                    y > y_max - edge_margin)
            
            position_desc = f"{v_pos} {h_pos}"
            if is_edge:
                position_desc += " (edge)"
            
            desc = f"Point {point_id}: Located at normalized coordinates ({norm_x:.2f}, {norm_y:.2f}), position: {position_desc}, confidence score: {score:.2f}"
            point_descriptions.append(desc)
        
        return point_descriptions

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
        # First gather all points to calculate bounding box
        point_positions = [point.position for point in point_objects.values()]
        labels = list(point_objects.keys())
        
        # Get object bounding box - either from object_info or calculate from points
        if object_info is not None and object_info.bbox is not None:
            x_min, y_min, x_max, y_max = object_info.bbox
        elif point_positions:
            # Calculate approximate bounding box from points
            x_points = [p[0] for p in point_positions]
            y_points = [p[1] for p in point_positions]
            # These are already normalized coordinates
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)
            # Add padding
            padding = 0.05  # 5% padding
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, x_min - padding * width)
            y_min = max(0, y_min - padding * height)
            x_max = min(1.0, x_max + padding * width)
            y_max = min(1.0, y_max + padding * height)
        else:
            # Fallback to full image if no points
            x_min, y_min = 0, 0
            x_max, y_max = 1.0, 1.0
        
        # Define regions of the bounding box
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
            x, y = point.position  # These are normalized coordinates
            
            # Determine horizontal position
            if x < left_bound:
                h_pos = "left"
            elif x > right_bound:
                h_pos = "right"
            else:
                h_pos = "center"
            
            # Determine vertical position
            if y < top_bound:
                v_pos = "top"
            elif y > bottom_bound:
                v_pos = "bottom"
            else:
                v_pos = "middle"
            
            # Check if point is near an edge
            is_edge = (x < x_min + edge_margin or 
                    x > x_max - edge_margin or 
                    y < y_min + edge_margin or 
                    y > y_max - edge_margin)
            
            position_desc = f"{v_pos} {h_pos}"
            if is_edge:
                position_desc += " (edge)"
            
            desc = (f"Point {label}: {point.description}, position: {position_desc}" 
                    if hasattr(point, 'description') and point.description 
                    else f"Point {label}: Located at normalized coordinates ({x:.2f}, {y:.2f}), position: {position_desc}")
            point_descriptions.append(desc)
        
        return point_descriptions


    def generate_skill(self, 
                    image: np.ndarray,
                    points_of_interest: Dict[str, Any],
                    abstract_action: str,
                    target_object: str,
                    object_info: Optional[ObjectInfo] = None) -> Optional[Skill]:
        """
        Generate a new skill using the LLM interface based on image input with labeled points and surfaces
        
        Args:
            image: Input image of the object
            points_of_interest: Dictionary of points information (from detect_regions_of_interest)
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
            object_info: ObjectInfo containing additional object data including surface_masks
        
        Returns:
            Generated skill if successful, None otherwise
        """
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
            surface_img, points_img = self._visualize_points_of_interest(image, object_info, points_of_interest)
        else:
            # Create the separate visualizations for surfaces with normals and points
            surface_img, points_img = self._visualize_points_of_interest(image, object_info, points_of_interest)
        
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
        
        # Save each visualization separately
        surface_img_path = self._save_image(surface_img, f"surface_{image_id}")
        points_img_path = self._save_image(points_img, f"points_{image_id}")
        
        if self.debug:
            self.get_logger().info(f"Saved surface image to {surface_img_path}")
            self.get_logger().info(f"Saved points image to {points_img_path}")
        
        # Also save the original image for reference
        self._save_image(image, image_id)
        
        # Create descriptions for prompt
        point_descriptions = []
        surface_descriptions = []

        # Handle points
        if 'pixel_coords' in points_of_interest:
            # New format from detect_regions_of_interest
            pixel_coords = points_of_interest['pixel_coords']
            scores = points_of_interest.get('scores', [1.0] * len(pixel_coords))
            ids = points_of_interest.get('ids', [f"{chr(97 + i)}" for i in range(len(pixel_coords))])
            
            # Use the new function to categorize points
            point_descriptions = self._categorize_points_in_bounding_box(
                image=image,
                pixel_coords=pixel_coords,
                object_info=object_info,
                ids=ids,
                scores=scores
            )
        else:
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
                area_percentage = (np.sum(surface_mask) / (image.shape[0] * image.shape[1])) * 100
                
                # Find centroid of the surface
                y_coords, x_coords = np.where(surface_mask)
                if len(y_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    
                    # Calculate normal for this surface
                    normal = self._calculate_surface_normals(surface_mask, object_info.depth_image)
                    
                    # Normalize coordinates for description
                    norm_x, norm_y = centroid_x / image.shape[1], centroid_y / image.shape[0]
                    
                    desc = f"Surface {alpha_id}: Normal vector ({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}), covers {area_percentage:.1f}% of image area"
                    surface_descriptions.append(desc)
        
        # Create prompt for skill definition
        combined_prompt = [{"role": "system", "content": f"""Analyze the object visually and generate a complete skill definition for performing {abstract_action} on a {target_object}.

                    You will be provided with two separate visualizations:
                    1. FIRST IMAGE: The object with surface segments highlighted in different colors, each with an alphabetical label and normal vectors shown as arrows
                    2. SECOND IMAGE: The same object with points of interest marked by colored circles, each with an alphabetical label

                    Points of interest:
                    {chr(10).join(point_descriptions)}

                    Surface segments with calculated normal vectors:
                    {chr(10).join(surface_descriptions)}

                    First, determine the specific subtype of the action based on the object's visual characteristics.
                    The skill name should follow the format: action_targetobject_mechanism

                    AVAILABLE ACTION PRIMITIVES AND PARAMETERS:

                    1. move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)
                    - point_label: The labeled point (a, b, c, etc.) from the SECOND IMAGE where the gripper should move to
                    - is_top_down_grasp: Boolean (true/false) indicating if the gripper should approach from above
                    - is_side_grasp: Boolean (true/false) indicating if the gripper should approach from the side
                    - Example: move_gripper_to_pose('a', true, false) - Move to point 'a' with a top-down approach

                    2. push('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                    - surface_label: The labeled surface (a, b, c, etc.) from the FIRST IMAGE that the robot should push against
                    - force_direction: Either 'perpendicular' (push directly into the surface) or 'parallel' (push along the surface)
                    - is_button: Boolean (true/false) indicating if this is a button push (short distance, low force)
                    - has_pivot: Boolean (true/false) indicating if the push should pivot around another point
                    - pivot_point_label: If has_pivot is true, the labeled point to pivot around; otherwise use empty string ''
                    - Example: push('b', 'perpendicular', true, false, '') - Push surface 'b' like a button

                    3. pull('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                    - surface_label: The labeled surface (a, b, c, etc.) from the FIRST IMAGE that the robot should pull
                    - force_direction: Either 'perpendicular' (pull directly away from the surface) or 'parallel' (pull along the surface)
                    - is_button: Boolean (true/false) indicating if this is a button-like pull (short distance, low force)
                    - has_pivot: Boolean (true/false) indicating if the pull should pivot around another point
                    - pivot_point_label: If has_pivot is true, the labeled point to pivot around; otherwise use empty string ''
                    - Example: pull('c', 'parallel', false, true, 'd') - Pull surface 'c' along the surface, pivoting around point 'd'

                    4. close_gripper()
                    - Closes the robot's gripper to grasp an object
                    - No parameters required
                    - Example: close_gripper()

                    5. open_gripper()
                    - Opens the robot's gripper to release an object
                    - No parameters required
                    - Example: open_gripper()

                    6. retract_gripper()
                    - Moves the gripper away from the current position to a safe position
                    - No parameters required
                    - Example: retract_gripper()

                    SKILL PARAMETERS:

                    1. force_threshold:
                    - low: For delicate objects or precise operations (1-5N)
                    - medium: For standard operations (5-15N)
                    - high: For operations requiring significant force (15-30N)

                    2. precision_required:
                    - low: For operations where exact positioning is not critical (±10mm)
                    - medium: For standard operations requiring good accuracy (±5mm)
                    - high: For operations requiring very precise positioning (±1mm)

                    3. speed_requirement:
                    - slow: For delicate operations or where safety is paramount (0.1-0.2m/s)
                    - medium: For standard operations (0.2-0.4m/s)
                    - high: For operations where time efficiency is important (0.4-0.6m/s)

                    IMPORTANT: You must output ONLY a valid JSON object with the following structure:

                    {{{{
                        "skill_name": "specific_action_name_with_mechanism",
                        "primitive_sequence": [
                            "EACH PRIMITIVE MUST USE EXACTLY ONE OF THESE FORMATS:",
                            "move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)",
                            "push('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')",
                            "pull('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')",
                            "close_gripper()",
                            "open_gripper()",
                            "retract_gripper()"
                        ],
                        "parameters": {{{{
                            "force_threshold": "low/medium/high",
                            "precision_required": "low/medium/high",
                            "speed_requirement": "slow/medium/fast"
                        }}}},
                        "prerequisites": [
                            "list of required conditions"
                        ],
                        "constraints": [
                            "list of safety limits and constraints"
                        ],
                        "explanations": [
                            "list of detailed explanations for primitives and parameter choices"
                        ]
                    }}}}

                    CRITICAL FORMATTING RULES:
                    1. All point and surface labels MUST be in single quotes (e.g., 'a', 'b', etc.)
                    2. force_direction MUST be in single quotes and be either 'parallel' or 'perpendicular'
                    3. is_button, has_pivot, is_top_down_grasp, is_side_grasp MUST be boolean values (true or false) WITHOUT quotes
                    4. pivot_point_label MUST be in single quotes, even if empty (e.g., '', 'c')
                    5. The syntax must match EXACTLY one of these patterns:
                    - move_gripper_to_pose('a', true, false)
                    - push('b', 'perpendicular', true, false, '')
                    - pull('c', 'parallel', false, true, 'd')
                    - close_gripper()
                    - open_gripper()
                    - retract_gripper()

                    IMPORTANT DISTINCTION:
                    - For move_gripper_to_pose: use a point label from the SECOND IMAGE (points of interest)
                    - For push and pull: use a surface label from the FIRST IMAGE (surface segments)
                    - Use the normal vector information shown in the FIRST IMAGE to determine the best direction for pushing or pulling

                    Note:
                    - The chosen point for move_gripper_to_pose should be the BEST location for grasping or manipulating the object
                    - The chosen surface for push/pull should be the CLEAREST INDICATOR of the surface to align the force with
                    - Use the calculated normal vectors shown as arrows to determine the appropriate force direction
                    - Prerequisites should include any conditions that must be met before execution
                    - Constraints should include any safety limits or operating constraints
                    - BE VERY CAREFUL SELECTING POINTS, always take your time and DOUBLE CHECK that they are in the spacial location you think they are in

                    Base all values on the visual appearance of the object.
                    Return only the raw JSON object with no additional text or formatting. the output should start with an open bracket and end with a close bracket."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Abstract Action: {abstract_action}
                Target Object: {target_object}
                
                Generate a complete skill definition for this task based on the object's visual appearance.
                First determine the specific subtype of action needed based on the object's
                characteristics, then generate the complete skill definition.
                
                FIRST IMAGE shows surface segments with alphabetical labels and normal vectors displayed as arrows.
                SECOND IMAGE shows points of interest with alphabetical labels.
                
                For move_gripper_to_pose: use a point label from the SECOND IMAGE.
                For push and pull: use a surface label from the FIRST IMAGE.
                Use the normal vector arrows to determine appropriate force directions.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(surface_img)}"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(points_img)}"
                }}
            ]}]
        pprint.pp(point_descriptions)
        # Get response
        skill_response = self.llm.query_llm_sync(combined_prompt)
        try:
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
        
        self._store_skill(skill)
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
            self._save_image(new_vis_img, new_image_id)
            
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
            
            self._store_skill(adapted_skill)
            return adapted_skill
            
        except json.JSONDecodeError:
            print("Error parsing adaptation response")
            return None

    def _store_skill(self, skill: Skill):
        """Store a skill both in cache and on disk"""
        self.skills_cache[skill.name] = skill
        skill_path = self.skills_dir / f"{skill.name}.json"
        with open(skill_path, 'w') as f:
            json.dump(asdict(skill), f, indent=2)

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