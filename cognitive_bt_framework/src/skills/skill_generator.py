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


     # Helper function to find optimal label position avoiding overlaps
    def find_optimal_label_position(self, px, py, alpha_id, all_points, placed_labels, img_width, img_height):
        """
        Find the best position for a label that doesn't overlap with points or other labels.
        Improved version with better overlap avoidance and more positioning options.
        """
        
        # Get label dimensions
        label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]  # Slightly smaller font
        label_width, label_height = label_bg_size[0] + 6, label_bg_size[1] + 6  # More padding
        
        # Define more candidate positions in concentric circles around the point
        base_distance = 25  # Base distance from point
        distances = [base_distance, base_distance * 1.5, base_distance * 2, base_distance * 2.5]  # Multiple radii
        
        candidate_positions = []
        
        # For each distance, try 8 directions
        for distance in distances:
            angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions in degrees
            for angle in angles:
                rad = np.radians(angle)
                offset_x = int(distance * np.cos(rad))
                offset_y = int(distance * np.sin(rad))
                
                label_x = px + offset_x
                label_y = py + offset_y
                
                candidate_positions.append((offset_x, offset_y, label_x, label_y, distance, angle))
        
        best_position = None
        best_score = -1
        
        for offset_x, offset_y, label_x, label_y, distance, angle in candidate_positions:
            # Check if label is within image bounds with margin
            margin = 10
            if (label_x < margin or label_y < margin or 
                label_x + label_width > img_width - margin or 
                label_y + label_height > img_height - margin):
                continue
            
            # Check distance from all points (including current one)
            min_point_distance = float('inf')
            for other_px, other_py in all_points:
                # Calculate distance from label corners to point to ensure no overlap with point markers
                label_corners = [
                    (label_x, label_y),
                    (label_x + label_width, label_y),
                    (label_x, label_y + label_height),
                    (label_x + label_width, label_y + label_height)
                ]
                
                corner_distances = [
                    np.sqrt((corner_x - other_px)**2 + (corner_y - other_py)**2)
                    for corner_x, corner_y in label_corners
                ]
                min_corner_distance = min(corner_distances)
                min_point_distance = min(min_point_distance, min_corner_distance)
            
            # Check overlap with all placed labels
            has_overlap = False
            min_label_distance = float('inf')
            
            for placed_x, placed_y, placed_w, placed_h in placed_labels:
                # Check for rectangle overlap with extra buffer
                buffer = 5  # Minimum space between labels
                if not (label_x + label_width + buffer < placed_x or 
                    placed_x + placed_w + buffer < label_x or
                    label_y + label_height + buffer < placed_y or 
                    placed_y + placed_h + buffer < label_y):
                    has_overlap = True
                    break
                else:
                    # Calculate minimum distance between label centers
                    label_center_x = label_x + label_width // 2
                    label_center_y = label_y + label_height // 2
                    placed_center_x = placed_x + placed_w // 2
                    placed_center_y = placed_y + placed_h // 2
                    center_dist = np.sqrt((label_center_x - placed_center_x)**2 + 
                                        (label_center_y - placed_center_y)**2)
                    min_label_distance = min(min_label_distance, center_dist)
            
            # Score this position (higher is better)
            if has_overlap:
                continue  # Skip overlapping positions entirely
            
            if min_point_distance < 15:  # Too close to a point marker
                score = -50 + min_point_distance
            else:
                # Good position, score based on multiple factors
                distance_score = min_point_distance  # Prefer further from points
                label_clearance_score = min_label_distance if min_label_distance != float('inf') else 100
                
                # Prefer certain directions (right and bottom-right are usually best)
                direction_preference = 0
                if 315 <= angle <= 45 or angle == 0:  # Right side
                    direction_preference = 20
                elif 45 < angle <= 135:  # Bottom side
                    direction_preference = 15
                elif 135 < angle <= 225:  # Left side  
                    direction_preference = 5
                else:  # Top side
                    direction_preference = 10
                
                # Prefer closer distances if possible
                distance_penalty = distance / 10
                
                score = distance_score + label_clearance_score + direction_preference - distance_penalty
            
            if score > best_score:
                best_score = score
                best_position = (label_x, label_y, offset_x, offset_y)
        
        # If no good position found, try a systematic grid search as fallback
        if best_position is None:
            best_position = self._grid_search_label_position(
                px, py, label_width, label_height, placed_labels, img_width, img_height
            )
        
        # Final fallback: place to the right with offset to minimize overlap
        if best_position is None:
            # Find a y-offset that minimizes overlap
            best_y_offset = -10
            min_overlap_count = float('inf')
            
            for y_offset in range(-50, 51, 10):
                label_x = px + 30
                label_y = py + y_offset
                
                if label_y < 0 or label_y + label_height > img_height:
                    continue
                    
                overlap_count = 0
                for placed_x, placed_y, placed_w, placed_h in placed_labels:
                    if not (label_x + label_width < placed_x or 
                        placed_x + placed_w < label_x or
                        label_y + label_height < placed_y or 
                        placed_y + placed_h < label_y):
                        overlap_count += 1
                
                if overlap_count < min_overlap_count:
                    min_overlap_count = overlap_count
                    best_y_offset = y_offset
            
            best_position = (px + 30, py + best_y_offset, 30, best_y_offset)
        
        return best_position

    def _grid_search_label_position(self, px, py, label_width, label_height, placed_labels, img_width, img_height):
        """
        Systematic grid search for label placement when other methods fail.
        """
        search_radius = 80
        step_size = 10
        
        best_position = None
        min_overlaps = float('inf')
        
        for dx in range(-search_radius, search_radius + 1, step_size):
            for dy in range(-search_radius, search_radius + 1, step_size):
                label_x = px + dx
                label_y = py + dy
                
                # Check bounds
                if (label_x < 0 or label_y < 0 or 
                    label_x + label_width > img_width or 
                    label_y + label_height > img_height):
                    continue
                
                # Count overlaps
                overlap_count = 0
                for placed_x, placed_y, placed_w, placed_h in placed_labels:
                    if not (label_x + label_width < placed_x or 
                        placed_x + placed_w < label_x or
                        label_y + label_height < placed_y or 
                        placed_y + placed_h < label_y):
                        overlap_count += 1
                
                # Prefer positions with fewer overlaps, and closer to original point
                distance = np.sqrt(dx**2 + dy**2)
                score = -overlap_count * 1000 - distance  # Heavily penalize overlaps
                
                if overlap_count < min_overlaps or (overlap_count == min_overlaps and distance < 50):
                    min_overlaps = overlap_count
                    best_position = (label_x, label_y, dx, dy)
                    
                    if overlap_count == 0:  # Found non-overlapping position
                        break
            
            if min_overlaps == 0:  # Found non-overlapping position
                break
        
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
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black outline
                        4
                    )
                    
                    # Add surface ID at centroid
                    cv2.putText(
                        surface_img,
                        alpha_id,
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
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

        if points is not None and len(points) > 0:
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
                
                # Draw each point with improved positioning
                for idx, i in enumerate(sorted_indices):
                    px, py = pixel_coords[i]
                    score = scores[i]
                    point_id = ids[i]
                    
                    # Get color for this point (cycle through available colors)
                    color = point_colors[idx % len(point_colors)]
                    
                    # Generate alphabetical ID if not already alphabetical
                    if not point_id.isalpha():
                        alpha_id = get_alpha_id(idx + 1)
                    else:
                        alpha_id = point_id
                    
                    # Draw circle for point with colored outline (slightly larger for better visibility)
                    cv2.circle(points_img, (px, py), radius=10, color=color, thickness=3)
                    # Draw white center for better visibility
                    cv2.circle(points_img, (px, py), radius=6, color=(255, 255, 255), thickness=-1)
                    # Add a small colored center dot
                    cv2.circle(points_img, (px, py), radius=3, color=color, thickness=-1)
                    
                    # Find optimal label position using improved algorithm
                    label_x, label_y, offset_x, offset_y = self.find_optimal_label_position(
                        px, py, alpha_id, all_point_coords, placed_labels, w, h
                    )
                    
                    # Get label dimensions for tracking
                    label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_width, label_height = label_bg_size[0] + 6, label_bg_size[1] + 6
                    
                    # Draw connecting line if label is far from point
                    if abs(offset_x) > 20 or abs(offset_y) > 20:
                        # Draw a thin line from point edge to label
                        line_start_x = px + (8 if offset_x > 0 else -8)
                        line_start_y = py + (8 if offset_y > 0 else -8)
                        line_end_x = label_x + label_width // 2
                        line_end_y = label_y + label_height // 2
                        
                        cv2.line(points_img, (line_start_x, line_start_y), 
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
                    
                    # Draw label text in white for contrast
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
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
                
                # Draw each point with improved positioning  
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
                    
                    # Draw circle for point with colored outline (slightly larger for better visibility)
                    cv2.circle(points_img, (px, py), radius=10, color=color, thickness=3)
                    # Draw white center for better visibility
                    cv2.circle(points_img, (px, py), radius=6, color=(255, 255, 255), thickness=-1)
                    # Add a small colored center dot
                    cv2.circle(points_img, (px, py), radius=3, color=color, thickness=-1)
                    
                    # Find optimal label position using improved algorithm
                    label_x, label_y, offset_x, offset_y = self.find_optimal_label_position(
                        px, py, alpha_id, all_point_coords, placed_labels, w, h
                    )
                    
                    # Get label dimensions for tracking
                    label_bg_size = cv2.getTextSize(alpha_id, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_width, label_height = label_bg_size[0] + 6, label_bg_size[1] + 6
                    
                    # Draw connecting line if label is far from point
                    if abs(offset_x) > 20 or abs(offset_y) > 20:
                        # Draw a thin line from point edge to label
                        line_start_x = px + (8 if offset_x > 0 else -8)
                        line_start_y = py + (8 if offset_y > 0 else -8)
                        line_end_x = label_x + label_width // 2
                        line_end_y = label_y + label_height // 2
                        
                        cv2.line(points_img, (line_start_x, line_start_y), 
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
                    
                    # Draw label text in white for contrast
                    cv2.putText(points_img, alpha_id, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Track this label's position for future overlap checking
                    placed_labels.append((label_rect[0], label_rect[1], 
                                        label_width + 6, label_height + 6))

        # Add title to the points image
        cv2.putText(points_img, "Points of Interest", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
                        depth_mm = depth * 1000  # Convert meters to mm
                        min_mm = min_val * 1000
                        max_mm = max_val * 1000
                        
                        cv2.putText(
                            depth_img,
                            f"Depth: {min_mm:.0f}-{max_mm:.0f}mm",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                    else:
                        cv2.putText(
                            depth_img,
                            f"Depth: {min_val:.3f}-{max_val:.3f}m",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                    
                    # Add title
                    cv2.putText(
                        depth_img,
                        "Processed Depth Visualization",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
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
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Save the depth image
        self._save_image(depth_img, f"depth_image_{image_id}")
        
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
                desc = f"Point {label}: Located at normalized coordinates ({norm_x:.2f}, {norm_y:.2f}), position: {position_desc}"
            
            point_descriptions.append(desc)
        
        return point_descriptions


    def generate_skill(self, 
                    image: np.ndarray,
                    points_of_interest: Dict[str, Any],
                    abstract_action: str,
                    target_object: str,
                    object_info: Optional[ObjectInfo] = None,
                    ) -> Optional[Skill]:
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
        
        if self.debug:
            self.get_logger().info(f"Saved surface image to {surface_img_path}")
            self.get_logger().info(f"Saved points image to {points_img_path}")
        
        # Also save the original image for reference
        self._save_image(image, image_id)
        
        # Create descriptions for prompt
        point_descriptions = []
        surface_descriptions = []

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
                    
                    desc = f"Surface {alpha_id}"
                    surface_descriptions.append(desc)
        
        # Create prompt for skill definition
        combined_prompt = [
            {
                "role": "system",
                "content": f"""You are a grounded visual skill planner.

                Your task is to generate a complete, structured skill definition for performing **{abstract_action}** on a **{target_object}**, using only the provided visual inputs.

                --- VISUAL INPUT FORMAT ---
                You are provided with two images:
                1. **FIRST IMAGE** – Surface segments (labeled aaa, aab, aac...) with **normal vectors** shown as arrows.
                2. **SECOND IMAGE** – Points of interest (labeled aaa, aab, aac...) indicated with colored circles.

                SURFACE LABELS → FIRST IMAGE ONLY  
                POINT LABELS → SECOND IMAGE ONLY  

                {surface_descriptions}

                --- TASK ---
                1. Determine the **appropriate subtype** of `{abstract_action}` based on object geometry.
                2. Select a **sequence of action primitives** to achieve it.
                3. Set **precise parameters** based only on the visible surfaces, normals, and geometry.
                4. Return a **single valid raw JSON** with no text or formatting outside of it.

                --- AVAILABLE PRIMITIVES ---
                - move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)
                - push('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                - pull('surface_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                - close_gripper()
                - open_gripper()
                - retract_gripper()
                - twist('direction') // 'clockwise' or 'counterclockwise'

                --- PUSH / PULL PARAMETER GUIDE ---

                **1. force_direction ('perpendicular' | 'parallel')**
                - Use `'perpendicular'` if the force should act **into or out of the surface** — e.g., pushing a button, pulling a latch straight out.
                - Use `'parallel'` if the interaction requires a **sliding or dragging** motion along the surface — e.g., sliding a door, rotating a lid.
                - Align the direction with the **normal vector** shown in the FIRST IMAGE.

                **2. is_button (true | false)**
                - Use `true` when the push/pull is:
                - **Short distance**
                - Requires **low force**
                - Used to **activate or toggle** a mechanism (e.g., press a button, release a spring)
                - Use `false` when the action involves sustained contact or continuous movement (e.g., sliding, opening).

                **3. has_pivot (true | false)**
                - Use `true` when the action requires **rotating around a fixed point** on the object (e.g., opening a hinged lid, rotating a handle).
                - Use `false` when the force applies evenly across the surface (no clear rotation axis).

                **4. pivot_point_label ('a', 'b', ..., or '')**
                - Provide a **point label from the SECOND IMAGE** if `has_pivot = true`, representing the center or hinge of rotation.
                - Use `''` (empty string) if `has_pivot = false`.

                --- EXAMPLES ---
                - push('a', 'perpendicular', true, false, '') → Push a surface straight in like a button
                - pull('b', 'parallel', false, true, 'd') → Slide or rotate a surface around point 'd'

                --- OUTPUT FORMAT ---

                <start_json>
                {{
                "skill_name": "action_targetobject_mechanism",
                "primitive_sequence": [
                    "move_gripper_to_pose('a', true, false)",
                    "push('b', 'perpendicular', true, false, '')",
                    "pull('c', 'parallel', false, true, 'd')",
                    ...
                ],
                "parameters": {{
                    "force_threshold": "low" | "medium" | "high",
                    "precision_required": "low" | "medium" | "high",
                    "speed_requirement": "slow" | "medium" | "high"
                }},
                "prerequisites": ["list of preconditions"],
                "constraints": ["list of safety or collision limits"],
                "explanations": ["detailed explaination for why you chose each primitive and each parameter chosen"]
                }}
                <end_json>

                --- CRITICAL RULES ---
                - DO NOT invent new functions or primitives.
                - ONLY use surface labels for push/pull, point labels for move_gripper_to_pose.
                - All parameters must match physical and geometric properties shown in the images.
                - close/open gripper are the only methods that affect the gripper they must be called individually no other method will close/open the gripper
                - NEVER include markdown, explanation outside of the JSON block, or invalid syntax.
                - Keep in mind you may need to close the gripper before pushing/pulling to interact with an object at the tcp
                - GRIPPER ORIENTATION RULE:
                    - For HORIZONTAL movements (drawers, pushing/pulling along surfaces, sliding): USE SIDE GRASP
                    - For VERTICAL movements (lifting, pressing down): USE TOP-DOWN GRASP
                    - Always orient the gripper's z-axis to maximize force transmission in the intended direction
                - ensure your explainations cover all parameter and action selections in detail verify your understanding of parameters before selection
                - when opening an object ensure its lid/ cover is completely removed from the top for example a bottle cap should be removed using "pull"
                - POINTS ARE COLOR CODED PAY ATTENTION TO WHERE THE POINT ACTUALLY IS
                    in your explaination include why you selected a particular point and how you determined the associated label
                    
                When in doubt, choose the best grounded option based on visible contact affordances and robot camera constraints (e.g., collision with the wrist camera).

                Carefully verify label references and syntax. Output must be complete and syntactically valid.
                """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                Abstract Action: {abstract_action}
                Target Object: {target_object}

                Please analyze the two visualizations:
                - FIRST IMAGE: Surface segments with normal vectors (for push/pull)
                - SECOND IMAGE: Points of interest (for grasping/manipulation)

                Use only the predefined action primitives.
                Output must be a raw JSON object between '{{' and '}}'. No extra text or formatting.
                """
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