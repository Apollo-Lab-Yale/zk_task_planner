# skill_utils.py in cognitive_bt_framework

import cv2
import numpy as np
import time
import base64
import uuid
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any


def create_points_of_interest_visualization(
    color_image: np.ndarray,
    object_info,
    points: List[Tuple[int, int]],
    target_points: List[List[float]] = None
) -> np.ndarray:
    """
    Create visualization of points of interest on the object
    
    Args:
        color_image: RGB image
        object_info: ObjectInfo for the detected object
        points: List of pixel coordinates (x, y)
        target_points: Optional list of 3D target points for execution
        
    Returns:
        Visualization image
    """
    # Start with a copy of the color image
    vis_img = color_image.copy()
    
    # Draw object bounding box and mask if available
    if object_info is not None:
        # Draw bounding box
        if hasattr(object_info, 'bbox') and object_info.bbox is not None:
            x, y, w, h = object_info.bbox
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with ID and name
            alpha_id = object_info.alpha_id if hasattr(object_info, 'alpha_id') and object_info.alpha_id is not None else ""
            obj_text = f"{alpha_id}: {object_info.name}"
            cv2.putText(
                vis_img,
                obj_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Draw mask as overlay
        if hasattr(object_info, 'mask') and object_info.mask is not None:
            mask_overlay = np.zeros_like(vis_img, dtype=np.uint8)
            mask_overlay[object_info.mask] = [0, 100, 0]  # Light green
            vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, 0.3, 0)
    
    # Draw points of interest
    if points is not None and len(points) > 0:
        for i, (px, py) in enumerate(points):
            # Generate point label (alphabetical)
            point_id = chr(97 + i % 26)
            
            # Draw circle for point
            cv2.circle(vis_img, (px, py), 5, (0, 0, 255), -1)
            
            # Draw label
            cv2.putText(
                vis_img,
                point_id,
                (px + 5, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
    
    # Draw action target points if available
    if target_points is not None and len(target_points) > 0:
        # Need to convert 3D points to pixel coordinates
        # This requires camera intrinsics which might not be available here
        # So we'll just use relative positions assuming typical camera parameters
        for i, point_3d in enumerate(target_points):
            # Skip if point is all zeros
            if np.all(np.isclose(point_3d, 0.0, atol=1e-6)):
                continue
                
            # Skip if point is None
            if point_3d is None:
                continue
                
            # Simple projection (would need actual intrinsics for accuracy)
            # Assuming fx=fy=500, cx=width/2, cy=height/2, and z is in meters
            h, w = vis_img.shape[:2]
            cx, cy = w // 2, h // 2
            fx, fy = 500, 500
            
            # Handle different point formats
            if isinstance(point_3d, list) or isinstance(point_3d, tuple):
                if len(point_3d) >= 3:
                    # If it's already in pixel coordinates
                    if abs(point_3d[0]) < w and abs(point_3d[1]) < h:
                        px, py = int(point_3d[0]), int(point_3d[1])
                    else:
                        # It's in 3D world coordinates
                        x, y, z = point_3d[:3]
                        if z > 0:
                            px = int(cx + (x * fx) / z)
                            py = int(cy + (y * fy) / z)
                        else:
                            continue
                else:
                    continue
            else:
                # Numpy array case
                if point_3d.shape[0] >= 3:
                    x, y, z = point_3d[:3]
                    if z > 0:
                        px = int(cx + (x * fx) / z)
                        py = int(cy + (y * fy) / z)
                    else:
                        continue
                else:
                    continue
            
            # Draw target point with different color and style
            cv2.circle(vis_img, (px, py), 8, (255, 0, 0), 2)
            cv2.drawMarker(vis_img, (px, py), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)
            
            # Draw label with action index
            cv2.putText(
                vis_img,
                f"A{i+1}",
                (px + 10, py + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
    
    # Add title
    cv2.putText(
        vis_img,
        "Points of Interest",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    return vis_img


def visualize_points_of_interest(
    image: np.ndarray, 
    object_info,
    points: Dict[str, Any] = None
) -> np.ndarray:
    """
    Create a visualization with two side-by-side images:
    1. Image with surface masks as an overlay
    2. Points of interest visualization
    
    Args:
        image: Input image
        object_info: Object information with surface masks
        points: Dictionary of points information (from detect_regions_of_interest)
        
    Returns:
        Combined side-by-side visualization
    """
    h, w = image.shape[:2]
    
    # Create the first image: Surface masks overlay
    surface_img = image.copy()
    
    # Draw object mask as a semi-transparent overlay
    if object_info.mask is not None:
        mask_overlay = np.zeros_like(surface_img, dtype=np.uint8)
        mask_overlay[object_info.mask] = [0, 255, 0]  # Green for object mask
        surface_img = cv2.addWeighted(surface_img, 1.0, mask_overlay, 0.2, 0)
        
        # Draw object label with alphabetical ID
        alpha_id = object_info.alpha_id if object_info.alpha_id else get_alpha_id(object_info.id)
        # Find centroid of the object mask for label placement
        y_coords, x_coords = np.where(object_info.mask)
        if len(y_coords) > 0:
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            # Add object ID and name at centroid
            cv2.putText(
                surface_img,
                f"{alpha_id}: {object_info.name}",
                (centroid_x, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2
            )
    
    # Draw surface masks with different colors if available
    if object_info.surface_masks is not None and len(object_info.surface_masks) > 0:
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
        
        # Create a separate overlay for all surfaces
        surface_overlay = np.zeros_like(surface_img, dtype=np.uint8)
        
        # Add each surface to the overlay with a different color
        for i, (surface_name, surface_mask) in enumerate(object_info.surface_masks.items()):
            color_idx = i % len(color_map)
            surface_overlay[surface_mask] = color_map[color_idx]
            
            # Find centroid of the surface for label placement
            y_coords, x_coords = np.where(surface_mask)
            if len(y_coords) > 0:
                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))
                
                # Extract alphabetical ID if it's in the format "surface_abc"
                if "_" in surface_name:
                    alpha_id = surface_name.split("_")[1]
                else:
                    alpha_id = get_alpha_id(i + 1)
                
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
        
        # Add surface overlay with transparency
        surface_img = cv2.addWeighted(surface_img, 1.0, surface_overlay, 0.3, 0)
        
        # Add legend for surfaces
        legend_y = 100
        for i, (surface_name, _) in enumerate(object_info.surface_masks.items()):
            color_idx = i % len(color_map)
            color = color_map[color_idx]
            
            # Extract alphabetical ID
            if "_" in surface_name:
                alpha_id = surface_name.split("_")[1]
            else:
                alpha_id = get_alpha_id(i + 1)
                
            # Draw color square
            cv2.rectangle(
                surface_img,
                (10, legend_y),
                (30, legend_y + 20),
                color,
                -1
            )
            
            # Add surface ID and name
            cv2.putText(
                surface_img,
                f"{alpha_id}: Surface {i+1}",
                (40, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )
            
            legend_y += 25
    
    # Add title to the surface image
    cv2.putText(
        surface_img,
        "Surface Segmentation",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Create the second image: Points of interest
    points_img = image.copy()
    
    # Draw object bounding box and mask
    if object_info.bbox is not None:
        x, y, w_box, h_box = object_info.bbox
        cv2.rectangle(points_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    # Draw semi-transparent mask
    if object_info.mask is not None:
        mask_overlay = np.zeros_like(points_img, dtype=np.uint8)
        mask_overlay[object_info.mask] = [0, 100, 0]  # Light green
        points_img = cv2.addWeighted(points_img, 1.0, mask_overlay, 0.3, 0)
    
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
                    thickness=-1
                )
                
                # Draw label (use alphabetical ID)
                if not point_id.isalpha():
                    # If ID is not already alphabetical, generate one
                    point_id = get_alpha_id(i + 1)
                    
                cv2.putText(
                    points_img,
                    point_id,
                    (px + 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Add legend for points
            legend_y = 100
            # Take at most 5 points for the legend to avoid clutter
            sample_indices = np.linspace(0, len(pixel_coords)-1, min(5, len(pixel_coords)), dtype=int)
            for idx in sample_indices:
                score = scores[idx]
                point_id = ids[idx]
                
                # Normalize score for color
                norm_score = (score - min_score) / score_range if score_range > 0 else 0.5
                color = (
                    int(255 * (1 - norm_score)),  # B
                    0,                           # G
                    int(255 * norm_score)         # R
                )
                
                # If ID is not already alphabetical, use a generated one
                if not point_id.isalpha():
                    point_id = get_alpha_id(idx + 1)
                
                # Draw color circle
                cv2.circle(
                    points_img,
                    (15, legend_y),
                    radius=5,
                    color=color,
                    thickness=-1
                )
                
                # Add point ID and score
                cv2.putText(
                    points_img,
                    f"{point_id}: {score:.2f}",
                    (30, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
                legend_y += 20
                
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
                    (px + 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
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
    alpha_id = object_info.alpha_id if object_info.alpha_id else get_alpha_id(object_info.id)
    obj_text = f"Object: {alpha_id} {object_info.name}"
    cv2.putText(
        points_img,
        obj_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
    
    # Create side-by-side image (horizontal concatenation)
    combined_img = np.hstack((surface_img, points_img))
    
    # Add a separator line between the two images
    cv2.line(
        combined_img, 
        (w, 0), 
        (w, h), 
        (255, 255, 255), 
        2
    )
    
    return combined_img


def update_visualizations(
    color_image: Optional[np.ndarray] = None,
    sam_masks: Optional[np.ndarray] = None,
    poi_image: Optional[np.ndarray] = None,
    display_function=None
):
    """
    Update visualization windows with the provided images
    
    Args:
        color_image: Optional RGB image to display
        sam_masks: Optional SAM mask visualization
        poi_image: Optional points of interest visualization
        display_function: Optional function to call for displaying images
    """
    # If no display function provided, use OpenCV's imshow
    if display_function is None:
        def default_display(name, img):
            cv2.imshow(name, img)
            cv2.waitKey(1)
        display_function = default_display
    
    # Display each provided image
    if color_image is not None:
        display_function("Camera Feed", color_image)
    
    if sam_masks is not None:
        display_function("SAM Segmentation", sam_masks)
    
    if poi_image is not None:
        display_function("Points of Interest", poi_image)


def get_alpha_id(num: int) -> str:
    """
    Generate an alphabetical ID from a number
    
    Args:
        num: Input number
    
    Returns:
        Alphabetical ID (a-z, then aa, ab, etc.)
    """
    if num <= 0:
        return 'a'
        
    # Generate ID for numbers 1-26 (a-z)
    if num <= 26:
        return chr(96 + num)
    
    # For larger numbers, generate multi-letter IDs
    result = ''
    while num > 0:
        rem = num % 26
        if rem == 0:
            result = 'z' + result
            num = (num // 26) - 1
        else:
            result = chr(96 + rem) + result
            num = num // 26
    
    return result


def save_image(image: np.ndarray, image_id: str, output_dir: str = "/tmp") -> str:
    """
    Save an image to disk
    
    Args:
        image: Image to save
        image_id: Unique ID for the image
        output_dir: Directory to save to
        
    Returns:
        Path to saved image
    """
    import os
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{image_id}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    return filepath


def encode_image(image: np.ndarray) -> str:
    """
    Encode an image to base64 string
    
    Args:
        image: Image to encode
        
    Returns:
        Base64 encoded string
    """
    # Convert to BGR if needed (OpenCV uses BGR, PIL uses RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_img = Image.fromarray(image_rgb)
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str


def generate_skill_visualization(
    color_image: np.ndarray,
    object_info,
    points_of_interest,
    skill
):
    """
    Generate a comprehensive visualization for a skill
    
    Args:
        color_image: RGB image
        object_info: ObjectInfo for the detected object
        points_of_interest: Points detected on the object
        skill: Generated skill
        
    Returns:
        Visualization image
    """
    h, w = color_image.shape[:2]
    
    # Create the surface/points visualization
    side_by_side = visualize_points_of_interest(
        color_image,
        object_info,
        points_of_interest
    )
    
    # Create skill visualization
    skill_img = color_image.copy()
    
    # Add skill name
    cv2.putText(
        skill_img,
        f"Skill: {skill.name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add primitive sequence
    y_pos = 70
    cv2.putText(
        skill_img,
        "Primitive Sequence:",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    for i, primitive in enumerate(skill.primitive_sequence):
        y_pos += 25
        cv2.putText(
            skill_img,
            f"{i+1}. {primitive}",
            (30, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
    
    # Add parameters
    y_pos += 40
    cv2.putText(
        skill_img,
        "Parameters:",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    for key, value in skill.parameters.items():
        y_pos += 25
        cv2.putText(
            skill_img,
            f"{key}: {value}",
            (30, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
    
    # Combine all visualizations into a final image
    # Place side-by-side on top and skill info below
    
    # Create a blank canvas with proper size
    final_height = side_by_side.shape[0] + skill_img.shape[0]
    final_width = side_by_side.shape[1]
    final_img = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    
    # Copy side-by-side into top portion
    final_img[:side_by_side.shape[0], :side_by_side.shape[1]] = side_by_side
    
    # Copy skill info into bottom portion
    y_offset = side_by_side.shape[0]
    h_skill = min(skill_img.shape[0], final_height - y_offset)
    w_skill = min(skill_img.shape[1], final_width)
    final_img[y_offset:y_offset+h_skill, :w_skill] = skill_img[:h_skill, :w_skill]
    
    # Add separator line
    cv2.line(
        final_img,
        (0, side_by_side.shape[0]),
        (final_width, side_by_side.shape[0]),
        (255, 255, 255),
        2
    )
    
    return final_img