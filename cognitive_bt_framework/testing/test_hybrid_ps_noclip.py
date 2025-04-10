#!/usr/bin/env python3
"""
Improved test script for the Hybrid Perception System.
Features:
- Uses both YOLO-World and FastSAM-CLIP for detection and segmentation
- Interactive object switching and parameter adjustment
- Enhanced visualization of detections and segmentations
- Performance monitoring and profiling
"""

import numpy as np
import cv2
import torch
import time
import argparse
import colorsys
from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMMaskGenerator, FastSAMConfig
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.perception_system import PerceptionSystem, get_alpha_id
import torch
#!/usr/bin/env python3
"""
Improved test script for the Perception System.
Features:
- Uses both YOLO-World and FastSAM for detection and segmentation
- Interactive object switching and parameter adjustment
- Enhanced visualization of detections and segmentations with alphabetical labeling
- Performance monitoring and profiling
"""


def wait_for_valid_frame(camera, max_attempts=30):
    """Wait for a valid frame from the camera with timeout"""
    print("Waiting for valid frame...")
    for _ in range(max_attempts):
        frames = camera.get_frames()
        if frames is not None:
            color_image, depth_image = frames
            if color_image is not None and depth_image is not None:
                print("Valid frame received")
                return frames
        time.sleep(0.1)
    raise RuntimeError(f"Failed to get valid frame after {max_attempts} attempts")


def create_optimized_configs():
    """Create optimized configurations for YOLO-World and FastSAM models"""
    # FastSAM configuration
    fast_sam_config = FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_image_size=640,
        conf_threshold=0.25,  # Lower threshold for better book segmentation
        iou_threshold=0.5,
        retina_masks=True,
        remove_small_regions=False,
        merge_overlapping=False,
        overlap_threshold=0.5,
        min_area=10.0,  # Smaller minimum area to capture book details
        draw_borders=True
    )
    
    # YOLO-World configuration
    yolo_model_path = "yolov8x-worldv2.pt"
    
    return yolo_model_path, fast_sam_config


def get_unique_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        # Use HSV color space for more distinct colors
        h = i / n
        s = 0.8
        v = 0.9
        # Convert to RGB and then to BGR (for OpenCV)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def create_segmentation_visualization(image, objects, labeled_masks=None, metadata=None):
    """
    Create a comprehensive visualization showing all objects and segmentations
    
    Args:
        image: Original RGB image
        objects: List of detected ObjectInfo objects
        labeled_masks: Optional labeled mask array from FastSAM
        metadata: Optional metadata dict from FastSAM
        
    Returns:
        Visualization image showing all detections with scores
    """
    # Create a copy of the image for visualization
    seg_vis = image.copy()
    
    # First, draw segmentation masks if available
    if labeled_masks is not None and metadata is not None:
        # Get unique mask IDs (excluding 0 which is background)
        mask_ids = [mid for mid in np.unique(labeled_masks) if mid > 0]
        
        # Generate unique colors for each mask
        colors = get_unique_colors(len(mask_ids))
        color_map = {mask_id: colors[i % len(colors)] for i, mask_id in enumerate(mask_ids)}
        
        # Create a transparent overlay for the masks
        overlay = np.zeros_like(seg_vis, dtype=np.uint8)
        
        # Draw each mask with its color and alpha ID
        for i, mask_id in enumerate(mask_ids):
            mask = labeled_masks == mask_id
            color = color_map[mask_id]
            
            # Add mask to overlay
            overlay[mask] = color
            
            # Find position for the label (center of mass of the mask)
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # Get alpha ID and area if available
                alpha_id = metadata.get(mask_id, {}).get('alpha_id', get_alpha_id(mask_id))
                area = metadata.get(mask_id, {}).get('area', 0)
                
                # Create compact label
                label = f"{alpha_id}:{area:.0f}"
                
                # Use smaller font size for the label
                font_scale = 0.4
                font_thickness = 1
                
                # Get text size for potential overlap detection
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Add a small semi-transparent background for the text
                text_bg = seg_vis.copy()
                cv2.rectangle(
                    text_bg,
                    (center_x - text_width//2 - 2, center_y - text_height - 2),
                    (center_x + text_width//2 + 2, center_y + 2),
                    color,
                    -1  # Filled rectangle
                )
                seg_vis = cv2.addWeighted(seg_vis, 0.7, text_bg, 0.3, 0)
                
                # Add the label
                cv2.putText(
                    seg_vis,
                    label,
                    (center_x - text_width//2, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    font_thickness
                )
        
        # Blend the mask overlay with the image
        seg_vis = cv2.addWeighted(seg_vis, 0.7, overlay, 0.3, 0)
    
    # Now draw all object bounding boxes
    if objects:
        # Generate colors for each object
        obj_colors = get_unique_colors(len(objects))
        
        for i, obj in enumerate(objects):
            # Draw bounding box
            x, y, w, h = obj.bbox
            color = obj_colors[i % len(obj_colors)]
            cv2.rectangle(seg_vis, (x, y), (x + w, y + h), color, 2)
            
            # Add label with confidence and alpha ID
            alpha_id = obj.alpha_id if hasattr(obj, 'alpha_id') and obj.alpha_id else ""
            label = f"{alpha_id}: {obj.name} ({obj.confidence:.2f})"
            
            # Add text with background for better visibility
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                seg_vis,
                (x, y - text_size[1] - 5),
                (x + text_size[0], y),
                color,
                -1
            )
            
            cv2.putText(
                seg_vis,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
            # Draw component boxes and labels if available
            if obj.components:
                for comp_id, comp in obj.components.items():
                    cx, cy, cw, ch = comp.bbox
                    comp_color = (255, 0, 0)  # Blue for components
                    
                    cv2.rectangle(seg_vis, (cx, cy), (cx + cw, cy + ch), comp_color, 1)
                    
                    # Add component label with alpha_id
                    alpha_id = comp.alpha_id if hasattr(comp, 'alpha_id') and comp.alpha_id else ""
                    comp_name = comp.name if hasattr(comp, 'name') else f"part_{comp_id}"
                    comp_label = f"{alpha_id}: {comp_name} ({comp.confidence:.2f})"
                    cv2.putText(
                        seg_vis,
                        comp_label,
                        (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        comp_color,
                        1
                    )
    
    # Add a title
    cv2.putText(
        seg_vis,
        "All Detections and Segmentations",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return seg_vis


def create_sam_mask_visualization(image, roi_coords, labeled_masks, metadata):
    """
    Create a visualization showing all masks generated by SAM in the region of interest
    
    Args:
        image: Original RGB image
        roi_coords: (x, y, w, h) coordinates of the region of interest
        labeled_masks: Labeled mask array from FastSAM
        metadata: Metadata dict from FastSAM
    
    Returns:
        Visualization image showing all masks with different colors
    """
    # Extract ROI coordinates
    x_roi, y_roi, w_roi, h_roi = roi_coords
    
    # Create a copy of the original image
    vis = image.copy()
    
    # Draw a rectangle around the ROI
    cv2.rectangle(vis, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (255, 255, 0), 2)
    
    # Get unique mask IDs (excluding 0 which is background)
    mask_ids = [mid for mid in np.unique(labeled_masks) if mid > 0]
    
    # Generate unique colors for each mask
    colors = get_unique_colors(len(mask_ids))
    
    # Create a separate visualization just for the ROI to show masks clearly
    roi_vis = image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
    mask_overlay = np.zeros_like(roi_vis, dtype=np.uint8)
    
    # Draw each mask with a different color
    for i, mask_id in enumerate(mask_ids):
        # Get mask within the ROI
        mask = labeled_masks == mask_id
        color = colors[i % len(colors)]
        
        # Add mask to overlay with semi-transparency
        mask_overlay[mask] = color
        
        # Get alpha ID and area if available
        alpha_id = metadata.get(mask_id, {}).get('alpha_id', get_alpha_id(mask_id))
        area = metadata.get(mask_id, {}).get('area', 0)
        
        # Find position for the label (center of mass of the mask)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            
            # Add label with ID and area
            label = f"{alpha_id}:{area:.0f}"
            cv2.putText(
                mask_overlay,
                label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
    
    # Blend the mask overlay with the ROI image
    roi_vis = cv2.addWeighted(roi_vis, 0.7, mask_overlay, 0.3, 0)
    
    # Create a title for the ROI visualization
    title = f"SAM Masks - {len(mask_ids)} segments detected"
    cv2.putText(
        roi_vis,
        title,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    # Insert the ROI visualization back into a full-sized image
    # First create a black canvas of the original image size
    full_vis = np.zeros_like(image)
    
    # Scale the ROI visualization to fit in the upper half of the screen
    target_height = full_vis.shape[0] // 2
    scale = target_height / h_roi if h_roi > 0 else 1.0
    target_width = int(w_roi * scale)
    
    if target_width > 0 and target_height > 0:
        scaled_roi_vis = cv2.resize(roi_vis, (target_width, target_height))
        
        # Position in upper center
        start_x = max(0, (full_vis.shape[1] - target_width) // 2)
        
        # Place the scaled ROI visualization
        full_vis[0:target_height, start_x:start_x+target_width] = scaled_roi_vis
    
    # Add instruction text
    cv2.putText(
        full_vis,
        "SAM Segmentation Masks in Region of Interest",
        (10, full_vis.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    cv2.putText(
        full_vis,
        f"Object: {len(mask_ids)} masks detected",
        (10, full_vis.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )
    
    return full_vis

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Perception System')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--conf', type=float, default=0.5, help='Default confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Initialize camera
    camera = Camera(width=args.width, height=args.height)
    if not camera.start():
        print("Failed to start camera")
        return
    
    try:
        # Wait for valid frame
        try:
            frames = wait_for_valid_frame(camera)
        except RuntimeError as e:
            print(e)
            return
        
        # Get camera matrix from RealSense intrinsics
        camera_matrix = np.array([
            [camera.intrinsics.fx, 0, camera.intrinsics.ppx],
            [0, camera.intrinsics.fy, camera.intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Initialize perception system
        yolo_model_path, fast_sam_config = create_optimized_configs()
        perception = PerceptionSystem(
            fast_sam_config=fast_sam_config,
            yolo_model_path=yolo_model_path,
            camera_matrix=camera_matrix,
            depth_scale=camera.depth_scale,
            debug=True
        )
        
        # Create direct access to FastSAM for raw segmentation
        fastsam = FastSAMMaskGenerator(fast_sam_config)
        
        # Get object classes from user or use defaults
        user_input = input("Enter objects to detect (comma-separated) or press Enter for defaults: ").strip()
        if user_input:
            target_objects = [obj.strip() for obj in user_input.split(",")]
        else:
            target_objects = ["person", "bottle", "laptop", "cup", "keyboard", "mouse", "book", "cell phone"]
        
        # Get confidence threshold
        try:
            conf_threshold = float(input("Enter confidence threshold (0.0-1.0) or press Enter for default 0.5: ").strip() or "0.5")
            conf_threshold = max(0.0, min(1.0, conf_threshold))  # Clamp to valid range
        except ValueError:
            conf_threshold = args.conf
            print(f"Invalid input, using default threshold of {conf_threshold}")
        
        # Set initial operating modes
        detection_mode = "objects"  # Options: "objects", "object", "parts"
        current_obj_idx = 0
        selected_object = None
        
        # Print instructions
        print(f"\nLooking for: {', '.join(target_objects)} with confidence threshold {conf_threshold}")
        print("Detection Modes:")
        print("  - 'objects': Detect all objects in the scene")
        print("  - 'object': Focus on a single object type")
        print("  - 'parts': Segment parts within a selected object")
        print("\nControls:")
        print("  - 'm': Switch between detection modes")
        print("  - 'n': Next object")
        print("  - 'p': Previous object")
        print("  - 'a': Analyze parts in the selected object")
        print("  - '+'/'-': Adjust confidence threshold")
        print("  - 's': Save current frame")
        print("  - 'd': Print detailed performance stats")
        print("  - 'r': Reset performance profiler")
        print("  - 'q': Quit")
        
        # Tracking variables
        save_enabled = False
        show_profiling = False
        frame_count = 0
        fps_history = []
        fps_smoothing = 10  # Number of frames to average for FPS calculation
        profiling_interval = 30  # Print profiling stats every N frames
        
        # Create windows
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("SAM Masks", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Interest Points", cv2.WINDOW_NORMAL)
        point_method = "orb"  # Default method for point detection
        point_methods = ["harris", "shi_tomasi", "sift", "orb", "fast"]
        max_points = 20
        # Main loop
        while True:
            # Get frames from camera
            frames = camera.get_frames()
            if frames is None:
                print("Failed to get frames")
                continue
            
            color_image, depth_image = frames
            frame_count += 1
            
            # Reset the profiler periodically for benchmarking
            if frame_count % 100 == 1:
                perception.reset_profiler()
                if args.debug:
                    print("\nReset profiler - starting new benchmark")
            
            # Start timing
            start_time = time.time()
            raw_segmentation_results = None
            objects = []
            
            # Detection logic based on current mode
            if detection_mode == "objects":
                # Detect all objects in the scene
                objects = perception.detect_objects(
                    image=color_image,
                    classes=target_objects,
                    conf=conf_threshold,
                    segment=True,
                    depth_image=depth_image
                )
                
                # Update the selected object if needed
                if objects and (selected_object is None or current_obj_idx >= len(objects)):
                    selected_object = objects[0]
                    current_obj_idx = 0
                elif objects:
                    selected_object = objects[current_obj_idx]
                
            elif detection_mode == "object":
                # Focus on a single object type
                current_target = target_objects[current_obj_idx]
                
                # Get the object detection
                object_info = perception.detect_object(
                    target_object=current_target,
                    image=color_image,
                    conf=conf_threshold,
                    segment=True,
                    depth_image=depth_image
                )
                if object_info is not None:
                    objects = [object_info]
                    selected_object = object_info
                else:
                    objects = []
                    selected_object = None
                
            elif detection_mode == "parts" and selected_object is not None:
                # Always keep showing the parent object in parts mode - set this first
                objects = [selected_object]
                
                # Part detection mode - segment all parts within the selected object
                if hasattr(perception, 'detect_object_parts'):
                    try:
                        parts = perception.detect_object_parts(
                            image=color_image,
                            bbox=selected_object.bbox,
                            conf_threshold=conf_threshold,
                            depth_image=depth_image
                        )
                        
                        # Store parts as components in the selected object
                        selected_object.components = parts
                        
                        # Get raw segmentation for visualization
                        x, y, w, h = selected_object.bbox
                        # Add padding
                        pad = int(max(w, h) * 0.05)
                        x_roi = max(0, x - pad)
                        y_roi = max(0, y - pad)
                        w_roi = min(color_image.shape[1] - x_roi, w + 2*pad)
                        h_roi = min(color_image.shape[0] - y_roi, h + 2*pad)
                        
                        # Extract ROI
                        roi = color_image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                        
                        # Generate raw segmentation
                        if roi.size > 0:
                            try:
                                labeled_masks, metadata = fastsam.generate_masks(
                                    roi,
                                    prompt_type="everything"
                                )
                                raw_segmentation_results = (labeled_masks, metadata, (x_roi, y_roi, w_roi, h_roi))
                            except Exception as e:
                                print(f"Error generating raw segmentation: {e}")
                                raw_segmentation_results = None
                    except Exception as e:
                        print(f"Error detecting parts: {e}")
                        # Ensure we still have the parent object even if part detection fails
                        raw_segmentation_results = None
                
                # Display parts info
                if hasattr(selected_object, 'components') and selected_object.components:
                    print(f"\nDetected {len(selected_object.components)} parts in {selected_object.name}")
                    for part_id, part in selected_object.components.items():
                        alpha_id = part.alpha_id if hasattr(part, 'alpha_id') else ""
                        print(f"  - Part {alpha_id}: Area={part.bbox[2] * part.bbox[3]} pixels")
                        if hasattr(part, 'pose') and part.pose is not None:
                            x, y, z = part.pose[:3]
                            print(f"    Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
                else:
                    print(f"\rAnalyzing parts in {selected_object.name}...", end="")
            
            # Calculate process time
            process_time = time.time() - start_time
            
            # Calculate FPS with smoothing
            fps_history.append(1.0/process_time if process_time > 0 else 0)
            if len(fps_history) > fps_smoothing:
                fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Display summary of detected objects
            if objects:
                print("\nDetected objects:")
                for i, obj in enumerate(objects):
                    alpha_id = obj.alpha_id if hasattr(obj, 'alpha_id') else ""
                    status = " (selected)" if obj == selected_object else ""
                    print(f"{i} [{alpha_id}]: {obj.name} - Conf: {obj.confidence:.3f}{status}")
                    if hasattr(obj, 'pose') and obj.pose is not None:
                        x, y, z = obj.pose[:3]
                        print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters")
                    
                    # Show component info if available
                    if hasattr(obj, 'components') and obj.components:
                        print(f"  Parts:")
                        for part_id, part in obj.components.items():
                            alpha_id = part.alpha_id if hasattr(part, 'alpha_id') else ""
                            print(f"  - Part {alpha_id}: Conf: {part.confidence:.3f}")
            
            # Print performance stats periodically or when manually requested
            if show_profiling or frame_count % profiling_interval == 0:
                # Print top 5 hotspots in compact format
                print("\n--- Performance Hotspots ---")
                perception.print_performance_summary(sort_by="current_total", top_n=5, compact=True)
                show_profiling = False
            
            # Create main detection visualization
            if objects:
                detection_vis = perception.visualize_detections(
                    image=color_image,
                    objects=objects,
                    show_masks=True,
                    show_parts=True,
                    show_poses=True
                )
            else:
                # Create a default visualization if no objects are detected
                detection_vis = color_image.copy()
                cv2.putText(
                    detection_vis,
                    "No objects detected",
                    (int(color_image.shape[1]/2 - 100), int(color_image.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            
            # Create segmentation visualization
            if raw_segmentation_results is not None:
                labeled_masks, metadata, roi_coords = raw_segmentation_results
                
                # Create a mask for the full image
                full_masks = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.int32)
                x_roi, y_roi, w_roi, h_roi = roi_coords
                
                # Place the ROI masks in the correct position
                try:
                    full_masks[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = labeled_masks
                except ValueError as e:
                    print(f"Error placing masks in full image: {e}")
                
                # Create full metadata
                full_metadata = {}
                for mask_id, meta in metadata.items():
                    # Update bbox to full image coordinates
                    if 'bbox' in meta:
                        bbox = meta['bbox']
                        full_bbox = [
                            bbox[0] + x_roi,
                            bbox[1] + y_roi,
                            bbox[2],
                            bbox[3]
                        ]
                        meta['bbox'] = full_bbox
                    full_metadata[mask_id] = meta
                
                # Create visualization
                seg_vis = create_segmentation_visualization(
                    color_image, 
                    objects, 
                    full_masks, 
                    full_metadata
                )
            else:
                # Just use the regular detection visualization without running FastSAM on the whole image
                seg_vis = create_segmentation_visualization(color_image, objects)
            
            # Create SAM mask visualization if in parts mode and raw segmentation results are available
            sam_mask_vis = None
            if detection_mode == "parts" and raw_segmentation_results is not None:
                labeled_masks, metadata, roi_coords = raw_segmentation_results
                
                # Create the SAM mask visualization
                sam_mask_vis = create_sam_mask_visualization(
                    color_image, 
                    roi_coords, 
                    labeled_masks, 
                    metadata
                )
            else:
                # Create an empty visualization or message when not in parts mode
                sam_mask_vis = np.zeros_like(color_image)
                cv2.putText(
                    sam_mask_vis,
                    "Switch to parts mode to see SAM masks",
                    (int(color_image.shape[1]/2 - 200), int(color_image.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            interest_points_vis = None
            if selected_object is not None:
                try:
                    # Check if our method exists
                    if hasattr(perception, 'detect_regions_of_interest'):
                        interest_points = perception.detect_regions_of_interest(
                            image=color_image,
                            obj_info=selected_object,
                            method=point_method,
                            max_points=max_points,
                            visualize=True
                        )
                        
                        # Store visualization
                        if 'visualization' in interest_points:
                            interest_points_vis = interest_points['visualization']
                            last_poi_frame = frame_count
                            
                            # Print POI info only in points mode or occasionally
                            if detection_mode == "points" or frame_count % (poi_update_interval * 5) == 0:
                                print(f"\nDetected {len(interest_points.get('keypoints', []))} interest points with {point_method.upper()} method")
                    else:
                        if detection_mode == "points":
                            print(f"\rPoint detection method not available in PerceptionSystem", end="")
                except Exception as e:
                    if detection_mode == "points":
                        print(f"Error detecting interest points: {e}")
            # Add mode and status information to main image
            mode_text = f"Mode: {detection_mode.capitalize()}"
            if detection_mode == "object":
                mode_text += f" - Target: {target_objects[current_obj_idx]}"
            elif detection_mode == "parts" and selected_object is not None:
                mode_text += f" - Object: {selected_object.name}"
            
            cv2.putText(
                detection_vis,
                mode_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            # Add confidence threshold info
            cv2.putText(
                detection_vis,
                f"Threshold: {conf_threshold:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Add FPS counter
            cv2.putText(
                detection_vis,
                f"FPS: {fps:.1f}",
                (10, detection_vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            if interest_points_vis is None:
                interest_points_vis = np.zeros_like(color_image)
                if selected_object is None:
                    msg = "Select an object first"
                else:
                    msg = f"Points of interest for {selected_object.name}"
                
                cv2.putText(
                    interest_points_vis,
                    msg,
                    (int(color_image.shape[1]/2 - 200), int(color_image.shape[0]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # Add information about the point method
            if interest_points_vis is not None and selected_object is not None:
                method_text = f"Method: {point_method.upper()}, Max Points: {max_points}"
                cv2.putText(
                    interest_points_vis,
                    method_text,
                    (10, interest_points_vis.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
            # Show the visualizations
            cv2.imshow("Detection", detection_vis)
            cv2.imshow("Segmentation", seg_vis)
            cv2.imshow("SAM Masks", sam_mask_vis)
            cv2.imshow("Points of Interest", interest_points_vis)
            
            # Save images if requested
            if save_enabled:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                detection_filename = f"detection_{timestamp}.jpg"
                segmentation_filename = f"segmentation_{timestamp}.jpg"
                sam_masks_filename = f"sam_masks_{timestamp}.jpg"
                
                cv2.imwrite(detection_filename, detection_vis)
                cv2.imwrite(segmentation_filename, seg_vis)
                cv2.imwrite(sam_masks_filename, sam_mask_vis)
                
                print(f"\nSaved images as {detection_filename}, {segmentation_filename}, and {sam_masks_filename}")
                save_enabled = False
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                break
                
            elif key == ord('m'):
                # Switch detection mode
                if detection_mode == "objects":
                    detection_mode = "object"
                    print(f"\nSwitched to single object mode - Target: {target_objects[current_obj_idx]}")
                elif detection_mode == "object":
                    if selected_object is not None:
                        detection_mode = "parts"
                        print(f"\nSwitched to part detection mode for {selected_object.name}")
                    else:
                        detection_mode = "objects"
                        print("\nSwitched to multi-object detection mode")
                else:  # parts mode
                    detection_mode = "objects"
                    print("\nSwitched to multi-object detection mode")
                    
            elif key == ord('n'):
                # Next object
                if detection_mode == "object":
                    current_obj_idx = (current_obj_idx + 1) % len(target_objects)
                    print(f"\nSwitched to target: {target_objects[current_obj_idx]}")
                elif detection_mode == "objects" and objects:
                    current_obj_idx = (current_obj_idx + 1) % len(objects)
                    selected_object = objects[current_obj_idx]
                    alpha_id = selected_object.alpha_id if hasattr(selected_object, 'alpha_id') else ""
                    print(f"\nSelected object {current_obj_idx} [{alpha_id}]: {selected_object.name}")
                    
            elif key == ord('p'):
                # Previous object
                if detection_mode == "object":
                    current_obj_idx = (current_obj_idx - 1) % len(target_objects)
                    print(f"\nSwitched to target: {target_objects[current_obj_idx]}")
                elif detection_mode == "objects" and objects:
                    current_obj_idx = (current_obj_idx - 1) % len(objects)
                    selected_object = objects[current_obj_idx]
                    alpha_id = selected_object.alpha_id if hasattr(selected_object, 'alpha_id') else ""
                    print(f"\nSelected object {current_obj_idx} [{alpha_id}]: {selected_object.name}")
                    
            elif key == ord('a'):
                # Analyze parts for selected object
                if selected_object is not None and detection_mode != "parts":
                    detection_mode = "parts"
                    print(f"\nAnalyzing parts for {selected_object.name}")
                elif detection_mode == "parts":
                    detection_mode = "objects" if len(objects) > 1 else "object"
                    print(f"\nExiting part analysis mode")
                else:
                    print("\nNo object selected for part analysis")
                    
            elif key == ord('+') or key == ord('='):
                # Increase confidence threshold
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"\nIncreased confidence threshold to {conf_threshold:.2f}")
                
            elif key == ord('-') or key == ord('_'):
                # Decrease confidence threshold
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"\nDecreased confidence threshold to {conf_threshold:.2f}")
                
            elif key == ord('s'):
                # Enable image saving on next frame
                save_enabled = True
                print("\nSaving next frame...")
                
            elif key == ord('d'):
                # Show detailed performance stats
                show_profiling = True
                print("\n=== Detailed Performance Summary ===")
                perception.print_performance_summary(sort_by="current_total", compact=False)
                
            elif key == ord('r'):
                # Reset profiler
                perception.reset_profiler()
                print("\nPerformance profiler has been reset")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Print final performance summary
        if perception.debug:
            print("\n=== Final Performance Summary ===")
            perception.print_performance_summary(sort_by="total", compact=False)
            
        # Release resources
        camera.stop()
        perception.release()
        cv2.destroyAllWindows()
        print("\nApplication terminated")

if __name__ == "__main__":
    main()