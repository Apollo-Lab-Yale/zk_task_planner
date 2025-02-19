import numpy as np
import cv2
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP, FastSAMConfig
import time
import torch

class ColorPalette:
    def __init__(self):
        self.colors = {}
        self.next_color_idx = 0
        self.palette = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 255, 0),  # Light green
            (0, 128, 255),  # Orange
            (255, 0, 128)   # Purple
        ]
    
    def get_color(self, mask_id):
        if mask_id not in self.colors:
            self.colors[mask_id] = self.palette[self.next_color_idx % len(self.palette)]
            self.next_color_idx += 1
        return self.colors[mask_id]

def create_optimized_config():
    return FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.1,
        iou_threshold=0.25,
        clip_threshold=0.6,
        min_area=1.0,
        merge_overlapping=False
    )

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

def process_frame(color_image, depth_image, query, fastsam, color_palette):
    """Process a single frame and create visualization"""
    # Process image with FastSAM
    print(f"Processing frame with query: {query}")
    results = fastsam.process_image(color_image, query)
    if results is None:
        return None
        
    labeled_masks, metadata = results
    
    # Create visualization
    vis_image = color_image.copy()
    overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    # Draw masks and labels
    for mask_id in np.unique(labeled_masks)[1:]:  # Skip 0 (background)
        if mask_id in metadata:
            mask = labeled_masks == mask_id
            color = color_palette.get_color(mask_id)
            overlay[mask] = color
            
            # Add label with score
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                label = f"r{mask_id}"
                if 'clip_score' in metadata[mask_id]:
                    label += f" ({metadata[mask_id]['clip_score']:.2f})"
                
                cv2.putText(
                    vis_image,
                    label,
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
    
    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    # Create depth visualization
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    
    # Stack images horizontally
    display_image = np.hstack((vis_image, depth_colormap))
    
    # Add query overlay
    cv2.putText(
        display_image,
        f"Query: {query}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Print detection information
    print(f"\nDetection Results:")
    print(f"Found {len(metadata)} masks")
    for mask_id, meta in metadata.items():
        print(f"Mask {mask_id}:")
        print(f"  Area: {meta['area']:.1f} pixels")
        print(f"  CLIP score: {meta.get('clip_score', 'N/A')}")
        print(f"  Bounding box: {meta['bbox']}")
    
    return display_image, results

def main():
    # Get query from user
    query = input("Enter object to detect: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    # Initialize camera
    camera = Camera(width=640, height=480)
    if not camera.start():
        print("Failed to start camera")
        return

    try:
        # Initialize systems
        config = create_optimized_config()
        fastsam = FastSAMWithCLIP(config)
        color_palette = ColorPalette()
        
        # Wait for valid frame
        try:
            frames = wait_for_valid_frame(camera)
        except RuntimeError as e:
            print(e)
            return
            
        color_image, depth_image = frames
        
        # Process frame
        result = process_frame(color_image, depth_image, query, fastsam, color_palette)
        if result is None:
            print("Failed to process frame")
            return
            
        display_image, (labeled_masks, metadata) = result
        
        # Save result
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"fastsam_result_{timestamp}.jpg"
        cv2.imwrite(filename, display_image)
        print(f"\nSaved result as {filename}")
        
        # Show result briefly
        cv2.imshow("Result", display_image)
        cv2.waitKey(3000)  # Show for 3 seconds

    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()