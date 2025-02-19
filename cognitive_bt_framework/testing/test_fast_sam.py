import numpy as np
import cv2
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP, FastSAMConfig
import time
import torch

class ColorPalette:
    def __init__(self):
        # Generate fixed colors for consistent visualization
        self.colors = {}
        self.next_color_idx = 0
        # Pre-defined color palette (BGR format)
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
    """Create FastSAM configuration optimized for more detections"""
    return FastSAMConfig(
        model_type="FastSAM-x",
        device="cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.1,
        iou_threshold=0.25,
        clip_threshold=0.6,
        min_area=1.0,
        merge_overlapping=False
    )

def print_controls_and_settings(config, current_query, clear_terminal=True):
    """Print current controls and settings to terminal"""
    if clear_terminal:
        print("\033[H\033[J")  # Clear terminal
    
    print("\n=== FastSAM + CLIP Test Controls ===")
    print("SPACE - Process current frame with FastSAM")
    print("i     - Input new query")
    print("c/C   - Decrease/Increase confidence threshold")
    print("t/T   - Decrease/Increase CLIP threshold")
    print("s     - Save current frame")
    print("q     - Quit")
    
    print("\n=== Current Settings ===")
    print(f"Query:            {current_query if current_query else 'No filter'}")
    print(f"Conf threshold:   {config.conf_threshold:.2f}")
    print(f"IoU threshold:    {config.iou_threshold:.2f}")
    print(f"CLIP threshold:   {config.clip_threshold:.2f}")
    print(f"Min area:         {config.min_area:.1f}")
    print("\nWaiting for input...")

def update_display(image, depth_image, color_palette, processing, last_processed_results, config, current_query):
    """Update the display with current frame and overlay"""
    if processing and last_processed_results is not None:
        labeled_masks, metadata = last_processed_results
        
        # Create visualization with consistent colors
        vis_image = image.copy()
        overlay = np.zeros_like(vis_image, dtype=np.uint8)
        
        for mask_id in np.unique(labeled_masks)[1:]:  # Skip 0 (background)
            if mask_id in metadata:
                mask = labeled_masks == mask_id
                color = color_palette.get_color(mask_id)
                overlay[mask] = color
                
                # Add labels with consistent colors
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
    else:
        vis_image = image.copy()

    # Show depth colormap alongside RGB
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    
    # Stack images horizontally
    display_image = np.hstack((vis_image, depth_colormap))
    
    # Add text overlays
    cv2.putText(
        display_image,
        f"Query: {current_query if current_query else 'No filter'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show configuration parameters
    params_text = [
        f"Conf thresh: {config.conf_threshold}",
        f"IoU thresh: {config.iou_threshold}",
        f"CLIP thresh: {config.clip_threshold}",
        f"Min area: {config.min_area}",
        "Press SPACE to process frame"
    ]
    
    for i, text in enumerate(params_text):
        cv2.putText(
            display_image,
            text,
            (10, 60 + i*25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

    # Show results if available
    if processing and last_processed_results is not None:
        _, metadata = last_processed_results
        cv2.putText(
            display_image,
            f"Found {len(metadata)} masks",
            (10, display_image.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Show the image
    cv2.imshow("FastSAM + CLIP Test", display_image)
    return display_image


def main():
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
        
        # List of queries to test
        queries = [
            "a container",
            "a bottle",
            "a box",
            "a package or container",
            "a rectangular object",
            None  # None will show all segments without CLIP filtering
        ]
        current_query_idx = 0
        
        # Processing state
        processing = False
        last_processed_results = None
        
        # Create window with a larger default size
        window_name = "FastSAM Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Set initial window size (1280 width to accommodate side-by-side display)
        cv2.resizeWindow(window_name, 1280, 480)

        # Print controls
        print("\n=== Controls ===")
        print("SPACE - Process current frame")
        print("n     - Next query")
        print("c/C   - Decrease/Increase confidence threshold")
        print("t/T   - Decrease/Increase CLIP threshold")
        print("s     - Save frame")
        print("+/-   - Increase/Decrease window size")
        print("q     - Quit")
        print(f"\nCurrent query: {queries[current_query_idx]}")

        # Window size state
        current_width = 1280
        current_height = 480

        while True:
            # Get frames
            frames = camera.get_frames()
            if frames is None:
                continue
                
            color_image, depth_image = frames
            
            # Create visualization
            if processing and last_processed_results is not None:
                labeled_masks, metadata = last_processed_results
                
                # Create mask visualization
                vis_image = color_image.copy()
                overlay = np.zeros_like(vis_image, dtype=np.uint8)
                
                for mask_id in np.unique(labeled_masks)[1:]:
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
            else:
                vis_image = color_image.copy()

            # Show depth alongside RGB
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Stack images horizontally
            display_image = np.hstack((vis_image, depth_colormap))
            
            # Add overlay text
            current_query = queries[current_query_idx]
            cv2.putText(
                display_image,
                f"Query: {current_query if current_query else 'No filter'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Show settings
            params_text = [
                f"Conf thresh: {config.conf_threshold:.2f}",
                f"IoU thresh: {config.iou_threshold:.2f}",
                f"CLIP thresh: {config.clip_threshold:.2f}",
                f"Min area: {config.min_area:.1f}",
                "Press SPACE to process",
                "+/- to resize window"
            ]
            
            for i, text in enumerate(params_text):
                cv2.putText(
                    display_image,
                    text,
                    (10, 60 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )

            # Show image
            cv2.imshow(window_name, display_image)
            
            # Handle keyboard input (30ms timeout)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('n'):
                current_query_idx = (current_query_idx + 1) % len(queries)
                processing = False
                print(f"\nSwitched to query: {queries[current_query_idx]}")
            elif key == ord('c'):
                config.conf_threshold = max(0.05, config.conf_threshold - 0.1)
                processing = False
                print(f"\nLowered confidence threshold to {config.conf_threshold:.2f}")
            elif key == ord('p'):
                config.conf_threshold = min(0.9, config.conf_threshold + 0.1)
                processing = False
                print(f"\nIncreased confidence threshold to {config.conf_threshold:.2f}")
            elif key == ord('t'):
                config.clip_threshold = max(0.1, config.clip_threshold - 0.1)
                processing = False
                print(f"\nLowered CLIP threshold to {config.clip_threshold:.2f}")
            elif key == ord('d'):
                config.clip_threshold = min(0.9, config.clip_threshold + 0.1)
                processing = False
                print(f"\nIncreased CLIP threshold to {config.clip_threshold:.2f}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"fastsam_test_{timestamp}.jpg"
                cv2.imwrite(filename, display_image)
                print(f"\nSaved image as {filename}")
            elif key == ord('+') or key == ord('='):
                current_width = int(current_width * 1.2)
                current_height = int(current_height * 1.2)
                cv2.resizeWindow(window_name, current_width, current_height)
                print(f"\nIncreased window size to {current_width}x{current_height}")
            elif key == ord('-') or key == ord('_'):
                current_width = max(640, int(current_width * 0.8))
                current_height = max(480, int(current_height * 0.8))
                cv2.resizeWindow(window_name, current_width, current_height)
                print(f"\nDecreased window size to {current_width}x{current_height}")
            elif key == ord(' '):
                print(f"\nProcessing frame with query: {current_query if current_query else 'No filter'}")
                try:
                    last_processed_results = fastsam.process_image(color_image, current_query)
                    labeled_masks, metadata = last_processed_results
                    
                    # Print detection information
                    print(f"Found {len(metadata)} masks")
                    for mask_id, meta in metadata.items():
                        print(f"Mask {mask_id}:")
                        print(f"  Area: {meta['area']:.1f} pixels")
                        print(f"  CLIP score: {meta.get('clip_score', 'N/A')}")
                        print(f"  Bounding box: {meta['bbox']}")
                    
                    processing = True
                except Exception as e:
                    print(f"Error processing: {e}")
                    processing = False

    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()