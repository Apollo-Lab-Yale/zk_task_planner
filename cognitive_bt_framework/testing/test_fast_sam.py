import numpy as np
import cv2
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from queue import Queue

# Import FastSAM and RealSense classes
from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMConfig, FastSAMMaskGenerator
from cognitive_bt_framework.src.vision.realsense import Camera
class RealTimePlotter:
    """Class to handle real-time plotting with matplotlib"""
    def __init__(self, fig_size=(15, 8)):
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(1, 3, figsize=fig_size)
        self.fig.canvas.manager.set_window_title('FastSAM with RealSense')
        
        # Initialize empty images
        self.rgb_img = self.axs[0].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.depth_img = self.axs[1].imshow(np.zeros((480, 640, 3), dtype=np.uint8), cmap='jet')
        self.mask_img = self.axs[2].imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Set titles
        self.axs[0].set_title('RGB Image')
        self.axs[1].set_title('Depth Image')
        self.axs[2].set_title('Segmentation Masks')
        
        # Turn off axes
        for ax in self.axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add status text
        self.status_text = self.fig.text(0.5, 0.01, "Initializing...", ha='center')
        self.fps_text = self.fig.text(0.9, 0.01, "FPS: --", ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Keep track of status
        self.running = True
        self.last_update_time = time.time()
        self.frame_count = 0
    
    def update_rgb(self, image):
        """Update RGB image"""
        self.rgb_img.set_data(image)
    
    def update_depth(self, image):
        """Update depth image with colormap"""
        self.depth_img.set_data(image)
    
    def update_masks(self, image):
        """Update mask visualization"""
        self.mask_img.set_data(image)
    
    def update_status(self, text):
        """Update status text"""
        self.status_text.set_text(text)
    
    def update_fps(self):
        """Update FPS counter"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_text.set_text(f"FPS: {fps:.1f}")
            self.last_update_time = current_time


def parse_args():
    parser = argparse.ArgumentParser(description="FastSAM Mask Generator with RealSense Camera Demo")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")
    parser.add_argument("--model_type", type=str, default="FastSAM-s", choices=["FastSAM-s", "FastSAM-x"], 
                        help="FastSAM model type")
    parser.add_argument("--max_image_size", type=int, default=640, help="Maximum image size for processing")
    parser.add_argument("--conf_threshold", type=float, default=0.4, help="Confidence threshold for detection")
    parser.add_argument("--depth_threshold", type=float, default=1.5, 
                        help="Maximum depth in meters for processing")
    parser.add_argument("--prompt_type", type=str, default="everything", 
                        choices=["everything", "points", "boxes", "text"],
                        help="Type of prompting to use for FastSAM")
    parser.add_argument("--prompt_text", type=str, default="", help="Text prompt for text-based prompting")
    parser.add_argument("--camera_width", type=int, default=640, help="RealSense camera width")
    parser.add_argument("--camera_height", type=int, default=480, help="RealSense camera height")
    parser.add_argument("--camera_fps", type=int, default=30, help="RealSense camera FPS")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage for FastSAM model")
    parser.add_argument("--process_every", type=int, default=5, 
                        help="Process every Nth frame with FastSAM")
    return parser.parse_args()


def process_frames(mask_generator, camera, args, result_queue, command_queue):
    """Process frames in a separate thread"""
    frame_count = 0
    
    while True:
        # Check for commands
        if not command_queue.empty():
            command = command_queue.get()
            if command == "stop":
                break
        
        # Get frames from camera
        frames = camera.get_frames()
        if frames is None:
            time.sleep(0.1)
            continue
        
        color_image, depth_image = frames
        
        # Skip if frames are invalid
        if color_image is None or depth_image is None:
            continue
        
        # Process depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # Put frames in result queue
        if not result_queue.full():
            result = {
                'color': color_image.copy(),
                'depth': depth_colormap,
                'masks': None,
                'metadata': None,
                'vis_image': None,
                'status': "Capturing"
            }
            result_queue.put(result)
        
        # Process with FastSAM every N frames
        if frame_count % args.process_every == 0:
            try:
                # Select prompt based on arguments
                prompt_args = {}
                if args.prompt_type == "points":
                    h, w = color_image.shape[:2]
                    prompt_args = {"points": [[w//2, h//2]]}
                elif args.prompt_type == "boxes":
                    h, w = color_image.shape[:2]
                    box_size = min(h, w) // 3
                    box = [
                        w//2 - box_size//2,
                        h//2 - box_size//2,
                        w//2 + box_size//2,
                        h//2 + box_size//2
                    ]
                    prompt_args = {"boxes": [box]}
                elif args.prompt_type == "text":
                    if args.prompt_text:
                        prompt_args = {"text": args.prompt_text}
                    else:
                        args.prompt_type = "everything"
                
                # Generate masks
                start_time = time.time()
                masks, metadata = mask_generator.generate_masks(
                    color_image,
                    prompt_type=args.prompt_type,
                    **prompt_args
                )
                proc_time = time.time() - start_time
                
                # Process depth information
                if camera.depth_scale and depth_image is not None:
                    for mask_id in list(metadata.keys()):
                        mask = masks == mask_id
                        if not mask.any():
                            continue
                        
                        # Extract depth values for this mask
                        region_depths = depth_image[mask]
                        valid_depths = region_depths[region_depths > 0]
                        
                        if len(valid_depths) > 0:
                            # Convert to meters
                            avg_depth_m = np.mean(valid_depths) * camera.depth_scale
                            min_depth_m = np.min(valid_depths) * camera.depth_scale
                            
                            # Add to metadata
                            metadata[mask_id]['avg_depth'] = avg_depth_m
                            metadata[mask_id]['min_depth'] = min_depth_m
                
                # Visualize masks
                vis_image = mask_generator.visualize_masks(
                    color_image,
                    masks,
                    metadata,
                    alpha=0.5
                )
                
                # Update result
                if not result_queue.full():
                    result = {
                        'color': color_image.copy(),
                        'depth': depth_colormap,
                        'masks': masks,
                        'metadata': metadata,
                        'vis_image': vis_image,
                        'status': f"Found {len(metadata)} objects, processing took {proc_time:.2f}s"
                    }
                    result_queue.put(result)
                
            except Exception as e:
                # Handle errors
                if not result_queue.full():
                    result = {
                        'color': color_image.copy(),
                        'depth': depth_colormap,
                        'masks': None,
                        'metadata': None,
                        'vis_image': None,
                        'status': f"Error: {str(e)}"
                    }
                    result_queue.put(result)
        
        frame_count += 1


def update_plot(frame, plotter, result_queue, args, output_dir):
    """Update function for matplotlib animation"""
    if not plotter.running:
        return
    
    try:
        # Get latest result
        if not result_queue.empty():
            result = result_queue.get()
            
            # Update images
            plotter.update_rgb(result['color'])
            plotter.update_depth(result['depth'])
            
            if result['vis_image'] is not None:
                plotter.update_masks(result['vis_image'])
            
            # Update status text
            plotter.update_status(result['status'])
            
            # Save result if needed
            if result['vis_image'] is not None and 'masks' in result and result['masks'] is not None:
                # Only save every 30th processed frame to avoid filling disk
                if plotter.frame_count % 30 == 0:
                    output_path = output_dir / f"frame_{plotter.frame_count:04d}.jpg"
                    cv2.imwrite(str(output_path), cv2.cvtColor(result['vis_image'], cv2.COLOR_RGB2BGR))
        
        # Update FPS
        plotter.update_fps()
        plotter.frame_count += 1
        
        # Redraw figure
        plotter.fig.canvas.draw_idle()
    
    except Exception as e:
        plotter.update_status(f"Plot error: {str(e)}")


def on_close(event, plotter, command_queue):
    """Handle figure close event"""
    plotter.running = False
    command_queue.put("stop")


def main():
    """Main function demonstrating FastSAM with RealSense Camera using Matplotlib"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize RealSense Camera
    print("Initializing RealSense Camera...")
    camera = Camera(
        width=args.camera_width, 
        height=args.camera_height, 
        fps=args.camera_fps,
        debug=True
    )
    
    # Start camera
    if not camera.start():
        print("Failed to start camera. Exiting.")
        return
    
    print("Camera started successfully.")
    
    # Initialize FastSAM
    print("Initializing FastSAM model...")
    device = "cpu" if args.use_cpu else "cuda"
    
    fastsam_config = FastSAMConfig(
        model_type="FastSAM-x",
        max_image_size=640,
        conf_threshold=0.5,  # Lower threshold for better book segmentation
        iou_threshold=0.5,
        retina_masks=True,
        remove_small_regions=False,
        merge_overlapping=False,
        overlap_threshold=0.5,
        min_area=10.0,  # Smaller minimum area to capture book details
        draw_borders=True
    )
    
    mask_generator = FastSAMMaskGenerator(fastsam_config)
    print(f"FastSAM model initialized with {args.model_type} on {device}.")
    
    # Create result queue and command queue for thread communication
    result_queue = Queue(maxsize=5)
    command_queue = Queue()
    
    # Create plotter
    plotter = RealTimePlotter()
    
    # Connect close event
    plotter.fig.canvas.mpl_connect('close_event', 
                                 lambda event: on_close(event, plotter, command_queue))
    
    # Start processing thread
    process_thread = threading.Thread(
        target=process_frames,
        args=(mask_generator, camera, args, result_queue, command_queue),
        daemon=True
    )
    process_thread.start()
    
    # Create animation
    ani = FuncAnimation(
        plotter.fig, 
        lambda frame: update_plot(frame, plotter, result_queue, args, output_dir),
        interval=33,  # ~30 FPS
        cache_frame_data=False
    )
    
    # Set up key event handling
    def on_key(event):
        if event.key == 'q':
            plt.close(plotter.fig)
        elif event.key == 's':
            # Save current figures
            save_time = int(time.time())
            if hasattr(plotter, 'current_vis_image') and plotter.current_vis_image is not None:
                save_path = output_dir / f"snapshot_{save_time}.jpg"
                cv2.imwrite(str(save_path), cv2.cvtColor(plotter.current_vis_image, cv2.COLOR_RGB2BGR))
                plotter.update_status(f"Saved snapshot to {save_path}")
    
    plotter.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show plot (this blocks until window is closed)
    plt.show()
    
    # Clean up
    plotter.running = False
    command_queue.put("stop")
    process_thread.join(timeout=1.0)
    camera.stop()
    print("Camera stopped and application closed")


if __name__ == "__main__":
    main()