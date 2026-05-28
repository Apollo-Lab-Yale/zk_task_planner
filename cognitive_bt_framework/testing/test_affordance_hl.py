#!/usr/bin/env python3
"""
Test script for visualizing action highlighting with RealSense camera.
This script simulates the neural network action detection for testing purposes.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageEnhance

# Import the RealSense camera
from cognitive_bt_framework.src.vision.realsense import Camera

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test RealSense action highlighting")
parser.add_argument("--width", type=int, default=640, help="Camera width")
parser.add_argument("--height", type=int, default=480, help="Camera height")
parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
parser.add_argument("--viz_size", type=int, default=256, help="Visualization size")
parser.add_argument("--out", default=None, help="Output directory for saved frames")
parser.add_argument("--mode", default="demo", choices=["demo", "manual"], 
                    help="Mode: 'demo' for automatic actions, 'manual' for keyboard-triggered")
args = parser.parse_args()

# Define actions and colors (similar to original script)
ACTIONS = ['hold', 'rotate', 'push']
COLORS = ['red', 'green', 'blue']
COLOR_MAP = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
            'cyan': [0, 1, 1], 'magenta': [1, 0, 1], 'yellow': [1, 1, 0]}

# Utility functions for visualization
def resize_tensor(tensor, sz):
    """Resize tensor to specified size"""
    tensor = F.interpolate(tensor, (sz, sz), mode='bilinear', align_corners=True)
    return tensor

def blur(tensor, sz, Z):
    """Apply Gaussian blur to tensor"""
    tensor = tensor.permute(1, 2, 0).numpy()
    k_size = int(np.sqrt(sz**2) / Z)
    if k_size % 2 == 0:
        k_size += 1
    tensor = cv2.GaussianBlur(tensor, (k_size, k_size), 0)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1)
    return tensor

def generate_color_map(hmaps, colors, sz):
    """Generate color map from heatmaps"""
    colors = [COLOR_MAP[c] for c in colors]
    colors = 1 - torch.FloatTensor(colors).unsqueeze(2).unsqueeze(2)  # invert colors

    vals, idx = torch.sort(hmaps, 0, descending=True)
    cmap = torch.zeros(hmaps.shape)
    for c in range(hmaps.shape[0]):
        cmap[c][idx[0] == c] = vals[0][idx[0] == c]

    cmap = cmap.unsqueeze(1).expand(cmap.shape[0], 3, cmap.shape[-1], cmap.shape[-1])
    cmap = [hmap * color for hmap, color in zip(cmap, colors)]
    cmap = torch.stack(cmap, 0)

    cmap = resize_tensor(cmap, sz)
    cmap, _ = cmap.max(0)

    # Blur the heatmap to make it smooth
    cmap = blur(cmap, sz, 9)
    cmap = 1 - cmap  # Invert heatmap: white background

    # Improve contrast for visibility
    cmap = transforms.ToPILImage()(cmap)
    cmap = ImageEnhance.Color(cmap).enhance(1.5)
    cmap = ImageEnhance.Contrast(cmap).enhance(1.5)
    cmap = transforms.ToTensor()(cmap)

    return cmap

def overlay_heatmaps(img_tensor, hmaps, colors, sz):
    """Overlay heatmaps on image tensor"""
    # Generate color map from heatmaps
    cmap = generate_color_map(hmaps, colors, sz)

    # Generate per-pixel alpha channel and overlay
    alpha = (1 - cmap).mean(0)
    overlay = (1 - alpha) * img_tensor + alpha * cmap

    return overlay

def generate_demo_heatmaps(frame, sz=256, mode="demo", active_actions=None):
    """
    Generate simulated heatmaps for demonstration
    
    In demo mode, generates automatic heatmap patterns
    In manual mode, uses active_actions to highlight specific regions
    """
    # Convert frame to tensor (normalize to 0-1)
    frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
    
    # Create a heatmap tensor for each action
    num_actions = len(ACTIONS)
    heatmaps = torch.zeros((num_actions, sz, sz))
    
    if mode == "demo":
        # Demo mode: Create simulated patterns that move over time
        t = time.time() * 0.5  # Time-based animation
        
        # Hold action (centered circle)
        center_x = sz // 2 + int(np.sin(t) * sz * 0.3)
        center_y = sz // 2 + int(np.cos(t) * sz * 0.2)
        for x in range(sz):
            for y in range(sz):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                heatmaps[0, y, x] = max(0, 1 - dist / (sz * 0.15))
        
        # Rotate action (arc)
        center_x, center_y = sz // 2, sz // 2
        angle = t % (2 * np.pi)
        for x in range(sz):
            for y in range(sz):
                dx, dy = x - center_x, y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                if sz * 0.2 < dist < sz * 0.4:
                    # Calculate angle of this pixel
                    pix_angle = np.arctan2(dy, dx) % (2 * np.pi)
                    # Check if pixel is in the current arc
                    angle_diff = min((pix_angle - angle) % (2 * np.pi), 
                                    (angle - pix_angle) % (2 * np.pi))
                    if angle_diff < 0.8:
                        heatmaps[1, y, x] = max(0, 1 - angle_diff / 0.8)
        
        # Push action (direction-based)
        direction_x = np.sin(t * 0.7)
        direction_y = np.cos(t * 0.7)
        for x in range(sz):
            for y in range(sz):
                nx, ny = (x / sz) * 2 - 1, (y / sz) * 2 - 1  # Normalize to -1, 1
                alignment = (nx * direction_x + ny * direction_y) * 0.5 + 0.5
                dist_from_edge = 1.0 - max(abs(nx), abs(ny))
                heatmaps[2, y, x] = alignment * dist_from_edge * 2
    
    else:  # Manual mode
        if active_actions:
            # Only activate specified actions with simple patterns
            for action_idx in active_actions:
                if action_idx == 0:  # Hold - center
                    center_x, center_y = sz // 2, sz // 2
                    for x in range(sz):
                        for y in range(sz):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            heatmaps[0, y, x] = max(0, 1 - dist / (sz * 0.25))
                
                elif action_idx == 1:  # Rotate - right side
                    for x in range(sz):
                        for y in range(sz):
                            nx = x / sz  # Normalize to 0-1
                            if nx > 0.6:  # Right side
                                heatmaps[1, y, x] = (nx - 0.6) / 0.4
                
                elif action_idx == 2:  # Push - bottom
                    for x in range(sz):
                        for y in range(sz):
                            ny = y / sz  # Normalize to 0-1
                            if ny > 0.6:  # Bottom part
                                heatmaps[2, y, x] = (ny - 0.6) / 0.4

    # Apply smoothing to heatmaps
    for i in range(num_actions):
        heatmaps[i] = torch.from_numpy(
            cv2.GaussianBlur(heatmaps[i].numpy(), (15, 15), 0)
        )
        # Normalize
        if heatmaps[i].max() > 0:
            heatmaps[i] = heatmaps[i] / heatmaps[i].max()
    
    return heatmaps

def convert_frame_to_tensor(frame, target_size):
    """Convert OpenCV frame to PyTorch tensor with resizing"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL for consistent resizing
    pil_img = Image.fromarray(frame_rgb)
    
    # Resize
    pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert to tensor (0-1 range)
    img_tensor = transforms.ToTensor()(pil_img)
    
    return img_tensor

def process_frame(frame, sz=256, mode="demo", active_actions=None):
    """Process a frame to create visualization with heatmaps"""
    # Convert frame to tensor and resize
    img_tensor = convert_frame_to_tensor(frame, sz)
    
    # Generate simulated heatmaps
    heatmaps = generate_demo_heatmaps(frame, sz, mode, active_actions)
    
    # Create overlay
    overlay = overlay_heatmaps(img_tensor, heatmaps, COLORS, sz)
    
    # Create side-by-side visualization
    viz_tensors = [img_tensor, overlay]
    grid = torchvision.utils.make_grid(viz_tensors, nrow=2, padding=4)
    
    # Convert back to numpy for OpenCV display
    grid_pil = transforms.ToPILImage()(grid)
    display_img = cv2.cvtColor(np.array(grid_pil), cv2.COLOR_RGB2BGR)
    
    return display_img

def main():
    """Main function"""
    print(f"Starting RealSense action visualization test in {args.mode} mode")
    print(f"Actions: {ACTIONS}")
    print(f"Press 1-3 to toggle actions in manual mode, 'q' to quit")
    
    # Initialize camera
    camera = Camera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        debug=True
    )
    
    if not camera.start():
        print("Failed to start the camera!")
        return
    
    # Create output directory if needed
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        print(f"Saving frames to {args.out}")
    
    # Track active actions for manual mode
    active_actions = set()
    
    # Main loop
    frame_count = 0
    try:
        while True:
            # Get frame from camera
            frames = camera.get_frames()
            if not frames:
                print("Failed to get frames, retrying...")
                time.sleep(0.1)
                continue
            
            color_frame, _ = frames  # We only need color frame
            
            # Process the frame
            viz_frame = process_frame(
                color_frame, 
                sz=args.viz_size, 
                mode=args.mode,
                active_actions=active_actions if args.mode == "manual" else None
            )
            
            # Add UI text
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Title
            cv2.putText(viz_frame, "RealSense Action Visualization Test", 
                       (10, 25), font, 0.7, (255, 255, 255), 2)
            
            # Mode
            cv2.putText(viz_frame, f"Mode: {args.mode}", 
                       (10, 50), font, 0.5, (255, 255, 255), 1)
            
            # Legend
            for i, (action, color) in enumerate(zip(ACTIONS, COLORS)):
                color_bgr = [int(c*255) for c in COLOR_MAP[color][::-1]]  # Convert to BGR
                status = "ON" if i in active_actions or args.mode == "demo" else "OFF"
                cv2.putText(viz_frame, f"{i+1}: {action} [{status}]", 
                           (10, 75 + i*25), font, 0.5, color_bgr, 2)
            
            # Display result
            cv2.imshow('RealSense Action Visualization Test', viz_frame)
            
            # Save frame if requested
            if args.out and frame_count % 30 == 0:  # Save every 30 frames
                filename = f"{args.out}/frame_{frame_count:06d}.jpg"
                cv2.imwrite(filename, viz_frame)
                print(f"Saved {filename}")
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC to quit
                break
            elif args.mode == "manual":
                # Number keys to toggle actions
                if ord('1') <= key <= ord('3'):
                    action_idx = key - ord('1')
                    if action_idx in active_actions:
                        active_actions.remove(action_idx)
                        print(f"Turned OFF {ACTIONS[action_idx]}")
                    else:
                        active_actions.add(action_idx)
                        print(f"Turned ON {ACTIONS[action_idx]}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        camera.stop()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()