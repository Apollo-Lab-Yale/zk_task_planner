import numpy as np
import torch
import cv2
import gc
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor

def overlaps(box1, box2):
        """Check if two bounding boxes overlap"""
        return not (box1[2] < box2[0] or
                    box1[0] > box2[2] or
                    box1[3] < box2[1] or 
                    box1[1] > box2[3])

def get_alpha_id(num):
    """
    Convert a numerical ID to an alphabetical ID (a, b, c, ... aa, ab, ...)
    For example:
    1 -> a, 2 -> b, ..., 26 -> z, 27 -> aa, 28 -> ab, ...
    """
    if num <= 0:
        return ""
    
    letters = ""
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        letters = chr(97 + remainder) + letters  # 97 is ASCII for 'a'
    
    return letters

@dataclass
class FastSAMConfig:
    """Configuration settings for Ultralytics FastSAM-based mask generation"""
    model_type: str = "FastSAM-x"  # FastSAM-s or FastSAM-x
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory management
    max_image_size: int = 640  # Input image size for processing
    enable_memory_efficient_attention: bool = False
    
    # Mask generation parameters
    conf_threshold: float = 0.4  # Confidence threshold for detection
    iou_threshold: float = 0.7   # IoU threshold for NMS
    retina_masks: bool = True    # Use high-quality mask output
    
    # Post-processing parameters
    remove_small_regions: bool = False
    merge_overlapping: bool = False
    overlap_threshold: float = 0.5
    min_area: float = 10.0  # Minimum area for mask retention
    draw_borders: bool = True

class FastSAMMaskGenerator:
    """
    Memory-optimized class for generating and managing masks using Ultralytics FastSAM.
    Supports different prompting methods: everything, points, boxes, and text.
    """
    
    def __init__(self, config: FastSAMConfig):
        """Initialize the FastSAM-based mask generator"""
        self.config = config
        self._setup_memory_config()
        self._setup_model()
        self._next_mask_id = 1
        self.debug = False
        
        self.last_masks = None

    def _setup_memory_config(self):
        """Configure memory management settings"""
        if self.config.device == "cuda":
            # Set PyTorch memory allocator settings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            
            if self.config.enable_memory_efficient_attention:
                os.environ["PYTORCH_ENABLE_MEM_EFF_ATTENTION"] = "1"

    def _setup_model(self):
        """Initialize and configure the FastSAM model"""
        try:
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Initialize FastSAM predictor with configuration
            predictor_overrides = {
                "conf": self.config.conf_threshold,
                "task": "segment",
                "mode": "predict",
                "model": f"{self.config.model_type}.pt",
                "save": False,
                "imgsz": self.config.max_image_size
            }
            
            self.predictor = FastSAMPredictor(overrides=predictor_overrides)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "GPU out of memory. Try reducing the input image size "
                    "or adjusting confidence/IoU thresholds."
                )
            raise e

    def generate_masks(
        self, 
        image: Union[str, np.ndarray],
        prompt_type: str = "everything",
        **prompt_args
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate masks for the input image using FastSAM with specified prompt type
        
        Args:
            image: Input image (path or numpy array)
            prompt_type: Type of prompt ("everything", "points", "boxes", or "text")
            **prompt_args: Additional arguments for specific prompt types:
                - points: List of [x, y] coordinates
                - boxes: List of [x1, y1, x2, y2] coordinates
                - text: String text prompt
            
        Returns:
            Tuple[np.ndarray, Dict]: Labeled mask array and metadata dictionary
        """
        try:
            # Store original image dimensions
            if isinstance(image, np.ndarray):
                original_height, original_width = image.shape[:2]
            else:
                # If it's a path, we'll get dimensions later
                original_height, original_width = None, None
            
            # Check if image needs to be resized
            if isinstance(image, np.ndarray) and (original_height > self.config.max_image_size or 
                                                original_width > self.config.max_image_size):
                # Calculate new dimensions
                if original_width > original_height:
                    new_width = self.config.max_image_size
                    new_height = int(original_height * (self.config.max_image_size / original_width))
                else:
                    new_height = self.config.max_image_size
                    new_width = int(original_width * (self.config.max_image_size / original_height))
                
                # Resize image for processing
                resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Resized image from {image.shape[:2]} to {resized_image.shape[:2]}")
                input_image = resized_image
            else:
                input_image = image
                
            # Get initial results
            start = time.time()
            everything_results = self.predictor(input_image)
            print(f"Inference time: {time.time() - start:.2f}s")
            
            # If we got dimensions from a file, store them now
            if original_height is None and original_width is None:
                original_height, original_width = everything_results[0].orig_shape
                
            # Apply specific prompt if needed
            if prompt_type != "everything":
                if prompt_type == "points" and "points" in prompt_args:
                    results = self.predictor.prompt(everything_results, points=prompt_args["points"])
                elif prompt_type == "boxes" and "boxes" in prompt_args:
                    results = self.predictor.prompt(everything_results, bboxes=prompt_args["boxes"])
                elif prompt_type == "text" and "text" in prompt_args:
                    results = self.predictor.prompt(everything_results, texts=prompt_args["text"])
                else:
                    raise ValueError(f"Invalid or missing arguments for prompt type: {prompt_type}")
            else:
                results = everything_results

            # Convert results to labeled masks and metadata
            labeled_masks, metadata = self._process_results(results)

            # If the image was resized, resize the masks back to original dimensions
            if isinstance(image, np.ndarray) and (original_height != labeled_masks.shape[0] or 
                                                original_width != labeled_masks.shape[1]):
                # Create a new labeled mask array of the original size
                original_labeled_masks = np.zeros((original_height, original_width), dtype=np.int32)
                
                # Update metadata bounding boxes and resize to original dimensions
                for mask_id in list(metadata.keys()):
                    # Create a binary mask for this ID
                    binary_mask = labeled_masks == mask_id
                    
                    # Resize binary mask to original dimensions
                    resized_binary_mask = cv2.resize(
                        binary_mask.astype(np.uint8), 
                        (original_width, original_height), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                    # Skip if the mask disappeared during resizing
                    if not np.any(resized_binary_mask):
                        del metadata[mask_id]
                        continue
                    
                    # Update the labeled mask
                    original_labeled_masks[resized_binary_mask] = mask_id
                    
                    # Update the bounding box in metadata
                    y_indices, x_indices = np.where(resized_binary_mask)
                    bbox = [
                        int(x_indices.min()),
                        int(y_indices.min()),
                        int(x_indices.max() - x_indices.min()),
                        int(y_indices.max() - y_indices.min())
                    ]
                    metadata[mask_id]['bbox'] = bbox
                    
                    # Update the area
                    metadata[mask_id]['area'] = float(resized_binary_mask.sum())
                    
                    # Recalculate contours if needed
                    if self.config.draw_borders:
                        contours, _ = cv2.findContours(
                            resized_binary_mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        metadata[mask_id]['contours'] = [
                            cv2.approxPolyDP(c, epsilon=0.01, closed=True).tolist()
                            for c in contours
                        ]
                self.last_masks = original_labeled_masks
                return original_labeled_masks, metadata
            else:
                return labeled_masks, metadata

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "Out of memory during mask generation. Try:\n"
                    "1. Reducing the input image size\n"
                    "2. Adjusting confidence threshold\n"
                    "3. Adjusting IoU threshold"
                )
            raise e

    def _process_results(self, results) -> Tuple[np.ndarray, Dict]:
        """Process FastSAM results into labeled masks and metadata"""
        # Get image dimensions from results
        height, width = results[0].orig_shape
        labeled_masks = np.zeros((height, width), dtype=np.int32)
        metadata = {}

        # Process each mask
        for i, mask_data in enumerate(results[0].masks, 1):
            # Convert mask tensor to boolean numpy array
            mask = mask_data.data.cpu().numpy()[0]
            original_shape = mask.shape
            
            # Resize mask to match original image dimensions if needed
            if mask.shape[0] != height or mask.shape[1] != width:
                if self.debug:
                    print(f"Resizing mask from {mask.shape} to {(height, width)}")
                mask = cv2.resize(
                    mask.astype(np.float32), 
                    (width, height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Convert to boolean mask after resizing
            mask = (mask > 0.5).astype(bool)
            
            # Skip if mask is empty
            if not mask.any():
                continue
                
            mask_id = self._next_mask_id
            self._next_mask_id += 1
            
            # Add mask to labeled array
            labeled_masks[mask] = mask_id
            
            # Calculate mask properties
            area = float(mask.sum())
            if area < self.config.min_area and self.config.remove_small_regions:
                continue
                
            # Get bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            
            bbox = [
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max() - x_indices.min()),
                int(y_indices.max() - y_indices.min())
            ]
            
            # Store metadata
            alpha_id = get_alpha_id(mask_id)  # Convert numeric ID to alphabetical ID
            metadata[mask_id] = {
                'area': area,
                'bbox': bbox,
                'alpha_id': alpha_id,  # Store alphabetical ID in metadata
                'original_shape': original_shape,
                # 'confidence': float(results[0].probs[i-1]) if len(results[0].probs) > 0 else 1.0,
            }
            
            # Add contours if enabled
            if self.config.draw_borders:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                metadata[mask_id]['contours'] = [
                    cv2.approxPolyDP(c, epsilon=0.01, closed=True).tolist()
                    for c in contours
                ]

        # Merge overlapping masks if configured
        if self.config.merge_overlapping:
            labeled_masks, metadata = self._merge_overlapping_masks(labeled_masks, metadata)
            
        return labeled_masks, metadata
    
    def _merge_overlapping_masks(
        self, 
        masks: np.ndarray, 
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Merge masks that have significant overlap"""
        new_masks = masks.copy()
        new_metadata = metadata.copy()
        
        mask_ids = list(metadata.keys())
        for i in range(len(mask_ids)):
            for j in range(i + 1, len(mask_ids)):
                id1, id2 = mask_ids[i], mask_ids[j]
                
                if id1 not in new_metadata or id2 not in new_metadata:
                    continue
                    
                mask1 = masks == id1
                mask2 = masks == id2
                
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                
                if union > 0:  # Avoid division by zero
                    iou = intersection / union
                    
                    if iou > self.config.overlap_threshold:
                        # Keep mask with higher confidence if available
                        if 'confidence' in new_metadata[id1] and 'confidence' in new_metadata[id2]:
                            conf1 = new_metadata[id1]['confidence']
                            conf2 = new_metadata[id2]['confidence']
                            
                            if conf1 >= conf2:
                                new_masks[mask2] = id1
                                del new_metadata[id2]
                            else:
                                new_masks[mask1] = id2
                                del new_metadata[id1]
                        else:
                            # If no confidence scores, keep the first mask
                            new_masks[mask2] = id1
                            del new_metadata[id2]
                        
        return new_masks, new_metadata

    def visualize_masks(
        self, 
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5
    ) -> np.ndarray:
        vis_image = image.copy()
        unique_masks = np.unique(masks)[1:]
        overlay = np.zeros_like(vis_image, dtype=np.float32)
        placed_boxes = []
        
        for mask_id in unique_masks:
            if mask_id not in metadata:
                continue
                
            mask = masks == mask_id
            color = np.random.random(3) * 255
            overlay[mask] = color
            
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                x_center = int(np.mean(x_coords))
                y_center = int(np.mean(y_coords))
                
                # Use alphabetical ID instead of numeric ID
                alpha_id = metadata[mask_id].get('alpha_id', get_alpha_id(mask_id))
                region_id = alpha_id
                
                text_size = cv2.getTextSize(region_id, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                text_x = x_center - text_size[0]//2
                text_y = y_center + text_size[1]//2
                
                text_box = (text_x - 5, text_y - text_size[1] - 5,
                          text_x + text_size[0] + 5, text_y + 5)
                
                # Find non-overlapping position
                offset_y = text_size[1] + 10
                while any(overlaps(text_box, box) for box in placed_boxes):
                    text_y += offset_y
                    text_box = (text_x - 5, text_y - text_size[1] - 5,
                              text_x + text_size[0] + 5, text_y + 5)
                
                placed_boxes.append(text_box)
                
                # Draw arrow
                cv2.arrowedLine(vis_image,
                              (x_center, y_center),
                              (text_x + text_size[0]//2, text_y - text_size[1]//2),
                              (0, 0, 0),
                              2,
                              tipLength=0.2)
                
                # Draw text with outline
                for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(vis_image, region_id, 
                              (text_x+dx, text_y+dy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                cv2.putText(vis_image, region_id,
                          (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            if self.config.draw_borders and 'contours' in metadata[mask_id]:
                contours = metadata[mask_id]['contours']
                cv2.drawContours(vis_image,
                               [np.array(c) for c in contours],
                               -1,
                               (0, 0, 0),
                               thickness=2)
        
        vis_image = cv2.addWeighted(vis_image, 1 - alpha,
                                   overlay.astype(np.uint8), alpha, 0)
        return vis_image.astype(np.uint8)
    
    def show_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        figsize: Tuple[int, int] = (20, 20),
        font_size: int = 10
    ) -> None:
        """
        Display masks using matplotlib with ID labels
        
        Args:
            image (np.ndarray): Original RGB image
            masks (np.ndarray): Labeled mask array
            metadata (Dict): Mask metadata dictionary
            figsize (Tuple[int, int]): Figure size for matplotlib
            font_size (int): Size of the font for ID labels
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Show original image
        plt.imshow(image)
        
        # Create mask overlay
        unique_masks = np.unique(masks)[1:]  # Skip 0 (background)
        mask_img = np.zeros((*image.shape[:2], 4))
        
        for mask_id in unique_masks:
            if mask_id not in metadata:
                continue
                
            mask = masks == mask_id
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            mask_img[mask] = color_mask
            
            # Calculate centroid
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                # Fallback to bbox center
                bbox = metadata[mask_id]['bbox']
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
            
            # Use alphabetical ID instead of numeric ID
            alpha_id = metadata[mask_id].get('alpha_id', get_alpha_id(mask_id))
            
            # Add ID label
            plt.text(
                cx, cy,
                alpha_id,
                color='white',
                fontsize=font_size,
                bbox=dict(
                    facecolor='black',
                    alpha=0.7,
                    edgecolor='none',
                    pad=1
                ),
                ha='center',
                va='center'
            )
            
            # Draw borders if enabled
            if self.config.draw_borders:
                contours = metadata[mask_id].get('contours')
                if contours:
                    plt.contour(
                        mask,
                        colors=['blue'],
                        alpha=0.4,
                        linewidths=1
                    )
        
        plt.imshow(mask_img)
        plt.axis('off')
        plt.show()