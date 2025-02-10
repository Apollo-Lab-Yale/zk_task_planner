import os
import gc
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
import cv2
import open_clip as clip
from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor

@dataclass
class FastSAMConfig:
    """Configuration settings for FastSAM with CLIP filtering"""
    model_type: str = "FastSAM-x"  # FastSAM-s or FastSAM-x
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Memory management
    max_image_size: int = 640
    enable_memory_efficient_attention: bool = False
    
    # FastSAM parameters
    conf_threshold: float = 0.4
    iou_threshold: float = 0.75
    retina_masks: bool = True
    
    # CLIP parameters
    clip_model: str = "ViT-B-32"
    clip_threshold: float = 0.85
    
    # Post-processing parameters
    remove_small_regions: bool = True
    merge_overlapping: bool = False
    overlap_threshold: float = 0.5
    min_area: float = 10.0
    draw_borders: bool = True

class FastSAMWithCLIP:
    """
    Memory-optimized class for generating masks using FastSAM and filtering with CLIP
    """
    
    def __init__(self, config: FastSAMConfig):
        self.config = config
        self._setup_memory_config()
        self._setup_models()
        self._next_mask_id = 1
        self.last_masks = []
        self.last_metadata = []

    def _setup_memory_config(self):
        """Configure memory management settings"""
        if self.config.device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            
            if self.config.enable_memory_efficient_attention:
                os.environ["PYTORCH_ENABLE_MEM_EFF_ATTENTION"] = "1"

    def _setup_models(self):
        """Initialize FastSAM and CLIP models"""
        try:
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            # Initialize FastSAM
            predictor_overrides = {
                "conf": self.config.conf_threshold,
                "task": "segment",
                "mode": "predict",
                "model": f"{self.config.model_type}.pt",
                "save": False,
                "imgsz": self.config.max_image_size
            }
            self.predictor = FastSAMPredictor(overrides=predictor_overrides)

            # Initialize CLIP
            self.clip_model, _, self.clip_preprocess = clip.create_model_and_transforms(
                self.config.clip_model, 
                device=self.config.device
            )

            self.clip_tokenizer = clip.get_tokenizer(self.config.clip_model)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "GPU out of memory. Try reducing input image size "
                    "or adjusting thresholds."
                )
            raise e

    @torch.no_grad()
    def get_clip_score(self, crop: Image.Image, query: str) -> float:
        """Calculate CLIP similarity score between image crop and text query"""
        # Prepare text prompts
        text_inputs = self.clip_tokenizer(
            [f"{query}", "a photo of background"]
        ).to(self.config.device)
        
        # Prepare image
        image_input = self.clip_preprocess(crop).unsqueeze(0).to(self.config.device)

        # Get similarity scores
        image_features, text_features, logits_per_image= self.clip_model(image_input, text_inputs)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return float(similarity[0][0].item())

    def process_image(
        self,
        image: Union[str, np.ndarray],
        query: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process image with FastSAM and optionally filter with CLIP query
        
        Args:
            image: Input image (path or numpy array)
            query: Optional text query for CLIP filtering
            
        Returns:
            Tuple[np.ndarray, Dict]: Labeled mask array and metadata dictionary
        """
        try:
            # Generate masks with FastSAM
            start = time.time()
            results = self.predictor(image)
            print(f"FastSAM inference time: {time.time() - start:.2f}s")

            # Convert results to labeled masks and metadata
            labeled_masks, metadata = self._process_results(results, image, query)
            self.last_masks = labeled_masks
            self.last_metadata = metadata
            return labeled_masks, metadata

        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                raise RuntimeError(
                    "Out of memory during processing. Try:\n"
                    "1. Reducing input image size\n"
                    "2. Adjusting confidence threshold\n"
                    "3. Adjusting IoU threshold"
                )
            raise e

    def _process_results(
        self, 
        results, 
        original_image: Union[str, np.ndarray],
        query: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process FastSAM results with fixed mask dimensions"""
        # Convert image to numpy if needed
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        height, width = original_image.shape[:2]
        labeled_masks = np.zeros((height, width), dtype=np.int32)
        metadata = {}

        # Print dimensions for debugging
        print(f"Original image dimensions: {original_image.shape}")
        if len(results) > 0 and hasattr(results[0], 'masks') and len(results[0].masks) > 0:
            print(f"First mask dimensions: {results[0].masks[0].data.shape}")

        # Process each mask
        for i, mask_data in enumerate(results[0].masks, 1):
            try:
                # Get mask and convert to numpy
                mask = mask_data.data.cpu().numpy()[0]
                mask = (mask > 0.5).astype(bool)
                
                print(f"Processing mask {i}, original shape: {mask.shape}")
                
                # Resize mask to match original image dimensions
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask.astype(np.uint8), 
                        (width, height), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    
                print(f"Mask {i} resized shape: {mask.shape}")
                
                if not mask.any():
                    print(f"Mask {i} is empty after resize")
                    continue
                    
                area = float(mask.sum())
                if area < self.config.min_area and self.config.remove_small_regions:
                    print(f"Mask {i} too small: {area} < {self.config.min_area}")
                    continue

                # Get bounding box
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    print(f"Mask {i} has no valid indices")
                    continue
                    
                x1, y1 = int(x_indices.min()), int(y_indices.min())
                x2, y2 = int(x_indices.max()), int(y_indices.max())
                bbox = [x1, y1, x2 - x1, y2 - y1]

                # Apply CLIP filtering if query provided
                clip_score = 1.0
                if query:
                    # Ensure crop coordinates are within bounds
                    x1 = max(0, min(x1, width-1))
                    y1 = max(0, min(y1, height-1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if x2 <= x1 or y2 <= y1:
                        print(f"Invalid crop dimensions for mask {i}: ({x1},{y1}) to ({x2},{y2})")
                        continue
                    
                    # Crop and convert to PIL
                    crop = original_image[y1:y2, x1:x2]
                    if crop.size == 0:
                        print(f"Empty crop for mask {i}")
                        continue
                        
                    crop_pil = Image.fromarray(crop)
                    
                    # Make crop square with padding
                    max_dim = max(crop.shape[0], crop.shape[1])
                    crop_square = Image.new('RGB', (max_dim, max_dim), (0,0,0))
                    paste_x = (max_dim - crop.shape[1]) // 2
                    paste_y = (max_dim - crop.shape[0]) // 2
                    crop_square.paste(crop_pil, (paste_x, paste_y))
                    
                    clip_score = self.get_clip_score(crop_square, query)
                    print(f"Mask {i} CLIP score: {clip_score}")
                    
                    if clip_score < self.config.clip_threshold:
                        print(f"Mask {i} filtered by CLIP score: {clip_score} < {self.config.clip_threshold}")
                        continue
                
                mask_id = self._next_mask_id
                self._next_mask_id += 1
                
                labeled_masks[mask] = mask_id
                metadata[mask_id] = {
                    'area': area,
                    'bbox': bbox,
                    'clip_score': clip_score
                }

                print(f"Successfully processed mask {i} with ID {mask_id}")

            except Exception as e:
                print(f"Error processing mask {i}: {str(e)}")
                continue

        print(f"Total valid masks after processing: {len(metadata)}")
        return labeled_masks, metadata

    def visualize_masks(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Visualize masks with labels and arrows"""
        return self._visualize_with_arrows(image, masks, metadata, alpha)

    def _visualize_with_arrows(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metadata: Dict,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Helper method for mask visualization with arrows and labels"""
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
                
                label = f"r{mask_id}"
                if 'clip_score' in metadata[mask_id]:
                    label += f" ({metadata[mask_id]['clip_score']:.2f})"
                
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                text_x = x_center - text_size[0]//2
                text_y = y_center + text_size[1]//2
                
                text_box = (
                    text_x - 5, 
                    text_y - text_size[1] - 5,
                    text_x + text_size[0] + 5, 
                    text_y + 5
                )
                
                # Find non-overlapping position
                offset_y = text_size[1] + 10
                while any(self._overlaps(text_box, box) for box in placed_boxes):
                    text_y += offset_y
                    text_box = (
                        text_x - 5,
                        text_y - text_size[1] - 5,
                        text_x + text_size[0] + 5,
                        text_y + 5
                    )
                
                placed_boxes.append(text_box)
                
                # Draw arrow and text
                cv2.arrowedLine(
                    vis_image,
                    (x_center, y_center),
                    (text_x + text_size[0]//2, text_y - text_size[1]//2),
                    (0, 0, 0),
                    2,
                    tipLength=0.2
                )
                
                # Draw text with outline
                for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    cv2.putText(
                        vis_image, 
                        label,
                        (text_x+dx, text_y+dy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,0),
                        2
                    )
                cv2.putText(
                    vis_image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )
            
            if self.config.draw_borders and 'contours' in metadata[mask_id]:
                contours = metadata[mask_id]['contours']
                cv2.drawContours(
                    vis_image,
                    [np.array(c) for c in contours],
                    -1,
                    (0, 0, 0),
                    thickness=2
                )
        
        vis_image = cv2.addWeighted(
            vis_image,
            1 - alpha,
            overlay.astype(np.uint8),
            alpha,
            0
        )
        return vis_image.astype(np.uint8)

    def _overlaps(self, box1, box2):
        """Check if two bounding boxes overlap"""
        return not (
            box1[2] < box2[0] or
            box1[0] > box2[2] or
            box1[3] < box2[1] or
            box1[1] > box2[3]
        )

    def _merge_overlapping_masks(
        self,
        masks: np.ndarray,
        metadata: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Merge masks with significant overlap"""
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
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > self.config.overlap_threshold:
                        # Keep mask with higher CLIP score if available
                        score1 = new_metadata[id1].get('clip_score', 0)
                        score2 = new_metadata[id2].get('clip_score', 0)
                        
                        if score1 >= score2:
                            new_masks[mask2] = id1
                            del new_metadata[id2]
                        else:
                            new_masks[mask1] = id2
                            del new_metadata[id1]
                        
        return new_masks, new_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastSAM with CLIP filtering")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--query", default=None, help="Text query for CLIP filtering")
    parser.add_argument("--output", default="output.jpg", help="Output path")
    parser.add_argument("--model-type", default="FastSAM-x", 
                        choices=["FastSAM-s", "FastSAM-s"],
                        help="FastSAM model type")
    parser.add_argument("--conf-threshold", type=float, default=0.4,
                        help="Confidence threshold for FastSAM")
    parser.add_argument("--clip-threshold", type=float, default=0.95,
                        help="Similarity threshold for CLIP filtering")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = FastSAMConfig(
        model_type=args.model_type,
        conf_threshold=args.conf_threshold,
        clip_threshold=args.clip_threshold
    )
    
    # Initialize processor
    processor = FastSAMWithCLIP(config)
    
    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    labeled_masks, metadata = processor.process_image(image, args.query)
    
    # Visualize results
    result = processor.visualize_masks(image, labeled_masks, metadata)
    
    # Save output
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    
    print(f"Processed image saved to {args.output}")
    if args.query:
        print(f"Found {len(metadata)} matches for query '{args.query}'")