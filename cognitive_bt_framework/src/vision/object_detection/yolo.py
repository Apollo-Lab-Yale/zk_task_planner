from ultralytics import YOLO, YOLOWorld, FastSAM
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from cognitive_bt_framework.src.vision.realsense import Camera
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch


class HybridYOLO:
    def __init__(self, yolo_world_model: str = "yolov8x-worldv2.pt", 
                 fastsam_model: str = "FastSAM-x.pt", 
                 is_sim=False, confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Hybrid YOLO combining YOLOWorld detection with FastSAM segmentation
        
        Args:
            yolo_world_model: Path to YOLOWorld model for detection
            fastsam_model: Path to FastSAM model for segmentation
            is_sim: Whether running in simulation mode
            confidence_threshold: Minimum confidence score for detections
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.is_sim = is_sim
        if not is_sim:
            self.camera = Camera()
        else:
            self.camera = None
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load models on specified device
        self.detector = YOLOWorld(yolo_world_model)
        self.segmenter = FastSAM(fastsam_model)
        
        # Move models to device if CUDA
        if self.device == "cuda":
            try:
                self.detector.to(self.device)
                self.segmenter.to(self.device)
                print("Models moved to GPU")
            except Exception as e:
                print(f"Failed to move models to GPU: {e}")
                self.device = "cpu"
        
        self.confidence_threshold = confidence_threshold
        self.class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}

    def start(self) -> bool:
        return self.camera.start() if self.camera else True

    def stop(self) -> None:
        if self.camera:
            self.camera.stop()

    def set_classes(self, classes: List[str]):
        """Set custom classes for YOLOWorld detection"""
        print(f"DEBUG: Setting classes to: {classes}")
        self.detector.set_classes(classes)
    
    def predict(self, image: np.ndarray, verbose: bool = False):
        """
        Compatibility method for perception system that expects predict() interface
        Returns actual ultralytics Results objects
        """
        print(f"DEBUG: predict() called with image shape: {image.shape}")
        
        # Get detections using our detect_objects method
        detections = self.detect_objects(image)
        
        print(f"DEBUG: detect_objects returned {len(detections)} detections")
        
        if len(detections) == 0:
            # Return empty ultralytics results using YOLOWorld
            return self.detector.predict(image, verbose=verbose)
        
        # Create actual ultralytics tensors from our detections
        bboxes = torch.tensor([det['bbox'] for det in detections], dtype=torch.float32)
        confs = torch.tensor([det['conf'] for det in detections], dtype=torch.float32)
        cls_ids = torch.tensor([det['cls'] for det in detections], dtype=torch.float32)
        
        # Use YOLOWorld to get a proper Results object structure
        yolo_results = self.detector.predict(image, verbose=verbose)
        
        # Replace the boxes and masks data with our hybrid results
        if len(yolo_results) > 0:
            result = yolo_results[0]
            
            # Create new Boxes object with our data
            from ultralytics.engine.results import Boxes, Masks
            result.boxes = Boxes(
                torch.cat([bboxes, confs.unsqueeze(1), cls_ids.unsqueeze(1)], dim=1),
                orig_shape=image.shape[:2]
            )
            
            # Create masks tensor from our segmentation masks
            if any('mask' in det for det in detections):
                masks_tensor = torch.stack([
                    torch.from_numpy(det['mask']).float() 
                    for det in detections
                ])
                result.masks = Masks(masks_tensor, orig_shape=image.shape[:2])
                print(f"DEBUG: Created {len(masks_tensor)} masks with shape {masks_tensor.shape}")
            else:
                result.masks = None
                print("DEBUG: No masks available")
        
        return yolo_results

    def _segment_bbox(self, image: np.ndarray, bbox_info: Dict) -> Dict:
        """
        Segment a single bounding box using FastSAM
        
        Args:
            image: Full image
            bbox_info: Dict with bbox coordinates, confidence, class info
            
        Returns:
            Dict with segmentation mask added
        """
        bbox = bbox_info['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return with rectangular mask
            full_mask = np.zeros(image.shape[:2], dtype=np.float32)
            bbox_info['mask'] = full_mask
            return bbox_info
        
        # Crop the region
        cropped_region = image[y1:y2, x1:x2]
        
        # Run FastSAM on the cropped region
        try:
            fastsam_results = self.segmenter(cropped_region, device=self.device, retina_masks=True, 
                                           imgsz=640, conf=0.4, iou=0.9, verbose=False)
            
            best_mask = None
            max_area = 0
            
            if len(fastsam_results) > 0 and fastsam_results[0].masks is not None:
                # Find the mask with the largest area (most likely the main object)
                for mask in fastsam_results[0].masks:
                    mask_data = mask.data[0].cpu().numpy()
                    
                    # Resize mask to match cropped region
                    if mask_data.shape != cropped_region.shape[:2]:
                        mask_data = cv2.resize(mask_data, (x2-x1, y2-y1))
                    
                    area = np.sum(mask_data > 0.5)
                    if area > max_area:
                        max_area = area
                        best_mask = mask_data
            
            # Create full-size mask
            if best_mask is not None and max_area > 100:  # Minimum area threshold
                full_mask = np.zeros(image.shape[:2], dtype=np.float32)
                full_mask[y1:y2, x1:x2] = best_mask
                print(f"DEBUG: Created FastSAM mask with {np.sum(best_mask > 0.5)} pixels in crop, {np.sum(full_mask > 0.5)} in full image")
            else:
                # Fallback to rectangular mask
                full_mask = np.zeros(image.shape[:2], dtype=np.float32)
                full_mask[y1:y2, x1:x2] = 1.0
                print(f"DEBUG: Created rectangular fallback mask with {np.sum(full_mask > 0.5)} pixels")
                
        except Exception as e:
            # If FastSAM fails, use rectangular mask
            full_mask = np.zeros(image.shape[:2], dtype=np.float32)
            full_mask[y1:y2, x1:x2] = 1.0
        
        bbox_info['mask'] = full_mask
        return bbox_info

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects using YOLOWorld, then get segmentation masks using FastSAM in parallel
        Returns same format as ObjectDetection.detect_objects()
        """
        detections = []
        
        print(f"DEBUG: detect_objects called with image shape: {image.shape}")
        
        # Step 1: Use YOLOWorld for detection
        world_results = self.detector(image, device=self.device, verbose=False)[0]
        
        print(f"DEBUG: YOLOWorld returned {len(world_results.boxes) if world_results.boxes is not None else 0} detections")
        
        if world_results is None or world_results.boxes is None:
            return detections
        
        # Step 2: Prepare bbox info for parallel processing
        bbox_infos = []
        for box in world_results.boxes:
            confidence = box.conf.item()
            print(f"DEBUG: Detection confidence: {confidence:.3f}, threshold: {self.confidence_threshold}")
            if confidence < self.confidence_threshold:
                print(f"DEBUG: Skipping detection due to low confidence")
                continue
                
            class_name = world_results.names[int(box.cls.item())]
            bbox = box.xyxy[0].cpu().numpy()
            
            bbox_infos.append({
                'bbox': bbox,
                'conf': confidence,
                'cls': box.cls.item(),
                'name': class_name
            })
        
        # Step 3: Run FastSAM on each bbox in parallel
        if len(bbox_infos) == 0:
            return detections
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(len(bbox_infos), 4)  # Limit concurrent threads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all segmentation tasks
            future_to_bbox = {
                executor.submit(self._segment_bbox, image, bbox_info): bbox_info 
                for bbox_info in bbox_infos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_bbox):
                try:
                    result = future.result()
                    detections.append(result)
                except Exception as e:
                    # If a segmentation fails, use the original bbox with rectangular mask
                    bbox_info = future_to_bbox[future]
                    x1, y1, x2, y2 = map(int, bbox_info['bbox'])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    full_mask = np.zeros(image.shape[:2], dtype=np.float32)
                    full_mask[y1:y2, x1:x2] = 1.0
                    bbox_info['mask'] = full_mask
                    detections.append(bbox_info)
            
        return detections

    def get_object_depth(self, depth_image: np.ndarray, mask: np.ndarray) -> float:
        masked_depth = depth_image[mask > 0.5]
        return np.mean(masked_depth[masked_depth > 0]) if masked_depth.size > 0 else 0

    def apply_mask_overlay(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
                           alpha: float = 0.5) -> np.ndarray:
        mask_binary = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary > 0] = color
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def process_frame(self, frames=None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.is_sim:
            frames = self.camera.get_frames()
        elif not frames:
            return None, None
        
        color_image, depth_image = frames
        detections = self.detect_objects(color_image)

        vis_color = color_image.copy()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        vis_depth = depth_colormap.copy()

        for det in detections:
            bbox = det['bbox']
            mask = det['mask']
            color = self.class_colors[int(det['cls'])]

            # Apply segmentation mask overlay
            vis_color = self.apply_mask_overlay(vis_color, mask, color)
            vis_depth = self.apply_mask_overlay(vis_depth, mask, color)

            depth = self.get_object_depth(depth_image, mask)
            depth_meters = depth * self.camera.depth_scale if self.camera.depth_scale else 0

            x1, y1 = map(int, bbox[:2])
            label = f"{det['name']} {depth_meters:.2f}m"

            for img in [vis_color, vis_depth]:
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_color, vis_depth

    def get_classes(self) -> List[str]:
        # Return available classes from YOLOWorld detector
        if hasattr(self.detector, 'names'):
            return list(self.detector.names.values())
        return []

    def run(self) -> None:
        if not self.start():
            return

        try:
            while True:
                result = self.process_frame()
                if result:
                    color_det, depth_det = result
                    combined = np.hstack((color_det, depth_det))
                    cv2.imshow('Hybrid YOLO Detection with Segmentation', combined)

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            self.stop()


if __name__ == "__main__":
    import sys
    
    print("Hybrid YOLO Detection - Combining YOLOWorld + YOLO-E")
    print("\nUsage:")
    print("  python yolo.py [custom_classes...]")
    print("\nExample:")
    print("  python yolo.py cabinet drawer handle")
    
    # Parse command line classes
    custom_classes = sys.argv[1:] if len(sys.argv) > 1 else []
    
    try:
        detector = HybridYOLO(confidence_threshold=0.3)
        
        if custom_classes:
            detector.set_classes(custom_classes)
            print(f"Set custom classes: {custom_classes}")
        else:
            print("Using default YOLOWorld classes")
            
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()