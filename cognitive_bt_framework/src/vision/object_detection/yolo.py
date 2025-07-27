from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from cognitive_bt_framework.src.vision.realsense import Camera
# from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv



class ObjectDetection:
    def __init__(self, model_path: str = "yolo11x-seg.pt", is_sim=False, 
                 confidence_threshold: float = 0.5):
        """
        Initialize YOLO object detection with configurable parameters
        
        Args:
            model_path: Path to YOLO model file
            is_sim: Whether running in simulation mode
            confidence_threshold: Minimum confidence score for detections
        """
        self.is_sim = is_sim
        if not is_sim:
            self.camera = Camera()
        else:
            self.camera = None
        self.model = YOLO(model_path)
        self.class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}
        
        # Configurable parameters
        self.confidence_threshold = confidence_threshold

    def start(self) -> bool:
        return self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, verbose=False)[0]
        detections = []
        
        if results is None or results.boxes is None or results.masks is None:
            return detections
            
        for box, mask in zip(results.boxes, results.masks):
            confidence = box.conf.item()
            class_name = results.names[int(box.cls.item())]
            
            # Apply confidence threshold - open vocabulary, no class filtering
            if confidence < self.confidence_threshold:
                continue
            
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy(),
                'conf': confidence,
                'cls': box.cls.item(),
                'name': class_name,
                'mask': mask.data[0].cpu().numpy()
            })
            
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

    def process_frame(self, frames = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
        return list(self.model.names.values())

    def run(self) -> None:
        if not self.start():
            return

        try:
            while True:
                result = self.process_frame()
                if result:
                    color_det, depth_det = result
                    combined = np.hstack((color_det, depth_det))
                    cv2.imshow('Object Detection with Segmentation', combined)

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            self.stop()


def test_cabinet_detection():
    """Test cabinet detection with RealSense camera on RGB frames"""
    print("=== Testing Cabinet Detection with RealSense Camera ===")
    
    # Cabinet-related keywords to look for
    cabinet_keywords = [
        'cabinet', 'drawer', 'cupboard', 'shelf', 'wardrobe', 
        'dresser', 'bookcase', 'furniture', 'storage', 'door'
    ]
    
    # Initialize RealSense camera
    camera = Camera()
    if not camera.start():
        print("Failed to start RealSense camera")
        return
    
    # Initialize detector with cabinet focus
    detector = ObjectDetection(
        confidence_threshold=0.1,  # Lower threshold for furniture
        is_sim=True  # Use external camera frames
    )
    
    print(f"Available YOLO classes: {detector.get_classes()}")
    
    # Find cabinet-like classes in YOLO model
    available_classes = detector.get_classes()
    cabinet_classes = []
    for class_name in available_classes:
        for keyword in cabinet_keywords:
            if keyword.lower() in class_name.lower():
                cabinet_classes.append(class_name)
                break
    
    print(f"Potential cabinet classes in YOLO model: {cabinet_classes}")
    print("Looking for cabinets in RGB frames... Press 'q' to quit")
    
    try:
        cabinet_detections = []
        frame_count = 0
        
        while True:
            # Get frames from RealSense camera
            frames = camera.get_frames()
            if not frames:
                continue
                
            color_image, depth_image = frames
            frame_count += 1
            
            # Run object detection on RGB frame
            detections = detector.detect_objects(color_image)
            
            # Create visualization
            vis_color = color_image.copy()
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            
            # Filter and visualize cabinet-like objects
            frame_cabinets = []
            for det in detections:
                class_name = det['name'].lower()
                is_cabinet = False
                
                for keyword in cabinet_keywords:
                    if keyword in class_name:
                        frame_cabinets.append(det)
                        is_cabinet = True
                        print(f"Frame {frame_count}: Cabinet detected - {det['name']} (confidence: {det['conf']:.2f})")
                        break
                
                # Draw all detections, highlight cabinets
                bbox = det['bbox']
                mask = det['mask']
                color = (0, 255, 0) if is_cabinet else (255, 255, 255)  # Green for cabinets, white for others
                
                # Apply mask overlay
                vis_color = detector.apply_mask_overlay(vis_color, mask, color)
                
                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_color, (x1, y1), (x2, y2), color, 2)
                
                label = f"{det['name']} {det['conf']:.2f}"
                if is_cabinet:
                    label += " [CABINET]"
                    
                cv2.putText(vis_color, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if frame_cabinets:
                cabinet_detections.extend(frame_cabinets)
            
            # Display results
            cv2.putText(vis_color, f"Cabinets found: {len(set((det['name'], tuple(det['bbox'])) for det in cabinet_detections))}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_color, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show RGB detection and depth
            combined = np.hstack((vis_color, depth_colormap))
            cv2.imshow('Cabinet Detection - RGB + Depth', combined)
            
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
                
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        
        # Summary
        print(f"\n=== Cabinet Detection Summary ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total cabinet detections: {len(cabinet_detections)}")
        if cabinet_detections:
            unique_classes = set(det['name'] for det in cabinet_detections)
            print(f"Unique cabinet types found: {list(unique_classes)}")
            
            # Show average confidence per class
            for class_name in unique_classes:
                class_detections = [det for det in cabinet_detections if det['name'] == class_name]
                avg_conf = sum(det['conf'] for det in class_detections) / len(class_detections)
                print(f"  {class_name}: {len(class_detections)} detections, avg confidence: {avg_conf:.3f}")
        else:
            print("No cabinet objects detected. Try pointing camera at furniture/cabinets.")


def test_configurable_detection():
    """Test configurable detection parameters"""
    print("=== Testing Configurable Detection Parameters ===")
    
    # Test with high confidence threshold
    print("\n1. Testing with high confidence threshold (0.8)")
    detector_high = ObjectDetection(confidence_threshold=0.8)
    print(f"High confidence detector created - threshold: {detector_high.confidence_threshold}")
    
    # Test with low confidence threshold
    print("\n2. Testing with low confidence threshold (0.2)")
    detector_low = ObjectDetection(confidence_threshold=0.2)
    print(f"Low confidence detector created - threshold: {detector_low.confidence_threshold}")
    
    # Show available classes (open vocabulary)
    print(f"\n3. Available YOLO classes (open vocabulary): {len(detector_high.get_classes())} classes")
    print("First 10 classes:", detector_high.get_classes()[:10])
    print("Detector configurations created successfully - all using open vocabulary")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "cabinet":
            test_cabinet_detection()
        elif sys.argv[1] == "config":
            test_configurable_detection()
        else:
            print("Available tests:")
            print("  python yolo.py cabinet  - Test cabinet detection")
            print("  python yolo.py config   - Test configurable parameters")
    else:
        print("YOLO Object Detection with Configurable Parameters")
        print("\nUsage:")
        print("  python yolo.py cabinet  - Test cabinet detection")
        print("  python yolo.py config   - Test configurable parameters")
        print("\nConfigurable parameters:")
        print("  - confidence_threshold: Minimum confidence score (default: 0.5)")
        print("  - Open vocabulary: Detects all YOLO classes without filtering")