import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
from transformers import OwlViTProcessor, OwlViTForObjectDetection

@dataclass
class OWLViTStateConfig:
    """Configuration for OWL-ViT-based state detection"""
    model_type: str = "google/owlvit-base-patch32"
    max_image_size: int = 640
    confidence_threshold: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Additional state detection thresholds
    state_confidence: float = 0.5
    spatial_iou_threshold: float = 0.5

class OWLViTStateDetector:
    """
    Class for detecting object states using OWL-ViT for detection and state analysis
    """
    def __init__(self, config: OWLViTStateConfig = None):
        """Initialize the state detector with OWL-ViT"""
        self.config = config or OWLViTStateConfig()
        
        # Initialize OWL-ViT
        self.processor = OwlViTProcessor.from_pretrained(self.config.model_type)
        self.model = OwlViTForObjectDetection.from_pretrained(self.config.model_type)
        self.model.to(self.config.device)
        
        # Split predicates into boolean and relational
        self.bool_preds = [
            'visible', 'receptacle', 'canSwitchOn', 'isSwitchedOn',
            'breakable', 'isBroken', 'canFillWithLiquid', 'isFilledWithLiquid',
            'dirtyable', 'isDirty', 'cookable', 'isCooked', 'sliceable', 'isSliced', 
            'openable', 'isOpen', 'pickupable', 'isPickedUp',
            'moveable'
        ]
        
        self.relational_preds = ['inRoom', 'isOnTop', 'isInside']
        
        # Combined predicate list
        self.predicate_list = self.bool_preds + self.relational_preds
        
        # Generate text queries for state detection
        self.state_queries = self._generate_state_queries()

    def _generate_state_queries(self) -> Dict[str, List[str]]:
        """Generate text queries for state detection"""
        queries = {
            'receptacle': ['container', 'receptacle', 'holder'],
            'canSwitchOn': ['switch', 'button', 'lamp', 'light'],
            'isSwitchedOn': ['powered on', 'lit up', 'switched on'],
            'breakable': ['fragile', 'glass', 'ceramic'],
            'isBroken': ['broken', 'damaged', 'shattered'],
            'canFillWithLiquid': ['container', 'glass', 'cup', 'bowl'],
            'isFilledWithLiquid': ['filled', 'containing liquid', 'not empty'],
            'dirtyable': ['dirty', 'clean', 'stained'],
            'isDirty': ['dirty', 'stained', 'unclean'],
            'cookable': ['raw food', 'uncooked', 'ingredients'],
            'isCooked': ['cooked', 'prepared', 'ready to eat'],
            'sliceable': ['whole', 'uncut', 'sliceable'],
            'isSliced': ['sliced', 'cut', 'chopped'],
            'openable': ['door', 'cabinet', 'container with lid'],
            'isOpen': ['open', 'opened', 'not closed'],
            'pickupable': ['small object', 'movable item', 'handheld'],
            'moveable': ['movable', 'portable', 'not fixed']
        }
        return queries

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Process image for OWL-ViT input"""
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        
        return image

    def _detect_objects(self, inputs) -> Tuple[List[Dict], torch.Tensor]:
        """Detect objects using OWL-ViT"""
        import time
        image = inputs
        # Generate text queries for object detection
        common_objects = [[
            "table", "chair", "couch", "cabinet", "counter",
            "sink", "refrigerator", "microwave", "oven", "dishwasher",
            "cup", "bowl", "plate", "fork", "knife", "spoon",
            "bottle", "can", "food", "fruit", "vegetable",
            "door", "window", "light", "lamp", "clock",
            "book", "remote", "phone", "laptop", "tv"
        ]]
        
        # Run object detection
        with torch.no_grad():
            inputs = self.processor(text=common_objects, images=inputs, return_tensors="pt")
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.Tensor([image.shape[:2]]).to(self.config.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.config.confidence_threshold
        )[0]
        
        # Convert to list of dictionaries
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score >= self.config.confidence_threshold:
                detections.append({
                    'score': score.item(),
                    'label': label,
                    'bbox': box.cpu().numpy()
                })
        
        return detections, outputs.image_embeds

    def _detect_states(self, image_embeds: torch.Tensor, detections: List[Dict]) -> List[Dict]:
        """Detect object states using OWL-ViT's text-image matching"""
        import time
        timing_stats = {'text_matching': 0, 'spatial': 0}
        states = []
        
        # Project image embeddings through class head
        batch_size, height, width, hidden_dim = image_embeds.shape
        image_feats = image_embeds.reshape(-1, hidden_dim)
        image_class_embeds = self.model.class_head.dense0(image_feats)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        
        for det in detections:
            state = {
                'name': det['label'],
                'bbox': det['bbox'],
                'predicates': {'visible': 1}
            }
            
            start_text = time.time()
            
            for pred in self.bool_preds:
                if pred == 'visible':
                    continue
                    
                if pred in self.state_queries:
                    queries = self.state_queries[pred]
                    text_inputs = self.processor(
                        text=queries,
                        return_tensors="pt",
                        padding=True
                    )
                    text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        text_embeds = self.model.owlvit.get_text_features(**text_inputs)
                        text_embeds = text_embeds / (torch.linalg.norm(text_embeds, dim=-1, keepdim=True) + 1e-6)
                        
                        # Use einsum for proper batch matmul
                        pred_logits = torch.einsum("pd,qd->pq", image_class_embeds, text_embeds)
                        max_sim = pred_logits.max().item()
                        
                        state['predicates'][pred] = 1 if max_sim > self.config.state_confidence else 0
                else:
                    state['predicates'][pred] = 0
            
            timing_stats['text_matching'] += time.time() - start_text
            
            start_spatial = time.time()
            state['predicates']['inRoom'] = {'value': 1, 'room': self._detect_room(image_embeds)}
            
            for other_det in detections:
                if other_det != det:
                    if self._check_on_top(det['bbox'], other_det['bbox']):
                        state['predicates']['isOnTop'] = {
                            'value': 1,
                            'object': other_det['label']
                        }
                        break
            if 'isOnTop' not in state['predicates']:
                state['predicates']['isOnTop'] = {'value': 0, 'object': None}
                
            state['predicates']['isInside'] = {'value': 0, 'object': None}
            for other_det in detections:
                if other_det != det and self._check_inside(det['bbox'], other_det['bbox']):
                    state['predicates']['isInside'] = {
                        'value': 1,
                        'object': other_det['label']
                    }
                    break
            
            timing_stats['spatial'] += time.time() - start_spatial
            states.append(state)
        
        print(f"State detection breakdown - Text matching: {timing_stats['text_matching']:.3f}s, Spatial: {timing_stats['spatial']:.3f}s")
        return states

    def _detect_room(self, image_embeds: torch.Tensor) -> str:
        """Detect room type using OWL-ViT"""
        room_types = ["kitchen", "living room", "bedroom", "bathroom", "office"]
        
        text_inputs = self.processor(text=room_types, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.config.device) for k, v in text_inputs.items()}
        
        # Project image embeddings through class head
        batch_size, height, width, hidden_dim = image_embeds.shape
        image_feats = image_embeds.reshape(-1, hidden_dim)
        image_class_embeds = self.model.class_head.dense0(image_feats)
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        
        with torch.no_grad():
            text_embeds = self.model.owlvit.get_text_features(**text_inputs)
            text_embeds = text_embeds / (torch.linalg.norm(text_embeds, dim=-1, keepdim=True) + 1e-6)
            
            pred_logits = torch.einsum("pd,qd->pq", image_class_embeds, text_embeds)
            most_likely_room = room_types[pred_logits.mean(dim=0).argmax().item()]
        
        return most_likely_room

    def _check_on_top(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """Check if bbox1 is on top of bbox2"""
        y_bottom_1 = bbox1[3]
        y_top_2 = bbox2[1]
        x_center_1 = (bbox1[0] + bbox1[2]) / 2
        x_left_2, x_right_2 = bbox2[0], bbox2[2]
        
        return (abs(y_bottom_1 - y_top_2) < 20 and 
                x_left_2 <= x_center_1 <= x_right_2)

    def _check_inside(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """Check if bbox1 is inside bbox2"""
        return (bbox1[0] > bbox2[0] and bbox1[2] < bbox2[2] and
                bbox1[1] > bbox2[1] and bbox1[3] < bbox2[3])

    def get_object_states(self, image: np.ndarray) -> Tuple[Dict, List[Dict]]:
        """
        Main method to get object states from an image
        
        Returns:
            Tuple[Dict, List[Dict]]: 
                - Object states dictionary
                - List of detections with bounding boxes
        """
        import time
        start_total = time.time()
        
        # Time image processing
        start = time.time()
        inputs = self._process_image(image)
        process_time = time.time() - start
        print(f"Image processing time: {process_time:.3f}s")
        
        # Time object detection
        start = time.time()
        detections, image_embeds = self._detect_objects(image)
        detect_time = time.time() - start
        print(f"Object detection time: {detect_time:.3f}s")
        print(f"Number of objects detected: {len(detections)}")
        
        if not detections:
            total_time = time.time() - start_total
            print(f"Total state generation time: {total_time:.3f}s")
            return {}, []
        
        # Time state detection
        start = time.time()
        states = self._detect_states(image_embeds, detections)
        state_time = time.time() - start
        print(f"State detection time: {state_time:.3f}s")
        
        # Convert to required format
        object_states = {}
        for i, state in enumerate(states, 1):
            object_states[f"region_{i}"] = state
        
        total_time = time.time() - start_total
        print(f"Total state generation time: {total_time:.3f}s")
        print(f"Breakdown - Processing: {process_time:.3f}s, Detection: {detect_time:.3f}s, States: {state_time:.3f}s")
        
        return object_states, detections

    def visualize_results(
        self,
        image: np.ndarray,
        object_states: Dict,
        detections: List[Dict]
    ) -> np.ndarray:
        """Visualize detection results with states"""
        vis_image = image.copy()
        
        # Draw bounding boxes and labels
        for det in detections:
            bbox = det['bbox'].astype(np.int32)
            cv2.rectangle(
                vis_image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )
        
        # Add text annotations
        y_offset = 30
        for region_id, state_info in object_states.items():
            # Handle both boolean and relational predicates
            active_predicates = []
            
            # Add boolean predicates
            active_predicates.extend([
                pred for pred in self.bool_preds
                if state_info['predicates'].get(pred) == 1
            ])
            
            # Add relational predicates with their targets
            for pred in self.relational_preds:
                pred_info = state_info['predicates'].get(pred)
                if pred_info and pred_info.get('value') == 1:
                    if pred == 'inRoom':
                        active_predicates.append(f"{pred}({pred_info['room']})")
                    else:
                        active_predicates.append(f"{pred}({pred_info['object']})")
            
            # Create text
            text = f"{state_info['name']}: {', '.join(active_predicates[:3])}"
            
            cv2.putText(
                vis_image, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            y_offset += 25
            
        return vis_image

def test_detector():
    """Test script for OWLViTStateDetector"""
    import matplotlib.pyplot as plt
    from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv
    
    # Initialize detector and simulation
    config = OWLViTStateConfig(
        max_image_size=640,
        confidence_threshold=0.1
    )
    detector = OWLViTStateDetector(config)
    sim = RobosuiteSimEnv()
    
    # Get and process image
    image = sim.get_camera_image()
    object_states, detections = detector.get_object_states(image)
    
    # Create visualization
    vis_image = detector.visualize_results(image, object_states, detections)
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # Print detailed results
    print("\nDetected Object States:")
    for region_id, state_info in object_states.items():
        print(f"\n{state_info['name']} ({region_id}):")
        for pred, val in state_info['predicates'].items():
            if isinstance(val, dict):
                if val['value'] == 1:
                    print(f"  {pred}: {val}")
            elif val == 1:
                print(f"  {pred}: {val}")

if __name__ == "__main__":
    test_detector()