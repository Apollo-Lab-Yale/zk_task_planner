from ultralytics import YOLO
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional
from cognitive_bt_framework.utils import get_claude_key
from cognitive_bt_framework.src.vision.object_detection.yolo import ObjectDetection
from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv

import anthropic

class ObjectStateDetector:
    def __init__(self, yolo_model_path: str = "yolo11x-seg.pt", claude_model: str = "claude-3-5-sonnet-20240620"):
        """
        Initialize the object state detector with YOLO and Claude interfaces
        """
        self.yolo = YOLO(yolo_model_path)
        self.client = anthropic.Anthropic(api_key=get_claude_key())
        self.claude_model = claude_model
        self.predicate_list = [
            'visible', 'receptacle', 'toggleable', 'breakable',
            'canFillWithLiquid', 'dirtyable', 'cookable', 'isHeatSource',
            'sliceable', 'openable', 'pickupable', 'moveable', 'isOpen',
            'isToggled', 'isBroken', 'isFilledWithLiquid', 'isDirty',
            'isCooked', 'isSliced', 'isPickedUp'
        ]

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the image using YOLO
        """
        results = self.yolo(image, verbose=False)[0]
        detections = []
        
        if results.boxes is not None and results.masks is not None:
            for box, mask in zip(results.boxes, results.masks):
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'conf': box.conf.item(),
                    'cls': box.cls.item(),
                    'name': results.names[int(box.cls.item())],
                    'mask': mask.data[0].cpu().numpy()
                })
        return detections

    def generate_state_query(self, image: np.ndarray, detections: List[Dict]) -> str:
        """
        Generate a prompt for Claude to analyze object states
        """
        detected_objects = [f"{det['name']} (confidence: {det['conf']:.2f})" for det in detections]
        objects_str = "\n".join(detected_objects)
        predicates_str = "\n".join(self.predicate_list)
        
        return f"""Please analyze this image and provide boolean values (1 for true, 0 for false) for each predicate 
        for each detected object. Consider the visual evidence carefully.

        Detected Objects:
        {objects_str}

        Predicates to evaluate:
        {predicates_str}

        For each object, provide a JSON-like structure with predicate values. Only include predicates that can be 
        reasonably determined from the image. If a predicate cannot be determined with reasonable confidence, 
        omit it from the results.

        Focus on clearly visible properties and states. For example:
        - 'visible' should be 1 for detected objects
        - 'receptacle' for objects that can contain other items
        - 'openable' for objects with visible hinges or lids
        - 'isOpen' for objects currently in an open state
        - 'isFilledWithLiquid' for containers with visible liquid

        Return the results in a simple dictionary format like this example:
        {{
            "object1": {{
                "visible": 1,
                "canFillWithLiquid": 1,
                "isFilledWithLiquid": 0
            }},
            "object2": {{
                "visible": 1,
                "receptacle": 1
            }}
        }}

        Only provide the dictionary/JSON response with no additional explanation or text."""

    def query_claude(self, image: np.ndarray, prompt: str) -> Dict:
        """
        Query Claude with the image and prompt, return parsed results
        """
        try:
            response = self.client.messages.create(
                model=self.claude_model,
                max_tokens=2000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode_image(image)
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            # Find JSON content between curly braces
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return eval(json_str)  # Using eval since we know the format is safe
            return {}
            
        except Exception as e:
            print(f"Error querying Claude: {e}")
            return {}

    def get_object_states(self, image: np.ndarray) -> Dict:
        """
        Main method to get object states from an image
        """
        # Detect objects using YOLO
        detections = self.detect_objects(image)
        
        if not detections:
            return {}
            
        # Generate query for Claude
        prompt = self.generate_state_query(image, detections)
        
        # Get state analysis from Claude
        object_states = self.query_claude(image, prompt)
        
        return object_states

    def visualize_results(self, image: np.ndarray, object_states: Dict) -> np.ndarray:
        """
        Visualize the detection results and object states
        """
        vis_image = image.copy()
        y_offset = 30
        
        for obj_name, states in object_states.items():
            # Add object name and states as text overlay
            text = f"{obj_name}: "
            true_predicates = [pred for pred, val in states.items() if val == 1]
            text += ", ".join(true_predicates)
            
            cv2.putText(vis_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
            
        return vis_image

if __name__ == "__main__":
    from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv  # Import your simulation environment
    import matplotlib.pyplot as plt
    # Initialize detector and simulation
    detector = ObjectStateDetector()
    sim = RobosuiteSimEnv()
    
    # Get and process image
    image = sim.get_camera_image()
    object_states = detector.get_object_states(image)
    
    # Create visualization
    fig = detector.visualize_results(image, object_states)
    
    # Display the results
    plt.show()
    
    # Optional: Save the figure
    # fig.savefig('object_states_visualization.png', bbox_inches='tight', dpi=300)
    
    # Print detailed results
    print("\nDetected Object States:")
    for obj_name, states in object_states.items():
        print(f"\n{obj_name}:")
        for pred, val in states.items():
            print(f"  {pred}: {val}")
    input()        
    plt.close()  # Clean up the figure when done
