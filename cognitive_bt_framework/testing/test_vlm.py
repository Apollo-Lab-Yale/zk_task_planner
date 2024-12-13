import os
from PIL import Image
import numpy as np
import numpy as np
import base64
import io
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
 # Disable GLFW warnings
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import InferenceClient

from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv
from cognitive_bt_framework.src.vision.object_detection.yolo import ObjectDetection
from cognitive_bt_framework.utils.llm_utils import get_hf_key


class PredicateDetector:
    def __init__(self, model_path="THUDM/cogvlm-chat-hf"):
        hf_key = get_hf_key()
        # Ensure CUDA is visible
        self.client = InferenceClient(api_key=hf_key)
        
        self.predicate_prompts = {
            'visible': "Is the {obj_name} clearly visible in region {bbox}?",
            'receptacle': "Looking at the {obj_name} in region {bbox}, can it contain or hold other objects?",
            'toggleable': "Can the {obj_name} in region {bbox} be turned on or off?", 
            'breakable': "Is the {obj_name} in region {bbox} fragile or breakable?",
            'canFillWithLiquid': "Can the {obj_name} in region {bbox} be filled with liquid?",
            'dirtyable': "Can the {obj_name} in region {bbox} become dirty or stained?",
            'cookable': "Can the {obj_name} in region {bbox} be cooked?",
            'isHeatSource': "Is the {obj_name} in region {bbox} a source of heat?",
            'sliceable': "Can the {obj_name} in region {bbox} be cut or sliced?",
            'openable': "Can the {obj_name} in region {bbox} be opened and closed?",
            'pickupable': "Can the {obj_name} in region {bbox} be picked up and carried?",
            'moveable': "Can the {obj_name} in region {bbox} be moved around?",
            'isOpen': "Is the {obj_name} in region {bbox} currently in an open state?",
            'isToggled': "Is the {obj_name} in region {bbox} currently turned on?",
            'isBroken': "Does the {obj_name} in region {bbox} appear to be broken?",
            'isFilledWithLiquid': "Does the {obj_name} in region {bbox} currently contain liquid?",
            'isDirty': "Does the {obj_name} in region {bbox} appear dirty or stained?",
            'isCooked': "Has the {obj_name} in region {bbox} been cooked?",
            'isSliced': "Has the {obj_name} in region {bbox} been cut or sliced?",
            'isPickedUp': "Is the {obj_name} in region {bbox} currently being held?"
        }

    def _encode_image(self, image):
        """Convert PIL Image or numpy array to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def get_object_states(self, image, bbox, class_name):
        try:
            image_b64 = self._encode_image(image)
            
            # Scale bbox
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                w, h = image.size
                
            scaled_bbox = [
                int(bbox[0] * 1000 / w),
                int(bbox[1] * 1000 / h),
                int(bbox[2] * 1000 / w),
                int(bbox[3] * 1000 / h)
            ]
            bbox_str = f"[[{scaled_bbox[0]},{scaled_bbox[1]},{scaled_bbox[2]},{scaled_bbox[3]}]]"
            
            # Construct combined prompt
            combined_prompt = (
                f"For the {class_name} in region {bbox_str}, answer the following questions with Yes or No only:\n"
                f"1. Is the {class_name} clearly visible?\n"
                f"2. Can the {class_name} contain or hold other objects?\n"
                f"3. Can the {class_name} be turned on or off?\n"
                f"4. Is the {class_name} fragile or breakable?\n"
                f"5. Can the {class_name} be filled with liquid?\n"
                f"6. Can the {class_name} become dirty?\n"
                f"7. Can the {class_name} be cooked?\n"
                f"8. Is the {class_name} a heat source?\n"
                f"9. Can the {class_name} be cut or sliced?\n"
                f"10. Can the {class_name} be opened and closed?\n"
                f"11. Can the {class_name} be picked up?\n"
                f"12. Can the {class_name} be moved?\n"
                f"13. Is the {class_name} currently open?\n"
                f"14. Is the {class_name} currently turned on?\n"
                f"15. Does the {class_name} appear broken?\n"
                f"16. Does the {class_name} contain liquid?\n"
                f"17. Does the {class_name} appear dirty?\n"
                f"18. Has the {class_name} been cooked?\n"
                f"19. Has the {class_name} been sliced?\n"
                f"20. Is the {class_name} currently being held?\n"
                "Respond with a numbered list of Yes/No answers only. "
                "Yes or no answers should be followed by - <explaination>"
            )

            # Make single API call
            stream = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }],
                max_tokens=1000,
                stream=True
            )

            # Collect response
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            # Parse responses into states dict
            states = {}
            predicate_list = list(self.predicate_prompts.keys())
            
            # Split response into lines and process each answer
            answers = [line.strip().lower() for line in response_text.split('\n') if line.strip()]
            
            for i, answer in enumerate(answers):
                print(f"______ " + answer)
                if i < len(predicate_list):
                    states[predicate_list[i]] = 'yes' in answer and 'no' not in answer
            states['name'] = class_name
            return states
                
        except Exception as e:
            print(f"Error in get_object_states: {str(e)}")
            return {}
        
    def _parse_response(self, response):
        """Convert model response to boolean."""
        pos_words = ['yes', 'true', 'correct', 'it is', 'it does']
        neg_words = ['no', 'false', 'incorrect', 'it is not', 'it does not']
        
        response = response.lower()
        
        is_positive = any(word in response for word in pos_words)
        is_negative = any(word in response for word in neg_words)
        
        if is_positive and not is_negative:
            return True
        return False

    def detect_spatial_relationships(self, image, detections):
        """Get spatial relationships between objects."""
        # Convert image to base64
        image_b64 = self._encode_image(image)
            
        # Format detections
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
            
        objects_desc = []
        for det in detections:
            bbox = [
                int(det['bbox'][0] * 1000 / w),
                int(det['bbox'][1] * 1000 / h),
                int(det['bbox'][2] * 1000 / w),
                int(det['bbox'][3] * 1000 / h)
            ]
            objects_desc.append(f"{det['name']} at [[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]]")
            
        prompt = "Describe the spatial relationships between these objects: " + ", ".join(objects_desc)
        
        stream = self.client.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }],
                    max_tokens=1000,
                    stream=True
                )
                
        # Collect response
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        return response_text
    
# Usage example:
if __name__ == "__main__":
    detector = PredicateDetector()
    obj_detector = ObjectDetection()
    sim = RobosuiteSimEnv()
    sim.start()
    
    # Get image and detections
    image = sim.get_camera_image()
    detections = obj_detector.detect_objects(image)

    # Process a single object
    states = detector.get_object_states(
        image,
        bbox=[100, 100, 200, 200],
        class_name="cup"
    )
    print('_______________')
    print(states)
    print('_______________')
    input()
    # Get relationships between objects
    relationships = detector.detect_spatial_relationships(
        image,
        detections=[
            {"name": "cup", "bbox": [100,100,200,200]},
            {"name": "table", "bbox": [50,150,300,300]}
        ])
    print(f"Spatial relationships: {relationships}")

    sim.stop()