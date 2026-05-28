import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import anthropic
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from cognitive_bt_framework.utils import get_claude_key
import torch
import time

from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMMaskGenerator, FastSAMConfig

@dataclass
class FastSAMStateConfig:
    """Configuration for FastSAM-based state detection"""
    model_type: str = "FastSAM-s"
    max_image_size: int = 640
    conf_threshold: float = 0.6
    iou_threshold: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model: str = "claude-3-5-sonnet-20240620"

class FastSAMStateDetector:
    """
    Class for detecting object states using FastSAM for segmentation and Claude for state analysis
    """
    def __init__(self, config: FastSAMStateConfig = None):
        """Initialize the state detector with FastSAM and Claude interfaces"""
        self.config = config or FastSAMStateConfig()
        
        # Initialize FastSAM
        self.mask_generator = FastSAMMaskGenerator(FastSAMConfig(
            model_type=self.config.model_type,
            device=self.config.device,
            max_image_size=self.config.max_image_size,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold
        ))
        
        # Initialize Claude client
        self.llm_interface = anthropic.Anthropic(api_key=get_claude_key())
        
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

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _crop_object(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract object crop using mask"""
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        crop = image[y:y+h, x:x+w].copy()
        mask_crop = mask[y:y+h, x:x+w]
        crop[~mask_crop] = 0
        return crop

    def generate_state_query(self, image: np.ndarray, masks: np.ndarray, metadata: Dict) -> str:
        """Generate a prompt for Claude to analyze object states"""
        # Create individual object crops
        object_crops = []
        for mask_id in np.unique(masks)[1:]:  # Skip 0 (background)
            if mask_id in metadata:
                mask = masks == mask_id
                crop = self._crop_object(image, mask)
                object_crops.append({
                    'id': mask_id,
                    'crop': crop,
                    'area': metadata[mask_id]['area']
                })

        # Sort objects by area (largest first)
        object_crops.sort(key=lambda x: x['area'], reverse=True)
        
        # Create prompt
        prompt = f"""Analyze this scene image to identify objects and their states. For each segmented region:

1. First identify what the object is, considering its visual appearance and context.
2. Then evaluate the following predicates for each identified object:

Boolean predicates (1 for true, 0 for false):
{', '.join(self.bool_preds)}

Relational predicates (requiring additional object or room information):
{', '.join(self.relational_preds)}

Please provide results in this JSON format:
{{
    "region_1": {{
        "name": "object_name",
        "predicates": {{
            "visible": 1,
            "receptacle": 0,
            "inRoom": {{"value": 1, "room": "kitchen"}},
            "isOnTop": {{"value": 1, "object": "counter"}},
            "isInside": {{"value": 0, "object": null}}
        }}
    }},
    ...
}}

For relational predicates:
- 'inRoom' should specify the room name
- 'isOnTop' should specify the object being rested upon
- 'isInside' should specify the containing object

Focus on clearly visible properties and spatial relationships. Consider:
- Physical properties (broken, sliced, cooked, etc.)
- Containment relationships (what objects can contain others)
- Spatial relationships (on top, inside, proximity)
- States (open/closed, switched on/off, filled)

Remember all of the following predicates should be evaluated for all detected objects and be present in the output state:
{', '.join(self.predicate_list)}

For boolean predicates, use 1 for true and 0 for false.
For relational predicates, include both the value (1/0) and the related object/room information."""
        return prompt

    def get_object_states(self, image: np.ndarray) -> Tuple[Dict, np.ndarray, Dict]:
        """
        Main method to get object states from an image
        
        Returns:
            Tuple[Dict, np.ndarray, Dict]: 
                - Object states dictionary
                - Labeled masks array
                - Mask metadata dictionary
        """
        # Get segmentation masks using FastSAM
        masks, metadata = self.mask_generator.generate_masks(image)
        
        if not np.any(masks):
            return {}, masks, metadata
            
        # Generate query for Claude
        prompt = self.generate_state_query(image, masks, metadata)
        
        # Get state analysis from Claude
        try:
            response = self.llm_interface.messages.create(
                model=self.config.llm_model,
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
            
            import json
            
            # Extract JSON from response
            response_text = response.content[0].text
            print(response_text)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    object_states = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Response text: {response_text}")
                    object_states = {}
            else:
                object_states = {}
                
        except Exception as e:
            print(f"Error querying Claude: {e}")
            return {}, masks, metadata
            
        # Add mask data to object states
        for region_id, state_info in object_states.items():
            if int(region_id.split('_')[1]) in metadata:
                mask_id = int(region_id.split('_')[1])
                state_info['mask'] = masks == mask_id
                state_info['bbox'] = metadata[mask_id]['bbox']
        
        return object_states, masks, metadata

    def visualize_results(
        self,
        image: np.ndarray,
        object_states: Dict,
        masks: np.ndarray,
        metadata: Dict
    ) -> np.ndarray:
        """Visualize detection results with masks and states"""
        # Create mask visualization
        vis_image = self.mask_generator.visualize_masks(image, masks, metadata, alpha=0.3)
        
        # Add text annotations
        y_offset = 30
        for state_info in object_states:
            # Add object name and active predicates
            text = f"{state_info['name']}: "
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
            
            text += ", ".join(active_predicates[:3])  # Show first 3 predicates
            
            cv2.putText(
                vis_image, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            y_offset += 25
            
        return vis_image

async def test_detector():
    import matplotlib.pyplot as plt
    from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv
    from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

    # Initialize components
    config = FastSAMStateConfig(
        model_type="FastSAM-x",
        max_image_size=640,
        conf_threshold=0.65
    )
    detector = FastSAMStateDetector(config)
    sim = RobosuiteSimEnv()
    llm_interface = LLMInterfaceOpenAI(model_name='gpt-4o')

    # Get and process image
    image = sim.get_camera_image()
    start = time.time()
    object_states, masks, metadata = await llm_interface.get_object_states(image, detector.mask_generator)
    print(f"It took {time.time() - start} seconds to get object states")
    
    # First show overall scene with all detections
    vis_image = detector.visualize_results(image, object_states, masks, metadata)
    plt.figure(figsize=(15, 10))
    plt.imshow(vis_image)
    plt.title("Complete Scene with All Detections")
    plt.axis('off')
    plt.show()

    # Now iterate through each detected object
    for i, state_info in enumerate(object_states):
        print(state_info)
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Get object name and region
        region_id = state_info['region_id']
        object_name = state_info['name']
        
        # Get object mask and apply it to image
        mask = state_info.get('mask', None)
        if mask is not None:
            # Show original image with mask overlay
            masked_image = image.copy()
            masked_image[~mask] = masked_image[~mask] // 4  # Dim non-masked regions
            ax1.imshow(masked_image)
            ax1.set_title(f"Object: {object_name}")
            ax1.axis('off')
            
            # Show state information as text
            state_text = f"Object: {object_name}\nRegion: {region_id}\n\nPredicates:\n"
            for pred, val in state_info['predicates'].items():
                if isinstance(val, dict):
                    # Handle relational predicates
                    if val['value'] == 1:
                        if 'room' in val:
                            state_text += f"- {pred}: {val['room']}\n"
                        elif 'object' in val and val['object']:
                            state_text += f"- {pred}: {val['object']}\n"
                elif val == 1:
                    # Handle boolean predicates
                    state_text += f"- {pred}\n"
            
            ax2.text(0.1, 0.9, state_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax2.axis('off')
            ax2.set_title("State Information")
            
            plt.tight_layout()
            plt.show()

            # Print detailed state information
            print(f"\nDetailed State Information for {object_name} ({region_id}):")
            print(state_text)
            
            # Optional: wait for user input before showing next object
            input("Press Enter to continue to next object...")
            plt.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_detector())
    input()