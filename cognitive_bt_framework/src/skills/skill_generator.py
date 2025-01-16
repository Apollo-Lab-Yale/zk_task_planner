from typing import Dict, List, Optional, Any
import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import time

from cognitive_bt_framework.src.llm_interface import LLMInterfaceOpenAI

@dataclass
class Skill:
    """Data class to store skill information"""
    name: str
    abstract_action: str
    target_object: str
    primitive_sequence: List[str]
    parameters: Dict[str, Any]
    prerequisites: List[str]
    constraints: List[str]
    image_id: str

class SkillGenerator:
    def __init__(self, llm_interface, skills_dir: str = "stored_skills"):
        """
        Initialize the skill generator with an LLM interface and storage directory
        
        Args:
            llm_interface: LLM interface for generating skills
            skills_dir: Directory to store generated skills
        """
        self.llm = llm_interface
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.image_dir = self.skills_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.skills_cache: Dict[str, Skill] = {}
        self._load_stored_skills()

    def _load_stored_skills(self):
        """Load previously stored skills from disk"""
        for skill_file in self.skills_dir.glob("*.json"):
            try:
                with open(skill_file, 'r') as f:
                    skill_data = json.load(f)
                    skill = Skill(**skill_data)
                    self.skills_cache[skill.name] = skill
            except Exception as e:
                print(f"Error loading skill from {skill_file}: {e}")

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_image(self, image: np.ndarray, image_id: str):
        """Save image to disk"""
        cv2.imwrite(str(self.image_dir / f"{image_id}.png"), image)

    def _load_image(self, image_id: str) -> Optional[np.ndarray]:
        """Load image from disk"""
        image_path = self.image_dir / f"{image_id}.png"
        if image_path.exists():
            return cv2.imread(str(image_path))
        return None

    async def generate_skill(self, 
                           image: np.ndarray,
                           mask: np.ndarray,
                           abstract_action: str,
                           target_object: str) -> Optional[Skill]:
        """
        Generate a new skill using the LLM interface based on image input
        
        Args:
            image: Input image of the object
            mask: Binary mask of the target object
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
        
        Returns:
            Generated skill if successful, None otherwise
        """
        # Create masked image
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        # Generate unique ID for this image
        image_id = f"{abstract_action}_{target_object}_{hash(mask.tobytes()) & 0xFFFFFF:06x}"
        self._save_image(masked_image, image_id)
        
        # Create prompt for skill definition
        combined_prompt = [
            {"role": "system", "content": f"""Analyze the object visually and generate a complete skill definition for performing {abstract_action} on a {target_object}.

            First, determine the specific subtype of the action based on the object's visual characteristics.
            The skill name should follow the format: action_targetobject_mechanism

            Return a JSON object with the following structure:
            {{
                "skill_name": "specific_action_name_with_mechanism",
                "primitive_sequence": [
                    "list of primitive actions using only these commands:",
                    "apply_force(direction, keywords)",
                        "direction: one of [up, down, left, right, push, pull] relative to object surface",
                        "keywords: list of descriptive words to identify the component (e.g., [handle, knob, button])",
                    "apply_torque(axis, keywords)",
                        "axis: one of [clockwise, counterclockwise] relative to object surface",
                        "keywords: list of descriptive words to identify the component",
                    "close_gripper()",
                    "release()",
                    "moveGripperToPose(keywords)",
                        "keywords: list of descriptive words for the target pose location",
                    "retractGripper()"
                ],
                "parameters": {{
                    "force_threshold": "low/medium/high",
                    "precision_required": "low/medium/high",
                    "speed_requirement": "slow/medium/fast"
                }},
                "prerequisites": [
                    "list of required conditions"
                ],
                "constraints": [
                    "list of safety limits and constraints"
                ]
            }}
            
            Base all values on the visual appearance of the object.
            Return only the raw JSON object with no additional text."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Abstract Action: {abstract_action}
                Target Object: {target_object}
                
                Generate a complete skill definition for this task based on the object's visual appearance.
                First determine the specific subtype of action needed based on the object's
                characteristics, then generate the complete skill definition including
                primitive sequence, parameters, prerequisites, and constraints.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(masked_image)}"
                }}
            ]}
        ]
        
        # Get response
        skill_response = await self.llm.query_llm(combined_prompt)
        try:
            skill_data = json.loads(skill_response)
        except json.JSONDecodeError:
            print("Error parsing skill definition")
            return None

        # Create skill using the skill_data
        skill = Skill(
            name=skill_data.get('skill_name', f"{abstract_action}_{target_object}_generic_{hash(image_id) & 0xFFFFFF:06x}"),
            abstract_action=abstract_action,
            target_object=target_object,
            primitive_sequence=skill_data.get('primitive_sequence', []),
            parameters=skill_data.get('parameters', {}),
            prerequisites=skill_data.get('prerequisites', []),
            constraints=skill_data.get('constraints', []),
            image_id=image_id
        )
        
        self._store_skill(skill)
        return skill

    async def find_similar_skill(self, 
                               image: np.ndarray,
                               mask: np.ndarray,
                               abstract_action: str,
                               target_object: str) -> Optional[Skill]:
        """
        Find a similar existing skill based on image
        
        Args:
            image: Input image of the object
            mask: Binary mask of the target object
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
        
        Returns:
            Most similar existing skill if found, None otherwise
        """
        if not self.skills_cache:
            return None
            
        masked_image = image.copy()
        masked_image[~mask] = 0
        
        similarity_prompt = [
            {"role": "system", "content": """Compare the given object visually with the existing skill.
            Return a similarity score between 0 and 1, where 1 means identical and 0 means completely different.
            Consider visual similarity of the mechanism and interaction points.
            Return only the numeric score."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                New Task:
                Abstract Action: {abstract_action}
                Target Object: {target_object}
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(masked_image)}"
                }}
            ]}
        ]

        best_score = 0
        best_skill = None
        
        for skill in self.skills_cache.values():
            if skill.abstract_action != abstract_action or skill.target_object != target_object:
                continue
                
            existing_image = self._load_image(skill.image_id)
            if existing_image is None:
                continue
                
            similarity_prompt[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(existing_image)}"
                }
            })
            
            score_response = await self.llm.query_llm(similarity_prompt)
            try:
                score = float(score_response)
                if score > best_score:
                    best_score = score
                    best_skill = skill
            except ValueError:
                continue

        return best_skill if best_score > 0.8 else None

    async def adapt_skill(self, 
                         base_skill: Skill,
                         new_image: np.ndarray,
                         new_mask: np.ndarray) -> Optional[Skill]:
        """
        Adapt an existing skill based on new image
        
        Args:
            base_skill: The skill to adapt
            new_image: Image of the new object
            new_mask: Mask of the new object
        
        Returns:
            Adapted skill if successful, None otherwise
        """
        masked_image = new_image.copy()
        masked_image[~new_mask] = 0
        base_image = self._load_image(base_skill.image_id)
        
        if base_image is None:
            return None
            
        adaptation_prompt = [
            {"role": "system", "content": """Adapt the existing skill's primitive sequence and parameters for the new object.
            Consider visual differences in the mechanism and interaction points.
            Return ONLY a JSON object with 'primitive_sequence' and 'parameters' fields. Return nothing else."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Original Skill:
                {json.dumps(asdict(base_skill), indent=2)}
                
                Adapt the skill based on visual differences.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(base_image)}"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(masked_image)}"
                }}
            ]}
        ]
        
        adaptation_response = await self.llm.query_llm(adaptation_prompt)
        try:
            adapted_data = json.loads(adaptation_response)
            
            # Generate new image ID and save
            new_image_id = f"{base_skill.abstract_action}_{base_skill.target_object}_{hash(new_mask.tobytes()) & 0xFFFFFF:06x}"
            self._save_image(masked_image, new_image_id)
            
            adapted_skill = Skill(
                name=f"{base_skill.name}_adapted_{hash(new_image_id) & 0xFFFFFF:06x}",
                abstract_action=base_skill.abstract_action,
                target_object=base_skill.target_object,
                primitive_sequence=adapted_data['primitive_sequence'],
                parameters=adapted_data['parameters'],
                prerequisites=base_skill.prerequisites,
                constraints=base_skill.constraints,
                image_id=new_image_id
            )
            
            self._store_skill(adapted_skill)
            return adapted_skill
            
        except json.JSONDecodeError:
            print("Error parsing adaptation response")
            return None

    def _store_skill(self, skill: Skill):
        """Store a skill both in cache and on disk"""
        self.skills_cache[skill.name] = skill
        skill_path = self.skills_dir / f"{skill.name}.json"
        with open(skill_path, 'w') as f:
            json.dump(asdict(skill), f, indent=2)

    def list_skills(self) -> List[str]:
        """Return a list of all available skill names"""
        return list(self.skills_cache.keys())

    def get_skill_details(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific skill"""
        skill = self.get_skill(skill_name)
        if skill:
            details = asdict(skill)
            image = self._load_image(skill.image_id)
            if image is not None:
                details['image'] = self._encode_image(image)
            return details
        return None

    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """Retrieve a skill by name"""
        return self.skills_cache.get(skill_name)

async def test_skill_generator():
    """
    Test function to demonstrate SkillGenerator functionality with different switch-on actions
    """
    # Initialize LLM interface and skill generator
    llm = LLMInterfaceOpenAI(model_name="gpt-4-turbo")
    generator = SkillGenerator(llm, skills_dir="test_skills")
    IMAGE_SIZE = 640

    # Test Case 1: Wall Light Switch (Toggle mechanism)
    print("\nTest Case 1: Wall Light Switch")
    print("-----------------------------------------")
    
    # Create wall switch image and mask
    switch_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    # Draw a rectangular switch plate
    cv2.rectangle(switch_image, (295, 270), (345, 370), (200, 200, 200), -1)  # Grey plate
    # Draw the switch lever
    cv2.rectangle(switch_image, (305, 300), (335, 340), (240, 240, 240), -1)  # White switch
    # Add texture to make it look like a toggle
    cv2.circle(switch_image, (320, 320), 5, (180, 180, 180), -1)  # Switch detail
    
    switch_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    switch_mask[270:370, 295:345] = True
    
    start = time.time()
    switch_skill = await generator.generate_skill(
        image=switch_image,
        mask=switch_mask,
        abstract_action="SwitchOn",
        target_object="light"
    )
    print(f"Time to generate skill: {time.time() - start}")

    if switch_skill:
        print(f"Generated Skill Name: {switch_skill.name}")
        print("Primitive Sequence:")
        for primitive in switch_skill.primitive_sequence:
            print(f"  - {primitive}")
        print(f"Parameters: {json.dumps(switch_skill.parameters, indent=2)}")

    # Test Case 2: Stove Burner (Rotary Knob)
    print("\nTest Case 2: Stove Burner Control")
    print("-------------------------------------------------")
    
    # Create stove knob image and mask
    stove_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    # Draw circular knob
    cv2.circle(stove_image, (320, 320), 30, (50, 50, 50), -1)  # Dark grey knob
    # Add indicator line and markings
    cv2.line(stove_image, (320, 320), (320, 300), (255, 255, 255), 2)
    # Add control markings
    for angle in range(0, 271, 90):
        pt1 = (int(320 + 25 * np.cos(np.radians(angle))), 
               int(320 + 25 * np.sin(np.radians(angle))))
        cv2.circle(stove_image, pt1, 2, (200, 200, 200), -1)
    
    stove_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    cv2.circle(stove_mask.astype(np.uint8), (320, 320), 30, 1, -1)
    stove_mask = stove_mask.astype(bool)
    
    stove_skill = await generator.generate_skill(
        image=stove_image,
        mask=stove_mask,
        abstract_action="SwitchOn",
        target_object="stove"
    )
    
    if stove_skill:
        print(f"Generated Skill Name: {stove_skill.name}")
        print("Primitive Sequence:")
        for primitive in stove_skill.primitive_sequence:
            print(f"  - {primitive}")
        print(f"Parameters: {json.dumps(stove_skill.parameters, indent=2)}")

    # Test Case 3: Push Button Lamp
    print("\nTest Case 3: Push Button Lamp")
    print("--------------------------------")
    
    # Create lamp with push button image and mask
    lamp_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    # Draw lamp base
    cv2.rectangle(lamp_image, (280, 300), (360, 340), (200, 200, 200), -1)
    # Draw push button with more detail
    cv2.circle(lamp_image, (320, 320), 12, (180, 180, 180), -1)  # Button surround
    cv2.circle(lamp_image, (320, 320), 10, (255, 0, 0), -1)      # Red button
    # Add button detail
    cv2.circle(lamp_image, (320, 320), 5, (220, 0, 0), -1)       # Button center
    
    lamp_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    lamp_mask[300:340, 280:360] = True
    
    
    lamp_skill = await generator.generate_skill(
        image=lamp_image,
        mask=lamp_mask,
        abstract_action="SwitchOn",
        target_object="lamp"
    )
    
    if lamp_skill:
        print(f"Generated Skill Name: {lamp_skill.name}")
        print("Primitive Sequence:")
        for primitive in lamp_skill.primitive_sequence:
            print(f"  - {primitive}")
        print(f"Parameters: {json.dumps(lamp_skill.parameters, indent=2)}")

    # Test Case 4: Touch-Sensitive Lamp (Adaptation)
    print("\nTest Case 4: Adapt Push Button to Touch Lamp")
    print("--------------------------------------------")
    if lamp_skill:
        # Create touch-sensitive lamp image
        touch_lamp_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        # Draw lamp base with touch-sensitive surface
        cv2.rectangle(touch_lamp_image, (280, 300), (360, 340), (220, 220, 220), -1)
        # Add touch sensor indication
        cv2.circle(touch_lamp_image, (320, 320), 15, (180, 180, 180), 2)
        cv2.circle(touch_lamp_image, (320, 320), 12, (160, 160, 160), -1)
        # Add touch symbol
        cv2.line(touch_lamp_image, (315, 315), (325, 325), (100, 100, 100), 2)
        cv2.line(touch_lamp_image, (315, 325), (325, 315), (100, 100, 100), 2)
        
        touch_lamp_mask = lamp_mask.copy()
        
        touch_lamp_states = lamp_states.copy()
        touch_lamp_states["pose"] = [0.6, 0.3, 0.7]
        
        adapted_skill = await generator.adapt_skill(
            lamp_skill,
            touch_lamp_image,
            touch_lamp_mask,
            touch_lamp_states
        )
        
        if adapted_skill:
            print(f"Adapted Skill Name: {adapted_skill.name}")
            print("\nOriginal Push Button Sequence:")
            for primitive in lamp_skill.primitive_sequence:
                print(f"  - {primitive}")
            print("\nAdapted Touch Lamp Sequence:")
            for primitive in adapted_skill.primitive_sequence:
                print(f"  - {primitive}")
            print(f"Parameters: {json.dumps(adapted_skill.parameters, indent=2)}")

    # Test Case 5: Skill Management
    print("\nTest Case 5: Skill Management")
    print("-----------------------------")
    print("Available Skills:", generator.list_skills())
    
    if switch_skill:
        retrieved_skill = generator.get_skill_details(switch_skill.name)
        if retrieved_skill:
            print(f"\nRetrieved Skill Details for {switch_skill.name}:")
            # Print everything except the image data
            details = {k: v for k, v in retrieved_skill.items() if k != 'image'}
            print(json.dumps(details, indent=2))

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_skill_generator())