from typing import Dict, List, Optional, Any, Tuple
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
from cognitive_bt_framework.src.vision.perception_system import ObjectInfo

@dataclass
class PointOfInterest:
    """Data class to store labeled point of interest"""
    label: str  # Alphabetical label (a, b, c, etc.)
    position: Tuple[float, float]  # Normalized (x, y) coordinates
    description: str = ""  # Optional description of the point
    
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
    points_of_interest: Dict[str, PointOfInterest]  # Dictionary of labeled points
    explanations: List[str]

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
        self.debug = False
        self._load_stored_skills()

    def _load_stored_skills(self):
        """Load previously stored skills from disk"""
        for skill_file in self.skills_dir.glob("*.json"):
            try:
                with open(skill_file, 'r') as f:
                    skill_data = json.load(f)
                    
                    # Convert points_of_interest dict to PointOfInterest objects
                    if "points_of_interest" in skill_data:
                        points = {}
                        for label, point_data in skill_data["points_of_interest"].items():
                            points[label] = PointOfInterest(
                                label=point_data["label"],
                                position=tuple(point_data["position"]),
                                description=point_data.get("description", "")
                            )
                        skill_data["points_of_interest"] = points
                    else:
                        skill_data["points_of_interest"] = {}
                        
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

    def _visualize_points_of_interest(self, image: np.ndarray, points: Dict[str, PointOfInterest]) -> np.ndarray:
        """
        Create a visualization of the image with labeled points of interest
        
        Args:
            image: Input image
            points: Dictionary of points of interest
            
        Returns:
            Image with visualized points of interest
        """
        vis_img = image.copy()
        h, w = image.shape[:2]
        
        # Draw labeled points
        for label, point in points.items():
            # Convert normalized coordinates to pixel coordinates
            px, py = int(point.position[0] * w), int(point.position[1] * h)
            
            # Draw circle for point
            cv2.circle(vis_img, (px, py), 5, (0, 0, 255), 3)
            
            # Draw label
            cv2.putText(
                vis_img,
                label,
                (px + 10, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return vis_img
    
    def generate_skill(self, 
                           image: np.ndarray,
                           points_of_interest: Dict[str, PointOfInterest],
                           abstract_action: str,
                           target_object: str,
                           object_info: Optional[ObjectInfo] = None) -> Optional[Skill]:
        """
        Generate a new skill using the LLM interface based on image input with labeled points
        
        Args:
            image: Input image of the object
            points_of_interest: Dictionary of labeled points of interest
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
            object_info: Optional ObjectInfo containing additional object data
        
        Returns:
            Generated skill if successful, None otherwise
        """
        # Create visualization of image with labeled points
        vis_img = self._visualize_points_of_interest(image, points_of_interest)
        
        # Generate unique ID for this image
        timestamp = int(time.time())
        point_hash = hash(str([(p.label, p.position) for p in points_of_interest.values()])) & 0xFFFFFF
        image_id = f"{abstract_action}_{target_object}_{timestamp}_{point_hash:06x}"
        img_path = self._save_image(vis_img, f"vis_{image_id}")
        if self.debug:
            self.get_logger().info(f"Saved visualization image to {img_path}")
        
        # Also save the original image for reference
        # Generate unique ID for this image
        point_hash = hash(str([(p.label, p.position) for p in points_of_interest.values()])) & 0xFFFFFF
        image_id = f"{abstract_action}_{target_object}_{point_hash:06x}"
        self._save_image(vis_img, image_id)
        
        # Create descriptions for prompt
        point_descriptions = []
        for label, point in points_of_interest.items():
            x, y = point.position
            desc = f"Point {label}: {point.description}" if point.description else f"Point {label}: Located at normalized coordinates ({x:.2f}, {y:.2f})"
            point_descriptions.append(desc)
        
        # Create prompt for skill definition
        combined_prompt = [{"role": "system", "content": f"""Analyze the object visually and generate a complete skill definition for performing {abstract_action} on a {target_object}.

                    The image contains an object with labeled points of interest (red circles with alphabetical labels).

                    Points of interest:
                    {chr(10).join(point_descriptions)}

                    First, determine the specific subtype of the action based on the object's visual characteristics.
                    The skill name should follow the format: action_targetobject_mechanism

                    AVAILABLE ACTION PRIMITIVES AND PARAMETERS:

                    1. move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)
                    - point_label: The labeled point (a, b, c, etc.) where the gripper should move to
                    - is_top_down_grasp: Boolean (true/false) indicating if the gripper should approach from above
                    - is_side_grasp: Boolean (true/false) indicating if the gripper should approach from the side
                    - Example: move_gripper_to_pose('a', true, false) - Move to point 'a' with a top-down approach

                    2. push('point_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                    - point_label: The labeled point that the vision system should use to detect the surface whos normal will guide the direction of the push
                    - force_direction: Either 'perpendicular' (push directly into the surface) or 'parallel' (push along the surface)
                    - is_button: Boolean (true/false) indicating if this is a button push (short distance, low force)
                    - has_pivot: Boolean (true/false) indicating if the push should pivot around another point
                    - pivot_point_label: If has_pivot is true, the labeled point to pivot around; otherwise use empty string ''
                    - Example: push('b', 'perpendicular', true, false, '') - Push point 'b' like a button

                    3. pull('point_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')
                    - point_label: The labeled point that the vision system should use to detect the surface whos normal will guide the direction of the pull
                    - force_direction: Either 'perpendicular' (pull directly away from the surface) or 'parallel' (pull along the surface)
                    - is_button: Boolean (true/false) indicating if this is a button-like pull (short distance, low force)
                    - has_pivot: Boolean (true/false) indicating if the pull should pivot around another point
                    - pivot_point_label: If has_pivot is true, the labeled point to pivot around; otherwise use empty string ''
                    - Example: pull('c', 'parallel', false, true, 'd') - Pull point 'c' along the surface, pivoting around point 'd'

                    4. close_gripper()
                    - Closes the robot's gripper to grasp an object
                    - No parameters required
                    - Example: close_gripper()

                    5. open_gripper()
                    - Opens the robot's gripper to release an object
                    - No parameters required
                    - Example: open_gripper()

                    6. retract_gripper()
                    - Moves the gripper away from the current position to a safe position
                    - No parameters required
                    - Example: retract_gripper()

                    SKILL PARAMETERS:

                    1. force_threshold:
                    - low: For delicate objects or precise operations (1-5N)
                    - medium: For standard operations (5-15N)
                    - high: For operations requiring significant force (15-30N)

                    2. precision_required:
                    - low: For operations where exact positioning is not critical (±10mm)
                    - medium: For standard operations requiring good accuracy (±5mm)
                    - high: For operations requiring very precise positioning (±1mm)

                    3. speed_requirement:
                    - slow: For delicate operations or where safety is paramount (0.1-0.2m/s)
                    - medium: For standard operations (0.2-0.4m/s)
                    - high: For operations where time efficiency is important (0.4-0.6m/s)

                    IMPORTANT: You must output ONLY a valid JSON object with the following structure:

                    {{{{
                        "skill_name": "specific_action_name_with_mechanism",
                        "primitive_sequence": [
                            "EACH PRIMITIVE MUST USE EXACTLY ONE OF THESE FORMATS:",
                            "move_gripper_to_pose('point_label', is_top_down_grasp, is_side_grasp)",
                            "push('point_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')",
                            "pull('point_label', 'force_direction', is_button, has_pivot, 'pivot_point_label')",
                            "close_gripper()",
                            "open_gripper()",
                            "retract_gripper()"
                        ],
                        "parameters": {{{{
                            "force_threshold": "low/medium/high",
                            "precision_required": "low/medium/high",
                            "speed_requirement": "slow/medium/fast"
                        }}}},
                        "prerequisites": [
                            "list of required conditions"
                        ],
                        "constraints": [
                            "list of safety limits and constraints"
                        ],
                        "explanations": [
                            "list of detailed explanations for primitives and parameter choices"
                        ]
                    }}}}

                    CRITICAL FORMATTING RULES:
                    1. All point labels MUST be in single quotes (e.g., 'a', 'b', etc.)
                    2. force_direction MUST be in single quotes and be either 'parallel' or 'perpendicular'
                    3. is_button, has_pivot, is_top_down_grasp, is_side_grasp MUST be boolean values (true or false) WITHOUT quotes
                    4. pivot_point_label MUST be in single quotes, even if empty (e.g., '', 'c')
                    5. The syntax must match EXACTLY one of these patterns:
                    - move_gripper_to_pose('a', true, false)
                    - push('b', 'perpendicular', true, false, '')
                    - pull('c', 'parallel', false, true, 'd')
                    - close_gripper()
                    - open_gripper()
                    - retract_gripper()

                    Note:
                    - For push and pull, the force_direction determines if the movement is perpendicular to the surface or parallel along it
                    - Surface normals will be calculated automatically from the environment
                    - When grasping an object, move_gripper_to_pose should specify the best point label for positioning
                    - All references to locations should use the labeled points (a, b, c, etc.)
                    - Prerequisites should include any conditions that must be met before execution (e.g., "object must be stationary")
                    - Constraints should include any safety limits (e.g., "maximum force must not exceed 15N")
                    
                    - The chosen point label for move_gripper_to_pose should indicate the BEST location for the robot to manipulate the object given 
                      the requested action: {abstract_action}
                    
                    - The chosen surface point label for push/pull should be the CLEAREST INDICATOR of the surface to align the force with
                        the goal be to should make it easy for the robot to calculate the normal direction.

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
                
                Use the labeled points (a, b, c, etc.) to reference specific locations on the object.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(vis_img)}"
                }}
            ]}]
        # Get response
        skill_response = self.llm.query_llm(combined_prompt)
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
            explanations=skill_data.get('explanations', []),
            image_id=image_id,
            points_of_interest=points_of_interest
        )
        
        self._store_skill(skill)
        return skill

    def find_similar_skill(self, 
                          image: np.ndarray,
                          points_of_interest: Dict[str, PointOfInterest],
                          abstract_action: str,
                          target_object: str) -> Optional[Skill]:
        """
        Find a similar existing skill based on points of interest by comparing against all eligible skills at once
        
        Args:
            image: Input image of the object
            points_of_interest: Dictionary of labeled points of interest
            abstract_action: The abstract action to perform
            target_object: The object to perform the action on
        
        Returns:
            Most similar existing skill if found, None otherwise
        """
        if not self.skills_cache:
            return None
            
        # Create visualization for comparison
        vis_img = self._visualize_points_of_interest(image, points_of_interest)
        
        # Filter relevant skills by action and object type
        relevant_skills = []
        for skill in self.skills_cache.values():
            if skill.abstract_action == abstract_action and skill.target_object == target_object:
                existing_image = self._load_image(skill.image_id)
                if existing_image is not None:
                    relevant_skills.append((skill, existing_image))
        
        # If no relevant skills, return early
        if not relevant_skills:
            return None
        
        # Prepare skill images for batch comparison
        skill_images = []
        for _, img in relevant_skills:
            skill_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(img)}"
                }
            })
        
        # Create comparison prompt with all skills
        comparison_prompt = [
            {"role": "system", "content": f"""Compare the first image (new object) with the {len(relevant_skills)} following images (existing skills).
            For each comparison, provide a similarity score between 0 and 1, where 1 means identical and 0 means completely different.
            Consider visual similarity of the mechanism, interaction points, and their spatial relationships.
            Focus particularly on the positions of the labeled points relative to each other.
            The action to perform is "{abstract_action}" on a "{target_object}".
            
            Return ONLY a raw JSON object with scores in this format with NO additional formatting:
            {{
                "scores": [0.75, 0.42, 0.91, ...],
                "best_match_index": 2,  // Index of the image with highest score (0-based)
                "best_match_score": 0.91  // The highest score value
            }}"""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Compare this new object with the {len(relevant_skills)} existing skills below.
                Return the similarity scores for each, and identify the best match if any are similar enough.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(vis_img)}"
                }}
            ]}
        ]
        
        # Add all skill images to the prompt
        for img_data in skill_images:
            comparison_prompt[1]["content"].append(img_data)
        
        # Query LLM for all comparisons at once
        comparison_response = self.llm.query_llm(comparison_prompt)
        
        try:
            # Parse the JSON response
            result = json.loads(comparison_response)
            
            # Validate response format
            if "scores" not in result or "best_match_index" not in result or "best_match_score" not in result:
                print("Invalid comparison response format")
                return None
            
            # Get best score and corresponding skill
            best_score = result["best_match_score"]
            best_index = result["best_match_index"]
            
            # Ensure index is valid
            if best_index < 0 or best_index >= len(relevant_skills):
                print(f"Invalid best match index: {best_index}")
                return None
            
            # Return the best skill if score is high enough
            if best_score > 0.8:
                return relevant_skills[best_index][0]
            else:
                return None
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing comparison response: {e}")
            print(f"Response: {comparison_response}")
            return None

    def adapt_skill(self, 
                         base_skill: Skill,
                         new_image: np.ndarray,
                         new_points: Dict[str, PointOfInterest]) -> Optional[Skill]:
        """
        Adapt an existing skill based on new image with different points of interest
        
        Args:
            base_skill: The skill to adapt
            new_image: Image of the new object
            new_points: Dictionary of labeled points of interest for the new object
        
        Returns:
            Adapted skill if successful, None otherwise
        """
        # Create visualization for the new object
        new_vis_img = self._visualize_points_of_interest(new_image, new_points)
        base_image = self._load_image(base_skill.image_id)
        
        if base_image is None:
            return None
            
        # Extract point descriptions for prompt
        new_point_descriptions = []
        for label, point in new_points.items():
            x, y = point.position
            desc = f"Point {label}: {point.description}" if point.description else f"Point {label}: Located at normalized coordinates ({x:.2f}, {y:.2f})"
            new_point_descriptions.append(desc)
            
        adaptation_prompt = [
            {"role": "system", "content": f"""Adapt the existing skill's primitive sequence and parameters for the new object based on the new labeled points.
            
            The new image contains an object with labeled points of interest (red circles with alphabetical labels).
            
            New points of interest:
            {chr(10).join(new_point_descriptions)}
            
            Consider visual differences in the mechanism and interaction points.
            Make sure to update the point labels in the primitive sequence to match the new object's labeled points.
            
            Return ONLY a JSON object with 'primitive_sequence' and 'parameters' fields. Return nothing else."""},
            {"role": "user", "content": [
                {"type": "text", "text": f"""
                Original Skill:
                {json.dumps(asdict(base_skill), indent=2)}
                
                Adapt the skill based on visual differences and the new labeled points.
                """},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(base_image)}"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(new_vis_img)}"
                }}
            ]}
        ]
        
        adaptation_response = self.llm.query_llm(adaptation_prompt)
        try:
            adapted_data = json.loads(adaptation_response)
            
            # Generate new image ID and save
            point_hash = hash(str([(p.label, p.position) for p in new_points.values()])) & 0xFFFFFF
            new_image_id = f"{base_skill.abstract_action}_{base_skill.target_object}_{point_hash:06x}"
            self._save_image(new_vis_img, new_image_id)
            
            adapted_skill = Skill(
                name=f"{base_skill.name}_adapted_{hash(new_image_id) & 0xFFFFFF:06x}",
                abstract_action=base_skill.abstract_action,
                target_object=base_skill.target_object,
                primitive_sequence=adapted_data['primitive_sequence'],
                parameters=adapted_data['parameters'],
                prerequisites=base_skill.prerequisites,
                constraints=base_skill.constraints,
                image_id=new_image_id,
                points_of_interest=new_points
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