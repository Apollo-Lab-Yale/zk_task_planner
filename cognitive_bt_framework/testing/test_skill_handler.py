import asyncio
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
import torch

from cognitive_bt_framework.src.llm_interface import LLMInterfaceOpenAI
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMConfig
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator
from cognitive_bt_framework.src.skills.skill_handler import SkillHandler

# Mock classes for testing
class MockObjectDetector:
    def detect_object(self, target_object):
        # Create a mock image and mask based on object type
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        mask = np.zeros((640, 640), dtype=bool)
        
        if target_object == "lamp":
            # Create lamp with push button
            cv2.rectangle(image, (280, 300), (360, 340), (200, 200, 200), -1)  # base
            cv2.circle(image, (320, 320), 12, (180, 180, 180), -1)  # button surround
            cv2.circle(image, (320, 320), 10, (255, 0, 0), -1)  # button
            mask[300:340, 280:360] = True
            
        elif target_object == "switch":
            # Create wall switch
            cv2.rectangle(image, (295, 270), (345, 370), (200, 200, 200), -1)  # plate
            cv2.rectangle(image, (305, 300), (335, 340), (240, 240, 240), -1)  # switch
            cv2.circle(image, (320, 320), 5, (180, 180, 180), -1)  # detail
            mask[270:370, 295:345] = True
            
        return {
            'image': image,
            'mask': mask
        }

class MockPerceptionSystem:
    def get_3d_point(self, point_2d):
        # Convert 2D point to mock 3D coordinates
        return np.array([point_2d[0] / 640.0, point_2d[1] / 640.0, 0.5])
        
    def get_object_pose(self, mask):
        # Return mock 6D pose [x, y, z, roll, pitch, yaw]
        y_coords, x_coords = np.where(mask)
        center_x = np.mean(x_coords) / 640.0
        center_y = np.mean(y_coords) / 640.0
        return np.array([center_x, center_y, 0.5, 0, 0, 0])

async def test_skill_handler():
    """Test the SkillHandler functionality with various test cases"""
    
    # Initialize components
    llm = LLMInterfaceOpenAI(model_name="gpt-4-turbo")
    skill_generator = SkillGenerator(llm, skills_dir="test_skills")
    object_detector = MockObjectDetector()
    perception_system = MockPerceptionSystem()
    
    # Configure FastSAM
    fast_sam_config = FastSAMConfig()
    
    # Initialize SkillHandler
    skill_handler = SkillHandler(
        skill_generator=skill_generator,
        object_detector=object_detector,
        perception_system=perception_system,
        fast_sam_config=fast_sam_config
    )
    
    # Test Case 1: Push Button Lamp
    print("\nTest Case 1: Push Button Lamp")
    print("-" * 50)
    
    lamp_skill = await skill_handler.instantiate_skill(
        abstract_action="SwitchOn",
        target_object="lamp"
    )
    
    if lamp_skill:
        print(f"Generated Skill Name: {lamp_skill.skill_name}")
        print("\nAction Sequence:")
        for i, action in enumerate(lamp_skill.action_sequence):
            print(f"\n{i+1}. {action.action_type}")
            print(f"   Position: {action.position}")
            print(f"   Orientation: {action.orientation}")
            print(f"   Parameters: {action.parameters}")
            
        print("\nExecution Parameters:")
        for key, value in lamp_skill.execution_parameters.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.tolist()}")
            else:
                print(f"   {key}: {value}")
                
        # Visualize the skill
        lamp_info = object_detector.detect_object("lamp")
        vis_image = skill_handler.visualize_skill(lamp_skill, lamp_info['image'])
        cv2.imwrite("lamp_skill_visualization.png", vis_image)
    
    # Test Case 2: Wall Switch
    print("\nTest Case 2: Wall Switch")
    print("-" * 50)
    
    switch_skill = await skill_handler.instantiate_skill(
        abstract_action="SwitchOn",
        target_object="switch"
    )
    
    if switch_skill:
        print(f"Generated Skill Name: {switch_skill.skill_name}")
        print("\nAction Sequence:")
        for i, action in enumerate(switch_skill.action_sequence):
            print(f"\n{i+1}. {action.action_type}")
            print(f"   Position: {action.position}")
            print(f"   Orientation: {action.orientation}")
            print(f"   Parameters: {action.parameters}")
            
        print("\nExecution Parameters:")
        for key, value in switch_skill.execution_parameters.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.tolist()}")
            else:
                print(f"   {key}: {value}")
                
        # Visualize the skill
        switch_info = object_detector.detect_object("switch")
        vis_image = skill_handler.visualize_skill(switch_skill, switch_info['image'])
        cv2.imwrite("switch_skill_visualization.png", vis_image)
    
    # Test error handling
    print("\nTest Case 3: Error Handling")
    print("-" * 50)
    
    # Test with non-existent object
    nonexistent_skill = await skill_handler.instantiate_skill(
        abstract_action="SwitchOn",
        target_object="nonexistent_object"
    )
    
    print(f"Handling non-existent object: {'Failed as expected' if nonexistent_skill is None else 'Unexpected success'}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_skill_handler())