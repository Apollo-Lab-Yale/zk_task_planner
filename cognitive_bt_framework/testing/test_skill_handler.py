import asyncio
import numpy as np
import cv2
from pathlib import Path

from cognitive_bt_framework.src.skills import SkillHandler, SkillGenerator
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMConfig, FastSAMWithCLIP
from cognitive_bt_framework.src.skills.primitive_parser import PrimitiveParser

def visualize_test_object(image: np.ndarray, mask: np.ndarray, title: str):
    """Helper function to visualize test objects"""
    # Create visualization
    vis_image = image.copy()
    vis_image[mask] = cv2.addWeighted(vis_image[mask], 0.7, np.full_like(vis_image[mask], [0, 255, 0]), 0.3, 0)
    
    # Add title
    cv2.putText(vis_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display image
    cv2.imshow(title, vis_image)
    cv2.waitKey(1000)  # Display for 1 second
    cv2.destroyAllWindows()

async def test_skill_handler():
    """Test function to demonstrate SkillHandler functionality with different objects"""
    
    # Initialize required components
    llm = LLMInterfaceOpenAI(model_name="gpt-4-turbo")
    parser = PrimitiveParser()
    
    # Initialize FastSAM config
    fast_sam_config = FastSAMConfig(
        model_type="FastSAM-x",
        conf_threshold=0.4,
        clip_threshold=0.85,
        min_area=10.0
    )
    
    # Mock perception system for testing
    class MockPerceptionSystem:
        def get_object_pose(self, mask):
            # Return mock 6D pose [x, y, z, roll, pitch, yaw]
            center_y, center_x = np.mean(np.where(mask), axis=1)
            return np.array([center_x/640, center_y/640, 0.5, 0, 0, 0])
            
        def get_3d_point(self, point_2d):
            # Convert 2D point to mock 3D coordinates
            return np.array([point_2d[0]/640, point_2d[1]/640, 0.5])
    
    # Mock object detector for testing
    class MockObjectDetector:
        def detect_object(self, object_name):
            # Create test images for different objects
            IMAGE_SIZE = 640
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
            
            if object_name == "coffee_machine":
                # Draw coffee machine with button and dial
                cv2.rectangle(image, (200, 200), (440, 400), (150, 150, 150), -1)  # Main body
                cv2.circle(image, (320, 250), 15, (255, 0, 0), -1)  # Power button
                cv2.circle(image, (320, 350), 20, (200, 200, 200), -1)  # Control dial
                cv2.line(image, (320, 350), (320, 335), (50, 50, 50), 2)  # Dial marker
                
                # Create mask for entire coffee machine
                mask[200:401, 200:441] = True
                
            elif object_name == "drawer":
                # Draw drawer with handle
                cv2.rectangle(image, (150, 250), (490, 390), (139, 69, 19), -1)  # Drawer body
                cv2.rectangle(image, (300, 300), (340, 340), (200, 200, 200), -1)  # Handle
                
                # Create mask for drawer
                mask[250:391, 150:491] = True
                
            elif object_name == "lamp":
                # Draw table lamp with touch sensor
                cv2.rectangle(image, (280, 200), (360, 400), (200, 200, 200), -1)  # Base
                cv2.circle(image, (320, 250), 15, (160, 160, 160), 2)  # Touch sensor
                cv2.circle(image, (320, 250), 12, (180, 180, 180), -1)
                
                # Create mask for lamp
                mask[200:401, 280:361] = True
                
            result = {'image': image, 'mask': mask}
            visualize_test_object(image, mask, object_name)
            return result
    
    # Initialize components
    skill_generator = SkillGenerator(llm, skills_dir="test_skills")
    object_detector = MockObjectDetector()
    perception_system = MockPerceptionSystem()
    
    # Initialize SkillHandler
    handler = SkillHandler(
        skill_generator,
        object_detector,
        perception_system,
        fast_sam_config
    )
    
    # Test cases with their expected primitive sequences
    test_cases = [
        {
            'name': "Coffee Machine",
            'action': "SwitchOn",
            'object': "coffee_machine",
            'expected_primitives': [
                "move_to(keywords=[power_button])",
                "apply_force(direction=push, keywords=[power_button])",
                "retractGripper()"
            ]
        },
        {
            'name': "Drawer",
            'action': "Open",
            'object': "drawer",
            'expected_primitives': [
                "move_to(keywords=[handle])",
                "grasp(keywords=[handle])",
                "apply_force(direction=pull, keywords=[handle])",
                "release()"
            ]
        },
        {
            'name': "Touch Lamp",
            'action': "SwitchOn",
            'object': "lamp",
            'expected_primitives': [
                "move_to(keywords=[touch_sensor])",
                "apply_force(direction=push, keywords=[touch_sensor])",
                "retractGripper()"
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 50)
        
        # First validate the expected primitives
        try:
            parsed_expected = parser.validate_primitive_sequence(test_case['expected_primitives'])
            print("Expected primitive sequence is valid")
            
            # Print parsed primitives
            for primitive in parsed_expected:
                print(f"  {primitive.action_type}:")
                for param, value in primitive.parameters.items():
                    print(f"    {param}: {value}")
                    
        except ValueError as e:
            print(f"Error in expected primitives: {e}")
            continue
        
        # Generate skill
        skill = await handler.instantiate_skill(
            abstract_action=test_case['action'],
            target_object=test_case['object']
        )
        
        if skill:
            print(f"\nGenerated Skill: {skill.skill_name}")
            print("\nAction Sequence:")
            for action in skill.action_sequence:
                print(f"  Action: {action.action_type}")
                print(f"  Position: {action.position}")
                print(f"  Parameters: {action.parameters}")
                print(f"  Orientation: {action.orientation}\n")
                
            # Validate generated action sequence
            try:
                primitive_strs = [
                    parser.format_primitive(
                        action.action_type,
                        **action.parameters
                    )
                    for action in skill.action_sequence
                ]
                parsed_generated = parser.validate_primitive_sequence(primitive_strs)
                print("Generated primitive sequence is valid")
                
            except ValueError as e:
                print(f"Error in generated primitives: {e}")
                
            # Visualize the skill
            object_info = object_detector.detect_object(test_case['object'])
            if object_info:
                vis_image = handler.visualize_skill(skill, object_info['image'])
                cv2.imshow(f"{test_case['name']} Skill Visualization", vis_image)
                cv2.waitKey(2000)  # Display for 2 seconds
                cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_skill_handler())