import asyncio
import numpy as np
import cv2
from pathlib import Path

# Import base configurations and utils first
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMConfig
from cognitive_bt_framework.src.llm_interface import LLMInterfaceOpenAI

# Then import perception and skill components
from cognitive_bt_framework.src.vision.perception_system import PerceptionSystem
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator
from cognitive_bt_framework.src.skills import skill_handler

async def test_stove_interaction():
    """Test stove switch-on skill generation using real image"""
    
    # Initialize components
    llm = LLMInterfaceOpenAI(model_name="gpt-4-turbo")
    skill_generator = SkillGenerator(llm, skills_dir="test_skills")
    
    # Configure FastSAM and perception
    fast_sam_config = FastSAMConfig(
        model_type="FastSAM-x",
        conf_threshold=0.4,
        iou_threshold=0.75,
        clip_threshold=0.3
    )
    
    # Initialize perception system
    perception_system = PerceptionSystem(fast_sam_config)
    
    # Initialize SkillHandler
    handler = skill_handler.SkillHandler(
        skill_generator=skill_generator,
        perception_system=perception_system
    )
    
    # Load image
    image_path = "/home/liam/dev/zk_task_planner/cognitive_bt_framework/testing/top_view/img_00000.png"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Starting stove interaction test...")
    print("-" * 50)
    
    # Create mock depth image matching RGB dimensions
    depth_image = np.ones_like(image[:, :, 0], dtype=np.float32) * 0.7  # Assume 0.7m tabletop height
    
    # Generate skill for switching on stove
    stove_skill = await handler.instantiate_skill(
        abstract_action="SwitchOn",
        target_object="stove",
        image=image,
        depth_image=depth_image
    )
    
    if stove_skill:
        print(f"Generated Skill Name: {stove_skill.skill_name}")
        print("\nAction Sequence:")
        for i, action in enumerate(stove_skill.action_sequence):
            print(f"\n{i+1}. {action.action_type}")
            print(f"   Position: {action.position}")
            print(f"   Orientation: {action.orientation}")
            print(f"   Parameters: {action.parameters}")
            
        print("\nExecution Parameters:")
        for key, value in stove_skill.execution_parameters.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.tolist()}")
            else:
                print(f"   {key}: {value}")
                
        # Visualize skill and detection
        vis_image = handler.visualize_skill(stove_skill, image)
        
        # Add perception visualization
        object_info = perception_system.detect_object("stove", image, depth_image)
        if object_info:
            vis_image = perception_system.visualize_detection(vis_image, object_info)
            
        # Save visualization
        output_path = "stove_skill_visualization.png"
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("Failed to generate skill for stove interaction")

if __name__ == "__main__":
    asyncio.run(test_stove_interaction())