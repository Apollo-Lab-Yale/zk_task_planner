import asyncio
import cv2
import numpy as np
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.src.vision.sam.fast_sam_clip import FastSAMWithCLIP
from cognitive_bt_framework.src.skills.skill_handler import SkillHandler

async def test_skill_handler():
    # Initialize camera
    camera = Camera(width=640, height=480)
    if not camera.start():
        print("Failed to start camera")
        return

    try:
        # Initialize components
        config = FastSAMConfig(
            model_type="FastSAM-x",
            conf_threshold=0.25,
            iou_threshold=0.5,
            clip_threshold=0.5,
            min_area=1.0,
            debug=False
        )
        
        perception_system = PerceptionSystem(config)
        llm_interface = LLMInterfaceOpenAI()
        skill_generator = SkillGenerator(llm_interface=llm_interface)
        skill_handler = SkillHandler(skill_generator, perception_system)

        print("\nWaiting for valid frame...")
        # Wait for valid frame
        frames = None
        for _ in range(30):  # Try for 3 seconds
            frames = camera.get_frames()
            if frames is not None:
                color_image, depth_image = frames
                if color_image is not None and depth_image is not None:
                    break
            await asyncio.sleep(0.1)

        if frames is None:
            print("Failed to get valid frames from camera")
            return

        color_image, depth_image = frames
        
        action = 'pickup'
        target_object = 'pen'

        print("\nProcessing request...")
        print(f"Target: {target_object}")
        print(f"Action: {action}")
        
        # First detect the object
        print("\nDetecting object...")
        object_info = perception_system.detect_object(
            target_object,
            color_image,
            depth_image=depth_image
        )
        
        if object_info is None:
            print(f"\nFailed to detect {target_object}")
            return

        # Try to instantiate skill
        try:
            skill = await skill_handler.instantiate_skill(
                action,
                target_object,
                color_image,
                depth_image
            )

            if skill is None:
                print("\nFailed to instantiate skill")
                return

            print("\nSuccessfully instantiated skill!")
            print(f"Skill name: {skill.skill_name}")
            print(f"Target object: {skill.target_object}")
            print("\nAction sequence:")
            for i, action in enumerate(skill.action_sequence):
                print(f"\nAction {i+1}:")
                print(f"  Type: {action.action_type}")
                print(f"  Position: {action.position}")
                print(f"  Orientation: {action.orientation}")
                print(f"  Parameters: {action.parameters}")
                if hasattr(action, 'pixel_position'):
                    print(f"  Pixel Position: {action.pixel_position}")

            # Visualize object detection
            object_vis = perception_system.visualize_detection(color_image, object_info)
            cv2.imshow("Object Detection", object_vis)

            # Visualize the skill
            vis_image = skill_handler.visualize_skill(skill, color_image, object_info)
            
            # Show visualization
            cv2.imshow("Skill Visualization", vis_image)
            print("\nPress any key to exit...")
            cv2.waitKey(0)

        except Exception as e:
            print(f"\nError during skill instantiation: {str(e)}")
            import traceback
            traceback.print_exc()
            
    finally:
        camera.stop()
        cv2.destroyAllWindows()

def main():
    # Set up asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_skill_handler())
    finally:
        loop.close()

if __name__ == "__main__":
    main()