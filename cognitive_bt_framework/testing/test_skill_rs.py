import cv2
import numpy as np
import time
from cognitive_bt_framework.src.vision.realsense import Camera
from cognitive_bt_framework.src.vision import PerceptionSystem, FastSAMConfig
from cognitive_bt_framework.src.skills.skill_generator import SkillGenerator, PointOfInterest
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.src.skills.skill_handler import SkillHandler, SyncSkillHandler
import traceback

def test_skill_handler_sync():
    # Initialize camera with debug enabled
    camera = Camera(width=640, height=480, debug=True)
    print("Initializing camera...")
    if not camera.start():
        print("Failed to start camera")
        return
    
    # Increased delay to allow camera time to fully initialize and start streaming
    print("Waiting for camera to initialize...")
    time.sleep(3)

    try:
        # Initialize components
        config = FastSAMConfig(
            model_type="FastSAM-x",
            conf_threshold=0.25,
            iou_threshold=0.5,
            min_area=1.0,
        )
        
        perception_system = PerceptionSystem(config)
        llm_interface = LLMInterfaceOpenAI()
        skill_generator = SkillGenerator(llm_interface=llm_interface)
        
        # Create synchronous wrapper for the skill handler
        sync_skill_handler = SyncSkillHandler(skill_generator, perception_system)

        print("\nWaiting for valid frame...")
        # Wait for valid frame with more attempts and debugging
        frames = None
        for attempt in range(60):  # Increased to 60 attempts (6 seconds)
            frames = camera.get_frames()
            if frames is not None:
                color_image, depth_image = frames
                if color_image is not None and depth_image is not None:
                    print(f"Got valid frames on attempt {attempt+1}")
                    print(f"Color image shape: {color_image.shape}")
                    print(f"Depth image shape: {depth_image.shape}")
                    break
                else:
                    print(f"Attempt {attempt+1}: Got frames but images are None")
            else:
                print(f"Attempt {attempt+1}: No frames received")
            time.sleep(0.1)

        if frames is None or color_image is None or depth_image is None:
            print("Failed to get valid frames from camera after multiple attempts")
            return

        # Test the camera by displaying the images
        print("Testing camera images...")
        cv2.imshow("Color Image", color_image)
        
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_colormap)
        cv2.waitKey(1000)  # Display for 1 second
        
        action = 'open'
        target_object = 'book'

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
            print("Trying with a different target...")
            target_object = 'object'  # Try with a more generic term
            object_info = perception_system.detect_object(
                target_object,
                color_image,
                depth_image=depth_image
            )
            
            if object_info is None:
                print(f"\nFailed to detect any object")
                return

        # Detect regions of interest on the object
        print("\nDetecting points of interest...")
        roi_results = perception_system.detect_regions_of_interest(
            image=color_image,
            obj_info=object_info,
            method='shi_tomasi',  # Use SIFT for better feature detection
            max_points=20,   # Limit to 8 key points to avoid clutter
            visualize=True
        )
        
        # Create points of interest dictionary
        points_of_interest = {}
        for i, (pixel_x, pixel_y) in enumerate(roi_results['pixel_coords']):
            # Create alphabetical label
            label = get_alpha_id(i + 1)  # a, b, c, ...
            
            # Convert to normalized coordinates
            norm_x = pixel_x / color_image.shape[1]
            norm_y = pixel_y / color_image.shape[0]
            
            # Store in dictionary
            points_of_interest[label] = PointOfInterest(
                label=label,
                position=(norm_x, norm_y),
                description=f"Interest point {label}"
            )
        
        # Display the points of interest
        if 'visualization' in roi_results:
            cv2.imshow("Points of Interest", roi_results['visualization'])
            cv2.waitKey(1)  # Brief display to not block execution
            
        print(f"\nFound {len(points_of_interest)} points of interest")
        for label, point in points_of_interest.items():
            print(f"Point {label}: position={point.position}, description={point.description}")

        # Try to instantiate skill
        try:
            # Using synchronous instantiation
            instantiated_skill = sync_skill_handler.instantiate_skill(
                action,
                target_object,
                color_image,
                depth_image
            )
            
            if instantiated_skill is None:
                print("\nFailed to instantiate skill")
                return

            print("\nSuccessfully instantiated skill!")
            print(f"Skill name: {instantiated_skill.skill_name}")
            print(f"Target object: {instantiated_skill.target_object}")
            print("\nAction sequence:")
            for i, action in enumerate(instantiated_skill.action_sequence):
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
            skill_handler = SkillHandler(skill_generator, perception_system)  # Need for visualization
            vis_image = skill_handler.visualize_skill(instantiated_skill, color_image, object_info)
            
            # Show visualization
            cv2.imshow("Skill Visualization", vis_image)
            print("\nPress any key to exit...")
            cv2.waitKey(0)

        except Exception as e:
            print(f"\nError during skill instantiation: {str(e)}")
            traceback.print_exc()
            
    finally:
        print("Stopping camera...")
        camera.stop()
        cv2.destroyAllWindows()

# Helper function to convert numeric ID to alphabetical label
def get_alpha_id(num):
    """
    Convert a number to an alphabetical ID (a, b, c, ... aa, ab, ...)
    """
    if num <= 0:
        return ""
    
    letters = ""
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        letters = chr(97 + remainder) + letters  # 97 is ASCII for 'a'
    
    return letters

def main():
    test_skill_handler_sync()

if __name__ == "__main__":
    main()