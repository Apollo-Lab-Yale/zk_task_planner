from cognitive_bt_framework.src.vision.sam.sam import ImageSegmenter
from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv

import cv2

# Initialize the segmenter
segmenter = ImageSegmenter(
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    checkpoint="/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/vision/sam/sam2.1_hiera_large.pt"
)
sim = RobosuiteSimEnv()
sim.start()



# Auto segment with default points
while True:
    # Load your image (assuming it's already in numpy array format)
    image = sim.get_camera_image()  # Should be RGB format
    labeled_image, masks = segmenter.segment_and_visualize(image)
    input("press any key to ctu")
    cv2.destroyAllWindows()