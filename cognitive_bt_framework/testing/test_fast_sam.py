from cognitive_bt_framework.src.vision.sam.fast_sam import FastSAMMaskGenerator, FastSAMConfig

from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv

import cv2
import time

config = FastSAMConfig(
    model_type="FastSAM-x",  # or "FastSAM-x"
    max_image_size=640,
    conf_threshold=0.6,
    iou_threshold=0.9
)

mask_gen = FastSAMMaskGenerator(config)

sim = RobosuiteSimEnv()
sim.start()



# Auto segment with default points
# Load your image (assuming it's already in numpy array format)
image = sim.get_camera_image()  # Should be RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (200,200))
# Generate masks
try:
    start = time.time()
    labeled_masks, metadata = mask_gen.generate_masks(image)
    print(f"it took {time.time() - start} seconds to gen masks")
    print('got masks')
    # Visualize results
    print(len(labeled_masks))
    viz_img = mask_gen.show_masks(image, labeled_masks, metadata)
    input()
except RuntimeError as e:
    print(f"Memory error: {e}")
    # Handle error or adjust parameters