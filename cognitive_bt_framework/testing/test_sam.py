from cognitive_bt_framework.src.vision.sam.sam import SAM2MaskGenerator, SAM2MaskConfig
from cognitive_bt_framework.src.sim.robosuite.robosuite_sim import RobosuiteSimEnv

import cv2
import time

# Create memory-optimized configuration
config = SAM2MaskConfig(
    model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    checkpoint_path="/home/liam/dev/zk_task_planner/cognitive_bt_framework/src/vision/sam/sam2.1_hiera_tiny.pt",
    max_image_size=1024,  # Limit image size
    points_per_batch=32,  # Reduce batch size
    points_per_side=16,   # Reduce points
    # enable_memory_efficient_attention=True
)

# Initialize generator
mask_gen = SAM2MaskGenerator(config)
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
    mask_gen.show_masks(image, labeled_masks, metadata)
    input()
except RuntimeError as e:
    print(f"Memory error: {e}")
    # Handle error or adjust parameters