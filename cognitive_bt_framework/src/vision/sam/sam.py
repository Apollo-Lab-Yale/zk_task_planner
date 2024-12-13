import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class ImageSegmenter:
    def __init__(self, model_cfg, checkpoint, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = build_sam2(model_cfg, checkpoint)
        self.predictor = SAM2ImagePredictor(self.model)
        
    def generate_points(self, image):
        h, w = image.shape[:2]
        points = []
        labels = []
        
        # Edge points
        edge_spacing = 50
        for x in range(0, w, edge_spacing):
            points.extend([[x, 0], [x, h-1]])
            labels.extend([1, 1])
        for y in range(0, h, edge_spacing):
            points.extend([[0, y], [w-1, y]])
            labels.extend([1, 1])
        
        # Grid points
        grid_size = 6
        x_step = w // grid_size
        y_step = h // grid_size
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                points.append([i * x_step, j * y_step])
                labels.append(1)
        
        # Center focus points
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 4
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append([x, y])
            labels.append(1)
            
        return np.array(points), np.array(labels)

    def segment_image(self, image, points=None, labels=None):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            
            if points is None:
                points, labels = self.generate_points(image)
                
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            print(masks.shape)
            return [mask > 0.99 for mask in masks], scores, logits
    
    def create_labeled_image(self, image, masks):
        labeled_image = image.copy()
        # Define a color palette
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255],
            [128, 0, 0], [0, 128, 0], [0, 0, 128]
        ]
        
        overlay = np.zeros_like(image)
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            print(color)
            color_mask = np.zeros_like(image)
            color_mask[mask] = color
            overlay = cv2.addWeighted(overlay, 1, color_mask, 1, 0)
            
            # Add contours
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_image, contours, -1, color, 2)
            
            # Add label in contrasting color
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                
                # Add white background for text
                text = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                print(text)
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # White background
                padding = 5
                cv2.rectangle(labeled_image, 
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (255, 255, 255),
                            -1)
                
                # Black text
                cv2.putText(labeled_image, text, (text_x, text_y),
                          font, font_scale, (0, 0, 0), thickness)
        
        # Blend the overlay with original image
        labeled_image = cv2.addWeighted(labeled_image, 0.7, overlay, 0.3, 0)
        
        return labeled_image.astype(np.uint8)
    
    def segment_and_visualize(self, image, points=None, labels=None, display=True):
        masks, scores, _ = self.segment_image(image, points, labels)
        labeled_img = self.create_labeled_image(image, masks)
        
        if display:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
        return labeled_img, masks