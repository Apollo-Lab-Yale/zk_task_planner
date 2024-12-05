from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from cognitive_bt_framework.src.vision.realsense import Camera



class ObjectDetection:
    def __init__(self, model_path: str = "yolo11x-seg.pt", is_sim=False):
        if not is_sim:
            self.camera = Camera()
        else:
            self.camera = None
        self.model = YOLO(model_path)
        self.class_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}

    def start(self) -> bool:
        return self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, verbose=False)[0]
        detections = []
        # print(f"RESULTS------------------------{results}")
        if results is None or results.boxes is None:
            return detections
        print(f"DETECTIONS---------------{detections}")
        for box, mask in zip(results.boxes, results.masks):
            detections.append({
                'bbox': box.xyxy[0].cpu().numpy(),
                'conf': box.conf.item(),
                'cls': box.cls.item(),
                'name': results.names[int(box.cls.item())],
                'mask': mask.data[0].cpu().numpy()
            })
        # for box in results.boxes:
        #     detections.append({
        #         'bbox': box.xyxy[0].cpu().numpy(),
        #         'conf': box.conf.item(),
        #         'cls': box.cls.item(),
        #         'name': results.names[int(box.cls.item())],
        #         'mask': mask.data[0].cpu().numpy()
        #     })
        return detections

    def get_object_depth(self, depth_image: np.ndarray, mask: np.ndarray) -> float:
        masked_depth = depth_image[mask] #> 0.5]
        return np.mean(masked_depth[masked_depth > 0]) if masked_depth.size > 0 else 0

    def apply_mask_overlay(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
                           alpha: float = 0.5) -> np.ndarray:
        mask_binary = (mask > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary > 0] = color

        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def process_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        frames = self.camera.get_frames()
        if not frames:
            return None

        color_image, depth_image = frames
        detections = self.detect_objects(color_image)

        vis_color = color_image.copy()
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        vis_depth = depth_colormap.copy()

        for det in detections:
            bbox = det['bbox']
            mask = det['mask']
            color = self.class_colors[int(det['cls'])]

            # Apply segmentation mask overlay
            vis_color = self.apply_mask_overlay(vis_color, mask, color)
            vis_depth = self.apply_mask_overlay(vis_depth, mask, color)

            depth = self.get_object_depth(depth_image, mask)
            depth_meters = depth * self.camera.depth_scale if self.camera.depth_scale else 0

            x1, y1 = map(int, bbox[:2])
            label = f"{det['name']} {depth_meters:.2f}m"

            for img in [vis_color, vis_depth]:
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_color, vis_depth

    def run(self) -> None:
        if not self.start():
            return

        try:
            while True:
                result = self.process_frame()
                if result:
                    color_det, depth_det = result
                    combined = np.hstack((color_det, depth_det))
                    cv2.imshow('Object Detection with Segmentation', combined)

                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        finally:
            self.stop()


if __name__ == "__main__":
    detector = ObjectDetection()
    detector.run()