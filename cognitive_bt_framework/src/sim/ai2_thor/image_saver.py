from __future__ import annotations

import threading
from PIL import Image
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

DIR = "/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data/videos"

class SaveImagesThread(threading.Thread):
    def __init__(self, controller, frame_rate=10, directory=DIR, planner="unknown", goal = "unknown"):
        threading.Thread.__init__(self, target=self.run)
        if directory == DIR:
            directory += datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = directory  # The directory to save the images
        self.controller = controller
        self.images = []
        self.actions = []
        self.frame_rate = frame_rate
        self.keep_running = True
        self.planner = planner
        self.goal = goal

    def overlay_text(self, image, meta, position=(3, 190), font_path="arial.ttf", font_size=12):
        # Initialize the drawing context
        draw = ImageDraw.Draw(image)

        # Choose text color based on success parameter
        text_color = "green" if meta['lastActionSuccess'] else "red"
        text = meta['lastAction']
        # Load a font
        font = ImageFont.truetype(font_path, font_size)
        font2 = ImageFont.truetype(font_path, 8)

        draw.rectangle((0, 10, 420, 10 + 12), fill="white")
        draw.text((160, 10), self.goal, fill="black", font=font)

        # Add text to image
        draw.rectangle((position[0], position[1], position[0] + 130, position[1] + 12), fill="white")
        draw.text(position, text, fill=text_color, font=font)
        return image

    def join(self, timeout: float | None = ...) -> None:
        self.keep_running = False
        self.save_images()

    def reset(self):
        self.images = []
        self.keep_running = True
        self.start()

    def run(self):
        while self.keep_running:
            time.sleep(1/self.frame_rate)
            last_event = self.controller.last_event
            self.images.append(Image.fromarray(last_event.frame).resize(size=(420, 210)))
            self.actions.append({"lastActionSuccess": last_event.metadata['lastActionSuccess'], "lastAction": last_event.metadata['lastAction']})

        self.directory+=f"_{self.planner}" +f"_{self.goal}"

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Save each image to the directory
        for idx, image in enumerate(self.images):
            img_number = f"{idx}".zfill(5)
            image_path = os.path.join(self.directory, f'image{img_number}.png')
            image = self.overlay_text(image, self.actions[idx])
            print(f" ACTION:  {self.actions[idx]}")
            image.save(image_path)
            print(f'Saved {image_path}')

    def save_images(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Save each image to the directory
        for idx, image in enumerate(self.images):
            image = self.overlay_text(image, self.actions[idx])
            print(f" ACTION:  {self.actions[idx]}")
            num_str = f'{idx}'.zfill(4)
            image_path = os.path.join(self.directory, f'image_{num_str}.png')
            image.save(image_path)
            print(f'Saved {image_path}, {self.actions[idx]}')

