from typing import NamedTuple

class ObservedState(object):
    def __init__(self, image_data, depth_data, detected_objects, timestamp):
        self.image_data = image_data
        self.depth_data = depth_data
        self.detected_objects = detected_objects
        self.timestamp = timestamp