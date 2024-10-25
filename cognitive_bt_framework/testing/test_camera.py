import unittest
import time
import numpy as np
from cognitive_bt_framework.src.vision.realsense import Camera  # Assuming the class is saved in camera_module.py


class TestCameraFunctions(unittest.TestCase):

    def setUp(self):
        """This method is called before each test."""
        self.cam = Camera()  # Create a Camera instance

    def tearDown(self):
        """This method is called after each test."""
        self.cam.stop_streaming()  # Ensure streaming is stopped after each test

    def test_start_and_stop_streaming(self):
        """Test if the camera starts and stops streaming correctly."""
        self.cam.start_streaming(queue_size=50, keep_frames=True)
        time.sleep(2)  # Let it stream for a couple of seconds
        self.assertTrue(self.cam._running, "Camera streaming did not start correctly.")

        self.cam.stop_streaming()
        self.assertFalse(self.cam._running, "Camera streaming did not stop correctly.")

    def test_get_frame(self):
        """Test if frames can be retrieved from the queue while streaming."""
        self.cam.start_streaming(queue_size=50, keep_frames=True)
        time.sleep(2)  # Let the stream run for a short period

        # Try to get a frame from the stream
        frame = self.cam.get_frame()
        self.assertIsNotNone(frame, "Failed to retrieve a frame from the stream.")

        frame_data = frame.get_data()
        self.assertIsInstance(frame_data, np.ndarray, "Frame data should be a numpy array.")
        self.assertGreater(frame.get_frame_number(), 0, "Frame number should be greater than 0.")

    def test_align_depth_to_color(self):
        """Test if depth-to-color alignment works correctly."""
        self.cam.start_streaming(queue_size=50, keep_frames=True)
        time.sleep(2)  # Let the stream run for a short period

        # Align depth to color and get the processed images
        bg_removed, depth_colormap = self.cam.align_depth_to_color()

        self.assertIsNotNone(bg_removed, "Background-removed image should not be None.")
        self.assertIsNotNone(depth_colormap, "Depth colormap image should not be None.")
        self.assertIsInstance(bg_removed, np.ndarray, "Background-removed image should be a numpy array.")
        self.assertIsInstance(depth_colormap, np.ndarray, "Depth colormap should be a numpy array.")
        self.assertEqual(bg_removed.shape[1], 640, "The width of the bg_removed image should be 640.")
        self.assertEqual(depth_colormap.shape[1], 640, "The width of the depth_colormap should be 640.")

    def test_stop_streaming(self):
        """Test if the camera stops streaming without errors."""
        self.cam.start_streaming(queue_size=50, keep_frames=True)
        time.sleep(2)  # Let the stream run for a short period

        self.cam.stop_streaming()
        self.assertFalse(self.cam._running, "Camera streaming did not stop correctly.")
        self.assertIsNone(self.cam.thread, "Camera thread should not be active after stopping streaming.")


if __name__ == '__main__':
    unittest.main()
