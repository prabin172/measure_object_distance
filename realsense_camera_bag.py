import pyrealsense2 as rs
import numpy as np

class RealsenseCameraBag:
    def __init__(self, bag_file_path):
        self.bag_file_path = bag_file_path
        self.pipeline = None
        self.align = rs.align(rs.stream.color)
        self.frame_counter = 0

    def start(self):
        print("Loading data from .bag file")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(self.bag_file_path, repeat_playback=False)
        self.pipeline.start(config)

    def stop(self):
        self.pipeline.stop()

    def get_frame_stream(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                print(f"Oopsies! Depth or Color frame not available.")
                return False, None, None, None  # Adding None for timestamp
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            timestamp = color_frame.get_timestamp()
            self.frame_counter += 1
            print(f"Frame #{self.frame_counter} processed at time: {timestamp}")
            return True, color_image, depth_image, timestamp
        except RuntimeError as e:
            print("No more frames to process.")
            return None, None, None, None  # Special flag indicating no more frames

    def release(self):
        if self.pipeline is not None:
            print(f"Total frames processed: {self.frame_counter}")
            self.pipeline.stop()
