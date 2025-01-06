import cv2
import time
from realsense_camera_bag import RealsenseCameraBag
from mask_rcnn import MaskRCNN

def skip_frames(rs_bag, num_frames):
    for _ in range(num_frames):
        ret, _, _, _ = rs_bag.get_frame_stream()
        if ret is None:
            break

# Start timing
start_time = time.time()

bag_file_path = './1.bag'
batch_size = 40  # Adjust this as needed

rs_bag = RealsenseCameraBag(bag_file_path)
mrcnn = MaskRCNN()

total_frames_processed = 200
output_file = "11-2--object_distances.txt"

try:
    with open(output_file, "w") as f:
        while True:
            frames_list = []
            rs_bag.start()  # Start the pipeline
            skip_frames(rs_bag, total_frames_processed)  # Skip already processed frames
            print(f"One Time this")
            # Capture a batch of frames
            for _ in range(batch_size):
                ret, bgr_frame, depth_frame, timestamp = rs_bag.get_frame_stream()
                bgr_frame = cv2.rotate(bgr_frame, cv2.ROTATE_180)
                depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
                if ret is None:
                    break
                frames_list.append((bgr_frame, depth_frame, timestamp))

            rs_bag.stop()  # Stop the pipeline

            if not frames_list:
                break

            # Process the batch
            for i, (bgr_frame, depth_frame, timestamp) in enumerate(frames_list):
                print(f"Processing frame {i+1 + total_frames_processed}/{total_frames_processed + len(frames_list)} at time: {timestamp}")

                boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)
                bgr_frame = mrcnn.draw_object_mask(bgr_frame)
                bgr_frame = mrcnn.draw_object_info(bgr_frame, depth_frame)
                cv2.imshow("BGR Frame with Contours", bgr_frame)

                f.write(f"Time: {timestamp}\n")
                for box, class_id, center in zip(boxes, classes, centers):
                    class_name = mrcnn.classes[int(class_id)]
                    cx, cy = center
                    distance = depth_frame[cy, cx]
                    #x, y, x2, y2 = box
                    # distance = depth_frame[y:y2, x:x2]
                    # flattened_distance = distance.flatten()
                    f.write(f"\tClass: {class_name}, Box: {box}, Distance: {distance}\n")

                #time.sleep(4)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key to break
                    break

            total_frames_processed += len(frames_list)
            print(total_frames_processed)
            if ret is None:
                break

finally:
    rs_bag.release()
    cv2.destroyAllWindows()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
print(f"Total frames processed: {total_frames_processed}")
