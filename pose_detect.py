import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str, nargs="?", default="inlab1", help="Name of the video file (without extension)")
parser.add_argument("--video_path", type=str, help="Full path to the video file (overrides file_name if provided)")
args = parser.parse_args()

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Initialize variables
frame_count = 0 # Timestep counter
waypoints = [] # Store waypoints
ped_id = 1.0 # Person ID

# Load video
file_name = args.file_name

video_path = f"./Recorded Datasets/video_samples/{file_name}.mp4"


if args.video_path:
    video_path = args.video_path
    file_name = os.path.splitext(os.path.basename(video_path))[0]
else:
    file_name = args.file_name
    video_path = f"./Recorded Datasets/video_samples/{file_name}.mp4"

print(video_path)
print(file_name)
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Make directory to save waypoints
output_dir = f"./Recorded Datasets/{file_name}"
os.makedirs(output_dir, exist_ok=True)

# Desired framerate
# desired_fps = 25
desired_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")
frame_delay = int(1000 / desired_fps)

print(desired_fps)

### Homography: define image & real-world points

## In Lab - Mata's Desk
print("** NOTE THAT TRANSFORMATION IS BEING APPLIED BASED ON THE VIDEOS IN NewTest FOLDER ***")
# Replace these values for the inlab_eval videos: [547, 132], [1135, 231], [1066, 590], [432, 512]
image_points = np.array([
    [564, 163], [1046, 244], [990, 538], [469, 474]
], dtype=np.float32)
real_world_points = np.array([
    [253, 151], [13, 91], [13, 271], [193, 331]
], dtype=np.float32)

    

H, _ = cv2.findHomography(image_points, real_world_points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Stop when video ends

    frame_count += 1
    timestamp = frame_count / desired_fps  # Calculate the timestamp

    # Run YOLO on the frame
    results = model(frame)

    # Extract detections
    for result in results:
        keypoints = result.keypoints
        boxes = result.boxes

        if keypoints.has_visible:
            # for box_conf, kps in zip(boxes.conf.cpu().numpy(), keypoints.data.cpu().numpy()):
            # Find the detection with the maximum box_conf
            max_index = np.argmax(boxes.conf.cpu().numpy())
            max_box_conf = boxes.conf.cpu().numpy()[max_index]
            max_kps = keypoints.data.cpu().numpy()[max_index]
            
            if max_box_conf >= 0.55:
                # continue
            # for kp in kps:
            # kp shape: (17, 3) => (x, y, confidence)
                # kp = kp.cpu().numpy()
                keypoint_confidences = max_kps[:, 2]
                avg_conf = np.mean(keypoint_confidences)
            
                left_ankle = max_kps[15]  # x, y, conf
                right_ankle = max_kps[16]

                if left_ankle[2] > 0.70 and right_ankle[2] > 0.70:
                    cx = (left_ankle[0] + right_ankle[0]) / 2
                    cy = (left_ankle[1] + right_ankle[1]) / 2

                    # Apply homography
                    # X, Y =  np.array([cx, cy], dtype=np.float32)
                    input_point = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                    transformed_point = cv2.perspectiveTransform(input_point, H)
                    X, Y = transformed_point[0][0]

                    waypoints.append((timestamp, X, Y))

                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    
                    # Display confidence value
                    confidence_text = f"box_conf: {max_box_conf:.2f}, {left_ankle[2]:.2f}, {right_ankle[2]:.2f}"
                    text_position = (int(cx) + 10, int(cy) - 10)
                    cv2.putText(frame, confidence_text, text_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                else:
                    waypoints.append((timestamp, None, None))
            else:
                waypoints.append((timestamp, None, None))
        else:
            waypoints.append((timestamp, None, None))

    # if frame_count % 10 == 0:  # Show every 10th frame
    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        print("Paused... Press 'p' to resume or 'q' to quit.")
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord('p'):
                print("Resumed.")
                break
            elif key2 == ord('q'):
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

# Save full-resolution waypoints
with open(os.path.join(output_dir, "pose_waypoints_full_raw.csv"), "w") as f:
    f.truncate(0)
    f.write("timestamp,ped_id,x,y\n") # Header
    for point in waypoints:
        f.write(f"{point[0]},{ped_id},{point[1]},{point[2]}\n")

print("Waypoints saved to", os.path.join(output_dir, "pose_waypoints_full_raw.csv"))

# Sampling
freq = 10  # 1 Hz
# sampling_interval = int(desired_fps / freq)
sampling_interval = 1 / freq
sampled_waypoints = []

next_sample_time = 0.0
for timestamp, X, Y in waypoints:
    if timestamp >= next_sample_time:
        sampled_waypoints.append((timestamp, X, Y))
        next_sample_time += sampling_interval

# for i in range(0, len(waypoints), sampling_interval):
#     group = waypoints[i:i + sampling_interval]
#     valid_points = [(X, Y) for _, X, Y in group if X is not None and Y is not None]

#     if valid_points:
#         avg_X = sum(p[0] for p in valid_points) / len(valid_points)
#         avg_Y = sum(p[1] for p in valid_points) / len(valid_points)
#         sampled_waypoints.append((group[0][0], avg_X, avg_Y))

# Save sampled waypoints
with open(os.path.join(output_dir, "pose_waypoints_sampled_10hz_raw.csv"), "w") as f:
    f.truncate(0)
    f.write("timestamp,ped_id,x,y\n") # Header
    for point in sampled_waypoints:
        f.write(f"{point[0]},{ped_id},{point[1]},{point[2]}\n")

print("Sampled waypoints saved to", os.path.join(output_dir, "pose_waypoints_sampled_10hz_raw.csv"))

# # Smoothing
# def moving_average(data, window_size=5):
#     smoothed = []
#     for i in range(len(data)):
#         window = data[max(0, i - window_size + 1):i + 1]
#         avg_x = sum(p[1] for p in window) / len(window)
#         avg_y = sum(p[2] for p in window) / len(window)
#         smoothed.append((data[i][0], avg_x, avg_y))
#     return smoothed

# smoothed_waypoints = moving_average(sampled_waypoints)

# # Save smoothed waypoints
# with open(os.path.join(output_dir, "pose_waypoints_smoothed_raw.csv"), "w") as f:
#    f.truncate(0)
#    f.write("timestamp,ped_id,x,y\n") # Header
#    for point in smoothed_waypoints:
#        f.write(f"{point[0]},{ped_id},{point[1]},{point[2]}\n")

# print("Smoothed waypoints saved to", os.path.join(output_dir, "pose_waypoints_smoothed_raw.csv"))


# Sampling
freq = 2.5  # 1 Hz
# sampling_interval = int(desired_fps / freq)
sampling_interval = 1 / freq
sampled_waypoints = []

next_sample_time = 0.0
for timestamp, X, Y in waypoints:
    if timestamp >= next_sample_time:
        sampled_waypoints.append((timestamp, X, Y))
        next_sample_time += sampling_interval

# Save sampled waypoints
with open(os.path.join(output_dir, "pose_waypoints_sampled_2.5hz_raw.csv"), "w") as f:
    f.truncate(0)
    f.write("timestamp,ped_id,x,y\n") # Header
    for point in sampled_waypoints:
        f.write(f"{point[0]},{ped_id},{point[1]},{point[2]}\n")

print("Sampled waypoints saved to", os.path.join(output_dir, "pose_waypoints_sampled_2.5hz_raw.csv"))