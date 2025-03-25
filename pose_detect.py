import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str, nargs="?", default="inlab1", help="Name of the video file (without extension)")
args = parser.parse_args()

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Initialize variables
frame_count = 0 # Timestep counter
waypoints = [] # Store waypoints
ped_id = 1.0 # Person ID

# Load video
file_name = args.file_name

video_path = f"/home/sina/env_prediction_project/trajectory_detection/video_samples/{file_name}.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Make directory to save waypoints
output_dir = f"./{file_name}"
os.makedirs(output_dir, exist_ok=True)

# Desired framerate
desired_fps = 25
frame_delay = int(1000 / desired_fps)

### Homography: define image & real-world points
# ## In Lab - Door corner
# image_points = np.array([
#     [319, 58], [547, 132], [382, 313], [127, 209]
# ], dtype=np.float32)
# real_world_points = np.array([
#     [72, 84], [192, 84], [192, -36], [72, -36]
# ], dtype=np.float32)

## In Lab - Mata's Desk
image_points = np.array([
    [547, 132], [1135, 231], [1066, 590], [432, 512]
], dtype=np.float32)
real_world_points = np.array([
    [253, 151], [13, 91], [13, 271], [193, 331]
], dtype=np.float32)

# # ## Big Aisle - Left side of orange cone
# image_points = np.array([
#     [224, 184], [504, 164], [544, 506], [154, 539]
# ], dtype=np.float32)
# real_world_points = np.array([
#     [0, 180], [120, 180], [120, 0], [0, 0]
# ], dtype=np.float32)

H, _ = cv2.findHomography(image_points, real_world_points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Stop when video ends

    frame_count += 1

    # Run YOLO on the frame
    results = model(frame)

    # Extract detections
    for result in results:
        keypoints = result.keypoints
        boxes = result.boxes

        if keypoints.has_visible:
            for box_conf, kps in zip(boxes.conf.cpu().numpy(), keypoints.data.cpu().numpy()):
                if box_conf < 0.4:
                    continue
                # for kp in kps:
                # kp shape: (17, 3) => (x, y, confidence)
                    # kp = kp.cpu().numpy()
                keypoint_confidences = kps[:, 2]
                avg_conf = np.mean(keypoint_confidences)
                # if avg_conf > 0.48:
                left_ankle = kps[15]  # x, y, conf
                right_ankle = kps[16]

                if left_ankle[2] > 0.50 and right_ankle[2] > 0.50:
                    cx = (left_ankle[0] + right_ankle[0]) / 2
                    cy = (left_ankle[1] + right_ankle[1]) / 2

                    # Apply homography
                    # X, Y =  np.array([cx, cy], dtype=np.float32)
                    input_point = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                    transformed_point = cv2.perspectiveTransform(input_point, H)
                    X, Y = transformed_point[0][0]

                    waypoints.append((frame_count, X, Y))

                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    
                    # Display confidence value
                    confidence_text = f"avg_conf: {box_conf:.2f}, {left_ankle[2]:.2f}, {right_ankle[2]:.2f}"
                    text_position = (int(cx) + 10, int(cy) - 10)
                    cv2.putText(frame, confidence_text, text_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                else:
                    waypoints.append((frame_count, None, None))
        else:
            waypoints.append((frame_count, None, None))

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(frame_delay) & 0xFF
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
with open(os.path.join(output_dir, "pose_waypoints.txt"), "w") as f:
    f.truncate(0)
    for point in waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]}\t{point[2]}\n")

print("Waypoints saved to", os.path.join(output_dir, "pose_waypoints.txt"))

# Sampling
freq = 2.5  # 1 Hz
sampling_interval = int(desired_fps / freq)
sampled_waypoints = []

for i in range(0, len(waypoints), sampling_interval):
    group = waypoints[i:i + sampling_interval]
    valid_points = [(X, Y) for _, X, Y in group if X is not None and Y is not None]

    if valid_points:
        avg_X = sum(p[0] for p in valid_points) / len(valid_points)
        avg_Y = sum(p[1] for p in valid_points) / len(valid_points)
        sampled_waypoints.append((group[0][0], avg_X, avg_Y))

# Save sampled waypoints
with open(os.path.join(output_dir, "pose_waypoints_sampled.txt"), "w") as f:
    f.truncate(0)
    for point in sampled_waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]:.2f}\t{point[2]:.2f}\n")

print("Sampled waypoints saved to", os.path.join(output_dir, "pose_waypoints_sampled.txt"))

# Smoothing
def moving_average(data, window_size=5):
    smoothed = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        avg_x = sum(p[1] for p in window) / len(window)
        avg_y = sum(p[2] for p in window) / len(window)
        smoothed.append((data[i][0], avg_x, avg_y))
    return smoothed

smoothed_waypoints = moving_average(sampled_waypoints)

# Save smoothed waypoints
with open(os.path.join(output_dir, "pose_waypoints_smoothed.txt"), "w") as f:
    for point in smoothed_waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]:.2f}\t{point[2]:.2f}\n")

print("Smoothed waypoints saved to", os.path.join(output_dir, "pose_waypoints_smoothed.txt"))