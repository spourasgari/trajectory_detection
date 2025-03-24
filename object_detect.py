import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str, nargs="?", default="inlab1", help="Name of the video file (without extension)")
args = parser.parse_args()

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize variables
frame_count = 0  # Timestep counter
waypoints = []  # Store waypoints
ped_id = 1.0 # Person ID

# Load the video
file_name = args.file_name
output_dir = f"./{file_name}"
os.makedirs(output_dir, exist_ok=True)

video_path = f"/home/sina/env_prediction_project/object_detection/video_samples/{file_name}.mp4" 
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Define the desired framerate
desired_fps = 25  # Desired framerate
frame_delay = int(1000 / desired_fps)  # Delay between frames in milliseconds

# Homography: Define four known points in image & real-world coordinates
image_points = np.array([
    [319, 58], [547, 132], [382, 313], [127, 209]
], dtype=np.float32)

real_world_points = np.array([
    [72, 84], [192, 84], [192, -36], [72, -36]
], dtype=np.float32)

H, _ = cv2.findHomography(image_points, real_world_points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    
    frame_count += 1  # Increment frame count (timestep)

    # Run YOLO on the frame
    results = model(frame)

    # Extract detections
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()

            if int(cls) == 0 and conf > 0.7:  # Class 0 = "person"
                # Compute center of the bounding box
                cx, cy = int((x1 + x2) / 2), int(y2) # Bottom-middle of the box

                # Apply homography to transform (cx, cy) -> (X, Y)
                input_point = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                transformed_point = cv2.perspectiveTransform(input_point, H)
                X, Y = transformed_point[0][0]

                waypoints.append((frame_count, X, Y))  # Store waypoint

                # Draw the bounding box and waypoint
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            else:
                waypoints.append((frame_count, None, None))

    # Show the video with detections
    cv2.imshow("Tracking", frame)

    # Pause if 'p' is pressed
    key = cv2.waitKey(frame_delay) & 0xFF
    if key == ord('p'):  # Press 'p' to pause
        print("Paused... Press 'p' to resume or 'q' to quit.")
        while True:
            key2 = cv2.waitKey(1) & 0xFF
            if key2 == ord('p'):  # Resume when 'p' is pressed again
                print("Resumed.")
                break
            elif key2 == ord('q'):  # Quit if 'q' is pressed while paused
                print("Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()

# Save full-resolution waypoints
with open(os.path.join(output_dir, "box_waypoints.txt"), "w") as f:
    f.truncate(0)
    for point in waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]}\t{point[2]}\n")

print("Waypoints saved to", os.path.join(output_dir, "box_waypoints.txt"))


## Sampling
freq = 2.5  # Sampling frequency in Hz
sampling_interval = int(desired_fps / freq)
sampled_waypoints = []

for i in range(0, len(waypoints), sampling_interval):
    group = waypoints[i:i + sampling_interval]
    valid_points = [(X, Y) for _, X, Y in group if X is not None and Y is not None]

    if valid_points:
        # Compute average of valid points
        avg_X = int(sum(p[0] for p in valid_points) / len(valid_points))
        avg_Y = int(sum(p[1] for p in valid_points) / len(valid_points))
        sampled_waypoints.append((group[0][0], avg_X, avg_Y))  # Use the first frame_count in the group

# Save sampled waypoints to a new file
with open(os.path.join(output_dir, "box_waypoints_sampled.txt"), "w") as f:
    f.truncate(0)
    for point in sampled_waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]}\t{point[2]}\n")

print("Sampled waypoints saved to", os.path.join(output_dir, "box_waypoints_sampled.txt"))

def moving_average(data, window_size=5):
    smoothed = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        avg_x = sum(p[1] for p in window) / len(window)
        avg_y = sum(p[2] for p in window) / len(window)
        smoothed.append((data[i][0], avg_x, avg_y))  # Keep original timestamp
    return smoothed

smoothed_waypoints = moving_average(sampled_waypoints, window_size=5)
# print(smoothed_waypoints)

# Save smoothed waypoints to a new file
with open(os.path.join(output_dir, "box_waypoints_smoothed.txt"), "w") as f:
    f.truncate(0)
    for point in smoothed_waypoints:
        f.write(f"{point[0]}\t{ped_id}\t{point[1]}\t{point[2]}\n")

print("Smoothed waypoints saved to", os.path.join(output_dir, "box_waypoints_smoothed.txt"))
