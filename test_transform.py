import cv2
import numpy as np

## TESTS SHOULD BE DONE ON THE IMAGE WITH THE SAME PIXEL COUNT ##

# Homography: define image & real-world points

# ## In Lab - Door corner
# image_points = np.array([
#     [319, 58], [547, 132], [382, 313], [127, 209]
# ], dtype=np.float32)

# real_world_points = np.array([
#     [72, 84], [192, 84], [192, -36], [72, -36]
# ], dtype=np.float32)

# ## In Lab - Mata's Desk
# image_points = np.array([
#     [547, 132], [1135, 231], [1066, 590], [432, 512]
# ], dtype=np.float32)

# real_world_points = np.array([
#     [253, 151], [13, 91], [13, 271], [193, 331]
# ], dtype=np.float32)

# ## Big Aisle - Left side of orange cone
image_points = np.array([
    [224, 184], [504, 164], [544, 506], [154, 539]
], dtype=np.float32)

real_world_points = np.array([
    [0, 180], [120, 180], [120, 0], [0, 0]
], dtype=np.float32)

H, _ = cv2.findHomography(image_points, real_world_points)
# print(H)

# Test the homography transformation
print("Pixel Coordinates -> Real-world Coordinates:")
test_pixels = [
    # (125, 89),
    # (231, 127),
    # (763, 320),
    # (432, 677),
    # Mata's desk reference:
    (731, 101), # 193, 91
    (763, 320) # 133, 211
]

for px, py in test_pixels:
    input_point = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(input_point, H)
    X, Y = transformed_point[0][0]
    print(f"({px}, {py}) -> ({X:.2f}, {Y:.2f}) centimeters")