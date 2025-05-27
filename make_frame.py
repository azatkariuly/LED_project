import cv2
import numpy as np

# Define image size and color
width, height = 64, 32
red_color = (0, 0, 255)  # BGR format in OpenCV

# Create the image
red_frame = np.full((height, width, 3), red_color, dtype=np.uint8)

# Save the image
cv2.imwrite("red_frame.png", red_frame)


# import cv2
# import numpy as np
# import random

# # Constants
# width, height = 64, 32
# rect_w, rect_h = 32, 32
# alpha_range = (0.8, 1.2)  # Random alpha between 0.8 and 1.2
# beta_range = (-30, 30)    # Random beta between -30 and 30

# # Create a red image
# red_color = (0, 0, 255)  # BGR
# frame = np.full((height, width, 3), red_color, dtype=np.uint8)

# # Copy frame to modify
# frame = cv2.imread('../1.png')

# frame_modified = frame.copy()

# # Calculate how many rectangles
# cols = width // rect_w
# rows = height // rect_h

# # File to save modified rectangles
# log_lines = []

# # Randomly modify some rectangles
# for row in range(rows):
#     for col in range(cols):
#         if random.random() < 0.2:  # 20% chance to modify a rectangle
#             x, y = col * rect_w, row * rect_h
#             x2, y2 = x + rect_w, y + rect_h

#             # Random alpha and beta
#             alpha = round(random.uniform(*alpha_range), 2)
#             beta = round(random.uniform(*beta_range), 2)

#             # Apply the filter
#             frame_modified[y:y2, x:x2] = np.clip(
#                 frame_modified[y:y2, x:x2] * (alpha - 0.2) + beta,
#                 0, 255
#             ).astype(np.uint8)

#             # Log the modification
#             log_lines.append(f"({x}, {y}, {rect_w}, {rect_h}) {alpha} {beta}")

# # Save the modified frame
# cv2.imwrite("modified_frame.png", frame_modified)

# # Save the log
# with open("modified_rectangles.txt", "w") as f:
#     f.write("\n".join(log_lines))

# print("Image and log saved.")