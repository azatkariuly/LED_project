import cv2
import numpy as np
import re

# Constants
rect_w, rect_h = 32, 32

# Load original red frame
frame = cv2.imread("../3.jpg")
if frame is None:
    raise FileNotFoundError("red_frame.png not found!")

frame = cv2.resize(frame, (2816, 192), interpolation=cv2.INTER_AREA)
# Read modification log
with open("modified_rectangles.txt", "r") as f:
    lines = f.readlines()

# Apply modifications from the log
for line in lines:
    # Parse the line using regex
    match = re.match(r"\((\d+), (\d+), (\d+), (\d+)\)\s+([\d.]+)\s+(-?\d+)", line.strip())
    if match:
        x, y, w, h = map(int, match.group(1, 2, 3, 4))
        alpha = float(match.group(5))
        beta = float(match.group(6))

        # Compute bottom right coordinates
        x2, y2 = x + w, y + h

        # Apply filter
        frame[y:y2, x:x2] = np.clip(
            frame[y:y2, x:x2] * (alpha - 0.2) + beta,
            0, 255
        ).astype(np.uint8)
    else:
        print(f"Invalid line format: {line.strip()}")

# Save the re-modified image
cv2.imwrite("reconstructed_modified_frame.png", frame)
print("Reconstructed image saved.")
