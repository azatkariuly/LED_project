from PIL import Image
import numpy as np

# Load the image
image_path = "suka2.png"  # Update with your image path
img = Image.open(image_path)
img_np = np.array(img)

# Get image dimensions
height, width, _ = img_np.shape

# Divide image into 4 quadrants
half_h, half_w = height // 2, width // 2
sub_images = {
    "top_left": img_np[0:half_h, 0:half_w],
    "top_right": img_np[0:half_h, half_w:width],
    "bottom_left": img_np[half_h:height, 0:half_w],
    "bottom_right": img_np[half_h:height, half_w:width],
}

print("Ssss",sub_images)

# Calculate average RGB color for each sub-image
average_colors = {
    name: sub_img.mean(axis=(0, 1)).astype(int)
    for name, sub_img in sub_images.items()
}

# Print the results
for name, color in average_colors.items():
    print(f"{name}: {tuple(color)}")
