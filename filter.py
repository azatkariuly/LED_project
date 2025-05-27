import cv2
import numpy as np

# Load the image
image = cv2.imread("cropped_image_0.0_24.0_1047.0_102.0_910.0_577.0_66.0_522.0.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for better color processing

# Function to convert RGB to HSV
def rgb_to_hsv(rgb):
    rgb_np = np.uint8([[rgb]])  # Convert to numpy format
    hsv_color = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)[0][0]
    return hsv_color

# Define the two main colors: Black & Background
black_hsv = rgb_to_hsv([0, 0, 0])
main_hsv = rgb_to_hsv([234, 84, 103])

# Define close-range HSV boundaries for the main colors (to exclude)
black_lower = np.array([0, 0, 0])  # Pure black
black_upper = np.array([180, 255, 50])  # Slightly dark shades

main_lower = np.array([main_hsv[0] - 20, 70, 90])  # Background color lower bound
main_upper = np.array([main_hsv[0] + 20, 95, 120])  # Background color upper bound

# Create masks for the two main colors
mask_black = cv2.inRange(hsv, black_lower, black_upper)
mask_main = cv2.inRange(hsv, main_lower, main_upper)

# Combine masks to exclude the main colors
mask_exclude = cv2.bitwise_or(mask_black, mask_main)
mask_different = cv2.bitwise_not(mask_exclude)  # Invert to get different pixels

# Define the two colors to detect: RGB(221,123,98) and RGB(168,62,69)
color1_hsv = rgb_to_hsv([221, 123, 98])
color2_hsv = rgb_to_hsv([168, 62, 69])

# Define close-range color boundaries for the different pixels
# color1_lower = np.array([color1_hsv[0] - 10, 50, 50])
# color1_upper = np.array([color1_hsv[0] + 10, 255, 255])

color1_lower = np.array([color2_hsv[0] - 10, 50, 50])
color1_upper = np.array([color2_hsv[0] + 10, 205, 205])

color2_lower = np.array([color2_hsv[0] - 10, 50, 50])
color2_upper = np.array([color2_hsv[0] + 10, 255, 255])

# Create masks for the two different colors
mask_color1 = cv2.inRange(hsv, color1_lower, color1_upper)
mask_color2 = cv2.inRange(hsv, color2_lower, color2_upper)

# Combine the detected color masks with the different-pixel mask
# final_mask = cv2.bitwise_and(mask_different, cv2.bitwise_or(mask_color1, mask_color2))
final_mask1 = cv2.bitwise_and(mask_different, mask_color1)
final_mask2 = cv2.bitwise_and(mask_different, mask_color2)

# Highlight detected pixels on the original image
output = image.copy()
# output[final_mask > 0] = [0, 255, 0]  # Color detected pixels in green

output[final_mask1 != 0 & (np.any(output == [0, 0, 0], axis=-1))] = [0, 255, 0]
output[final_mask2 == 0 | (np.any(output == [0, 0, 0], axis=-1))] = [0, 255, 0]

# Show the result
cv2.imshow("Detected Different Pixels", output)

# Save the result
# cv2.imwrite("detected_pixels.png", output)
cv2.waitKey(0)
cv2.destroyAllWindows()














# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV for better color processing

# # Convert given RGB colors to HSV
# def rgb_to_hsv(rgb):
#     rgb_np = np.uint8([[rgb]])  # Convert to numpy format
#     hsv_color = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)[0][0]
#     return hsv_color

# # Main color (background) - RGB(234,84,103)
# main_hsv = rgb_to_hsv([234, 84, 103])

# # Define color ranges for the main color (to exclude)
# main_lower = np.array([main_hsv[0] - 10, 84, 100])  # Lower bound
# main_upper = np.array([main_hsv[0] + 10, 90, 113])  # Upper bound

# # Create a mask for the main color
# mask_main = cv2.inRange(hsv, main_lower, main_upper)

# # Invert the mask to detect different colors
# mask_different = cv2.bitwise_not(mask_main)

# print('sss', mask_different)

# # Define the two colors to detect
# color1_hsv = rgb_to_hsv([221, 123, 98])
# color2_hsv = rgb_to_hsv([168, 62, 69])

# # Define close-range color boundaries for different pixels
# color1_lower = np.array([color1_hsv[0] - 10, 50, 50])
# color1_upper = np.array([color1_hsv[0] + 10, 255, 255])

# color2_lower = np.array([color2_hsv[0] - 10, 50, 50])
# color2_upper = np.array([color2_hsv[0] + 10, 255, 255])

# # Create masks for the two target colors
# mask_color1 = cv2.inRange(hsv, color1_lower, color1_upper)
# mask_color2 = cv2.inRange(hsv, color2_lower, color2_upper)

# # Combine detected masks with the difference mask
# final_mask = cv2.bitwise_and(mask_different, cv2.bitwise_or(mask_color1, mask_color2))

# # Highlight detected pixels in the original image
# output = image.copy()
# output[final_mask == 0] = [0, 255, 0]  # Color detected pixels in green

# # Show the result
# cv2.imshow("Detected Different Pixels", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
