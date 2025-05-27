import cv2
import numpy as np

def calculate_brightness_contrast(image):
    # Convert the image to grayscale (simplifies calculation)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness (mean of pixel values)
    brightness = np.mean(gray_image)
    
    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(gray_image)
    
    return brightness, contrast

def adjust_brightness_contrast(image, brightness_diff, contrast_diff):
    # Convert image to grayscale for calculations
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply contrast adjustment (scaling the image by contrast_diff)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_diff, beta=brightness_diff)

    return adjusted_image

# Load the two images
image2 = cv2.imread('broken.jpg')  # First image (to be adjusted)
image1 = cv2.imread('correct.jpg')  # Second image (target image)

# Calculate brightness and contrast for both images
brightness1, contrast1 = calculate_brightness_contrast(image1)
brightness2, contrast2 = calculate_brightness_contrast(image2)

# Calculate the difference in brightness and contrast
brightness_diff = brightness2 - brightness1
contrast_diff = contrast2 / contrast1 if contrast1 != 0 else 1  # Avoid division by zero

# Apply the brightness and contrast adjustments to the first image
adjusted_image1 = adjust_brightness_contrast(image1, brightness_diff, contrast_diff)

# Show the images
cv2.imshow("Original Image", image1)
cv2.imshow("Target Image", image2)
cv2.imshow("Adjusted Image", adjusted_image1)

cv2.waitKey(0)
cv2.destroyAllWindows()
