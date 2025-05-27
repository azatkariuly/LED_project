import cv2
import numpy as np
# from skimage import exposure

def calculate_brightness_contrast(image1, image2):
    """
    Calculate brightness and contrast difference to adjust image1 to match image2.
    
    Parameters:
    - image1: np.ndarray (Grayscale image that needs correction)
    - image2: np.ndarray (Reference grayscale image)
    
    Returns:
    - brightness_diff: float (Brightness adjustment needed)
    - contrast_factor: float (Contrast scaling factor needed)
    """
    # Convert to grayscale if images are in color
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute mean brightness difference
    brightness_diff = np.mean(image2) - np.mean(image1)
    
    # Compute contrast scaling factor
    std_image1 = np.std(image1)
    std_image2 = np.std(image2)
    contrast_factor = std_image2 / std_image1 if std_image1 != 0 else 1
    
    return brightness_diff, contrast_factor

# Example usage
broken_img = cv2.imread("dark_broken.png", cv2.IMREAD_GRAYSCALE)
original_img = cv2.imread("dark_orig.png", cv2.IMREAD_GRAYSCALE)
brightness, contrast = calculate_brightness_contrast(broken_img, original_img)
print(f"Brightness adjustment: {brightness}, Contrast scaling: {contrast}")
