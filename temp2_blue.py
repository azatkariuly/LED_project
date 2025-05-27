# import numpy as np 
# import cv2

# #load the image
# image = cv2.imread("modules/res_blue.png")

import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_fill_almost_closed_shapes(image, low_threshold=50, high_threshold=150, dilation_iterations=3):

    if image is None:
        print("Error loading image.")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Dilation to close small gaps in edges
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for dilation
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)

    # Find contours in the dilated edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for filling shapes
    mask = np.zeros_like(gray)

    # Fill the detected contours
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Create a result image where filled regions are highlighted
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # result[result > 0] = 255
    # cv2.imshow('d', result)
    # cv2.waitKey(0)
    
    return result

    # # Display the images
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # plt.subplot(1, 4, 2), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Detection (Canny)'), plt.xticks([]), plt.yticks([])

    # plt.subplot(1, 4, 3), plt.imshow(dilated_edges, cmap='gray')
    # plt.title('Dilated Edges'), plt.xticks([]), plt.yticks([])

    # plt.subplot(1, 4, 4), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.title('Filled Shapes'), plt.xticks([]), plt.yticks([])

    # plt.show()

def select_broken_modules(image, mask, rect_width=32, rect_height=16):
    # Load the image
    height, width, _ = image.shape
    
    selected_rectangles = []
    for y in range(0, height, rect_height):
        for x in range(0, width, rect_width):
            # Extract the rectangle
            # roi = image[y:y+rect_height, x:x+rect_width]
            roi = mask[y:y+rect_height, x:x+rect_width]
            
            # Count green pixels (assume green is [0, 255, 0])
            green_mask = (roi[:, :, 0] > 0) & (roi[:, :, 1] > 0) & (roi[:, :, 2] > 0)
            green_ratio = np.sum(green_mask) / (rect_width * rect_height)
            
            # Select if at least 50% green
            if green_ratio >= 0.5:
                image[y:y+rect_height, x:x+rect_width] = [255, 0, 0]  # Color blue
                selected_rectangles.append((x, y, rect_width, rect_height))
    
    return image, selected_rectangles

def find_adjacent_non_blue(blue_rectangles, width=736, height=416):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    res = []
    for x, y, rect_width, rect_height in blue_rectangles:
        adjacent_non_blue = []
        for dx, dy in directions:
            adj_x, adj_y = x + dx * rect_width, y + dy * rect_height
            if 0 <= adj_x < width and 0 <= adj_y < height and (adj_x, adj_y, rect_width, rect_height) not in blue_rectangles:
                adjacent_non_blue.append((adj_x, adj_y, rect_width, rect_height))
                # image[adj_y:adj_y+rect_height, adj_x:adj_x+rect_width] = [255, 255, 255]
        res += [[(x, y, rect_width,rect_height), adjacent_non_blue]]
    return res

def calculate_brightness_contrast(incorrect_pixel, desired_pixel):
    incorrect_pixel = np.array(incorrect_pixel, dtype=np.float32)
    desired_pixel = np.array(desired_pixel, dtype=np.float32)
    
    # Solve for alpha and beta
    A = np.vstack([incorrect_pixel, np.ones_like(incorrect_pixel)]).T
    x, _, _, _ = np.linalg.lstsq(A, desired_pixel, rcond=None)
    
    alpha, beta = x
    return alpha, beta

def restore_broken_modules(image, res_image, broken_module, white_modules):
    # Convert to grayscale if image1 is in color
    x, y, w, h = broken_module
    image1 = image[y:y+h, x:x+w]
    
    # broken_brightness, broken_contrast = calculate_brightness_contrast(image1)
    broken_rgb = np.mean(image1, axis=(0, 1))
    # print('broken', x, y, np.mean(image1, axis=(0, 1)))

    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    avg_brightness, avg_contrast = [], []
    avg_rgb = []
    for i, white_module in enumerate(white_modules):
        x_w, y_w, w_w, h_w = white_module
        ref_img = image[y_w:y_w+h_w, x_w:x_w+w_w]
        
        avg_rgb.append(ref_img)
        
        # ref_brightness, ref_contrast = calculate_brightness_contrast(ref_img)
        # avg_brightness.append(ref_brightness)
        # avg_contrast.append(ref_contrast)
    
    avg_rgb = np.mean(np.mean(avg_rgb, axis=0), axis=(0, 1))
    diff_rgb = avg_rgb - broken_rgb
    # print('arg', avg_rgb, avg_rgb - broken_rgb)
    
    # brightness_diff = np.mean(avg_brightness) - broken_brightness
    # contrast_diff = np.mean(avg_contrast) / broken_contrast if broken_contrast != 0 else 1
    
    
    # if 288 <= x <= 320 and y == 96:
    #     print('do', broken_rgb, avg_rgb)
    alpha, beta = calculate_brightness_contrast(broken_rgb, avg_rgb)
    print(alpha, beta, is_image_dark(res_image[y:y+h, x:x+w]))
    res_image[y:y+h, x:x+w] = np.clip(res_image[y:y+h, x:x+w] * (alpha-0.2) + beta, 0, 255)
    # print('broken', broken_brightness, broken_contrast, brightness_diff, contrast_diff)
    
    # res_image[y:y+h, x:x+w] = cv2.convertScaleAbs(res_image[y:y+h, x:x+w], alpha=contrast_diff, beta=brightness_diff)
    # res_image[y:y+h, x:x+w] = np.clip(res_image[y:y+h, x:x+w] - np.array([60, 60, 60]), 0, 255)
    return

def is_image_dark(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    avg_brightness = np.mean(gray)  # Calculate average pixel intensity
    return avg_brightness < threshold

# Example usage:
file_name = 'modules/res_red.png'
# Load the image
image = cv2.imread(file_name)
broken_mask = detect_and_fill_almost_closed_shapes(image, low_threshold=50, high_threshold=100, dilation_iterations=2)
image, selected = select_broken_modules(image, broken_mask)

s_image = cv2.imread(file_name)
s_image_orig = s_image.copy()

output_image = np.zeros((416, 736, 3), dtype=np.uint8)
output_image[:, :, 2] = 255

blues_with_whites = find_adjacent_non_blue(selected)
# cv2.imshow('d', image)
# cv2.waitKey(0)


for broken_modules in blues_with_whites:
    broken_module, white_modules = broken_modules[0], broken_modules[1:][0]
    restore_broken_modules(s_image_orig, output_image, broken_module, white_modules)


cv2.imwrite('test2.jpg', output_image)
# cv2.imshow('d1', s_image_orig)
# cv2.imshow('d', output_image)
# cv2.waitKey(0)