import matplotlib.pyplot as plt

import cv2
import numpy as np

output_image = np.zeros((416, 736, 3), dtype=np.uint8)
output_image[:, :, 2] = 255
# cv2.imwrite('red.jpg', output_image)

# output_image = cv2.imread('TTT.jpeg')
# output_image = cv2.resize(output_image, (736, 416))


def stretch_image(image_path, src_pts, width, height):
    """
    Stretches an image to fill a rectangle by applying a perspective transformation.
    
    :param image_path: Path to the input image
    :param width: Target width of the stretched image
    :param height: Target height of the stretched image
    :return: Transformed image
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Define the original curved quadrilateral points (manually identified)
    # src_pts = np.array([[0, 22], [1048, 100], [908, 575], [66, 519]], dtype=np.float32)
    
    # Define the target rectangle points (full stretch)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transformation
    stretched_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return stretched_image

file_name = '391_143_1436_719_0.0_23.0_1045.0_100.0_908.0_576.0_66.0_522.0'
split_coords = file_name.split('_')
rec_x1, rec_y1, recx2, rec_y2 = split_coords[:4]
print('rec_x1, rec_y1, recx2, rec_y2', rec_x1, rec_y1, recx2, rec_y2)
split_coords = split_coords[6:]

# Iterate two items at a time

corner_points = []

for x, y in zip(split_coords[::2], split_coords[1::2]):
    print(f"Point: ({x}, {y})")
    # x, y = float(x), float(y)

    corner_points += [[float(x), float(y)]]
    
width_led = 32 * 23
height_led = 16 * 26

# s_image = stretch_image(f"{file_name}.png", np.array(corner_points, dtype=np.float32), width_led, height_led)
s_image = cv2.imread('modules/res_red.png')
s_image_orig = s_image.copy()

hsv = cv2.cvtColor(s_image, cv2.COLOR_BGR2HSV)

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

locations = np.column_stack(np.where(final_mask1 != 0 & np.any(s_image == [0, 0, 0], axis=-1)))
for location in locations:
    x, y = location
    s_image[x, y] = [0, 255, 0]

locations = np.column_stack(np.where(final_mask2 == 0 | np.any(s_image == [0, 0, 0], axis=-1)))
for location in locations:
    x, y = location
    s_image[x, y] = [0, 255, 0]

# cv2.imshow('suka', s_image)
# cv2.imshow('suka1', s_image_orig)

# cv2.waitKey(0)

# plt.imshow(cv2.cvtColor(s_image, cv2.COLOR_BGR2RGB))
# # Divide the image into a 32x16 grid
# h_step = 16
# w_step = 32

# for i in range(0, height_led, h_step):
#     plt.axhline(y=i, color='white', linestyle='--', linewidth=0.5)
# for j in range(0, width_led, w_step):
#     plt.axvline(x=j, color='white', linestyle='--', linewidth=0.5)
# plt.show()


def select_green_rectangles(image):
    """
    Divides an image into 26x23 rectangles and selects those containing at least 50% green pixels,
    then colors them blue.
    
    :param image_path: Path to the input image
    :return: Processed image with selected rectangles colored blue
    """
    # Load the image
    height, width, _ = image.shape
    
    rect_width, rect_height = 32, 16
    selected_rectangles = []
    for y in range(0, height, rect_height):
        for x in range(0, width, rect_width):
            # Extract the rectangle
            roi = image[y:y+rect_height, x:x+rect_width]
            
            # Count green pixels (assume green is [0, 255, 0])
            green_mask = (roi[:, :, 0] == 0) & (roi[:, :, 1] == 255) & (roi[:, :, 2] == 0)
            green_ratio = np.sum(green_mask) / (rect_width * rect_height)
            
            # Select if at least 50% green
            if green_ratio >= 0.3:
                image[y:y+rect_height, x:x+rect_width] = [255, 0, 0]  # Color blue
                selected_rectangles.append((x, y, rect_width, rect_height))
    
    return image, selected_rectangles

def find_adjacent_non_blue(blue_rectangles, width, height):
    """
    For each blue rectangle, prints all adjacent rectangles that are not blue.
    
    :param blue_rectangles: Set of coordinates of blue rectangles
    :param width: Image width
    :param height: Image height
    :param rect_width: Width of each rectangle
    :param rect_height: Height of each rectangle
    """
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    res = []
    for x, y, rect_width, rect_height in blue_rectangles:
        adjacent_non_blue = []
        for dx, dy in directions:
            adj_x, adj_y = x + dx * rect_width, y + dy * rect_height
            if 0 <= adj_x < width and 0 <= adj_y < height and (adj_x, adj_y, rect_width, rect_height) not in blue_rectangles:
                adjacent_non_blue.append((adj_x, adj_y, rect_width, rect_height))
                s_image[adj_y:adj_y+rect_height, adj_x:adj_x+rect_width] = [255, 255, 255]
        
        res += [[(x, y, rect_width,rect_height), adjacent_non_blue]]
    return res


m_image, selected = select_green_rectangles(s_image)
# print(selected)  # List of rectangles with at least 50% green


plt.imshow(cv2.cvtColor(m_image, cv2.COLOR_BGR2RGB))
# Divide the image into a 32x16 grid
h_step = 16
w_step = 32

# for i in range(0, height_led, h_step):
#     plt.axhline(y=i, color='white', linestyle='--', linewidth=0.5)
# for j in range(0, width_led, w_step):
#     plt.axvline(x=j, color='white', linestyle='--', linewidth=0.5)
    
# plt.show()




blues_with_whites = find_adjacent_non_blue(selected, width_led, height_led)

# print('resss', blues_with_whites)

def calculate_brightness_contrast(image, res_image, broken_module, white_modules):
    x, y, w, h = broken_module
    image1 = image[y:y+h, x:x+w]
    # broken = np.mean(image1, axis=(0, 1))

    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        
    avg_brightness_broken = np.mean(image1)
    avg_contrast_broken = np.std(image1) / avg_brightness_broken    
    
    brightness_avg = []
    contrast_avg = []
    
    for i, white_module in enumerate(white_modules):
        x_w, y_w, w_w, h_w = white_module
        ref_img = image[y_w:y_w+h_w, x_w:x_w+w_w]
        
        gray_image = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_image)
        avg_contrast = np.std(gray_image) / avg_brightness
        
        brightness_avg.append(avg_brightness)
        contrast_avg.append(avg_contrast)
    
    brightness_avg = np.mean(brightness_avg)
    contrast_avg = np.mean(contrast_avg)
    
    # print('broken', avg_brightness_broken, avg_contrast_broken)
    # print('correct', brightness_avg, contrast_avg)

    # print(image[y:y+h, x:x+w].shape)
    # print('cccc', image[y:y+h, x:x+w], np.clip(image[y:y+h, x:x+w] * contrast_factor + brightness_diff, 0, 255))
    # res_image[y:y+h, x:x+w] = np.clip(res_image[y:y+h, x:x+w] * contrast_factor + brightness_diff, 0, 255)
    
    # print('image1', image1.shape, np.mean(image1, axis=(0, 1)))
    # res_image[y:y+h, x:x+w] = np.clip(res_image[y:y+h, x:x+w] * 1.0125 + 3.05, 0, 255)
    # if y == 0:
        # res_image[y:y+h, x:x+w] = np.clip(res_image[y:y+h, x:x+w] - np.array([45, 45, 45]), 0, 255)
    res_image[y:y+h, x:x+w] = [0, 0, 190]
    
    
    return 0, 0

def adjust_pixel(rgb_pixel, contrast_factor, brightness_diff):
    new_pixel = np.clip(contrast_factor * np.array(rgb_pixel) + brightness_diff, 0, 255)
    return tuple(new_pixel.astype(np.uint8))

# cv2.imwrite('test.jpg', s_image_orig)
for broken_modules in blues_with_whites:
    broken_module, white_modules = broken_modules[0], broken_modules[1:][0]
    # print('ssssusuuuuussuusus', s_image_orig.shape)
    calculate_brightness_contrast(s_image_orig, output_image, broken_module, white_modules)

cv2.imshow('frame1', s_image_orig)
cv2.imshow('frame', output_image)

cv2.waitKey(0)

# plt.show()
# # red_image[..., [0, 2]] = red_image[..., [2, 0]]
# cv2.imwrite('test2.jpg', output_image)

# new_image = cv2.imread('TTT.jpeg')
# new_image = cv2.resize(new_image, (width_led, height_led))
# for broken_modules in blues_with_whites:
#     broken_module, white_modules = broken_modules[0], broken_modules[1:][0]
#     # print('ssssusuuuuussuusus', s_image_orig.shape)
#     calculate_brightness_contrast(s_image_orig, new_image, broken_module, white_modules)
# cv2.imwrite('test2.jpg', new_image)

