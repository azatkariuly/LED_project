import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import zoom
import cv2

# Open the image file
file_name = 'cropped_image_390_144_1438_719_0.0_22.0_1048.0_100.0_908.0_575.0_66.0_519.0'
image = Image.open(f"{file_name}.png")

# # Create a drawing object
# draw = ImageDraw.Draw(image)

split_coords = file_name.split('_')
rec_x1, rec_y1, recx2, rec_y2 = split_coords[2:6]
print('rec_x1, rec_y1, recx2, rec_y2', rec_x1, rec_y1, recx2, rec_y2)
split_coords = split_coords[6:]
print(split_coords)

# Iterate two items at a time

corner_points = []

for x, y in zip(split_coords[::2], split_coords[1::2]):
    print(f"Point: ({x}, {y})")

    corner_points.append((float(x), float(y)))

    # # Define the point and the size of the dot
    # point = (float(x), float(y))  # Coordinates of the dot (x, y)
    # dot_radius = 5  # Radius of the dot
    # dot_color = "red"  # Color of the dot

    # # Draw the dot (as a filled circle)
    # # draw.ellipse(
    # #     [
    # #         (point[0] - dot_radius, point[1] - dot_radius),  # Top-left corner
    # #         (point[0] + dot_radius, point[1] + dot_radius),  # Bottom-right corner
    # #     ],
    # #     fill=dot_color,
    # # )

# # Save or show the image
# image.show()

# upper_left_point = (0, 24)
# upper_right_point = (1047, 102)
# bottom_right_point = (910, 577)
# bottom_left_point = (66, 522)

print('AA', corner_points)

image_array = np.array(image)
image_height, image_width, _ = image_array.shape
# print(image_array.shape)  # (height, width, channels)
transformed_array = np.zeros((577, 1047, 4), dtype=np.uint8)
fixed_array = np.zeros((577, 1047))

# upper_line = []
# lower_line = []
# left_line = []
# right_line = []

edge_lines = []

# TODO: Implement the transformation logic here

def resize_pixel_array(pixels, new_size):
    # Resizing the array along the 1st dimension (number of pixels)
    zoom_factors = (new_size / pixels.shape[0], 1)  # Only zoom along the number of pixels
    resized_pixels = zoom(pixels, zoom_factors, order=1)  # Bilinear interpolation
    return resized_pixels


temp = None
upper_line = []
pixel_position = int(corner_points[0][1])
for i in range(int(corner_points[0][0]), int(corner_points[1][0])):
    for j in range(-1,2):
        if pixel_position+j < image_height:
            pixel = image_array[pixel_position+j, i]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                upper_line += [[pixel_position, i]]
                if temp is None:
                    temp = image_array[pixel_position, i]
                else:
                    temp = np.vstack((temp, image_array[pixel_position, i]))
                # image_array[pixel_position, i] = [0, 0, 255, 255]
                break

edge_lines.append(upper_line)

temp = None
right_line = []
pixel_position = int(corner_points[1][0])
for i in range(int(corner_points[1][1]), int(corner_points[2][1])):
    for j in range(1,-2,-1):
        if pixel_position+j < image_width:
            pixel = image_array[i, pixel_position+j]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                right_line += [[i, pixel_position]]
                if temp is None:
                    temp = image_array[i, pixel_position]
                else:
                    temp = np.vstack((temp, image_array[i, pixel_position]))
                # image_array[i, pixel_position] = [0, 255, 0, 255]
                break

edge_lines.append(right_line)

temp = None
bottom_line = []
pixel_position = int(corner_points[3][1])
for i in range(int(corner_points[3][0]), int(corner_points[2][0])):
    for j in range(1,-2,-1):
        if pixel_position+j < image_height:
            pixel = image_array[pixel_position+j, i]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                bottom_line += [[pixel_position, i]]
                if temp is None:
                    temp = image_array[pixel_position, i]
                else:
                    temp = np.vstack((temp, image_array[pixel_position, i]))
                # image_array[pixel_position, i] = [0, 0, 255, 255]
                break

edge_lines.append(bottom_line)

temp = None
left_line = []
pixel_position = int(corner_points[0][0])
for i in range(int(corner_points[0][1]), int(corner_points[3][1])):
    for j in range(-1,2):
        if pixel_position+j < image_width:
            pixel = image_array[i, pixel_position+j]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                left_line += [[i, pixel_position]]
                if temp is None:
                    temp = image_array[i, pixel_position]
                else:
                    temp = np.vstack((temp, image_array[i, pixel_position]))
                # image_array[i, pixel_position] = [0, 255, 0, 255]
                break
            
edge_lines.append(left_line)


print('suak edge lines', edge_lines)

# transformed_array[1] = resize_pixel_array(temp, 1047)

# print(image_array[22,1])
# transformed_array[0, 0] = image_array[24, 0]
# print(image_array[22, 2],image_array[0,0].shape)

def map_to_rectangle(point, height, width):
    a, b = point
    x1, y1 = int(corner_points[0][0]), int(corner_points[0][1])
    x2, y2 = int(corner_points[1][0]), int(corner_points[1][1])
    x3, y3 = int(corner_points[2][0]), int(corner_points[2][1])
    x4, y4 = int(corner_points[3][0]), int(corner_points[3][1])

    a1, b1, a2, b2, a3, b3, a4, b4 = None, None, None, None, None, None, None, None

    if x2 > a > x1:
        a1, b1 = edge_lines[0][a-x1]
    elif a <= x1:
        a1, b1 = x1, y1
    else:
        a1, b1 = x2, y2

    if y3 > b > y2:
        a2, b2 = edge_lines[1][b-y2]
    elif b <= y2:
        a2, b2 = x2, y2
    else:
        a2, b2 = x3, y3

    if x3 > a > x4:
        a3, b3 = edge_lines[2][a-x4]
    elif a <= x4:
        a3, b3 = x4, y4
    else:
        a3, b3 = x3, y3

    if y4 > b > y1:
        # print(y4, b, y1, len(left_line))
        a4, b4 = edge_lines[3][b-y1]
    elif b <= y1:
        a4, b4 = x1, y1
    else:
        a4, b4 = x4, y4

    if a1 >= 0 and b1 >= 0 and a2 >= 0 and b2 >= 0 and a3 >= 0 and b3 >= 0 and a4 >= 0 and b4 >=0:
        coord_x = height * ((a1-x1)/(x2-x1) + (a-a1)/(a3-a1) * ((a3-x4)/(x3-x4) - (a1-x1)/(x2-x1)))
        coord_y = width * ((b4-y1)/(y4-y1) + (b-b4)/(b2-b4) * ((b2-y2)/(y3-y2) - (b4-y1)/(y4-y1)))
        return min(max(0, int(coord_x)), width-1), min(max(0, int(coord_y)), height-1)
    return None


# image = cv2.imread("cropped_image_0.0_24.0_1047.0_102.0_910.0_577.0_66.0_522.0.png")
image = cv2.imread("cropped_image_390_144_1436_719_0.0_22.0_1046.0_99.0_909.0_575.0_66.0_520.0.png")
image_test = cv2.imread('frame.jpg')
cv2.imshow('Orign', image_test)
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

def adjust_pixel(rgb_pixel, contrast_factor, brightness_diff):
    new_pixel = np.clip(contrast_factor * np.array(rgb_pixel) + brightness_diff, 0, 255)
    return tuple(new_pixel.astype(np.uint8))

def fake_filter(image, point):
    neighbor_color = []
    y, x = point
    y_orig, x_orig = y-int(rec_y1), x-int(rec_x1)
    height, width, _ = image.shape

    # print(y_orig, x_orig, image_array[y_orig, x_orig], fixed_array[y_orig, x_orig])

    # top_left
    if y_orig > 0 and x_orig > 0:
        _, _, _, t = image_array[y_orig - 1, x_orig - 1]
        if [y_orig - 1, x_orig - 1] not in locations or fixed_array[y_orig - 1, x_orig - 1] == 1:
            if t == 255:
                neighbor_color += [image[y-1, x-1].tolist()]

    
    # # top_middle
    # if y > 0:
    #     neighbor_color += [image[y-1, x].tolist()]

    # # top_left
    # if y > 0 and x > 0:
    #     neighbor_color += [image[y-1, x-1].tolist()]
    # # top_middle
    # if y > 0:
    #     neighbor_color += [image[y-1, x].tolist()]
    # # top_right
    # if y > 0 and x < width - 1:
    #     neighbor_color += [image[y-1, x+1].tolist()]
    # # middle_left
    # if x > 0:
    #     neighbor_color += [image[y, x-1].tolist()]
    # # middle_right
    # if x < width - 1:
    #     neighbor_color += [image[y, x+1].tolist()]
    # # bottom_left
    # if y < height - 1 and x > 0:
    #     neighbor_color += [image[y+1, x-1].tolist()]
    # # bottom_middle
    # if y < height - 1:
    #     neighbor_color += [image[y+1, x].tolist()]
    # # bottom_right
    # if y < height - 1 and x < width - 1:
    #     neighbor_color += [image[y+1, x+1].tolist()]

    fixed_array[y_orig, x_orig] = 1
    if neighbor_color:
        r_avg = sum(color[2] for color in neighbor_color) / len(neighbor_color)
        g_avg = sum(color[1] for color in neighbor_color) / len(neighbor_color)
        b_avg = sum(color[0] for color in neighbor_color) / len(neighbor_color)

        res = [int(b_avg), int(g_avg), int(r_avg)]
        return res
    return None



# Get locations where final_mask1 is 1
locations = np.column_stack(np.where(final_mask1 != 0 & np.any(image == [0, 0, 0], axis=-1)))
for location in locations:
    x, y = location
    # # image_array[x, y]  = [0, 255, 0, 255]
    # mapped_point = map_to_rectangle((x, y), 1047, 577)
    # # transformed_array[mapped_point] = [0, 255, 0, 255]
    l1, l2, l3, _ = image_array[x, y]
    r1, r2, r3 = image_test[x+int(rec_y1), y+int(rec_x1)]
    
    # if r1 != 0 and l1 != 0:
    #     # print('r1', r1, l1, type(r1), type(l1), np.int32(r1) * np.int32(l1))
    #     # print(int(np.int32(r1) * np.int32(l1) / np.int32(231)))
    #     r1 = max(0, min(255, int(np.int32(r1) * 105 / 255)))
    # if r2 != 0 and l2 != 0:
    #     r2 = max(0, min(255, int(np.int32(r2) * 105 / 150)))
    # if r3 != 0 and l3 != 0:
    #     r3 = max(0, min(255, int(np.int32(r3) * 105 / 150)))

    # r1 = min(0, max(255, int(l1 * r1 / 231)))
    # r2 = min(0, max(255, int(l2 * r2 / 83)))
    # r3 = min(0, max(255, int(l3 * r3 / 99)))
    # # print('image_array', image_array[x, y])

    # image_test[mapped_point] = [r1, r2, r3]
    # image_test[x+int(rec_y1), y+int(rec_x1)] = [0, 0, 255]

    old_pixel = image_test[x+int(rec_y1), y+int(rec_x1)]
    # fake_color = fake_filter(image_test, (x+int(rec_y1), y+int(rec_x1)))
    # old_pixel = (100, 150, 200)  # Example RGB pixel
    new_pixel = adjust_pixel(old_pixel, 1.0329, 5.05)
    if new_pixel:
        image_test[x+int(rec_y1), y+int(rec_x1)] = new_pixel

locations = np.column_stack(np.where(final_mask2 == 0 | np.any(image == [0, 0, 0], axis=-1)))
for location in locations:
    x, y = location
    # # image_array[x, y]  = [0, 255, 0, 255]
    # mapped_point = map_to_rectangle((x, y), 1047, 577)
    # # transformed_array[mapped_point] = [0, 255, 0, 255]
    l1, l2, l3, _ = image_array[x, y]
    r1, r2, r3 = image_test[x+int(rec_y1), y+int(rec_x1)]    

    # if r1 != 0 and l1 != 0:
    #     # print('r1', r1, l1, type(r1), type(l1), np.int32(r1) * np.int32(l1))
    #     # print(int(np.int32(r1) * np.int32(l1) / np.int32(231)))
    #     r1 = max(0, min(255, int(np.int32(r1) * 231 / 255)))
    # if r2 != 0 and l2 != 0:
    #     r2 = max(0, min(255, int(np.int32(r2) * 225 / 255)))
    # if r3 != 0 and l3 != 0:
    #     r3 = max(0, min(255, int(np.int32(r3) * 220 / 255)))

    # image_test[mapped_point] = [r1, r2, r3]
    # image_test[x+int(rec_y1), y+int(rec_x1)] = [255,0,0]
    old_pixel = image_test[x+int(rec_y1), y+int(rec_x1)]
    # fake_color = fake_filter(image_test, (x+int(rec_y1), y+int(rec_x1)))
    # if fake_color:
    #     image_test[x+int(rec_y1), y+int(rec_x1)] = fake_color
    new_pixel = adjust_pixel(old_pixel, 0.56, 36.22)
    if new_pixel:
        image_test[x+int(rec_y1), y+int(rec_x1)] = new_pixel

# Show the result
cv2.imshow("Detected Different Pixels", image_test)
# print(image_array.shape, type(image_array))
# image_array[..., [0, 2]] = image_array[..., [2, 0]]
cv2.imwrite("tony2.jpg", image_test)

# Save the result
# cv2.imwrite("detected_pixels.png", output)
cv2.waitKey(0)
cv2.destroyAllWindows()





# point = (450, 350)
# mapped_point = map_to_rectangle(point, 1047, 577)
# # transformed_array[mapped_point] = [255, 0, 0, 255]

# trans_img = Image.fromarray(transformed_array)
# draw1 = ImageDraw.Draw(trans_img)

# # # Create a drawing object
# # draw = ImageDraw.Draw(image)

# # Define the point and the size of the dot
# dot_radius = 5  # Radius of the dot
# dot_color = "red"  # Color of the dot

# # Draw the dot (as a filled circle)
# draw1.ellipse(
#     [
#         (mapped_point[1] - dot_radius, mapped_point[0] - dot_radius),  # Top-left corner
#         (mapped_point[1] + dot_radius, mapped_point[0] + dot_radius),  # Bottom-right corner
#     ],
#     fill=dot_color,
# )
# trans_img.show()

# # # Create a drawing object
# draw2 = ImageDraw.Draw(image)

# # Define the point and the size of the dot
# dot_radius = 5  # Radius of the dot
# dot_color = "blue"  # Color of the dot

# # Draw the dot (as a filled circle)
# draw2.ellipse(
#     [
#         (point[1] - dot_radius, point[0] - dot_radius),  # Top-left corner
#         (point[1] + dot_radius, point[0] + dot_radius),  # Bottom-right corner
#     ],
#     fill=dot_color,
# )
# image.show()


# image1 = Image.fromarray(transformed_array)
# image1.save('test.png')
# image1.show()

# # image_array[point] = [255, 255, 255, 255]
# image = Image.fromarray(image_array)
# image.show()
# image.save('tt.png')