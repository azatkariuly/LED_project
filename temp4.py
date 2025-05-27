import numpy as np
from PIL import Image, ImageDraw
import cv2

def stretch_image(image, src_pts, width=32, height=16):
    """
    Stretches an image to fill a rectangle by applying a perspective transformation.
    
    :param image_path: Path to the input image
    :param width: Target width of the stretched image
    :param height: Target height of the stretched image
    :return: Transformed image
    """
    # # Load the image
    # image = cv2.imread(image_path)
    
    # Define the original curved quadrilateral points (manually identified)
    # src_pts = np.array([[0, 22], [1048, 100], [908, 575], [66, 519]], dtype=np.float32)
    
    # Define the target rectangle points (full stretch)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    
    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transformation
    stretched_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return stretched_image


# file_name = 'cropped_image_390_144_1438_719_0.0_22.0_1048.0_100.0_908.0_575.0_66.0_519.0'
file_name = '390_143_1438_719_0.0_24.0_1048.0_101.0_910.0_576.0_67.0_522.0'
image = Image.open(f"cropped/cropped_green_{file_name}.png")
draw = ImageDraw.Draw(image)
split_coords = file_name.split('_')
rec_x1, rec_y1, recx2, rec_y2 = split_coords[:4]
print('rec_x1, rec_y1, recx2, rec_y2', rec_x1, rec_y1, recx2, rec_y2)
split_coords = split_coords[4:]
print(split_coords)

# Iterate two items at a time

corner_points = []

for x, y in zip(split_coords[::2], split_coords[1::2]):
    print(f"Point: ({x}, {y})")

    corner_points.append((float(x), float(y)))
    
image_array = np.array(image)
image_height, image_width, _ = image_array.shape

upper_line = []
pixel_position = int(corner_points[0][1])
for i in range(int(corner_points[0][0]), int(corner_points[1][0])):
    for j in range(-1,2):
        if pixel_position+j < image_height:
            pixel = image_array[pixel_position+j, i]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                upper_line += [[pixel_position, i]]
                break

bottom_line = []
pixel_position = int(corner_points[3][1])
for i in range(int(corner_points[3][0]), int(corner_points[2][0])):
    for j in range(1,-2,-1):
        if pixel_position+j < image_height:
            pixel = image_array[pixel_position+j, i]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                bottom_line += [[pixel_position, i]]
                break

left_line = []
pixel_position = int(corner_points[0][0])
for i in range(int(corner_points[0][1]), int(corner_points[3][1])):
    for j in range(-1,2):
        if pixel_position+j < image_width:
            pixel = image_array[i, pixel_position+j]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                left_line += [[i, pixel_position]]
                break

right_line = []
pixel_position = int(corner_points[1][0])
for i in range(int(corner_points[1][1]), int(corner_points[2][1])):
    for j in range(1,-2,-1):
        if pixel_position+j < image_width:
            pixel = image_array[i, pixel_position+j]
            if pixel[-1] != 0:
                pixel_position = pixel_position+j
                right_line += [[i, pixel_position]]
                break
            
def draw_circles(circles, color="green"):
    dot_radius = 4  # Radius of the dot
    for point in circles:
        draw.ellipse(
            [
                (point[1] - dot_radius, point[0] - dot_radius),  # Top-left corner
                (point[1] + dot_radius, point[0] + dot_radius),  # Bottom-right corner
            ],
            fill=color,
        )

def divide_line(point1, point2, K):
    x1, y1 = point1
    x2, y2 = point2
    
    points = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in [i / K for i in range(K + 1)]]
    return points

def draw_lines(x_array, y_array, X):
    for i, j in zip(x_array, y_array):
        a1, b1 = i
        a2, b2 = j
        draw.line([(b1, a1), (b2, a2)], fill='green', width=2)
        
        for i in range(1, X):
            t = i / X  # Interpolation factor
            x = int(b1 + t * (b1 - b1))
            y = int(a1 + t * (a2 - a1))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")

def get_line_coordinates(p1, p2):
    """
    Returns an array containing all pixel coordinates forming a line between p1 and p2.
    
    :param p1: (x1, y1) - First point
    :param p2: (x2, y2) - Second point
    :return: A NumPy array of shape (N, 2) containing (x, y) coordinates
    """
    x1, y1 = p1
    x2, y2 = p2

    # Use Bresenham’s line algorithm
    coordinates = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        coordinates.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return np.array(coordinates)            

def generate_zoom_array(points, axis=0, start=0.75, center=1.0, end=0.65, max=True):
    y_values = np.array([p[axis] for p in points])  # Extract y values
    if max:
        max_index = np.argmax(y_values)
    else:
        max_index = np.argmin(y_values)  # Find index of max y
    max_index = len(points) // 2
    
    # Create an array with the same length as points
    result = np.zeros(len(points))
    result[max_index] = 1  # Set max y location to 1

    # Left side: Uniformly increasing from 0.75 to 1
    if max_index > 0:
        left_values = np.linspace(start, center, max_index+ 1)[:-1]  # Exclude 1 at max_index
        result[:max_index] = left_values
        

    # Right side: Uniformly decreasing from 1 to 0.65
    if max_index < len(points) - 1:
        right_values = np.linspace(center, end, len(points) - max_index)
        result[max_index:] = right_values

    return result.tolist()

def distance_curve(points, S, zoom=None):
    points = np.array(points)
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))  # Compute segment lengths
    # print('distance', distances)
    total_length = np.sum(distances)  # Total length of the curve
    segment_length = total_length / S
    # print('zoom', zoom)
    
    res = []
    curr_length = 0
    for i, point in enumerate(points[1:-1]):
        curr_length += distances[i]
        loc_segment_len = segment_length * zoom[i] if zoom else segment_length
        if curr_length >= loc_segment_len:
            res += [point]
            curr_length -= loc_segment_len
            if len(res) == S - 1:
                return res
        
    return res

# # Example Usage
# curve_points = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3), (8, 2), (9, 1)]  # Example curved line

K = 23
L = 26
zoom_upper = generate_zoom_array(upper_line, start=0.75, center=1.2, end=0.9, max=False)
circles_upper = distance_curve(upper_line, K, zoom=zoom_upper)

# draw_circles(circles_upper)

# zoom_bottom = generate_zoom_array(bottom_line, start=0.76, center=1.35, end=0.85, max=False)
zoom_bottom = generate_zoom_array(bottom_line, start=0.75, center=1.26, end=0.86, max=False)
circles_bottom = distance_curve(bottom_line, K, zoom=zoom_bottom)

# draw_circles(circles_bottom)

# zoom_left = generate_zoom_array(left_line, start=1, center=1.18, end=0.9, axis=1)
zoom_left = generate_zoom_array(left_line, start=1.12, center=1.01, end=0.94, axis=1)
circles_left = distance_curve(left_line, L, zoom=zoom_left)

# draw_circles(circles_left)

zoom_right = generate_zoom_array(right_line, start=1.12, center=1.02, end=0.92, axis=1)
circles_right = distance_curve(right_line, L, zoom=zoom_right)

# draw_circles(circles_right)

h_dots = []
v_dots = []


for i, j in zip(circles_upper, circles_bottom):
    # print(i, j, get_line_coordinates(i, j))
    h_line = get_line_coordinates(i, j)
    zoom_h = generate_zoom_array(h_line, start=1.25, center=1, end=0.85, axis=1)
    circles_h = distance_curve(h_line, L, zoom=zoom_h)
    # temp = divide_line(i, j, L)

    v_dots += [circles_h]
    # draw_circles(circles_h, color='blue')

# image.show()
for i, j in zip(circles_left, circles_right):
    temp = divide_line(i, j, K)
    
    h_dots += [temp[1:-1]]

    # draw_circles(temp[1:-1], color='white')

# h_dots = np.array(h_dots)
# v_dots = np.array(v_dots).transpose(1, 0, 2)


# for h_dot, v_dot in zip(h_dots, v_dots):
#     count = 0
#     for i, j in zip(h_dot, v_dot):
#         count += 1
#         draw_circles([i], color='blue')
#         draw_circles([j], color='white')
#     break

# draw_lines(circles_upper, circles_bottom, X=L)
# draw_lines(circles_left, circles_right, X=K)
rectangle_points = []

# first col
res = []
res += [upper_line[0]]
res += circles_left

res += [bottom_line[0]]

rectangle_points += [res]

# middle cols
for i in range(len(v_dots)):
    res = []
    res += [circles_upper[i]]
    res += v_dots[i]
    # print('sssss', h_dots)
    res += [circles_bottom[i]]

    rectangle_points += [res]

# last col
res = []
res += [upper_line[-1]]
res += circles_right

res += [bottom_line[-1]]

rectangle_points += [res]

rectangle_points = np.array(rectangle_points)

# print('final', rectangle_points.shape)

# for col in rectangle_points:
#     draw_circles(col, color='white')

# image.show()


h, w, _ = rectangle_points.shape
module_loc = []
for i in range(h - 1):
    res = []
    for j in range(w - 1):
        res += [[rectangle_points[i][j][::-1], rectangle_points[i+1][j][::-1], rectangle_points[i+1][j+1][::-1], rectangle_points[i][j+1][::-1]]]
    module_loc += [res]

module_loc = np.array(module_loc)

w_module = 32
h_module = 16
w, h, _, _ = module_loc.shape
out_image = np.zeros((h_module * h, w_module * w, 4), dtype=np.uint8)


for i in range(h):
    for j in range(w):
        stretched_module = stretch_image(image_array, np.array(module_loc[j][i], dtype=np.float32))
        out_image[i*h_module:i*h_module+h_module, j*w_module:j*w_module+w_module] = stretched_module
        # # print('ss', stretched_module)
        # cv2.imwrite(f"modules/{i}-{j}.png", stretched_module)
out_image[..., [0, 2]] = out_image[..., [2, 0]]
print('out', out_image.shape)
cv2.imwrite(f"modules/0-0-0.png", out_image)