import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QGraphicsView, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap, QPainterPath, QTransform, QImage

video_width = 64
video_height = 32

module_width = 32
module_height = 16


class CurvedLineItem(QGraphicsItem):
    def __init__(self, start, end):
        super().__init__()
        self.start_point = start
        self.end_point = end
        self.control_point = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.is_selected = False
        self.is_curved = False
        self.is_saved = False  # Track if the line has been saved

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        path = QPainterPath()
        path.moveTo(self.start_point)
        if self.is_curved:
            path.quadTo(self.control_point, self.end_point)
        else:
            path.lineTo(self.end_point)
        return path

    def paint(self, painter, option, widget=None):
        pen = QPen(Qt.black, 2)
        if self.is_selected:
            pen.setColor(Qt.red)  # Highlight the selected line in red
        if self.is_saved:  # Draw the line in green after saving
            pen.setColor(Qt.green)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.transparent))
        painter.drawPath(self.shape())

    def setControlPoint(self, point):
        if not self.is_saved:  # Prevent changes after saving
            self.control_point = point
            self.update()

    def setSelected(self, selected):
        self.is_selected = selected
        self.update()

    def contains(self, pos):
        # Check if the mouse position is close to the line (bounding box around the line)
        return self.boundingRect().contains(pos)

    def save(self):
        self.is_saved = True  # Mark the line as saved
        self.setFlag(QGraphicsItem.ItemIsMovable, False)  # Prevent moving the line

class CurvedLineScene(QGraphicsScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_item = None
        self.points = []
        self.line_items = []
        self.selected_line = None
        self.is_drawing = False
        self.saved_point = None  # Track the last saved point for drawing new lines
        self.quadrilateral_points = []  # To store the points of the quadrilateral
        
    def reset(self):
        self.clear()
        self.image_item = None
        self.points = []
        self.line_items = []
        self.selected_line = None
        self.is_drawing = False
        self.saved_point = None  # Track the last saved point for drawing new lines
        self.quadrilateral_points = []  # To store the points of the quadrilateral
        
    def mousePressEvent(self, event):
        pos = event.scenePos()

        # If the user is drawing the line (selecting points)
        if self.is_drawing and len(self.points) < 4:
            self.points.append(pos)
            if len(self.points) > 1:
                # Draw line from the last saved point if applicable
                if self.saved_point:
                    self.draw_line(self.saved_point, self.points[-1])
                else:
                    self.draw_line(self.points[-2], self.points[-1])

            # After the 4th point, close the quadrilateral by drawing the last line
            if len(self.points) == 4:
                self.draw_line(self.points[-1], self.points[0])
                self.quadrilateral_points = self.points  # Save the points for the quadrilateral

        # Check if the user clicked on an existing line
        clicked_item = self.itemAt(pos, QTransform())  # Fixed the call to itemAt with QTransform()
        if clicked_item and isinstance(clicked_item, CurvedLineItem):
            if self.selected_line:
                self.selected_line.setSelected(False)  # Unselect the previous line
            clicked_item.setSelected(True)
            self.selected_line = clicked_item

    def mouseMoveEvent(self, event):
        if self.selected_line and event.buttons() == Qt.LeftButton:
            # Update the control point of the selected line when dragging
            if not self.selected_line.is_saved:  # Allow dragging only if not saved
                self.selected_line.setControlPoint(event.scenePos())

    def draw_line(self, start, end):
        line_item = CurvedLineItem(start, end)
        self.addItem(line_item)
        self.line_items.append(line_item)
        self.saved_point = end  # Update the last saved point to the end of the current line

    def load_image(self, file_name, array=False):
        self.clear()
        if array:
            pixmap = QPixmap.fromImage(file_name)
        else:
            pixmap = QPixmap(file_name)

        # # Resize the pixmap (image) to fit the scene or to a desired size
        # new_width = 800  # Desired width of the image
        # new_height = 600  # Desired height of the image

        # pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create QGraphicsPixmapItem to hold the image in the scene
        self.image_item = QGraphicsPixmapItem(pixmap)

        # Add the image to the scene
        self.addItem(self.image_item)

    def start_drawing(self):
        self.is_drawing = True
        self.points = []  # Reset points

    def enable_curving(self):
        if self.selected_line:
            self.selected_line.is_curved = True
            self.selected_line.update()

    def save_line_position(self):
        if self.selected_line:
            self.selected_line.save()  # Save the selected line's position

            print('self.selected_line', self.line_items[0].start_point, self.line_items[0].end_point)

    def create_quadrilateral_path(self, line_items):
        path = QPainterPath()
        path.moveTo(line_items[0].start_point)

        for line in line_items:
            if line.is_curved:
                path.quadTo(line.control_point, line.end_point)
            else:
                path.lineTo(line.end_point)

        path.closeSubpath()  # Close the path to form a closed quadrilateral shape
        
        return path

    def cut_and_save_image(self):
        if len(self.quadrilateral_points) != 4:
            print("Error: The quadrilateral is not complete.")
            return

        path = self.create_quadrilateral_path(self.line_items)
        print('path', path)

        # Get the pixmap currently displayed in the QLabel
        label_pixmap = self.image_item.pixmap()  # Assuming `self.image_label` is your QLabel
        if label_pixmap is None:
            print("Error: No image loaded in the label.")
            return

        # Convert the QPixmap to QImage
        source_image = label_pixmap.toImage()

        # Create a mask of the same size as the image
        mask = QImage(source_image.size(), QImage.Format_ARGB32)
        mask.fill(Qt.transparent)  # Initially set the mask as transparent

        # Create a painter to paint the mask
        painter = QPainter(mask)
        painter.setBrush(QBrush(Qt.white))  # Set the brush to fill the shape
        painter.setPen(QPen(Qt.white))  # Set the pen to draw the outline of the shape
        painter.drawPath(path)  # Draw the path on the mask
        painter.end()

        # Now, apply the mask to the image
        result_image = source_image.copy()  # Copy the image to modify it
        result_image.setAlphaChannel(mask)  # Set the alpha channel to the mask

        bounding_rect = path.boundingRect().toRect()
        print(f"Bounding Rectangle Shape: Width={bounding_rect.width()}, Height={bounding_rect.height()}")
        print(f"Bounding Rectangle Coordinates: Top-Left=({bounding_rect.x()}, {bounding_rect.y()})")

        print("Line Endpoints:")
        for line in self.line_items:
            print(f"Start: ({line.start_point.x()}, {line.start_point.y()}), End: ({line.end_point.x()}, {line.end_point.y()})")

        cropped_image = result_image.copy(bounding_rect)

        name = f"cropped_image_{bounding_rect.x()}_{bounding_rect.y()}_{bounding_rect.x()+bounding_rect.width()}_{bounding_rect.y() + bounding_rect.height()}"
        for line in self.line_items:
            first_point = line.start_point.x() - bounding_rect.x(), line.start_point.y() - bounding_rect.y()
            # second_point = line.end_point.x() - bounding_rect.x(), line.end_point.y() - bounding_rect.y()
            name += f'_{first_point[0]}_{first_point[1]}'

        cropped_image.save(f"{name}.png")
        return name

class Tab3(QWidget):
    def __init__(self, video_width=video_width, video_height=video_height, module_width=module_width, module_height=module_height):
        super().__init__()
        self.video_width = video_width
        self.video_height = video_height
        self.module_width = module_width
        self.module_height = module_height

        self.view = QGraphicsView(self)
        self.scene = CurvedLineScene(self)

        self.view.setScene(self.scene)

        self.setWindowTitle("Draw and Curve Line on Image")
        self.setGeometry(100, 100, 1000, 600)

        # Layout to hold the image and buttons horizontally
        self.layout = QHBoxLayout(self)

        # Create a container for the image
        self.image_container = QWidget(self)
        self.image_container.setLayout(QVBoxLayout())
        self.image_container.layout().addWidget(self.view)

        # Create a container for the buttons
        self.button_container = QWidget(self)
        self.button_container.setLayout(QVBoxLayout())

        # Buttons to start drawing and curving
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_file)
        self.button_container.layout().addWidget(self.upload_button)
        
        self.draw_button = QPushButton("Start Drawing", self)
        self.draw_button.clicked.connect(self.start_drawing)
        self.draw_button.setVisible(False)
        self.button_container.layout().addWidget(self.draw_button)

        self.curve_button = QPushButton("Enable Curving", self)
        self.curve_button.clicked.connect(self.enable_curving)
        self.curve_button.setVisible(False)
        self.button_container.layout().addWidget(self.curve_button)

        self.save_button = QPushButton("Save Line Position", self)
        self.save_button.clicked.connect(self.save_line_position)
        self.save_button.setVisible(False)
        self.button_container.layout().addWidget(self.save_button)

        self.cut_button = QPushButton("Cut and Save Image", self)
        self.cut_button.clicked.connect(self.cut_and_save_image)
        self.cut_button.setVisible(False)
        self.button_container.layout().addWidget(self.cut_button)
        
        self.detect_edges_btn = QPushButton("Detect Edges", self)
        self.detect_edges_btn.clicked.connect(self.detect_edges_dots)
        self.detect_edges_btn.setVisible(False)
        self.button_container.layout().addWidget(self.detect_edges_btn)

        self.transform_button = QPushButton("Transform to rectangle", self)
        self.transform_button.clicked.connect(self.transform_to_rectangle)
        self.transform_button.setVisible(False)
        self.button_container.layout().addWidget(self.transform_button)
        
        self.detect_broken_pixels_btn = QPushButton("Detect broken modules", self)
        self.detect_broken_pixels_btn.clicked.connect(self.detect_broken_pixels)
        self.detect_broken_pixels_btn.setVisible(False)
        self.button_container.layout().addWidget(self.detect_broken_pixels_btn)
        
        
        self.reset_button = QPushButton("Reset state", self)
        self.reset_button.clicked.connect(self.reset_state)
        self.reset_button.setVisible(False)
        self.button_container.layout().addWidget(self.reset_button)

        # Add image and button containers to the main layout
        self.layout.addWidget(self.image_container)
        self.layout.addWidget(self.button_container)

    def upload_file(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
        
        if file_path:
            self.scene.load_image(file_path)
            self.upload_button.setVisible(False)
            self.cut_button.setVisible(True)
            self.draw_button.setVisible(True)
            self.curve_button.setVisible(True)
            self.save_button.setVisible(True)
            self.reset_button.setVisible(True)
    
    def start_drawing(self):
        self.scene.start_drawing()

    def enable_curving(self):
        self.scene.enable_curving()

    def save_line_position(self):
        self.scene.save_line_position()
        
    def reset_state(self):
        self.scene.clear()
        self.upload_button.setVisible(True)
        self.cut_button.setVisible(False)
        self.draw_button.setVisible(False)
        self.curve_button.setVisible(False)
        self.save_button.setVisible(False)
        self.cut_button.setVisible(False)
        self.detect_edges_btn.setVisible(False)
        self.transform_button.setVisible(False)
        self.detect_broken_pixels_btn.setVisible(False)
        self.reset_button.setVisible(False)

    def cut_and_save_image(self):
        self.file_name = self.scene.cut_and_save_image()
        
        if self.file_name:
            self.scene.load_image(f"{self.file_name}.png")
            self.cut_button.setVisible(False)
            self.draw_button.setVisible(False)
            self.curve_button.setVisible(False)
            self.save_button.setVisible(False)
            self.detect_edges_btn.setVisible(True)

    def detect_edges(self, image_array, corner_points):
        image_height, image_width, _ = image_array.shape

        edge_lines = []
        
        upper_line = []
        pixel_position = int(corner_points[0][1])
        for i in range(int(corner_points[0][0]), int(corner_points[1][0])):
            for j in range(-1,2):
                if -1 < pixel_position+j < image_height:
                    pixel = image_array[pixel_position+j, i]
                    if pixel[-1] != 0:
                        pixel_position = pixel_position+j
                        upper_line += [[pixel_position, i]]
                        # image_array[pixel_position, i] = [0, 0, 255, 255]
                        break
        
        edge_lines.append(upper_line)

        right_line = []
        pixel_position = int(corner_points[1][0])
        for i in range(int(corner_points[1][1]), int(corner_points[2][1])):
            for j in range(1,-2,-1):
                if -1 < pixel_position+j < image_width:
                    pixel = image_array[i, pixel_position+j]
                    if pixel[-1] != 0:
                        pixel_position = pixel_position+j
                        right_line += [[i, pixel_position]]
                        # image_array[i, pixel_position] = [0, 255, 0, 255]
                        break
                    
        edge_lines.append(right_line)
        
        bottom_line = []
        pixel_position = int(corner_points[3][1])
        for i in range(int(corner_points[3][0]), int(corner_points[2][0])):
            for j in range(1,-2,-1):
                if -1 < pixel_position+j < image_height:
                    pixel = image_array[pixel_position+j, i]
                    if pixel[-1] != 0:
                        pixel_position = pixel_position+j
                        bottom_line += [[pixel_position, i]]
                        # image_array[pixel_position, i] = [0, 0, 255, 255]
                        break

        edge_lines.append(bottom_line)
        
        left_line = []
        pixel_position = int(corner_points[0][0])
        for i in range(int(corner_points[0][1]), int(corner_points[3][1])):
            for j in range(-1,2):
                if -1 < pixel_position+j < image_width:
                    pixel = image_array[i, pixel_position+j]
                    if pixel[-1] != 0:
                        pixel_position = pixel_position+j
                        left_line += [[i, pixel_position]]
                        # image_array[i, pixel_position] = [0, 255, 0, 255]
                        break
                    
        edge_lines.append(left_line)
        
        return edge_lines
    
    def stretch_image(self, image, src_pts, width=32, height=16):
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply the transformation
        stretched_image = cv2.warpPerspective(image, matrix, (width, height))
        
        return stretched_image
    
    def generate_zoom_array(self, points, axis=0, start=0.75, center=1.0, end=0.65, max=True):
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

    def distance_curve(self, points, S, zoom=None):
        points = np.array(points)
        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))  # Compute segment lengths
        # print('distance', distances)
        total_length = np.sum(distances)  # Total length of the curve
        segment_length = total_length / S
        print("SE", total_length, S, segment_length)
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

    def get_line_coordinates(self, p1, p2):
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

    def divide_line(self, point1, point2, K):
        x1, y1 = point1
        x2, y2 = point2
        
        points = [(int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))) for t in [i / K for i in range(K + 1)]]
        return points

    def detect_edges_dots(self):
        self.image = cv2.imread(f"{self.file_name}.png")
        
        self.image[..., [0, 2]] = self.image[..., [2, 0]]
        split_coords = self.file_name.split('_')
        rec_x1, rec_y1, recx2, rec_y2 = split_coords[2:6]
        split_coords = split_coords[6:]
        
        corner_points = []

        for x, y in zip(split_coords[::2], split_coords[1::2]):
            corner_points.append((float(x), float(y)))

        edge_lines = self.detect_edges(self.image, corner_points)
        
        # for i in edge_lines[3]:
        #     x, y = i
        #     cv2.circle(self.image, (y, x), 2, (0, 255, 0), 4)
        
        K = self.video_width // self.module_height - 2
        L = self.video_height // self.module_height - 1

        zoom_upper = self.generate_zoom_array(edge_lines[0], start=1, center=1, end=1, max=False)
        circles_upper = self.distance_curve(edge_lines[0], K, zoom=zoom_upper)

        zoom_bottom = self.generate_zoom_array(edge_lines[2], start=1, center=1, end=1, max=False)
        circles_bottom = self.distance_curve(edge_lines[2], K, zoom=zoom_bottom)
        
        zoom_left = self.generate_zoom_array(edge_lines[3], start=1, center=1, end=1, axis=1)
        circles_left = self.distance_curve(edge_lines[3], L, zoom=zoom_left)
        
        zoom_right = self.generate_zoom_array(edge_lines[1], start=1, center=1, end=1, axis=1)
        circles_right = self.distance_curve(edge_lines[1], L, zoom=zoom_right)
        
        h_dots = []
        v_dots = []

        for i, j in zip(circles_upper, circles_bottom):
            h_line = self.get_line_coordinates(i, j)
            zoom_h = self.generate_zoom_array(h_line, start=1.25, center=1, end=0.85, axis=1)
            circles_h = self.distance_curve(h_line, L, zoom=zoom_h)

            v_dots += [circles_h]

        for i, j in zip(circles_left, circles_right):
            temp = self.divide_line(i, j, K)
            
            h_dots += [temp[1:-1]]

        rectangle_points = []
        
        # first col
        res = []
        res += [edge_lines[0][0]]
        res += circles_left

        res += [edge_lines[2][0]]

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
        res += [edge_lines[0][-1]]
        res += circles_right

        res += [edge_lines[1][-1]]
        
        print("RES", res)

        rectangle_points += [res]

        self.rectangle_points = np.array(rectangle_points)
        
        image_copy = self.image.copy()
        
        for col in rectangle_points:
            for point in col:
                x, y = point
                cv2.circle(image_copy, (y, x), 2, (255, 255, 255), 4)
                
        height, width, channels = image_copy.shape
        q_image = QImage(image_copy.data, width, height, width * channels, QImage.Format_RGB888)

        self.scene.load_image(q_image, array=True)
        
        self.detect_edges_btn.setVisible(False)
        self.transform_button.setVisible(True)

    def transform_to_rectangle(self):
        h, w, _ = self.rectangle_points.shape
        module_loc = []
        for i in range(h - 1):
            res = []
            for j in range(w - 1):
                res += [[self.rectangle_points[i][j][::-1], self.rectangle_points[i+1][j][::-1], self.rectangle_points[i+1][j+1][::-1], self.rectangle_points[i][j+1][::-1]]]
            module_loc += [res]
            
        module_loc = np.array(module_loc)

        w_module = 32
        h_module = 16
        w, h, _, _ = module_loc.shape
        self.out_image = np.zeros((h_module * h, w_module * w, 3), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                stretched_module = self.stretch_image(self.image, np.array(module_loc[j][i], dtype=np.float32))
                self.out_image[i*h_module:i*h_module+h_module, j*w_module:j*w_module+w_module] = stretched_module

        # self.out_image[..., [0, 2]] = self.out_image[..., [2, 0]]
        cv2.imwrite(f"suka.png", self.out_image)
        height, width, channels = self.out_image.shape
        q_image = QImage(self.out_image.data, width, height, width * channels, QImage.Format_RGB888)

        self.scene.load_image(q_image, array=True)
        
        self.transform_button.setVisible(False)
        self.detect_broken_pixels_btn.setVisible(True)

    def select_green_rectangles(self, image):
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

    def find_adjacent_non_blue(self, blue_rectangles, width, height):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        res = []
        for x, y, rect_width, rect_height in blue_rectangles:
            adjacent_non_blue = []
            for dx, dy in directions:
                adj_x, adj_y = x + dx * rect_width, y + dy * rect_height
                if 0 <= adj_x < width and 0 <= adj_y < height and (adj_x, adj_y, rect_width, rect_height) not in blue_rectangles:
                    adjacent_non_blue.append((adj_x, adj_y, rect_width, rect_height))
                    # s_image[adj_y:adj_y+rect_height, adj_x:adj_x+rect_width] = [255, 255, 255]
            
            res += [[(x, y, rect_width,rect_height), adjacent_non_blue]]
        return res

    def calculate_brightness_contrast(self, incorrect_pixel, desired_pixel):
        incorrect_pixel = np.array(incorrect_pixel, dtype=np.float32)
        desired_pixel = np.array(desired_pixel, dtype=np.float32)
        
        # print('incorrect_pixel', incorrect_pixel)
        # print('desired_pixel', desired_pixel)
        
        # Solve for alpha and beta
        A = np.vstack([incorrect_pixel, np.ones_like(incorrect_pixel)]).T
        x, _, _, _ = np.linalg.lstsq(A, desired_pixel, rcond=None)
        
        alpha, beta = x
        return alpha, beta
    
    def restore_broken_modules(self, image, res_image, broken_module, white_modules, alpha_offset=0.2):
        # Convert to grayscale if image1 is in color
        x, y, w, h = broken_module
        image1 = image[y:y+h, x:x+w]
        
        broken_rgb = np.mean(image1, axis=(0, 1))
        
        print('broken_rgb', broken_rgb)

        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        
        avg_rgb = []
        for i, white_module in enumerate(white_modules):
            x_w, y_w, w_w, h_w = white_module
            ref_img = image[y_w:y_w+h_w, x_w:x_w+w_w]
            
            avg_rgb.append(ref_img)
            
        avg_rgb = np.array(avg_rgb)

        alpha, beta = None, None

        if len(avg_rgb) > 0:
            avg_rgb = np.mean(np.mean(avg_rgb, axis=0), axis=(0, 1))
            
            alpha, beta = self.calculate_brightness_contrast(broken_rgb, avg_rgb)
        
        return alpha, beta

    def detect_broken_pixels(self):
        image = self.out_image.copy()
        # image[..., [0, 2]] = image[..., [2, 0]]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        for contour in contours:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            if 12 < h < 20:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), -1)
                
        height, width, channels = image.shape
        q_image = QImage(image.data, width, height, width * channels, QImage.Format_RGB888)

        self.scene.load_image(q_image, array=True)
    
        module_width, w_num = 32, 23
        module_height, h_num = 16, 26
        
        width_led = module_width * w_num
        height_led = module_height * h_num
        
        m_image, selected = self.select_green_rectangles(image)
        blues_with_whites = self.find_adjacent_non_blue(selected, width_led, height_led)
        
        red_image = np.zeros((416, 736, 3), dtype=np.uint8)

        # Set the red channel to 255
        red_image[:, :, 2] = 255
        
        with open('output.txt', "w") as file:        
            for broken_modules in blues_with_whites:
                broken_module, white_modules = broken_modules[0], broken_modules[1:][0]
                x, y, w, h = broken_module
                
                alpha, beta = self.restore_broken_modules(self.out_image, red_image, broken_module, white_modules)
                
                if alpha:
                    file.write(f"({x}, {y}, {w}, {h}) {alpha} {beta}\n")
        
        # cv2.imwrite('test2.jpg', red_image)
        
    def detect_broken_pixels_new(self):
        image = self.out_image
        
        image[..., [0, 2]] = image[..., [2, 0]]
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
        
        locations = np.column_stack(np.where(final_mask1 != 0 & np.any(image == [0, 0, 0], axis=-1)))
        for location in locations:
            x, y = location
            
            image[x, y] = [0, 255, 0]
            
        locations = np.column_stack(np.where(final_mask2 == 0 | np.any(image == [0, 0, 0], axis=-1)))
        for location in locations:
            x, y = location
            image[x, y] = [0, 255, 0]
            
        height, width, channels = image.shape
        q_image = QImage(image.data, width, height, width * channels, QImage.Format_RGB888)

        self.scene.load_image(q_image, array=True)
        
    
        module_width, w_num = 32, 23
        module_height, h_num = 16, 26
        
        width_led = module_width * w_num
        height_led = module_height * h_num
        
        m_image, selected = self.select_green_rectangles(image)
        blues_with_whites = self.find_adjacent_non_blue(selected, width_led, height_led)
        
        red_image = np.zeros((416, 736, 3), dtype=np.uint8)

        # Set the red channel to 255
        red_image[:, :, 2] = 255
        
        with open('output.txt', "w") as file:        
            for broken_modules in blues_with_whites:
                broken_module, white_modules = broken_modules[0], broken_modules[1:][0]
                x, y, w, h = broken_module
                
                alpha, beta = self.restore_broken_modules(self.out_image, red_image, broken_module, white_modules)
                
                if alpha:
                    file.write(f"({x}, {y}, {w}, {h}) {alpha} {beta}\n")
        
        cv2.imwrite('test2.jpg', red_image)