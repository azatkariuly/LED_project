import sys
import cv2
from PIL import Image
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRect, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap, QPainterPath, QTransform, QImage, QPolygonF, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QFileDialog, QGraphicsPixmapItem, QPushButton, QVBoxLayout, QWidget, QHBoxLayout

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.view = QGraphicsView(self)
        self.scene = CurvedLineScene(self)

        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.setWindowTitle("Draw and Curve Line on Image")
        self.setGeometry(100, 100, 1000, 600)  # Adjust the width to fit image and buttons

        # Layout to hold the image and buttons horizontally
        self.layout = QHBoxLayout()

        # Create a container for the image
        self.image_container = QWidget(self)
        self.image_container.setLayout(QVBoxLayout())  # Add vertical layout for the image
        self.image_container.layout().addWidget(self.view)

        # Create a container for the buttons
        self.button_container = QWidget(self)
        self.button_container.setLayout(QVBoxLayout())  # Add vertical layout for buttons

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
        
        self.detect_broken_pixels_btn = QPushButton("Detect broken pixels", self)
        self.detect_broken_pixels_btn.clicked.connect(self.detect_broken_pixels)
        self.detect_broken_pixels_btn.setVisible(False)
        self.button_container.layout().addWidget(self.detect_broken_pixels_btn)

        # Add image and button containers to the main layout
        self.layout.addWidget(self.image_container)
        self.layout.addWidget(self.button_container)

        # Set the main layout of the window
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def upload_file(self):
        # Open file dialog to select image
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.bmp *.jpeg)")
        
        if file_path:
            self.scene.load_image(file_path)
            self.upload_button.setVisible(False)
            self.cut_button.setVisible(True)
            self.draw_button.setVisible(True)
            self.curve_button.setVisible(True)
            self.save_button.setVisible(True)
    
    def start_drawing(self):
        self.scene.start_drawing()

    def enable_curving(self):
        self.scene.enable_curving()

    def save_line_position(self):
        self.scene.save_line_position()

    def cut_and_save_image(self):
        self.file_name = self.scene.cut_and_save_image()
        
        if self.file_name:
            self.scene.load_image(f"{self.file_name}.png")
            self.cut_button.setVisible(False)
            self.draw_button.setVisible(False)
            self.curve_button.setVisible(False)
            self.save_button.setVisible(False)
            self.detect_broken_pixels_btn.setVisible(True)
            
    def detect_edges(self, image_array, corner_points):
        image_height, image_width, _ = image_array.shape

        edge_lines = []
        
        upper_line = []
        pixel_position = int(corner_points[0][1])
        for i in range(int(corner_points[0][0]), int(corner_points[1][0])):
            for j in range(-1,2):
                if pixel_position+j < image_height:
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
                if pixel_position+j < image_width:
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
                if pixel_position+j < image_height:
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
                if pixel_position+j < image_width:
                    pixel = image_array[i, pixel_position+j]
                    if pixel[-1] != 0:
                        pixel_position = pixel_position+j
                        left_line += [[i, pixel_position]]
                        # image_array[i, pixel_position] = [0, 255, 0, 255]
                        break
                    
        edge_lines.append(left_line)
        
        return edge_lines
            
    def detect_broken_pixels(self):
        # image = Image.open(f"{self.file_name}.png")
        image = cv2.imread(f"{self.file_name}.png")
        split_coords = self.file_name.split('_')
        rec_x1, rec_y1, recx2, rec_y2 = split_coords[2:6]
        split_coords = split_coords[6:]
        
        corner_points = []

        for x, y in zip(split_coords[::2], split_coords[1::2]):
            corner_points.append((float(x), float(y)))

        edge_lines = self.detect_edges(image, corner_points)
        
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
            
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
