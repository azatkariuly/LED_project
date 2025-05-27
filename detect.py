import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRect, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap, QPainterPath, QTransform, QImage, QPolygonF, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QFileDialog, QGraphicsPixmapItem, QPushButton, QVBoxLayout, QWidget

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

    def load_image(self):
        # Open file dialog to select an image
        self.image_path = ['rgb/red.jpg', 'rgb/green.jpg', 'rgb/blue.jpg']
        # self.image_path = 'raw_image/3.jpg'  # Use a hardcoded file name for demonstration
        # self.image_path = 'tt.png'
        if self.image_path:
            pixmap = QPixmap(self.image_path[0])

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
        for i in range(len(self.image_path)):
            # source_image = label_pixmap.toImage()
            source_image = QImage(self.image_path[i])
            print('source iamge', source_image)

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

            cropped_image.save(f"{name}_{i}.png")
        return 

    def stretch_and_save_curved_image_2(self):
        if len(self.quadrilateral_points) != 4:
            print("Error: The quadrilateral is not complete.")
            return

        # Create the path based on the quadrilateral points
        path = self.create_quadrilateral_path(self.line_items)

        # Get the pixmap currently displayed in the QLabel
        label_pixmap = self.image_item.pixmap()  # Assuming `self.image_item` is your QLabel
        if label_pixmap.isNull():
            print("Error: No image loaded in the label.")
            return

        # Convert the QPixmap to QImage
        source_image = label_pixmap.toImage()

    def stretch_and_save_curved_image(self):
        if len(self.quadrilateral_points) != 4:
            print("Error: The quadrilateral is not complete.")
            return

        # Get the pixmap currently displayed in the QLabel
        label_pixmap = self.image_item.pixmap()  # Assuming `self.image_label` is your QLabel
        if label_pixmap is None:
            print("Error: No image loaded in the label.")
            return

        # Convert the QPixmap to QImage and then to a NumPy array for OpenCV processing
        source_image = label_pixmap.toImage()
        width = source_image.width()
        height = source_image.height()
        source_image = source_image.convertToFormat(QImage.Format_RGB32)

        ptr = source_image.bits()
        ptr.setsize(source_image.byteCount())
        img_array = np.array(ptr).reshape((height, width, 4))[:, :, :3]  # Drop the alpha channel

        # Define the source points (quadrilateral points)
        src_points = np.array([
            [self.quadrilateral_points[0].x(), self.quadrilateral_points[0].y()],
            [self.quadrilateral_points[1].x(), self.quadrilateral_points[1].y()],
            [self.quadrilateral_points[2].x(), self.quadrilateral_points[2].y()],
            [self.quadrilateral_points[3].x(), self.quadrilateral_points[3].y()]
        ], dtype=np.float32)

        # Define the destination points (rectangle)
        rect_width = int(max(
            np.linalg.norm(src_points[0] - src_points[1]),
            np.linalg.norm(src_points[2] - src_points[3])
        ))
        rect_height = int(max(
            np.linalg.norm(src_points[0] - src_points[3]),
            np.linalg.norm(src_points[1] - src_points[2])
        ))

        dst_points = np.array([
            [0, 0],
            [rect_width - 1, 0],
            [rect_width - 1, rect_height - 1],
            [0, rect_height - 1]
        ], dtype=np.float32)

        # Compute the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(img_array, transformation_matrix, (rect_width, rect_height))

        # Save the transformed image
        save_path = "stretched_image.png"
        if not cv2.imwrite(save_path, warped_image):
            print("Error: Failed to save the stretched image.")
        else:
            print(f"Stretched image saved to {save_path}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.view = QGraphicsView(self)
        self.scene = CurvedLineScene(self)

        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)

        self.setWindowTitle("Draw and Curve Line on Image")
        self.setGeometry(100, 100, 800, 600)

        # Layout to hold buttons
        self.layout = QVBoxLayout()

        # Buttons to start drawing and curving
        self.draw_button = QPushButton("Start Drawing", self)
        self.draw_button.clicked.connect(self.start_drawing)
        self.layout.addWidget(self.draw_button)

        self.curve_button = QPushButton("Enable Curving", self)
        self.curve_button.clicked.connect(self.enable_curving)
        self.layout.addWidget(self.curve_button)

        self.save_button = QPushButton("Save Line Position", self)
        self.save_button.clicked.connect(self.save_line_position)
        self.layout.addWidget(self.save_button)

        self.cut_button = QPushButton("Cut and Save Image", self)
        self.cut_button.clicked.connect(self.cut_and_save_image)
        self.layout.addWidget(self.cut_button)

        self.save_stretch_button = QPushButton("Stretch and Save Image", self)
        self.save_stretch_button.clicked.connect(self.scene.stretch_and_save_curved_image)
        self.layout.addWidget(self.save_stretch_button)

        # Widget to hold layout and set it in the window
        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setMenuWidget(self.container)

        # Load the image
        self.scene.load_image()

    def start_drawing(self):
        self.scene.start_drawing()

    def enable_curving(self):
        self.scene.enable_curving()

    def save_line_position(self):
        self.scene.save_line_position()

    def cut_and_save_image(self):
        self.scene.cut_and_save_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
