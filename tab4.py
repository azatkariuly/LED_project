import cv2
import os
import re
import math
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QTableWidget,
    QProgressBar,
    QSpacerItem,
    QSizePolicy,
    QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor

video_width = 640
video_height = 320


class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)  # Ensures image is centered if QLabel is bigger

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Get the position of the click relative to the QLabel
            click_x, click_y = event.pos().x(), event.pos().y()
            print(f"Clicked at QLabel coords: ({click_x}, {click_y})")

            pixmap = self.pixmap()
            if pixmap:
                img_width, img_height = pixmap.width(), pixmap.height()
                label_width, label_height = self.width(), self.height()

                # Calculate margins if the image is smaller than QLabel
                margin_x = (label_width - img_width) // 2 if img_width < label_width else 0
                margin_y = (label_height - img_height) // 2 if img_height < label_height else 0

                # Convert QLabel coordinates to image coordinates
                img_x = click_x - margin_x
                img_y = click_y - margin_y

                # Ensure coordinates are within the image bounds
                if 0 <= img_x < img_width and 0 <= img_y < img_height:
                    print(f"Clicked at Image coords: ({img_x}, {img_y})")
                else:
                    print("Click was outside the image!")

class Tab4(QWidget):
    def __init__(self):
        super().__init__()

        self.multiple_videos_paths = None
        self.multiple_images_paths = None
        
        # Video display labels with reduced size
        self.video_left_label = ClickableLabel(self)
        self.video_right_label = QLabel(self)
        self.video_left_label.setFixedSize(500, 400)
        self.video_right_label.setFixedSize(500, 400)

        # Load video button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.clicked.connect(self.load_image)

        # Square size selector
        self.size_selector = QComboBox(self)
        self.size_selector.addItems(["32x16"])
        self.size_selector.currentIndexChanged.connect(self.update_square_size)
        # self.size_selector.setVisible(False)

        # Brightness adjustment buttons
        self.brightness_input_field = QLineEdit(self)
        self.brightness_input_field.setPlaceholderText('0')
        self.brightness_input_field.textChanged.connect(self.validate_brightness_input)

        self.contrast_input_field = QLineEdit(self)
        self.contrast_input_field.setPlaceholderText('1')
        self.contrast_input_field.textChanged.connect(self.validate_contrast_input)

        # Play and Pause buttons
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setVisible(False)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setVisible(False)

        self.multiple_videos = QPushButton("Apply to Multiple Videos", self)
        self.multiple_videos.clicked.connect(self.apply_to_multiple_videos)
        self.multiple_images = QPushButton("Apply to Multiple Images", self)
        self.multiple_images.clicked.connect(self.apply_to_multiple_images)
        
        self.load_data_txt_button = QPushButton("Load From TXT", self)
        self.load_data_txt_button.clicked.connect(self.load_data_txt)

        # Table to display coordinates
        self.coordinates_table = QTableWidget(self)
        self.coordinates_table.cellClicked.connect(self.toggle_cell_color)

        # Save directory selection and display
        self.save_dir = os.path.join(os.path.expanduser("~"), "Desktop")  # Default to Desktop
        self.save_dir_button = QPushButton("Select Save Directory", self)
        self.save_dir_button.clicked.connect(self.select_save_dir)
        self.save_dir_label = QLabel(self.save_dir, self)


        # Placeholder button for future functionality
        self.placeholder_button = QPushButton("Save", self)
        self.placeholder_button.clicked.connect(self.save_video)
        
        # Progress bar for operations
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setEnabled(False)  # Make it non-clickable by disabling it
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.load_image_button)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_left_label)
        video_layout.addWidget(self.video_right_label)

        # Right side layout for buttons and combo box
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.size_selector)  # Move size selector to button layout
        brightness_label = QLabel('Brightness (-255, 255)', self)
        button_layout.addWidget(brightness_label)
        button_layout.addWidget(self.brightness_input_field)
        contrast_label = QLabel('Contrast (0, 3)', self)
        button_layout.addWidget(contrast_label)
        button_layout.addWidget(self.contrast_input_field)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.multiple_videos)
        button_layout.addWidget(self.multiple_images)
        button_layout.addWidget(self.load_data_txt_button)
        
        # Spacer to align video with buttons
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        button_layout.addItem(spacer)

        video_layout.addLayout(button_layout)
        layout.addLayout(video_layout)
        
        # Add the coordinates table to the layout
        layout.addWidget(self.coordinates_table)

        # Save directory layout
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_button)
        save_dir_layout.addWidget(self.save_dir_label)
        save_dir_layout.addStretch()
        save_dir_layout.addWidget(self.placeholder_button)
        
        layout.addLayout(save_dir_layout)

        # Progress bar layout
        progress_layout = QVBoxLayout()
        self.progress_label = QLabel(None, self)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)

        self.setLayout(layout)

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)

        # OpenCV Video Capture and variables
        self.video_path = None
        self.cap = None
        self.frame = None
        self.video_width = None
        self.video_height = None
        self.fps = None
        # self.selected_rect = set()
        self.selected_rect = {}
        self.square_size_width = 320
        self.square_size_height = 160
        self.brightness_factor = 0
        self.contrast_factor = 1  # Variable to hold the brightness factor
        self.is_playing = True  # Track play/pause state

    def resize_frame(self, frame, max_width=500, max_height=400):
        # Get original dimensions
        original_height, original_width = frame.shape[:2]
        
        # Calculate the scaling factors for width and height
        width_scale = max_width / original_width
        height_scale = max_height / original_height
        
        # Use the smaller scale to maintain aspect ratio
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_frame

    def validate_brightness_input(self):
        text = self.brightness_input_field.text()
        
        # Allow negative numbers and digits
        if text.startswith('-'):
            if len(text) > 1:
                if text[1:].isdigit():  # Check if the rest of the string after '-' is a number
                    value = int(text)
                    if -255 <= value <= 255:
                        self.brightness_factor = value
                    else:
                        self.brightness_input_field.setText(text[:-1])  # Remove last character
                        self.show_error("Please enter a number between -255 and 255.")
                else:
                    self.brightness_input_field.setText(text[:-1])  # Remove last character
                    self.show_error("Only numeric values are allowed.")
        elif text.isdigit():
            value = int(text)
            if -255 <= value <= 255:
                self.brightness_factor = value
            else:
                self.brightness_input_field.setText(text[:-1])  # Remove last character
                self.show_error("Please enter a number between -255 and 255.")
        elif text:
            self.brightness_input_field.setText(text[:-1])  # Remove last character
            self.show_error("Only numeric values are allowed.")

        self.update_frame(self.frame)

    def is_float_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def validate_contrast_input(self):
        text = self.contrast_input_field.text()
        
        # Allow negative numbers and digits
        if self.is_float_number(text):
            value = float(text)
            if 0 <= value <= 3:
                self.contrast_factor = value
            else:
                self.contrast_input_field.setText(text[:-1])  # Remove last character
                self.show_error("Please enter a number between 0 and 3.")
        elif text:
            self.contrast_input_field.setText(text[:-1])  # Remove last character
            self.show_error("Only numeric values are allowed.")

        self.update_frame(self.frame)

    def show_error(self, message):
        QMessageBox.warning(self, "Invalid Input", message)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        self.image_path = None
        
        if self.video_path:
            self.pause_button.setVisible(True)
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.timer.start(self.fps)
            # self.size_selector.setVisible(True)
            self.selected_rect.clear()
            self.populate_coordinates_table()

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg)")

        if self.image_path:
            self.play_button.setVisible(False)
            self.pause_button.setVisible(False)

            self.video_path = None
            if self.cap:
                self.cap.release()

            # self.frame = cv2.imread(self.image_path)
            self.frame = np.array(Image.open(self.image_path))
            self.frame = self.frame[..., [2, 1, 0]]

            self.update_frame(self.frame)
            self.populate_coordinates_table()

            # frame_left, frame_right = self.modify_left_right_videos(self.frame)
            # frame_left = self.resize_frame(frame_left)
            # frame_right = self.resize_frame(frame_right)
            # # frame_left = cv2.resize(frame_left, (400, 400))
            # # frame_right = cv2.resize(frame_right, (400, 400))

            # left_display = self.convert_frame_to_qimage(frame_left)
            # self.video_left_label.setPixmap(QPixmap.fromImage(left_display))

            # right_display = self.convert_frame_to_qimage(frame_right)
            # self.video_right_label.setPixmap(QPixmap.fromImage(right_display))

    def load_data_txt(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open TXT File", "", "Text Files (*.txt);;All Files (*)", options=options)
        
        if filePath:
            self.selected_rect.clear()
            
            with open(filePath, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.match(r"\((\d+), (\d+), (\d+), (\d+)\) ([\d.]+) (-?[\d.]+)", line)
                    if match:
                        x, y, w, h = map(int, match.groups()[:4])  # Convert to integers
                        a = float(match.group(5))  # Convert to float
                        b = float(match.group(6))  # Convert to float
                        
                        self.selected_rect[(x, y)] = [a, b]
                        
                if self.frame is not None and self.selected_rect:
                    self.update_frame(self.frame)
                             
    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", self.save_dir)
        if dir_path:
            self.save_dir = dir_path
            self.save_dir_label.setText(self.save_dir)

    def apply_to_multiple_videos(self):
        self.multiple_videos_paths, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        self.multiple_images_paths = None

    def apply_to_multiple_images(self):
        self.multiple_images_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Image Files (*.png *.jpg *.jpeg)")
        self.multiple_videos_paths = None

    def save_video(self):
        global video_width, video_height

        if self.multiple_images_paths:
            counter = -1
            self.progress_bar.setValue(0)
            self.progress_bar.setEnabled(True)
            for i, v in enumerate(self.multiple_images_paths):
                counter += 1
                self.progress_label.setText(f'Done {i}/{len(self.multiple_images_paths)}')

                save_frame = cv2.imread(v)
                _, save_frame = self.modify_left_right_videos(save_frame)

                cv2.imwrite(os.path.join(self.save_dir, f'test{i+1}.jpg'), save_frame)
                progress = int(counter / len(self.multiple_images_paths))
                self.progress_bar.setValue(progress)
            self.progress_label.setText(f'Finished')
        elif self.multiple_videos_paths or self.video_path:

            if not self.multiple_videos_paths and self.video_path:
                self.multiple_videos_paths = [self.video_path]

            if self.multiple_videos_paths:
                for i, v in enumerate(self.multiple_videos_paths):
                    self.progress_label.setText(f'Done {i}/{len(self.multiple_videos_paths)}')

                    cap = cv2.VideoCapture(v)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    counter = -1

                    self.progress_bar.setValue(0)
                    self.progress_bar.setEnabled(True)  # Enable the progress bar to show it's active

                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(os.path.join(self.save_dir, f'test{i+1}.mp4'), fourcc, fps, (video_width, video_height))

                    while cap.isOpened():
                        counter += 1
                        save_ret, save_frame = cap.read()
                        if not save_ret:
                            break

                        # Update the progress bar
                        progress = int(counter * 100 / total_frames)
                        self.progress_bar.setValue(progress)
                        QApplication.processEvents()  # Update the UI to reflect progress

                        # Brightness adjustment for selected squares
                        _, save_frame = self.modify_left_right_videos(save_frame)
                        # if self.selected_rect:
                        #     for rect in self.selected_rect:
                        #         x, y = rect
                        #         save_frame[y:y + self.square_size, x:x + self.square_size] = cv2.convertScaleAbs(
                        #             save_frame[y:y + self.square_size, x:x + self.square_size],
                        #             alpha=self.brightness_factor,
                        #             beta=50
                        #         )

                        # Write the frame
                        out.write(save_frame)

                    # Release resources after the saving is complete
                    cap.release()
                    out.release()
                    
                    # Set progress to 100% and disable the progress bar
                    self.progress_bar.setValue(100)
                    self.progress_bar.setEnabled(False)

                self.progress_label.setText(f'Finished')
        elif self.frame is not None:
            _, save_frame = self.modify_left_right_videos(self.frame)
            cv2.imwrite(os.path.join(self.save_dir, f'test1.jpg'), save_frame)
            self.progress_label.setText(f'Finished')
    
    def populate_coordinates_table(self):
        # Calculate the number of rows and columns
        
        rows = math.ceil(video_height/self.square_size_height)
        cols = math.ceil(video_width/self.square_size_width)

        # Set table dimensions
        self.coordinates_table.setRowCount(rows)
        self.coordinates_table.setColumnCount(cols)

        # Populate table with coordinates
        for row in range(rows):
            for col in range(cols):
                coordinate_text_x = col * self.square_size_width if col * self.square_size_width <= video_width else video_width
                coordinate_text_y = row * self.square_size_height if row * self.square_size_height <= video_height else video_height
                coordinate_text = f"({coordinate_text_x}, {coordinate_text_y})"
                cell_item = QTableWidgetItem(coordinate_text)
                cell_item.setTextAlignment(Qt.AlignCenter)
                self.coordinates_table.setItem(row, col, cell_item)

    def toggle_cell_color(self, row, col):
        item = self.coordinates_table.item(row, col)
        coord = (col * self.square_size_width, row * self.square_size_height)

        if coord in self.selected_rect:
            item.setBackground(QColor('transparent'))
            self.selected_rect.pop(coord)
        else:
            item.setBackground(QColor('blue'))
            self.selected_rect[coord] = [1, 0]

        if self.frame is not None and self.selected_rect:
            self.update_frame(self.frame)

    def modify_left_right_videos(self, frame):
        global video_width, video_height

        frame = cv2.resize(frame, (video_width, video_height))
        frame_left = np.copy(frame)
        frame_right = np.copy(frame)

        if self.selected_rect:
            for rect in self.selected_rect:
                x, y = rect
                x, y = y, x
                alpha, beta = self.selected_rect[rect]

                x = x * self.square_size_width
                y = y * self.square_size_height

                x2 = x + self.square_size_width
                y2 = y + self.square_size_height
                x2_1 = x + self.square_size_width
                y2_1 = y + self.square_size_height

                # x2 = x + self.square_size_width if x + self.square_size_width <= video_width-1 else video_width-1
                # y2 = y + self.square_size_height if y + self.square_size_height <= video_height-1 else video_height-1

                # x2_1 = x + self.square_size_width if x + self.square_size_width <= video_width else video_width
                # y2_1 = y + self.square_size_height if y + self.square_size_height <= video_height else video_height
                

                frame_left = cv2.rectangle(frame_left, (x, y), (x2, y2), (0, 255, 0), 2)
                # frame_right[y:y2_1, x:x2_1] = cv2.addWeighted(frame_right[y:y2_1, x:x2_1], self.contrast_factor, frame_right[y:y2_1, x:x2_1], 0, self.brightness_factor)
                # frame_right[y:y2_1, x:x2_1] = np.clip(frame_right[y:y2_1, x:x2_1] * (alpha-0.2) + beta, 0, 255)
                frame_right[y:y2_1, x:x2_1] = np.clip((frame_right[y:y2_1, x:x2_1] - beta) // (alpha-0.2), 0, 255)

        return frame_left, frame_right

    def update_frame(self, frame):
        frame_left, frame_right = self.modify_left_right_videos(frame)
        frame_left = self.resize_frame(frame_left)
        frame_right = self.resize_frame(frame_right)

        left_display = self.convert_frame_to_qimage(frame_left)
        self.video_left_label.setPixmap(QPixmap.fromImage(left_display))

        right_display = self.convert_frame_to_qimage(frame_right)
        self.video_right_label.setPixmap(QPixmap.fromImage(right_display))

    def update_video_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, self.frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

            self.update_frame(self.frame)

    def convert_frame_to_qimage(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        return QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

    def update_square_size(self):
        # Update square size based on selection
        self.selected_rect.clear()
        selected_size = self.size_selector.currentText()
        self.square_size_width = int(selected_size.split('x')[0])
        self.square_size_height = int(selected_size.split('x')[1])
        self.populate_coordinates_table()  # Update table when square size changes
        self.update_frame(self.frame)

    def play_video(self):
        self.is_playing = True
        self.pause_button.setVisible(True)
        self.play_button.setVisible(False)

    def pause_video(self):
        self.is_playing = False
        self.pause_button.setVisible(False)
        self.play_button.setVisible(True)
