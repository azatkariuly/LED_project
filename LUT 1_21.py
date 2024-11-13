import sys
import cv2
import random
import os
import math
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QLineEdit, QFormLayout, QPushButton, QFileDialog, QMessageBox, QCheckBox, QComboBox, QTableWidget, QProgressBar, QSpacerItem, QSizePolicy, QTableWidgetItem
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor

video_width = 800
video_height = 448

class VideoPlayer(QWidget):
    def __init__(self, video_paths, x_axis, y_axis, video_width, video_height, power_consumption=False, power_consumption_square=None, power_brightness=None, power_contrast=None, display_images=False, image_display_frequency=None):
        super().__init__()

        self.video_width = video_width
        self.video_height = video_height


        self.power_consumption = power_consumption
        self.power_consumption_square = power_consumption_square
        self.power_brightness = power_brightness
        self.power_contrast = power_contrast

        # Set the initial position on screen, remove title bar, and keep the window on top
        self.setGeometry(x_axis, y_axis, self.video_width, self.video_height)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Store the list of video paths
        self.video_paths = video_paths
        self.current_video_index = -1
        self.image_display_frequency = image_display_frequency

        # Open the first video using OpenCV
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_index])
        if not self.cap.isOpened():
            print("Error: Couldn't open video.")
            sys.exit()

        # Label to show video frames
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.video_width, self.video_height)

        # Start the timer to update the frame every 30ms (approx 30 fps)
        self.timer = QTimer(self)
        if display_images:
            self.timer.timeout.connect(self.update_image_frame)
            self.timer.start(int(image_display_frequency))
        else:
            self.timer.timeout.connect(self.update_video_frame)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.timer.start(int(1000 / self.fps))

    def play_video(self):
        """Start or resume the video playback."""
        if self.image_display_frequency:
            self.timer.start(int(self.image_display_frequency))
        else:
            self.timer.start(int(1000 / self.fps))

    def pause_video(self):
        """Pause the video playback."""
        self.timer.stop()

    def update_image_frame(self):
        self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
        frame = cv2.imread(self.video_paths[self.current_video_index])

        # Resize the frame
        frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

        # Convert the frame to QImage for displaying in the QLabel
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        if self.power_consumption:
            random_h, random_w = random.randint(0, h - self.power_consumption_square - 1), random.randint(0, w - self.power_consumption_square - 1)

            frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square] = cv2.addWeighted(
                frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square],
                self.power_contrast,
                frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square],
                0,
                self.power_brightness
            )
    
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display the QImage on the label
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def update_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # If the current video ends, move to the next video
            self.current_video_index = (self.current_video_index + 1) % len(self.video_paths)
            self.cap = cv2.VideoCapture(self.video_paths[self.current_video_index])
            return

        # Resize the frame
        frame_resized = cv2.resize(frame, (self.video_width, self.video_height))

        # Convert the frame to QImage for displaying in the QLabel
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        if self.power_consumption:
            random_h, random_w = random.randint(0, h - self.power_consumption_square - 1), random.randint(0, w - self.power_consumption_square - 1)

            frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square] = cv2.addWeighted(
                frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square],
                self.power_contrast,
                frame_rgb[random_h:random_h + self.power_consumption_square, random_w:random_w + self.power_consumption_square],
                0,
                self.power_brightness
            )
    
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display the QImage on the label
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        # Release the video capture when closing the window
        self.cap.release()
        event.accept()

    def cleanup(self):
        # Stop the timer if it's running
        if self.timer.isActive():
            self.timer.stop()

    def __del__(self):
        # Ensure resources are cleaned up if instance is deleted
        self.cleanup()

class TabbedApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # Initialize window
        self.setWindowTitle("LED Control")
        self.setGeometry(100, 100, 800, 600)

        # Layout for tabs
        self.layout = QVBoxLayout(self)
        
        # Tab widget
        self.tabs = QTabWidget(self)
        self.layout.addWidget(self.tabs)

        # Create tabs with text content
        self.create_tabs()
        
    def create_tabs(self):
        # Tab 1: Display some text
        self.tab1 = Tab1()
        self.tabs.addTab(self.tab1, "Setting Part")
        
        # Tab 2: Display different text
        self.tab2 = Tab2()
        self.tabs.addTab(self.tab2, "Solved Part_LUT")

    def closeEvent(self, event):
        # # Close the video player if it is open
        # if hasattr(self, 'tab1', 'video_player'):
        #     self.tab1.video_player.close()
        # if hasattr(self, 'tab2', 'cap'):
        #     self.tab2.cap.release()
        # if hasattr(self, 'tab2', 'timer'):
        #     self.tab2.timer.stop()

        if hasattr(self.tab1, 'cleanup'):
            self.tab1.cleanup()
        
        # Clean up tab2 if it has any resources to stop
        if hasattr(self.tab2, 'cleanup'):
            self.tab2.cleanup()

        event.accept()
        event.accept()

class Tab1(QWidget):
    def __init__(self):
        super().__init__()

        self.x_axis = 0
        self.y_axis = 0

        self.video_paths = []
        self.video_player = None

        self.image_loop_time = 100 # ms

        self.power_consumption = False
        self.power_square_size = 16
        self.power_brightness = 0
        self.power_contrast = 1.0

        self.create_tab()

    def create_tab(self):
        tab1_layout = QHBoxLayout()

        # First column layout with QFormLayout
        col1_layout = QFormLayout()
        label_x_axis = QLabel("x시작점:", self)
        self.input_x_axis = QLineEdit(self)
        self.input_x_axis.setPlaceholderText(str(self.x_axis))
        self.input_x_axis.textChanged.connect(self.validate_x_axis)
        
        label_y_axis = QLabel("y시작점:", self)
        self.input_y_axis = QLineEdit(self)
        self.input_y_axis.setPlaceholderText(str(self.y_axis))
        self.input_y_axis.textChanged.connect(self.validate_y_axis)

        label_video_width = QLabel("넓이:", self)
        self.input_video_width = QLineEdit(self)
        global video_width
        self.input_video_width.setPlaceholderText(str(video_width))
        self.input_video_width.textChanged.connect(self.validate_video_width)

        label_video_height = QLabel("높이:", self)
        self.input_video_height = QLineEdit(self)
        global video_height
        self.input_video_height.setPlaceholderText(str(video_height))
        self.input_video_height.textChanged.connect(self.validate_video_height)

        col1_layout.addRow(label_x_axis, self.input_x_axis)
        col1_layout.addRow(label_y_axis, self.input_y_axis)
        col1_layout.addRow(label_video_width, self.input_video_width)
        col1_layout.addRow(label_video_height, self.input_video_height)

        layout_image_loop = QLabel("Image loop time:", self)
        self.input_image_loop = QLineEdit(self)
        self.input_image_loop.setPlaceholderText(str(self.image_loop_time))
        self.input_image_loop.textChanged.connect(self.validate_image_loop_time)
        col1_layout.addRow(layout_image_loop, self.input_image_loop)
        
        # Add buttons below col1_layout
        col1_buttons_layout = QVBoxLayout()

        # Play and Pause buttons
        load_button = QPushButton("Load Videos", self)
        load_button.clicked.connect(self.load_videos)
        load_images_button = QPushButton("Load Images", self)
        load_images_button.clicked.connect(self.load_images)
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setVisible(False)
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setVisible(False)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setVisible(False)

        col1_buttons_layout.addWidget(load_button)
        col1_buttons_layout.addWidget(load_images_button)
        col1_buttons_layout.addWidget(self.play_button)
        col1_buttons_layout.addWidget(self.pause_button)
        col1_buttons_layout.addWidget(self.stop_button)
        col1_layout.addRow(col1_buttons_layout)

        # Second column layout with QFormLayout
        col2_layout = QFormLayout()
        self.checkbox_power = QCheckBox("소비전력", self)
        self.checkbox_power.stateChanged.connect(self.checkbox_power_state_changed)
        col2_layout.addRow(self.checkbox_power)

        label_power_size_selector = QLabel("번동영역:", self)
        self.power_size_selector = QLineEdit(self)
        self.power_size_selector.setPlaceholderText(str(self.power_square_size))
        # self.power_size_selector.currentIndexChanged.connect(self.power_update_square_size)
        self.power_size_selector.textChanged.connect(self.validate_power_square)

        label_power_brightness = QLabel("Brightness:", self)
        self.input_power_brigthness = QLineEdit(self)
        self.input_power_brigthness.setPlaceholderText(str(self.power_brightness))
        self.input_power_brigthness.textChanged.connect(self.validate_power_brightness)

        label_power_contrast = QLabel("Contrast:", self)
        self.input_power_contrast = QLineEdit(self)
        self.input_power_contrast.setPlaceholderText(str(self.power_contrast))
        self.input_power_contrast.textChanged.connect(self.validate_power_contrast)
        col2_layout.addRow(label_power_size_selector, self.power_size_selector)
        col2_layout.addRow(label_power_brightness, self.input_power_brigthness)
        col2_layout.addRow(label_power_contrast, self.input_power_contrast)

        # Add both form layouts to the horizontal layout
        tab1_layout.addLayout(col1_layout)
        tab1_layout.addLayout(col2_layout)

        self.setLayout(tab1_layout)

    def validate_power_square(self):
        text = self.power_size_selector.text()
        if text.isdigit():
            self.power_square_size = int(text)
        elif text:
            self.power_size_selector.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            self.power_square_sizet = 16
        # selected_size = self.power_size_selector.currentText()
        # self.power_square_size = int(selected_size.split('x')[0])

    def checkbox_power_state_changed(self):
        if self.checkbox_power.isChecked():
            self.power_consumption = True
        else:
            self.power_consumption = False

    def validate_x_axis(self):
        text = self.input_x_axis.text()
        if text.isdigit():
            self.x_axis = int(text)
        elif text:
            self.input_x_axis.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            self.x_axis = 0

    def validate_y_axis(self):
        text = self.input_y_axis.text()
        if text.isdigit():
            self.y_axis = int(text)
        elif text:
            self.input_y_axis.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            self.y_axis = 0

    def validate_video_width(self):
        text = self.input_video_width.text()
        global video_width
        if text.isdigit():
            video_width = int(text)
        elif text:
            self.input_video_width.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            video_width = 800

    def validate_video_height(self):
        text = self.input_video_height.text()
        global video_height
        if text.isdigit():
            video_height = int(text)
        elif text:
            self.input_video_height.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            video_height = 448

    def validate_image_loop_time(self):
        text = self.input_image_loop.text()
        if text.isdigit():
            self.image_loop_time = int(text)
        elif text:
            self.input_image_loop.setText(text[:-1])
            self.show_error("Only numeric values are allowed.")
        else:
            self.image_loop_time = 100

    def validate_power_brightness(self):
        text = self.input_power_brigthness.text()
        if text.startswith('-'):
            if len(text) > 1:
                if text[1:].isdigit():  # Check if the rest of the string after '-' is a number
                    value = int(text)
                    if -255 <= value <= 255:
                        self.power_brightness = value
                    else:
                        self.input_power_brigthness.setText(text[:-1])  # Remove last character
                        self.show_error("Please enter a number between -255 and 255.")
                else:
                    self.input_power_brigthness.setText(text[:-1])  # Remove last character
                    self.show_error("Only numeric values are allowed.")
        elif text.isdigit():
            value = int(text)
            if -255 <= value <= 255:
                self.power_brightness = value
            else:
                self.input_power_brigthness.setText(text[:-1])  # Remove last character
                self.show_error("Please enter a number between -255 and 255.")
        elif text:
            self.input_power_brigthness.setText(text[:-1])  # Remove last character
            self.show_error("Only numeric values are allowed.")

    def is_float_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def validate_power_contrast(self):
        text = self.input_power_contrast.text()
        if self.is_float_number(text):
            value = float(text)
            self.power_contrast = value
        elif text:
            self.input_power_contrast.setText(text[:-1])  # Remove last character
            self.show_error("Only numeric values are allowed.")
        else:
            self.power_contrast = 1.0

    def load_images(self):
        self.video_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        global video_width, video_height
        if self.video_player:
            del self.video_player
            self.stop_button.setVisible(False)
            self.pause_button.setVisible(False)
            self.play_button.setVisible(False)

        if self.video_paths:
            if self.power_consumption and (video_height < self.power_square_size or video_width < self.power_square_size):
                self.show_error('power consumption square size cannot be bigger than video size')
            else:
                self.video_player = VideoPlayer(
                    self.video_paths,
                    self.x_axis,
                    self.y_axis,
                    video_width,
                    video_height,
                    self.power_consumption,
                    self.power_square_size,
                    self.power_brightness,
                    self.power_contrast,
                    display_images=True,
                    image_display_frequency=int(self.image_loop_time)
                )
                self.video_player.show()

                self.pause_button.setVisible(True)
                self.stop_button.setVisible(True)

    def load_videos(self):
        self.video_paths, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        global video_width, video_height
        if self.video_player:
            del self.video_player
            self.stop_button.setVisible(False)
            self.pause_button.setVisible(False)
            self.play_button.setVisible(False)

        if self.video_paths:
            if self.power_consumption and (video_height < self.power_square_size or video_width < self.power_square_size):
                self.show_error('power consumption square size cannot be bigger than video size')
            else:
                self.video_player = VideoPlayer(
                    self.video_paths,
                    self.x_axis,
                    self.y_axis,
                    video_width,
                    video_height,
                    self.power_consumption,
                    self.power_square_size,
                    self.power_brightness,
                    self.power_contrast
                )
                self.video_player.show()

                self.pause_button.setVisible(True)
                self.stop_button.setVisible(True)

    def play_video(self):
        if hasattr(self, 'video_player'):
            self.video_player.play_video()
            self.pause_button.setVisible(True)
            self.play_button.setVisible(False)

    def pause_video(self):
        if hasattr(self, 'video_player'):
            self.video_player.pause_video()
            self.pause_button.setVisible(False)
            self.play_button.setVisible(True)

    def stop_video(self):
        if hasattr(self, 'video_player'):
            self.video_player.close()
            self.video_paths = None
            self.stop_button.setVisible(False)
            self.pause_button.setVisible(False)
            self.play_button.setVisible(False)

    def show_error(self, message):
        QMessageBox.warning(self, "Invalid Input", message)

    def cleanup(self):
        if self.video_player:
            self.video_player.close()
            self.video_player.deleteLater()

class Tab2(QWidget):
    def __init__(self):
        super().__init__()

        self.multiple_videos_paths = None
        self.multiple_images_paths = None
        
        # Video display labels with reduced size
        self.video_left_label = QLabel(self)
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
        self.size_selector.addItems(["16x16", "20x20", "64x64", "128x128", "256x256"])
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
        self.selected_rect = set()
        self.square_size = 16
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

            self.frame = cv2.imread(self.image_path)

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
        global video_width, video_height
        rows = math.ceil(video_height/self.square_size)
        cols = math.ceil(video_width/self.square_size)

        # Set table dimensions
        self.coordinates_table.setRowCount(rows)
        self.coordinates_table.setColumnCount(cols)

        # Populate table with coordinates
        for row in range(rows):
            for col in range(cols):
                coordinate_text_x = col * self.square_size if col * self.square_size <= video_width else video_width
                coordinate_text_y = row * self.square_size if row * self.square_size <= video_height else video_height
                coordinate_text = f"({coordinate_text_x}, {coordinate_text_y})"
                cell_item = QTableWidgetItem(coordinate_text)
                cell_item.setTextAlignment(Qt.AlignCenter)
                self.coordinates_table.setItem(row, col, cell_item)

    def toggle_cell_color(self, row, col):
        item = self.coordinates_table.item(row, col)
        coord = (col * self.square_size, row * self.square_size)

        if coord in self.selected_rect:
            item.setBackground(QColor('transparent'))
            self.selected_rect.remove(coord)
        else:
            item.setBackground(QColor('blue'))
            self.selected_rect.add(coord)

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

                x2 = x + self.square_size if x + self.square_size <= video_width-1 else video_width-1
                y2 = y + self.square_size if y + self.square_size <= video_height-1 else video_height-1

                x2_1 = x + self.square_size if x + self.square_size <= video_width else video_width
                y2_1 = y + self.square_size if y + self.square_size <= video_height else video_height

                frame_left = cv2.rectangle(frame_left, (x, y), (x2, y2), (0, 255, 0), 2)
                frame_right[y:y2_1, x:x2_1] = cv2.addWeighted(frame_right[y:y2_1, x:x2_1], self.contrast_factor, frame_right[y:y2_1, x:x2_1], 0, self.brightness_factor)

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
        self.square_size = int(selected_size.split('x')[0])
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TabbedApp()
    window.show()
    sys.exit(app.exec_())
