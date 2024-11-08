import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout, QComboBox, QSpacerItem, QSizePolicy, QTableWidget, QTableWidgetItem, QProgressBar, QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer, QRect, QPoint, Qt

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Selectable Brightness Adjustment")

        # Video display labels with reduced size
        self.video_left_label = QLabel(self)
        self.video_right_label = QLabel(self)
        self.display_width = None
        self.display_height = None
        self.display_ratio = None

        # Load video button
        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)

        # Square size selector
        self.size_selector = QComboBox(self)
        self.size_selector.addItems(["16x16", "20x20", "64x64", "128x128", "256x256"])
        self.size_selector.currentIndexChanged.connect(self.update_square_size)
        self.size_selector.setVisible(False)

        # Brightness adjustment buttons
        # self.increase_brightness_button = QPushButton("Increase Brightness", self)
        self.brightness_input_field = QLineEdit(self)
        self.brightness_input_field.setPlaceholderText('0')
        self.brightness_input_field.textChanged.connect(self.validate_brightness_input)

        self.contrast_input_field = QLineEdit(self)
        self.contrast_input_field.setPlaceholderText('1')
        self.contrast_input_field.textChanged.connect(self.validate_contrast_input)

        # Play and Pause buttons
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_video)

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
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)

        self.setLayout(layout)

        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # OpenCV Video Capture and variables
        self.video_path = None
        self.cap = None
        self.curr_frame = None
        self.video_width = None
        self.video_height = None
        self.fps = None
        self.selected_rect = set()
        self.square_size = 16
        self.brightness_factor = 0
        self.contrast_factor = 1  # Variable to hold the brightness factor
        self.is_playing = True  # Track play/pause state

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

    def is_float_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def validate_contrast_input(self):
        text = self.contrast_input_field.text()

        print('te', text)
        
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

    def show_error(self, message):
        QMessageBox.warning(self, "Invalid Input", message)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)

            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.display_ratio = self.video_width // 360
            self.display_width = self.video_width // self.display_ratio
            self.display_height = self.video_height // self.display_ratio

            self.video_left_label.setFixedSize(self.display_width, self.display_height)
            self.video_right_label.setFixedSize(self.display_width, self.display_height)

            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.timer.start(self.fps)
            self.size_selector.setVisible(True)
            self.selected_rect.clear()
            self.populate_coordinates_table()

    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", self.save_dir)
        if dir_path:
            self.save_dir = dir_path
            self.save_dir_label.setText(self.save_dir)

    def save_video(self):
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            counter = -1

            self.progress_bar.setValue(0)
            self.progress_bar.setEnabled(True)  # Enable the progress bar to show it's active

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(self.save_dir, 'test.mp4'), fourcc, self.fps, (self.video_width, self.video_height))

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

    def populate_coordinates_table(self):
        # Calculate the number of rows and columns
        rows = self.video_height // self.square_size
        cols = self.video_width // self.square_size

        # Set table dimensions
        self.coordinates_table.setRowCount(rows)
        self.coordinates_table.setColumnCount(cols)

        # Populate table with coordinates
        for row in range(rows):
            for col in range(cols):
                coordinate_text = f"({col * self.square_size}, {row * self.square_size})"
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

        if self.curr_frame is not None and self.selected_rect:
            frame_left, frame_right = self.modify_left_right_videos(self.curr_frame)
            frame_left = cv2.resize(frame_left, (self.display_width, self.display_height))
            frame_right = cv2.resize(frame_right, (self.display_width, self.display_height))

            left_display = self.convert_frame_to_qimage(frame_left)
            self.video_left_label.setPixmap(QPixmap.fromImage(left_display))

            right_display = self.convert_frame_to_qimage(frame_right)
            self.video_right_label.setPixmap(QPixmap.fromImage(right_display))

    def modify_left_right_videos(self, frame):
        frame_left = np.copy(frame)
        frame_right = np.copy(frame)

        if self.selected_rect:
            for rect in self.selected_rect:
                x, y = rect

                frame_left = cv2.rectangle(frame_left, (x, y), (x + self.square_size, y + self.square_size), (0, 255, 0), 2)
                # frame_right[y:y + self.square_size, x:x + self.square_size] = cv2.convertScaleAbs(frame_right[y:y + self.square_size, x:x + self.square_size], alpha=self.brightness_factor, beta=50)
                # frame_right[y:y + self.square_size, x:x + self.square_size] = cv2.convertScaleAbs(frame_right[y:y + self.square_size, x:x + self.square_size], alpha=self.contrast_factor, beta=self.brightness_factor)
                frame_right[y:y + self.square_size, x:x + self.square_size] = cv2.addWeighted(frame_right[y:y + self.square_size, x:x + self.square_size], self.contrast_factor, frame_right[y:y + self.square_size, x:x + self.square_size], 0, self.brightness_factor)
    

        return frame_left, frame_right

    def update_frame(self):
        if self.cap and self.cap.isOpened() and self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.timer.stop()
                return

            self.curr_frame = frame

            frame_left, frame_right = self.modify_left_right_videos(frame)
            frame_left = cv2.resize(frame_left, (self.display_width, self.display_height))
            frame_right = cv2.resize(frame_right, (self.display_width, self.display_height))

            left_display = self.convert_frame_to_qimage(frame_left)
            self.video_left_label.setPixmap(QPixmap.fromImage(left_display))

            right_display = self.convert_frame_to_qimage(frame_right)
            self.video_right_label.setPixmap(QPixmap.fromImage(right_display))

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

    def play_video(self):
        self.is_playing = True  # Set play state to True
        print("Video playback started.")

    def pause_video(self):
        self.is_playing = False  # Set play state to False
        print("Video playback paused.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
