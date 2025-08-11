# yolo_app.py
# Implements the YOLO-based video processing UI with real-time object detection
# and video source selection

from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSlider
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from ultralytics import YOLO
import cv2
import numpy as np
from video_source_dialog import VideoSourceDialog
from utils import class_aliases, get_color, draw_text_pil
from ColorAnnotation import DefectColorPanel
from PySide6.QtCore import QObject


def rgb_to_hex(bgr_tuple):
    """将 OpenCV 的 BGR 颜色转换为 #RRGGBB 的 hex 字符串"""
    b, g, r = bgr_tuple
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


class YOLOApp(QWidget):
    """YOLO application widget for real-time video processing and display."""

    def __init__(self):
        """Initialize the YOLO app with UI setup and model loading."""
        super().__init__()
        self.is_playing = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.setup_ui()
        self.load_model()

        self.color_panel = DefectColorPanel()  # ← 实例化颜色注释面板
        self.color_panel.hide()  # 如果不希望自动显示

    def setup_ui(self):
        """Set up the UI components for video display and controls."""
        layout = QVBoxLayout()

        # Original and result video labels
        self.label_original = QLabel("原始视频帧")
        self.label_original.setStyleSheet("color: white; background-color: black;")
        self.label_original.setScaledContents(True)

        self.label_result = QLabel("AI标注帧")
        self.label_result.setStyleSheet("color: white; background-color: black;")
        self.label_result.setScaledContents(True)

        # Button for opening video source dialog
        self.btn = QPushButton("视频源切换")
        self.btn.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #444;
                border: 1px solid #666;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)

        # Video seek slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.seek_video)

        # Layout for video display
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_result)

        # Play/pause button
        button_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("暂停/播放")
        self.play_pause_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #444;
                border: 1px solid #666;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        button_layout.addWidget(self.play_pause_button)
        button_layout.addStretch()

        # Assemble main layout
        layout.addLayout(image_layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.btn)
        layout.addLayout(button_layout)

        self.btn.clicked.connect(self.open_video_source_dialog)

        self.setLayout(layout)



    def toggle_play_pause(self):
        """Toggle play/pause state of the video stream."""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText("播放")
        else:
            if self.cap and self.cap.isOpened():
                self.timer.start(33)
                self.is_playing = True
                self.play_pause_button.setText("暂停")

    def load_model(self):
        """Load the YOLO model for object detection."""
        print("[INFO] Loading model...")
        self.model = YOLO("runs/train/exp_yolov8s_xiashuidao2/weights/best.pt")
        print("[INFO] Model loaded successfully")

    def open_video_source_dialog(self):
        """Open dialog for selecting video source (RTSP, camera, or file)."""
        dialog = VideoSourceDialog(self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            source = dialog.get_video_source()

            if source["type"] == "rtsp":
                self.cap = cv2.VideoCapture(source["url"])
                self.slider.setEnabled(False)
            elif source["type"] == "camera":
                self.cap = cv2.VideoCapture(source["index"])
                self.slider.setEnabled(False)
            else:  # file
                self.cap = cv2.VideoCapture(source["path"])
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.slider.setMaximum(total_frames)
                self.slider.setEnabled(True)
            if self.cap.isOpened():
                self.timer.start(33)  # ~30 FPS

    def inject_color_updater(self, panel: DefectColorPanel, emitter: QObject):
        self.color_panel = panel
        self.color_emitter = emitter

    def update_frame(self):
        """Update video frame with YOLO detection results."""
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Unable to read video frame")
            self.timer.stop()
            return

        result = self.model(frame)[0]
        img_copy = frame.copy()

        # Draw detection results (OBB or AABB)
        if hasattr(result, "obb") and result.obb and hasattr(result.obb, "xyxyxyxy"):
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            clss = result.obb.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                name = self.model.names[cls_id]
                alias = class_aliases.get(name, name)
                text = f"{alias} {conf:.1f}"
                color = get_color(cls_id)

                color_hex = rgb_to_hex(color)  # 转为 "#ff0000"
                if hasattr(self, 'color_emitter'):
                    self.color_emitter.update_color.emit(name, color_hex)

                pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_copy, [pts], True, color, 2)
                x, y = pts[0][0]
                img_copy = draw_text_pil(img_copy, text, (x, y - 20), color=color, font_size=50)

        elif hasattr(result, "boxes") and result.boxes and hasattr(result.boxes, "xyxy"):
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                name = self.model.names[cls_id]
                alias = class_aliases.get(name, name)
                text = f"{alias} {conf:.1f}"
                color = get_color(cls_id)

                # class_names = list(self.model.names.values())
                # print("模型可识别的类别名称：", class_names)

                color_hex = rgb_to_hex(color)  # 转为 "#ff0000"
                if hasattr(self, 'color_emitter'):
                    self.color_emitter.update_color.emit(name, color_hex)

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                img_copy = draw_text_pil(img_copy, text, (x1, y1 - 20), color=color, font_size=20)

        else:
            print("[WARN] No objects detected")

        self.set_pixmap(self.label_original, frame)
        self.set_pixmap(self.label_result, img_copy)

        # Update slider position for video files
        if hasattr(self, 'slider') and self.slider.isEnabled():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)



    def seek_video(self, frame):
        """Seek to a specific frame in the video."""
        if self.cap and self.cap.isOpened() and self.slider.isEnabled():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.update_frame()

    def set_pixmap(self, label, cv_img):
        """Convert OpenCV image to QPixmap and set it to a QLabel."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        """Clean up resources on widget close."""
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()