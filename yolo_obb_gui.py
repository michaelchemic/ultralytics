import sys
import threading
from datetime import timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import subprocess

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QUrl, Qt, QProcess, QCoreApplication, QDateTime
from PySide6.QtGui import QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                               QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QProgressDialog, QDialog, QLineEdit,
                               QRadioButton, QButtonGroup, QDialogButtonBox, QInputDialog, QSlider)
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickWidgets import QQuickWidget
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
from ultralytics import YOLO


class Backend(QObject):
    buttonClicked = Signal(str)
    videoProcessingComplete = Signal(int, int, str)
    updateProgress = Signal(int, str)
    showMessageBox = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.label_studio_script = os.path.join("LableStudioRUN.bat")
        self.process = None  # 用于保存 QProcess 实例

        self.font_path = "fonts/NotoSansSC-Regular.ttf"
        self.output_path = os.path.abspath("output_report.pdf")
        self.progress_dialog = None
        self.register_font()

    def register_font(self):
        if not os.path.exists(self.font_path):
            raise FileNotFoundError(f"[错误] 找不到字体文件：{self.font_path}")
        pdfmetrics.registerFont(TTFont("Chinese", self.font_path))
        print("[INFO] 中文字体注册成功")

    @Slot(str)
    def handleButtonClick(self, button_text):
        print(f"Python 端接收到按钮点击: {button_text}")

        if button_text == "数据标定":
            subprocess.Popen(
                ['cmd.exe', '/c', self.label_studio_script],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            # 检查脚本是否存在
            if not os.path.exists(self.label_studio_script):
                print(f"错误：找不到脚本 {self.label_studio_script}")
                return

        if button_text == "视频处理":
            self.process_video()

        if button_text == "照片处理":
            self.process_images()

        if button_text in "生成报告":
            self.generatePdf()

        if button_text in "截图":
            self.captureScreen()

    def captureScreen(self):
        # 获取主屏幕对象
        screen = QGuiApplication.primaryScreen()
        if not screen:
            QMessageBox.critical(None, "错误", "无法获取屏幕对象")
            return

        # 弹出输入框让用户输入截图名称
        default_name = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        text, ok = QInputDialog.getText(None, "输入截图名称", "请输入文件名（无需扩展名）", text=default_name)
        if not ok or not text.strip():
            QMessageBox.information(None, "取消", "截图已取消")
            return

        # 创建 screenshots 目录（如不存在）
        save_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(save_dir, exist_ok=True)

        # 拼接完整路径
        file_path = os.path.join(save_dir, f"{text.strip()}.png")

        # 进行截图并保存
        screenshot = screen.grabWindow(0)
        if screenshot.save(file_path, "png"):
            print(f"截图已保存为：{file_path}")
            QMessageBox.information(None, "截图成功", f"截图已保存为：\n{file_path}")
        else:
            print("截图保存失败")
            QMessageBox.warning(None, "保存失败", "无法保存截图。")

    def process_images(self):
        input_folder = QFileDialog.getExistingDirectory(None, "选择图片输入文件夹")
        if not input_folder:
            return

        output_folder = QFileDialog.getExistingDirectory(None, "选择输出文件夹")
        if not output_folder:
            return

        model = YOLO("runs/train/exp_yolov8s_xiashuidao2/weights/best.pt")
        image_paths = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))

        if not image_paths:
            QMessageBox.information(None, "提示", "未在选中的文件夹中找到图片！")
            return

        total = len(image_paths)

        # 创建进度对话框
        progress = QProgressDialog("正在处理图片...", "取消", 0, total)
        progress.setWindowTitle("批量照片处理进度")
        progress.setAutoClose(True)
        progress.setMinimumDuration(0)
        progress.show()

        print(f"[INFO] 开始处理 {total} 张图片...")

        # 类别缩写映射
        class_aliases = {
            "沉积物": "CJ", "sediment": "CJ",
            "结垢": "JG", "scaling": "JG",
            "障碍物": "ZW", "obstacle": "ZW",
            "残墙": "CQ", "坝根": "CQ", "residual wall": "CQ", "dam base": "CQ",
            "树根": "SG", "root": "SG",
            "渗漏": "SL", "leakage": "SL",
            "支管暗接": "AJ", "hidden branch": "AJ",
            "异物穿入": "CR", "foreign body": "CR",
            "接口材料脱落": "TL", "material loss": "TL",
            "脱节": "TJ", "dislocation": "TJ",
            "起伏": "QF", "unevenness": "QF",
            "错口": "CK", "misalignment": "CK",
            "腐蚀": "FS", "corrosion": "FS",
            "变形": "BX", "deformation": "BX",
            "破裂": "PL", "fracture": "PL",
            "浮渣": "FZ", "scum": "FZ"
        }

        def get_color(cls_id):
            np.random.seed(cls_id)
            return tuple(np.random.randint(0, 255, size=3).tolist())

        def draw_text_pil(image, text, position, color=(255, 0, 0), font_size=None):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)

            if font_size is None:
                font_size = max(12, int(image.shape[0] * 0.06))

            try:
                font = ImageFont.truetype("simhei.ttf", font_size)
            except Exception as e:
                print(f"[WARN] 字体加载失败：{e}")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 处理图片循环
        for idx, img_path in enumerate(image_paths, 1):
            QCoreApplication.processEvents()
            if progress.wasCanceled():
                break

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 跳过无法读取的文件：{img_path}")
                continue

            result = model(img, imgsz=640, device="cpu")[0]
            img_copy = img.copy()

            if hasattr(result, "boxes") and result.boxes and hasattr(result.boxes, "xyxy"):
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = box
                    name = model.names[cls_id]
                    alias = class_aliases.get(name, name)
                    text = f"{alias} {conf:.1f}"
                    color = get_color(cls_id)

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    img_copy = draw_text_pil(img_copy, text, (x1, y1 - 20), color=color, font_size=20)

            else:
                print(f"[{idx}] 未检测到目标：{img_path.name}")

            out_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(out_path, img_copy)

            progress.setValue(idx)
            progress.setLabelText(f"处理第 {idx}/{total} 张图片: {img_path.name}")
            print(f"[{idx}/{total}] 已处理: {img_path.name}")

        progress.close()

        QMessageBox.information(None, "完成", f"共处理 {total} 张图片\n结果已保存至：\n{output_folder}")

    def process_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            None, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)")

        if not video_path:
            return

        out_dir = os.path.join(os.path.dirname(video_path), "VideoFrame")
        os.makedirs(out_dir, exist_ok=True)

        self.progress_dialog = QProgressDialog("正在处理视频...", "取消", 0, 100)
        self.progress_dialog.setWindowTitle("视频处理进度")
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        self.updateProgress.connect(self._update_progress_ui)
        self.showMessageBox.connect(self._show_message_box)

        threading.Thread(
            target=self._run_video_processing,
            args=(video_path, out_dir),
            daemon=True
        ).start()

    def _update_progress_ui(self, percent, text):
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(text)

    def _show_message_box(self, title, text):
        QMessageBox.information(None, title, text)

    def _run_video_processing(self, video_path, out_dir):
        try:
            model = YOLO("runs/train/exp_yolov8s_xiashuidao2/weights/best.pt")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.showMessageBox.emit("错误", "无法打开视频文件")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = 0
            save_count = 0

            # 类别缩写映射（与前面一致）
            class_aliases = {
                "沉积物": "CJ", "sediment": "CJ",
                "结垢": "JG", "scaling": "JG",
                "障碍物": "ZW", "obstacle": "ZW",
                "残墙": "CQ", "坝根": "CQ", "residual wall": "CQ", "dam base": "CQ",
                "树根": "SG", "root": "SG",
                "渗漏": "SL", "leakage": "SL",
                "支管暗接": "AJ", "hidden branch": "AJ",
                "异物穿入": "CR", "foreign body": "CR",
                "接口材料脱落": "TL", "material loss": "TL",
                "脱节": "TJ", "dislocation": "TJ",
                "起伏": "QF", "unevenness": "QF",
                "错口": "CK", "misalignment": "CK",
                "腐蚀": "FS", "corrosion": "FS",
                "变形": "BX", "deformation": "BX",
                "破裂": "PL", "fracture": "PL",
                "浮渣": "FZ", "scum": "FZ"
            }

            def get_color(cls_id):
                np.random.seed(cls_id)
                return tuple(np.random.randint(0, 255, size=3).tolist())

            def draw_text_pil(image, text, position, color=(255, 0, 0), font_size=20):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("simhei.ttf", font_size)
                except Exception as e:
                    print(f"[WARN] 字体加载失败：{e}")
                    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                draw.text(position, text, font=font, fill=color)
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_index / fps
                ts = str(timedelta(seconds=timestamp))
                percent = int((frame_index / total_frames) * 100)
                status_text = f"处理中: 帧 {frame_index}/{total_frames} [{ts}]"
                self.updateProgress.emit(percent, status_text)

                if self.progress_dialog.wasCanceled():
                    break

                if frame is None:
                    frame_index += 1
                    continue

                result = model(frame)[0]
                boxes = result.boxes

                if boxes is not None and hasattr(boxes, "xyxy") and len(boxes) > 0:
                    img_copy = frame.copy()
                    for box, conf, cls_id in zip(boxes.xyxy.cpu().numpy().astype(int),
                                                 boxes.conf.cpu().numpy(),
                                                 boxes.cls.cpu().numpy().astype(int)):
                        x1, y1, x2, y2 = box
                        name = model.names[cls_id]
                        alias = class_aliases.get(name, name)
                        text = f"{alias} {conf:.1f}"
                        color = get_color(cls_id)

                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                        img_copy = draw_text_pil(img_copy, text, (x1, y1 - 20), color=color)

                    filename = f"frame_{frame_index}_t{ts.replace(':', '-')}.jpg"
                    out_path = os.path.join(out_dir, filename)
                    cv2.imwrite(out_path, img_copy)
                    save_count += 1

                frame_index += 1

            cap.release()
            self.videoProcessingComplete.emit(save_count, total_frames, out_dir)
            self.updateProgress.emit(100, "处理完成")
            self.showMessageBox.emit(
                "处理完成",
                f"视频处理完成!\n总帧数: {total_frames}\n检测到病害的帧数: {save_count}\n输出目录: {out_dir}"
            )

        except Exception as e:
            self.showMessageBox.emit("处理错误", f"发生错误: {str(e)}")

    def generatePdf(self):
        print("正在生成 PDF...")
        title = "表：排水管道缺陷统计表"
        paragraphs = [
            "这是一个使用 reportlab 库生成的 PDF 示例。",
            "本示例支持中文内容自动换行与分页，采用标准 A4 纸张，左上角 25mm 边距。",
            "你可以将此功能用于生成目标检测报告、自动化文档、日志打印等任务。",
            "感谢使用。"
        ]

        c = canvas.Canvas(self.output_path, pagesize=A4)
        width, height = A4
        margin = 25 * mm
        x = margin
        y = height - margin

        c.setFont("Chinese", 14)
        c.drawString(x, y, title)
        y -= 20
        c.setFont("Chinese", 12)

        for para in paragraphs:
            lines = self.split_lines(para, max_chars=38)
            for line in lines:
                if y < 40:
                    c.showPage()
                    c.setFont("Chinese", 12)
                    y = height - margin
                c.drawString(x, y, line)
                y -= 18
            y -= 12

        c.save()
        print(f"[INFO] PDF 文件已保存到：{self.output_path}")
        self.open_output_folder()

    def split_lines(self, text, max_chars=38):
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    def open_output_folder(self):
        folder = os.path.dirname(self.output_path)
        if sys.platform == 'win32':
            os.startfile(folder)
        elif sys.platform == 'darwin':
            subprocess.call(['open', folder])
        else:
            subprocess.call(['xdg-open', folder])


class VideoSourceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False

        self.setWindowTitle("视频源选择")
        self.setStyleSheet("""
            QDialog {
                background-color: black;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                font-size: 14px;
            }
            QRadioButton {
                color: white;
                font-size: 14px;
                spacing: 8px;
            }
            QDialogButtonBox {
                button-layout: 2;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #666;
                padding: 5px 15px;
                min-width: 80px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
        """)

        self.source_type = "rtsp"
        self.setup_ui()



    def setup_ui(self):
        layout = QVBoxLayout()

        # 视频源类型选择
        layout.addWidget(QLabel("选择视频源类型:"))

        self.rtsp_radio = QRadioButton("RTSP网络流")
        self.camera_radio = QRadioButton("本地摄像头")
        self.file_radio = QRadioButton("本地视频文件")
        self.rtsp_radio.setChecked(True)

        self.source_group = QButtonGroup()
        self.source_group.addButton(self.rtsp_radio, 0)
        self.source_group.addButton(self.camera_radio, 1)
        self.source_group.addButton(self.file_radio, 2)

        layout.addWidget(self.rtsp_radio)
        layout.addWidget(self.camera_radio)
        layout.addWidget(self.file_radio)

        # RTSP地址输入
        self.rtsp_label = QLabel("RTSP地址:")
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("例如: rtsp://username:password@ip:port/stream")
        layout.addWidget(self.rtsp_label)
        layout.addWidget(self.rtsp_input)

        # 摄像头索引输入
        self.camera_label = QLabel("摄像头索引:")
        self.camera_input = QLineEdit()
        self.camera_input.setPlaceholderText("例如: 0 (默认摄像头)")
        self.camera_input.setText("0")
        self.camera_label.setVisible(False)
        self.camera_input.setVisible(False)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.camera_input)

        # 本地视频文件输入
        self.file_label = QLabel("视频文件路径:")
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("点击选择视频文件")
        self.file_input.setReadOnly(True)
        self.file_button = QPushButton("选择文件")
        self.file_button.clicked.connect(self.select_video_file)
        self.file_label.setVisible(False)
        self.file_input.setVisible(False)
        self.file_button.setVisible(False)
        layout.addWidget(self.file_label)
        layout.addWidget(self.file_input)
        layout.addWidget(self.file_button)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 连接信号
        self.source_group.buttonClicked.connect(self.on_source_changed)

        self.setLayout(layout)



    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.file_input.setText(file_path)



    def on_source_changed(self, button):
        if button == self.rtsp_radio:
            self.source_type = "rtsp"
            self.rtsp_label.setVisible(True)
            self.rtsp_input.setVisible(True)
            self.camera_label.setVisible(False)
            self.camera_input.setVisible(False)
            self.file_label.setVisible(False)
            self.file_input.setVisible(False)
            self.file_button.setVisible(False)
        elif button == self.camera_radio:
            self.source_type = "camera"
            self.rtsp_label.setVisible(False)
            self.rtsp_input.setVisible(False)
            self.camera_label.setVisible(True)
            self.camera_input.setVisible(True)
            self.file_label.setVisible(False)
            self.file_input.setVisible(False)
            self.file_button.setVisible(False)
        else:
            self.source_type = "file"
            self.rtsp_label.setVisible(False)
            self.rtsp_input.setVisible(False)
            self.camera_label.setVisible(False)
            self.camera_input.setVisible(False)
            self.file_label.setVisible(True)
            self.file_input.setVisible(True)
            self.file_button.setVisible(True)

    def get_video_source(self):
        if self.source_type == "rtsp":
            return {
                "type": "rtsp",
                "url": self.rtsp_input.text().strip()
            }
        elif self.source_type == "camera":
            return {
                "type": "camera",
                "index": int(self.camera_input.text().strip()) if self.camera_input.text().strip().isdigit() else 0
            }
        else:
            return {
                "type": "file",
                "path": self.file_input.text().strip()
            }

class YOLOApp(QWidget):
    updateConfidence = Signal(float)
    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.label_original = QLabel("原始画面")
        self.label_original.setStyleSheet("color: white; background-color: black;")
        self.label_original.setScaledContents(True)

        self.label_result = QLabel("AI识别结果")
        self.label_result.setStyleSheet("color: white; background-color: black;")
        self.label_result.setScaledContents(True)

        self.btn = QPushButton("开启实时视频流AI标识")
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

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.seek_video)



        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_result)

        # 添加各控件到最终 layout
        layout.addLayout(image_layout)
        layout.addWidget(self.slider)  # 滑条加在图像下方
        layout.addWidget(self.btn)

        # 播放/暂停按钮
        button_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("播放")
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
        layout.addLayout(button_layout)

        self.btn.clicked.connect(self.open_video_source_dialog)

        self.setLayout(layout)

    def toggle_play_pause(self):
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
        print("[INFO] 加载模型...")
        self.model = YOLO("runs/train/exp_yolov8s_xiashuidao2/weights/best.pt")
        print("[INFO] 模型加载完成")

    def open_video_source_dialog(self):
        dialog = VideoSourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
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


    def start_video_stream(self, rtsp_url):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            print("[ERROR] 无法打开RTSP流")
            return
        self.timer.start(33)  # ~30 fps

    def start_camera_stream(self, camera_index=0):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] 无法打开摄像头索引 {camera_index}")
            return
        self.timer.start(33)  # ~30 fps

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] 无法读取视频帧")
            self.timer.stop()
            return

        result = self.model(frame)[0]
        img_copy = frame.copy()


        #置信度处理
        results = self.model(frame)
        boxes = results[0].boxes
        if boxes and hasattr(boxes, 'conf') and len(boxes.conf) > 0:
            max_conf = float(boxes.conf.max().item())
            self.updateConfidence.emit(max_conf)
            print("置信度",max_conf)
        else:
            self.updateConfidence.emit(0.0)

        # 类别缩写映射（按你的定义）
        class_aliases = {
            "沉积物": "CJ", "sediment": "CJ",
            "结垢": "JG", "scaling": "JG",
            "障碍物": "ZW", "obstacle": "ZW",
            "残墙": "CQ", "坝根": "CQ", "residual wall": "CQ", "dam base": "CQ",
            "树根": "SG", "root": "SG",
            "渗漏": "SL", "leakage": "SL",
            "支管暗接": "AJ", "hidden branch": "AJ",
            "异物穿入": "CR", "foreign body": "CR",
            "接口材料脱落": "TL", "material loss": "TL",
            "脱节": "TJ", "dislocation": "TJ",
            "起伏": "QF", "unevenness": "QF",
            "错口": "CK", "misalignment": "CK",
            "腐蚀": "FS", "corrosion": "FS",
            "变形": "BX", "deformation": "BX",
            "破裂": "PL", "fracture": "PL",
            "浮渣": "FZ", "scum": "FZ"
        }

        # 彩色标签生成函数
        def get_color(cls_id):
            np.random.seed(cls_id)
            return tuple(np.random.randint(0, 255, size=3).tolist())

        # PIL 中文绘制函数
        def draw_text_pil(image, text, position, color=(255, 0, 0), font_size=None):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)

            if font_size is None:
                # 自动根据图像高度计算字体大小（可调比例 0.03）
                font_size = max(12, int(image.shape[0] * 0.06))

            try:
                font = ImageFont.truetype("simhei.ttf", font_size)
            except Exception as e:
                print(f"[WARN] 字体加载失败：{e}")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 绘制 OBB 或 AABB
        if hasattr(result, "obb") and result.obb and hasattr(result.obb, "xyxyxyxy"):
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            confs = result.obb.conf.cpu().numpy()
            clss = result.obb.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                name = self.model.names[cls_id]
                alias = class_aliases.get(name, name)
                text = f"{alias} {conf:.1f}"
                color = get_color(cls_id)

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

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                img_copy = draw_text_pil(img_copy, text, (x1, y1 - 20), color=color, font_size=20)


        else:
            print("[WARN] 未检测到任何目标")

        self.set_pixmap(self.label_original, frame)
        self.set_pixmap(self.label_result, img_copy)

        # Update slider position for video files
        if hasattr(self, 'slider') and self.slider.isEnabled():
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setValue(current_frame)
            self.slider.blockSignals(False)

    def seek_video(self, frame):
        if self.cap and self.cap.isOpened() and self.slider.isEnabled():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.update_frame()

    def set_pixmap(self, label, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    backend = Backend()
    yolo_app = YOLO()

    qml_widget = QQuickWidget()
    qml_widget.setSource(QUrl.fromLocalFile("Pipeline_Defect_IdentificationContent/MainForm.ui.qml"))
    qml_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    qml_widget.rootContext().setContextProperty("backend", backend)

    #qml_widget.rootContext().setContextProperty("yolo_app", yolo_app)


    root = qml_widget.rootObject()
    if not root:
        print("[ERROR] QML 加载失败")
        sys.exit(-1)

    image_container = root.findChild(QObject, "imageContainer")
    if image_container is None:
        print("[ERROR] 找不到 imageContainer 区域")
        sys.exit(-1)

    container = QWidget()
    container.setWindowTitle("OBB 检测系统 - 混合界面")
    container.resize(1920, 1080)

    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(qml_widget)

    yolo_widget = YOLOApp()
    yolo_widget.setParent(container)
    yolo_widget.setStyleSheet("background-color: transparent;")

    def sync_geometry():
        x = image_container.property("x")
        y = image_container.property("y")
        w = image_container.property("width")
        h = image_container.property("height")
        yolo_widget.setGeometry(int(x), int(y), int(w), int(h))

    timer = QTimer()
    timer.timeout.connect(sync_geometry)
    timer.start(100)

    container.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()