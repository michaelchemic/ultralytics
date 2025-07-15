import sys
import threading
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import os
import subprocess

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QUrl, Qt, QProcess, QCoreApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                               QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QProgressDialog)
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

    from PySide6.QtWidgets import QFileDialog, QProgressDialog, QMessageBox
    from PySide6.QtCore import QCoreApplication
    import cv2
    import numpy as np
    from pathlib import Path
    from ultralytics import YOLO
    import os

    def process_images(self):
        input_folder = QFileDialog.getExistingDirectory(None, "选择图片输入文件夹")
        if not input_folder:
            return

        output_folder = QFileDialog.getExistingDirectory(None, "选择输出文件夹")
        if not output_folder:
            return

        model = YOLO("runs/train/yolo_obb_sewer10/weights/best.pt")
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

        for idx, img_path in enumerate(image_paths, 1):
            QCoreApplication.processEvents()  # 保证界面实时刷新
            if progress.wasCanceled():
                break

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 跳过无法读取的文件：{img_path}")
                continue

            results = model(img, imgsz=640, device="cpu")[0]

            for box in results.obb:
                if not hasattr(box, 'xyxyxyxy'):
                    continue
                points = box.xyxyxyxy.cpu().numpy().reshape(-1, 2).astype(int)
                label = f"{model.names[int(box.cls)]} {box.conf.item():.2f}"
                cv2.polylines(img, [points], True, (0, 255, 0), 2)
                cv2.putText(img, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            out_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(out_path, img)

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
            model = YOLO("runs/train/yolo_obb_sewer10/weights/best.pt")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.showMessageBox.emit("错误", "无法打开视频文件")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = 0
            save_count = 0

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

                results = model(frame)
                obb = results[0].obb

                if obb is not None and hasattr(obb, 'xyxyxyxy') and len(obb) > 0:
                    filename = f"frame_{frame_index}_t{ts.replace(':', '-')}.jpg"
                    out_path = os.path.join(out_dir, filename)
                    cv2.imwrite(out_path, frame)
                    save_count += 1

                frame_index += 1

            cap.release()
            self.videoProcessingComplete.emit(save_count, total_frames, out_dir)
            self.updateProgress.emit(100, "处理完成")
            self.showMessageBox.emit("处理完成",
                                     f"视频处理完成!\n总帧数: {total_frames}\n检测到病害的帧数: {save_count}\n输出目录: {out_dir}")

        except Exception as e:
            self.showMessageBox.emit("处理错误", f"发生错误: {str(e)}")

    def generatePdf(self):
        print("正在生成 PDF...")
        title = "AI 检测报告示例"
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


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        self.label_original = QLabel("原始图片")
        self.label_original.setStyleSheet("color: white;")
        self.label_original.setScaledContents(True)

        self.label_result = QLabel("检测结果")
        self.label_result.setStyleSheet("color: white;")
        self.label_result.setScaledContents(True)

        self.btn = QPushButton("选择图片")
        self.btn.setStyleSheet("color: white;")
        self.btn.clicked.connect(self.load_image)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_result)

        layout = QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addWidget(self.btn)
        self.setLayout(layout)

    def load_model(self):
        print("[INFO] 加载模型...")
        self.model = YOLO("runs/train/yolo_obb_sewer10/weights/best.pt")
        print("[INFO] 模型加载完成")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path)
            if img is None:
                print("[ERROR] 图片加载失败")
                return

            result = self.model(img)[0]
            img_copy = img.copy()

            if hasattr(result.obb, 'xyxyxyxy'):
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                for box in boxes:
                    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)

            self.set_pixmap(self.label_original, img)
            self.set_pixmap(self.label_result, img_copy)

    def set_pixmap(self, label, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio))


def main():
    app = QApplication(sys.argv)
    backend = Backend()

    qml_widget = QQuickWidget()
    qml_widget.setSource(QUrl.fromLocalFile("Pipeline_Defect_IdentificationContent/MainForm.ui.qml"))
    qml_widget.setResizeMode(QQuickWidget.SizeRootObjectToView)
    qml_widget.rootContext().setContextProperty("backend", backend)

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
