import sys
import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtQuick import QQuickItem

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QUrl, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtQuickWidgets import QQuickWidget


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        #self.setMinimumSize(1000, 500)

        self.label_original = QLabel("原始图片")
        self.label_original.setStyleSheet("color: white;")
        self.label_original.setScaledContents(True)

        self.label_result = QLabel("检测结果")
        self.label_result.setStyleSheet("color: white;")
        self.label_result.setScaledContents(True)

        self.btn = QPushButton("选择图片")
        self.btn.setStyleSheet("color: white;")
        self.btn.clicked.connect(self.load_image)

        # ⬅️ 新增水平布局：图片显示区域左右排列
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_result)

        # ⬇️ 总体布局：图片区域 + 按钮
        layout = QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addWidget(self.btn)

        self.setLayout(layout)

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

    qml_widget = QQuickWidget()
    qml_widget.setSource(QUrl.fromLocalFile("Pipeline_Defect_IdentificationContent/MainForm.ui.qml"))
    qml_widget.setResizeMode(QQuickWidget.SizeRootObjectToView)

    root = qml_widget.rootObject()
    image_container = root.findChild(QQuickItem, "imageContainer")
    if image_container is None:
        print("[ERROR] 找不到 imageContainer 区域")
        sys.exit(-1)

    container = QWidget()
    container.setWindowTitle("OBB 检测系统 - 混合界面")
    container.resize(1920, 1080)

    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(qml_widget)

    # 创建 YOLO 界面
    yolo_widget = YOLOApp()
    yolo_widget.setParent(container)
    yolo_widget.setStyleSheet("background-color: transparent;")

    def sync_geometry():
        # 实时获取 QML 区域尺寸
        x = image_container.x()
        y = image_container.y()
        w = image_container.width()
        h = image_container.height()
        yolo_widget.setGeometry(int(x), int(y), int(w), int(h))

    # 每 100ms 检查一次 QML 尺寸变化（也可使用绑定机制）
    timer = QTimer()
    timer.timeout.connect(sync_geometry)
    timer.start(100)

    container.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
