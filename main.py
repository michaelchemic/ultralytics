# main.py
# Entry point for the OBB Detection System application

import sys
from PySide6.QtCore import QTimer, QUrl, QObject
from PySide6.QtGui import QIcon
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication

from LoadingScreen import LoadingScreen
from backend import Backend
from system_monitor import Monitor
from yolo_app import YOLOApp
from ColorAnnotation import DefectColorPanel

# 在文件开头添加：
from PySide6.QtCore import Signal, QObject

# 定义颜色更新信号类
class ColorUpdateEmitter(QObject):
    update_color = Signal(str, str)

class MainWindow(QWidget):
    """
    MainWindow class for the OBB Detection System application.
    Manages the QML frontend, YOLOApp widget, and DefectColorPanel synchronization.
    """

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_backend()
        self.setup_yolo()
        self.setup_defect_panel()
        self.setup_geometry_sync()
        self.setup_color_signals()

    def setup_color_signals(self):
        self.color_emitter = ColorUpdateEmitter()
        self.color_emitter.update_color.connect(self.color_panel.update_defect_color)
        self.yolo_widget.inject_color_updater(self.color_panel, self.color_emitter)

    def setup_ui(self):
        """Set up the main window and QML widget."""
        self.setWindowTitle("管道病害识别系统 V1.0")
        self.resize(1920, 1080)

        # Load QML frontend
        self.qml_widget = QQuickWidget()
        self.qml_widget.setSource(QUrl.fromLocalFile("Pipeline_Defect_IdentificationContent/MainForm.ui.qml"))
        self.qml_widget.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)

        # Add QML widget to layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.qml_widget)

    def setup_backend(self):
        """Set up backend and monitor context properties."""
        self.backend = Backend()
        self.monitor = Monitor()
        self.qml_widget.rootContext().setContextProperty("backend", self.backend)
        self.qml_widget.rootContext().setContextProperty("monitor", self.monitor)

        # Verify QML loading
        self.root = self.qml_widget.rootObject()
        if not self.root:
            print("[ERROR] QML 加载失败")
            sys.exit(-1)

        # 找到 QML 中的图像展示区域和颜色标注容器
        self.image_container = self.root.findChild(QObject, "imageContainer")
        self.color_panel_container = self.root.findChild(QObject, "colorPanelContainer")
        self.right_panel = self.root.findChild(QObject, "rightPanel")

        if self.image_container is None:
            print("[ERROR] 找不到 imageContainer 对象")
            sys.exit(-1)


        if self.right_panel is None:
            print("[ERROR] Could not find rightPanel in QML")
            sys.exit(-1)

    def setup_yolo(self):
        """Set up the YOLOApp widget."""
        self.yolo_widget = YOLOApp()
        self.yolo_widget.setParent(self)
        self.yolo_widget.setStyleSheet("background-color: transparent;")

    def setup_defect_panel(self):
        """Set up the color annotation panel."""
        self.color_panel = DefectColorPanel()
        self.color_panel.setParent(self)
        self.color_panel.setStyleSheet("background-color: transparent;")


        # 初始化时隐藏标题栏窗口特征（可选）
        # self.defect_panel.setWindowFlags(self.windowFlags())  # 嵌入 QML 区域，不能显示为浮动窗口

    def setup_geometry_sync(self):
        """Set up timer to synchronize YOLOApp and DefectColorPanel geometry."""

        def sync_geometry():
            # 同步图像窗口
            x = self.image_container.property("x")
            y = self.image_container.property("y")
            w = self.image_container.property("width")
            h = self.image_container.property("height")
            self.yolo_widget.setGeometry(int(x), int(y), int(w), int(h))

            # 同步 DefectColorPanel 区域（缩小并居中）
            rx = self.right_panel.property("x")
            ry = self.right_panel.property("y")
            rw = self.right_panel.property("width")
            rh = self.right_panel.property("height")

            # 设置尺寸缩小，例如 90% 宽、80% 高
            panel_width = int(rw * 1)
            panel_height = int(rh * 0.6)

            # 居中偏移
            offset_x = (rw - panel_width) // 20
            offset_y = (rh - panel_height) // 25

            self.color_panel.setGeometry(
                int(rx + offset_x),
                int(ry + offset_y),
                panel_width,
                panel_height
            )

        self.timer = QTimer(self)
        self.timer.timeout.connect(sync_geometry)
        self.timer.start(100)  # 每 100ms 同步一次位置


def main():
    """
    Main function to initialize and run the application.
    Displays loading screen and then shows main window.
    """
    app = QApplication(sys.argv)

    loading = LoadingScreen()
    loading.finished.connect(lambda: show_main_window(app))
    loading.show()

    sys.exit(app.exec())


def show_main_window(app):
    main_window = MainWindow()
    main_window.setWindowIcon(QIcon("icons/logo.png"))
    main_window.show()
    app.main_window = main_window  # 防止被垃圾回收


if __name__ == "__main__":
    main()
