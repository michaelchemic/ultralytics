import sys
import time
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QMainWindow
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


# 加载线程，模拟初始化任务
class LoaderThread(QThread):
    finished = pyqtSignal()

    def run(self):
        time.sleep(3)  # 模拟耗时加载
        self.finished.emit()


# 加载界面
class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("加载中")
        self.setFixedSize(300, 150)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout()
        label = QLabel("正在加载，请稍候...")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)


# 主程序窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主程序窗口")
        self.setGeometry(200, 200, 500, 400)
        label = QLabel("欢迎进入主程序！", self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(label)


def main():
    app = QApplication(sys.argv)

    # 显示加载界面
    loading = LoadingScreen()
    loading.show()

    # 加载线程
    loader = LoaderThread()

    def on_loaded():
        loading.close()
        main_window = MainWindow()
        main_window.show()
        app.main_window = main_window  # 防止被垃圾回收

    loader.finished.connect(on_loaded)
    loader.start()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
