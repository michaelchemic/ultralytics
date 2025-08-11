# dxf_editor_app.py
# PySide6 DXF 编辑器：可打开、编辑、保存 DXF，具有基础绘图功能，界面风格参考 AutoCAD

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QGraphicsView, QGraphicsScene,
    QToolBar, QStatusBar, QMessageBox
)
from PySide6.QtGui import QIcon, QActionGroup, QPainter, QAction
from PySide6.QtCore import Qt, QPointF
import ezdxf

class DxfEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DXF CAD 编辑器 (PySide6)")
        self.resize(1200, 800)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.setCentralWidget(self.view)

        self.dxf_doc = None
        self.current_file = None

        self._create_toolbar()
        self._create_statusbar()

    def _create_toolbar(self):
        toolbar = QToolBar("工具栏")
        self.addToolBar(toolbar)

        open_action = QAction(QIcon.fromTheme("document-open"), "打开 DXF", self)
        open_action.triggered.connect(self.open_dxf)
        toolbar.addAction(open_action)

        save_action = QAction(QIcon.fromTheme("document-save"), "保存 DXF", self)
        save_action.triggered.connect(self.save_dxf)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # 示例：画线工具（未来可扩展）
        self.draw_mode = None
        draw_line_action = QAction("画线", self)
        draw_line_action.setCheckable(True)
        draw_line_action.triggered.connect(lambda: self.set_draw_mode("line"))
        toolbar.addAction(draw_line_action)

        self.action_group = QActionGroup(self)
        self.action_group.setExclusive(True)
        self.action_group.addAction(draw_line_action)

    def _create_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪")

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        self.status_bar.showMessage(f"当前模式：{mode}")

    def open_dxf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开 DXF 文件", "", "DXF 文件 (*.dxf)")
        if file_name:
            try:
                self.dxf_doc = ezdxf.readfile(file_name)
                self.current_file = file_name
                self._load_to_scene()
                self.status_bar.showMessage(f"打开文件：{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法打开文件：{str(e)}")

    def _load_to_scene(self):
        self.scene.clear()
        msp = self.dxf_doc.modelspace()
        for e in msp:
            try:
                t = e.dxftype()
                if t == 'LINE':
                    start = QPointF(float(e.dxf.start[0]), float(e.dxf.start[1]))
                    end = QPointF(float(e.dxf.end[0]), float(e.dxf.end[1]))
                    self.scene.addLine(start.x(), -start.y(), end.x(), -end.y())
                elif t == 'LWPOLYLINE':
                    points = [QPointF(x, -y) for x, y in e.get_points()]
                    for i in range(len(points) - 1):
                        self.scene.addLine(points[i].x(), points[i].y(), points[i + 1].x(), points[i + 1].y())
                    if e.closed:
                        self.scene.addLine(points[-1].x(), points[-1].y(), points[0].x(), points[0].y())
                elif t == 'CIRCLE':
                    center = e.dxf.center
                    radius = e.dxf.radius
                    self.scene.addEllipse(center[0] - radius, -center[1] - radius, 2 * radius, 2 * radius)
                elif t == 'ELLIPSE':
                    center = e.dxf.center
                    major = e.dxf.major_axis
                    ratio = e.dxf.radius_ratio
                    rx = (major[0] ** 2 + major[1] ** 2) ** 0.5
                    ry = rx * ratio
                    self.scene.addEllipse(center[0] - rx, -center[1] - ry, 2 * rx, 2 * ry)
                elif t == 'ARC':
                    center = e.dxf.center
                    radius = e.dxf.radius
                    start_angle = e.dxf.start_angle
                    end_angle = e.dxf.end_angle
                    rect = (center[0] - radius, -center[1] - radius, 2 * radius, 2 * radius)
                    # PyQt的角度是16倍度数，从3点钟方向逆时针，DXF角度从X轴正方向顺时针
                    start = 360 - end_angle
                    span = end_angle - start_angle
                    self.scene.addArc(*rect, int(start * 16), int(span * 16))
            except Exception as ex:
                print(f"绘制实体失败: {e.dxftype()}，错误: {ex}")

    def save_dxf(self):
        if not self.dxf_doc:
            QMessageBox.warning(self, "未加载", "请先打开一个 DXF 文件")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "保存 DXF 文件", self.current_file or "output.dxf", "DXF 文件 (*.dxf)")
        if file_name:
            try:
                self.dxf_doc.saveas(file_name)
                self.status_bar.showMessage(f"保存文件：{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DxfEditor()
    win.show()
    sys.exit(app.exec())
