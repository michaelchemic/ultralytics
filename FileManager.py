import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTreeView,
    QVBoxLayout, QHBoxLayout, QListView, QLabel, QSplitter, QLineEdit,
    QToolBar, QPushButton, QFileDialog, QMessageBox, QMenu, QAbstractItemView
)
from PyQt6.QtGui import QIcon, QAction, QPalette, QColor, QCursor, QFileSystemModel
from PyQt6.QtCore import Qt, QDir, QModelIndex, QSize


class MacStyleFileManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Finder - File Manager")
        self.setGeometry(200, 100, 1200, 700)
        self.setUnifiedTitleAndToolBarOnMac(True)

        self._init_ui()

    def _init_ui(self):
        self._setup_toolbar()
        self._setup_file_browser()
        self._apply_mac_style()

    def _setup_toolbar(self):
        toolbar = QToolBar("导航")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        self.back_action = QAction(QIcon.fromTheme("go-previous"), "返回", self)
        self.forward_action = QAction(QIcon.fromTheme("go-next"), "前进", self)
        self.up_action = QAction(QIcon.fromTheme("go-up"), "向上", self)

        self.path_edit = QLineEdit()
        self.path_edit.returnPressed.connect(self._go_to_path)

        refresh_action = QAction(QIcon.fromTheme("view-refresh"), "刷新", self)
        open_action = QAction(QIcon.fromTheme("document-open"), "打开", self)
        toggle_view_action = QAction(QIcon.fromTheme("view-list-icons"), "切换视图", self)

        self.back_action.triggered.connect(self._go_back)
        self.forward_action.triggered.connect(self._go_forward)
        self.up_action.triggered.connect(self._go_up)
        refresh_action.triggered.connect(self._refresh)
        open_action.triggered.connect(self._open_folder)
        toggle_view_action.triggered.connect(self._toggle_view_mode)

        toolbar.addAction(self.back_action)
        toolbar.addAction(self.forward_action)
        toolbar.addAction(self.up_action)
        toolbar.addWidget(self.path_edit)
        toolbar.addAction(refresh_action)
        toolbar.addAction(open_action)
        toolbar.addAction(toggle_view_action)

    def _setup_file_browser(self):
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.homePath()))
        self.tree.doubleClicked.connect(self._on_double_click)
        self.tree.setColumnWidth(0, 250)
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setDragEnabled(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        self.current_view = "tree"

        self.path_edit.setText(QDir.homePath())

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _apply_mac_style(self):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f5f5f7"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#007aff"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)

        self.setStyleSheet("""
            QTreeView, QListView {
                background-color: #ffffff;
                selection-background-color: #007aff;
                font-size: 14px;
            }
            QToolBar {
                background-color: #e5e5ea;
                border-bottom: 1px solid #ccc;
            }
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 4px 8px;
            }
        """)

    def _on_double_click(self, index: QModelIndex):
        if not self.model.isDir(index):
            file_path = self.model.filePath(index)
            os.startfile(file_path)
        else:
            self.tree.setRootIndex(index)
            self.path_edit.setText(self.model.filePath(index))

    def _go_to_path(self):
        path = self.path_edit.text()
        if os.path.exists(path):
            self.tree.setRootIndex(self.model.index(path))
        else:
            QMessageBox.warning(self, "路径错误", "该路径不存在")

    def _go_back(self):
        self.tree.setRootIndex(self.model.index(QDir.homePath()))
        self.path_edit.setText(QDir.homePath())

    def _go_forward(self):
        pass

    def _go_up(self):
        current_path = self.path_edit.text()
        parent_path = os.path.dirname(current_path)
        if os.path.exists(parent_path):
            self.tree.setRootIndex(self.model.index(parent_path))
            self.path_edit.setText(parent_path)

    def _refresh(self):
        current_path = self.path_edit.text()
        self.model.refresh(self.model.index(current_path))

    def _open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", QDir.homePath())
        if path:
            self.tree.setRootIndex(self.model.index(path))
            self.path_edit.setText(path)

    def _toggle_view_mode(self):
        current_index = self.tree.rootIndex()
        path = self.model.filePath(current_index)

        layout = self.centralWidget().layout()
        layout.takeAt(0).widget().deleteLater()

        if self.current_view == "tree":
            self.icon_view = QListView()
            self.icon_view.setViewMode(QListView.ViewMode.IconMode)
            self.icon_view.setModel(self.model)
            self.icon_view.setRootIndex(self.model.index(path))
            self.icon_view.setIconSize(QSize(64, 64))
            self.icon_view.setSpacing(10)
            self.icon_view.setMovement(QListView.Movement.Static)
            self.icon_view.setResizeMode(QListView.ResizeMode.Adjust)
            self.icon_view.doubleClicked.connect(self._on_double_click)
            self.icon_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.icon_view.setDragEnabled(True)
            self.icon_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.icon_view.customContextMenuRequested.connect(self._show_context_menu)
            layout.addWidget(self.icon_view)
            self.current_view = "icon"
        else:
            layout.addWidget(self.tree)
            self.current_view = "tree"

    def _show_context_menu(self, position):
        index = self.sender().indexAt(position)
        if not index.isValid():
            return

        file_path = self.model.filePath(index)
        menu = QMenu()
        open_action = QAction("打开", self)
        delete_action = QAction("删除", self)
        rename_action = QAction("重命名", self)
        open_in_explorer_action = QAction("在资源管理器中打开", self)

        open_action.triggered.connect(lambda: os.startfile(file_path))
        delete_action.triggered.connect(lambda: self._delete_file(file_path))
        rename_action.triggered.connect(lambda: self._rename_file(file_path))
        open_in_explorer_action.triggered.connect(lambda: os.system(f'explorer /select,"{file_path}"'))

        menu.addAction(open_action)
        menu.addAction(delete_action)
        menu.addAction(rename_action)
        menu.addSeparator()
        menu.addAction(open_in_explorer_action)
        menu.exec(QCursor.pos())

    def _delete_file(self, path):
        reply = QMessageBox.question(self, "确认删除", f"确定要删除：\n{path}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if os.path.isdir(path):
                    os.rmdir(path)
                else:
                    os.remove(path)
                self._refresh()
            except Exception as e:
                QMessageBox.critical(self, "删除失败", str(e))

    def _rename_file(self, path):
        name, ok = QFileDialog.getSaveFileName(self, "重命名为", path)
        if ok and name:
            try:
                os.rename(path, name)
                self._refresh()
            except Exception as e:
                QMessageBox.critical(self, "重命名失败", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    file_manager = MacStyleFileManager()
    file_manager.show()
    sys.exit(app.exec())