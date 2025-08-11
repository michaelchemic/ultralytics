import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QApplication


class DefectColorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("病害颜色面板")
        self.setStyleSheet("color: white; background-color: #1c3e52;")

        self.color_labels = {}  # 颜色标签
        self.clear_timers = {}  # 清除定时器

        self.structural_defects = ["SL(渗漏)", "AJ(支管暗接)", "CR(异物穿入)", "TL(接口材料脱落)",
                                   "TJ(脱节)", "QF(起伏)", "CK(错口)", "FS(腐蚀)", "BX(变形)", "PL(破裂)", "LF(裂缝)"]
        self.functional_defects = ["CJ(沉积物)", "JG(结垢)", "ZW(障碍物)", "CQ(残墙坝根)", "SG(树根)", "FZ(浮渣)","LJ(垃圾)"]

        # 正确初始化为字典
        self.color_labels = {}

        # 可选：缺陷别名映射
        self.alias_map = {
            "CJ": "CJ", "沉积物": "CJ", "sediment": "CJ",
            "JG": "JG", "结垢": "JG", "scaling": "JG",
            "ZW": "ZW", "障碍物": "ZW", "obstacle": "ZW",
            "CQ": "CQ", "残墙": "CQ", "坝根": "CQ", "residual wall": "CQ", "dam base": "CQ",
            "SG": "SG", "树根": "SG", "root": "SG",
            "SL": "SL", "渗漏": "SL", "leakage": "SL",
            "AJ": "AJ", "支管暗接": "AJ", "hidden branch": "AJ",
            "CR": "CR", "异物穿入": "CR", "foreign body": "CR","穿入": "CR",
            "TL": "TL", "接口材料脱落": "TL", "material loss": "TL",
            "TJ": "TJ", "脱节": "TJ", "dislocation": "TJ",
            "QF": "QF", "起伏": "QF", "unevenness": "QF",
            "CK": "CK", "错口": "CK", "misalignment": "CK",
            "FS": "FS", "腐蚀": "FS", "corrosion": "FS",
            "BX": "BX", "变形": "BX", "deformation": "BX",
            "PL": "PL", "破裂": "PL", "fracture": "PL",
            "LF":"LF" , "裂缝": "LF", "crack": "LF",
            "FZ": "FZ", "浮渣": "FZ", "scum": "FZ",
            "LJ": "LJ", "垃圾": "LJ", "garbage": "LJ"
        }

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(self.create_defect_group("结构性缺陷", self.structural_defects))
        main_layout.addWidget(self.create_defect_group("功能性缺陷", self.functional_defects))

    def create_defect_group(self, title, defect_list):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: white;
            }
            QLabel {
                color: white;
            }
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(5)

        for defect in defect_list:
            pure_name = defect.split("(")[0]  # 如 "SL"
            row = QHBoxLayout()

            # 左边颜色块
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet("background-color: gray; border: 1px solid white;")
            self.color_labels[pure_name] = color_label

            # 右边标签
            text_label = QLabel(defect)
            text_label.setStyleSheet("""
                color: white;
                font-size: 15px;
                font-family: 'Microsoft YaHei';
            """)

            row.addWidget(color_label)
            row.addSpacing(10)
            row.addWidget(text_label)
            layout.addLayout(row)

        group.setLayout(layout)
        return group

    # def update_defect_color(self, defect_name: str, color_hex: str):
    #     """
    #     槽函数：更新某个病害的小方块颜色
    #     """
    #     main_key = self.alias_map.get(defect_name, defect_name)
    #     if main_key in self.color_labels:
    #         self.color_labels[main_key].setStyleSheet(
    #             f"background-color: {color_hex}; border: 1px solid white;"
    #         )
    #     else:
    #         print(f"[警告] 未找到缺陷名: {defect_name}（映射后为 {main_key}）")


    def update_defect_color(self, defect_name: str, color_hex: str, duration_ms: int = 1000):
        """
        更新缺陷颜色并在一定时间后自动还原为灰色
        """
        main_key = self.alias_map.get(defect_name, defect_name)

        if main_key in self.color_labels:
            label = self.color_labels[main_key]
            label.setStyleSheet(f"background-color: {color_hex}; border: 1px solid white;")

            # 如果已有清除定时器，先取消
            if main_key in self.clear_timers:
                self.clear_timers[main_key].stop()

            # 设置定时器，延迟 duration_ms 后清除颜色
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda key=main_key: self.clear_color(key))
            timer.start(duration_ms)

            self.clear_timers[main_key] = timer
        else:
            print(f"[警告] 未找到缺陷名: {defect_name}（映射后为 {main_key}）")


    def clear_color(self, key):
        """
        将指定颜色块重置为灰色
        """
        if key in self.color_labels:
            self.color_labels[key].setStyleSheet("background-color: gray; border: 1px solid white;")
            # 移除定时器引用
            if key in self.clear_timers:
                self.clear_timers[key].deleteLater()
                del self.clear_timers[key]

# 测试入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = DefectColorPanel()
    panel.show()

    # 测试映射
    panel.update_defect_color("SL", "#ff0000")         # 简写
    panel.update_defect_color("渗漏", "#00ffff")       # 中文
    panel.update_defect_color("leakage", "#00ff00")    # 英文别名

    sys.exit(app.exec())
