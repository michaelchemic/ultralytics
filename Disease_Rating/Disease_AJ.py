import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
from pathlib import Path
try:
    import win32com.client
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

class DiseaseAJ:
    def __init__(self, input_dir, output_dir):
        self.input_dir = str(Path(input_dir))
        self.output_dir = str(Path(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipe_diameter = None
        self.defect_standards = {
            "AJ": {
                "phenomena": [
                    "接口位突出, 但主管未受损伤",
                    "接口位突出, 且主管受损出现裂痕",
                    "接口位突出, 且主管受损出现破裂",
                    "支管未插入, 且主管受损出现破裂"
                ],
                "level_a": "支管未伸入主管内",
                "level_b": "支管伸入主管内的长度等于主管直径10%",
                "level_c": "支管伸入主管内的长度等于主管直径20%",
                "measure_desc": "支管突出长度为管径的{:.1f}%"
            }
        }

    def _get_short_path(self, long_path):
        if not WIN32_AVAILABLE:
            return str(Path(long_path))
        try:
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(long_path + ".lnk")
            return shortcut.TargetPath
        except Exception as e:
            print(f"❌ 无法获取短路径: {e}")
            return str(Path(long_path))

    def _load_image(self, img_path, grayscale=False):
        img_path = str(Path(img_path))
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),
                             cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception as e:
            print(f"❌ UTF-8 加载失败: {e}")

        if WIN32_AVAILABLE:
            short_path = self._get_short_path(img_path)
            img = cv2.imread(short_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ 无法加载图像: {img_path} (短路径: {short_path})")
            return img
        else:
            print(f"❌ 无法加载图像: {img_path} (pywin32 未安装)")
            return None

    def _save_debug_image(self, image, step_name, img_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"{img_name}_{step_name}_{timestamp}.jpg")
        cv2.imwrite(save_path, image)
        return save_path

    def _auto_estimate_pipe_diameter(self, img_path):
        img_path = str(Path(img_path))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path, grayscale=True)
        if img is None:
            return None

        self._save_debug_image(img, "01_original_gray", img_name)
        img_blurred = cv2.medianBlur(img, 5)
        self._save_debug_image(img_blurred, "02_blurred", img_name)
        edges = cv2.Canny(img_blurred, 50, 150)
        self._save_debug_image(edges, "03_edges", img_name)

        circles = cv2.HoughCircles(
            img_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=int(img.shape[0] * 0.2),
            maxRadius=int(img.shape[0] * 0.6)
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_circle = max(circles[0], key=lambda item: item[2])
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_img, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
            cv2.putText(debug_img, f"Radius: {best_circle[2]}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self._save_debug_image(debug_img, "04_circle_detected", img_name)

            self.pipe_diameter = best_circle[2] * 2
            print(f"🔵 检测到主管道 - 半径: {best_circle[2]}px, 直径: {self.pipe_diameter}px")
            return best_circle

        print("🟠 未检测到主管道圆形")
        return None

    def _measure_intrusion_length(self, img, img_name):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        self._save_debug_image(mask, "05_branch_mask", img_name)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)
        self._save_debug_image(contour_img, "06_all_contours", img_name)

        min_area = (img.shape[0] * img.shape[1]) * 0.01
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        if not valid_contours:
            print("🟠 未检测到有效的支管轮廓")
            return 0

        branch_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(branch_contour)

        branch_img = img.copy()
        cv2.drawContours(branch_img, [branch_contour], -1, (0, 0, 255), 2)
        cv2.rectangle(branch_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self._save_debug_image(branch_img, "07_selected_branch", img_name)

        intrusion_px = w
        print(f"📏 测量支管突出长度: {intrusion_px:.1f}px")
        return intrusion_px

    def detect_defect(self, img_path):
        img_path = str(Path(img_path))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n🔍 开始分析图像: {img_name}")

        img = self._load_image(img_path)
        if img is None:
            print(f"❌ 无法加载图像: {img_path}")
            return {"error": f"无法加载图像: {img_path}"}

        self._save_debug_image(img, "00_original_color", img_name)
        circle_params = self._auto_estimate_pipe_diameter(img_path)
        if circle_params is None:
            return {"error": "无法识别主管道"}

        intrusion_px = self._measure_intrusion_length(img, img_name)
        intrusion_ratio = (intrusion_px / self.pipe_diameter) * 100

        conclusion = self._classify_defect(intrusion_px, intrusion_ratio)
        debug_report = {
            "pipe_diameter_px": float(self.pipe_diameter),
            "branch_intrusion_px": float(intrusion_px),
            "intrusion_ratio_percent": float(intrusion_ratio),
            "debug_images": [
                f for f in os.listdir(self.output_dir)
                if f.startswith(img_name) and f.endswith(".jpg")
            ]
        }
        conclusion.update(debug_report)

        return conclusion

    def _classify_defect(self, intrusion_px, intrusion_ratio):
        intrusion_px = float(intrusion_px)
        intrusion_ratio = float(intrusion_ratio)
        conclusion = {}

        if intrusion_px <= 0:
            conclusion["code"] = "AJ"
            conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][3]
            conclusion["description"] = self.defect_standards["AJ"]["level_a"]
            conclusion["location_size"] = "支管未插入"
        elif 0 < intrusion_ratio <= 10:
            conclusion["code"] = "AJ"
            conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
            conclusion["description"] = self.defect_standards["AJ"]["level_b"]
            conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
        elif 10 < intrusion_ratio < 20:
            conclusion["code"] = "AJ"
            conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
            conclusion["description"] = f"支管伸入长度介于10%-20%之间"
            conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
        elif intrusion_ratio >= 20:
            conclusion["code"] = "AJ"
            conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
            conclusion["description"] = self.defect_standards["AJ"]["level_c"]
            conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
        else:
            conclusion["conclusion"] = "未检测到明显的支管暗接缺陷。"

        return conclusion

    def visualize(self, img_path, result):
        img_path = str(Path(img_path))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            print(f"❌ 无法加载图像用于可视化: {img_path}")
            return None, None

        circle_params = self._auto_estimate_pipe_diameter(img_path)
        if circle_params is not None:
            (x, y, r) = circle_params
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.putText(img, f"Main Pipe (D={self.pipe_diameter}px)",
                        (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        text_y = 30
        report_text = ["检测报告:"]
        for key, value in result.items():
            if key not in ["debug_images"]:
                text = f"{key}: {float(value):.2f}" if isinstance(value, (np.floating, np.integer)) else f"{key}: {value}"
                cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                report_text.append(text)
                text_y += 25

        result_filename = f"{img_name}_result.jpg"
        final_path = os.path.join(self.output_dir, result_filename)
        cv2.imwrite(final_path, img)

        text_filename = f"{img_name}_result.txt"
        text_path = os.path.join(self.output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_text))

        self._display_debug_images(img_name)
        return final_path, text_path

    def _display_debug_images(self, img_name):
        debug_files = [f for f in os.listdir(self.output_dir) if f.startswith(img_name)]
        if not debug_files:
            return

        debug_files.sort()
        plt.figure(figsize=(15, 10))
        for i, filename in enumerate(debug_files[:8]):
            img = cv2.imread(os.path.join(self.output_dir, filename))
            if img is not None:
                plt.subplot(2, 4, i + 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(' '.join(filename.split('_')[1:3]))
                plt.axis('off')

        plt.tight_layout()
        plt.show()

    def process_directory(self):
        from pathlib import Path
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))

        if not image_files:
            print(f"❌ 输入目录 {self.input_dir} 中未找到图像文件")
            return

        print(f"找到的图像文件: {image_files}")
        for img_path in image_files:
            try:
                img_path = str(Path(img_path))
                print(f"\n📷 处理图像: {img_path}")
                if not os.path.exists(img_path):
                    print(f"❌ 文件不存在: {img_path}")
                    continue

                analysis_result = self.detect_defect(img_path)
                if "error" in analysis_result:
                    print(f"❌ 错误: {analysis_result['error']}")
                    continue

                result_img_path, result_txt_path = self.visualize(img_path, analysis_result)
                print(f"✅ 生成结果图像: {result_img_path}")
                print(f"✅ 生成报告文本: {result_txt_path}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")