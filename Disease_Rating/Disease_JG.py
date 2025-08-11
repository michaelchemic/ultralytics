import cv2
import numpy as np
import os
import glob
from pathlib import Path

class DiseaseJG:
    def __init__(self, input_dir="TEST_PHOTO", output_dir="JG_debug_output"):
        """Initialize the detector with input and output directories.

        Args:
            input_dir (str): Directory containing input images. Defaults to 'TEST_PHOTO'.
            output_dir (str): Directory to save debug images and results. Defaults to 'JG_debug_output'.
        """
        self.input_dir = str(Path(input_dir))
        self.output_dir = str(Path(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_image(self, img_path, grayscale=False):
        """Load image with support for UTF-8 paths."""
        img_path = str(Path(img_path))
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8),
                             cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"❌ Unable to load image: {img_path}")
            return img
        except Exception as e:
            print(f"❌ Failed to load image {img_path}: {str(e)}")
            return None

    def _save_debug_image(self, img, name):
        """Save debug image to the specified directory."""
        save_path = os.path.join(self.output_dir, name)
        if len(img.shape) == 2:  # Convert single-channel image to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(save_path, img)
        return save_path

    def detect_pipe(self, img_gray):
        """Detect pipe boundary using Hough Circle Transform.

        Args:
            img_gray: Grayscale image.

        Returns:
            tuple: (center_x, center_y, radius) of the detected pipe.
        """
        print("🔍 检测管道边界...")
        circles = cv2.HoughCircles(
            img_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=30,
            maxRadius=0
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            c = circles[0, 0]
            print(f"✅ 管道边界圆: center=({c[0]}, {c[1]}), radius={c[2]}")
            return c[0], c[1], c[2]
        else:
            raise RuntimeError("❌ 未能检测到管道边界圆")

    def detect_blockage(self, img_path):
        """Detect pipe blockage and calculate its ratio, save result image and text report.

        Args:
            img_path (str): Path to the input image.

        Returns:
            tuple: (result dictionary, result image path, result text path)
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            raise ValueError("无法加载图像")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, f"{img_name}_01_gray.jpg")

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cx, cy, r = self.detect_pipe(blurred)

        # Create pipe circular mask
        mask_pipe = np.zeros_like(gray)
        cv2.circle(mask_pipe, (cx, cy), r, 255, -1)
        self._save_debug_image(mask_pipe, f"{img_name}_02_pipe_mask.jpg")

        # Threshold to segment blockage area (dark regions)
        _, mask_scale = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        self._save_debug_image(mask_scale, f"{img_name}_03_scale_mask.jpg")
        scale_only = cv2.bitwise_and(mask_scale, mask_pipe)
        self._save_debug_image(scale_only, f"{img_name}_04_scale_only.jpg")

        # Invert within pipe mask
        scale_only_inv = np.zeros_like(scale_only)
        scale_only_inv[mask_pipe > 0] = 255 - scale_only[mask_pipe > 0]

        # Calculate areas
        pipe_area = np.count_nonzero(mask_pipe)
        scale_area = np.count_nonzero(scale_only_inv)
        loss_ratio = scale_area / pipe_area if pipe_area > 0 else 0

        # Grade defect
        score, level = self.grade_defect(loss_ratio)

        # Visualize result
        debug_img = img.copy()
        cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)  # Green pipe boundary

        # Create transparent green overlay for blockage
        overlay = debug_img.copy()
        green_color = (0, 255, 0)  # Green
        alpha = 0.4  # Transparency
        mask_combined = np.logical_and(scale_only_inv > 0, mask_pipe > 0)
        overlay[mask_combined] = green_color
        cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)

        # Add text annotations
        info = [
            f"Blockage Ratio: {loss_ratio:.2%}",
            f"Pipe Diameter: {2 * r}px",
            f"Defect Level: {level}",
            f"Score: {score}"
        ]
        for i, text in enumerate(info):
            cv2.putText(debug_img, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save result image
        result_img_path = os.path.join(self.output_dir, f"{img_name}_result.jpg")
        cv2.imwrite(result_img_path, debug_img)

        # Save report to text file
        report_text = ["检测报告:"]
        report_text.extend(info)
        result_txt_path = os.path.join(self.output_dir, f"{img_name}_result.txt")
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_text))

        print(f"📊 管道结构结垢率: {loss_ratio:.2%}, 等级: {level}, 评分: {score}")

        return {
            "image": img_path,
            "blockage_ratio": loss_ratio,
            "level": level,
            "score": score,
            "pipe_diameter": 2 * r,
            "result_image": result_img_path,
            "result_text": result_txt_path,
            "debug_images": [
                f"{img_name}_01_gray.jpg",
                f"{img_name}_02_pipe_mask.jpg",
                f"{img_name}_03_scale_mask.jpg",
                f"{img_name}_04_scale_only.jpg",
                f"{img_name}_result.jpg"
            ]
        }

    def grade_defect(self, ratio):
        """Grade the defect based on blockage ratio."""
        if ratio <= 0.15:
            return 0.5, 1
        elif ratio <= 0.25:
            return 2, 2
        elif ratio <= 0.5:
            return 5, 3
        else:
            return 10, 4

    def process_directory(self):
        """Process all JPG and PNG images in the input directory."""
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

                result = self.detect_blockage(img_path)
                print(f"✅ 生成结果图像: {result['result_image']}")
                print(f"✅ 生成报告文本: {result['result_text']}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")

# Example usage
if __name__ == "__main__":
    detector = DiseaseJG(input_dir="TEST_PHOTO")
    detector.process_directory()

# import cv2
# import numpy as np
# import os
# import subprocess
# import matplotlib.pyplot as plt
#
# class DiseaseJG:
#     def __init__(self, debug_dir="JG_debug_output"):
#         self.debug_dir = debug_dir
#         os.makedirs(debug_dir, exist_ok=True)
#
#     def detect_pipe(self, img_gray):
#         print("🔍 检测管道边界...")
#         circles = cv2.HoughCircles(
#             img_gray,
#             cv2.HOUGH_GRADIENT,
#             dp=1.2,
#             minDist=50,
#             param1=100,
#             param2=30,
#             minRadius=30,
#             maxRadius=0
#         )
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             c = circles[0, 0]
#             print(f"✅ 管道边界圆: center=({c[0]}, {c[1]}), radius={c[2]}")
#             return c[0], c[1], c[2]
#         else:
#             raise RuntimeError("❌ 未能检测到管道边界圆")
#
#     def calculate_blockage_ratio(self, img_path):
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("无法加载图像")
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         cx, cy, r = self.detect_pipe(blurred)
#
#         # 创建管道圆形掩膜
#         mask_pipe = np.zeros_like(gray)
#         cv2.circle(mask_pipe, (cx, cy), r, 255, -1)
#
#         # 阈值分割出结垢区域（暗区域）
#         _, mask_scale = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
#         scale_only = cv2.bitwise_and(mask_scale, mask_pipe)
#
#         # 取反（仅限管道内区域）
#         scale_only_inv = np.zeros_like(scale_only)
#         scale_only_inv[mask_pipe > 0] = 255 - scale_only[mask_pipe > 0]
#
#         # 仅统计管道圆内区域的像素
#         pipe_area = np.count_nonzero(mask_pipe)
#         scale_area = np.count_nonzero(scale_only_inv)
#         loss_ratio = scale_area / pipe_area
#
#         # 绘制调试图像
#         debug_img = img.copy()
#         cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)
#         cv2.putText(debug_img, f"Blockage: {loss_ratio:.2%}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#         # 创建透明绿色图层
#         overlay = debug_img.copy()
#         green_color = (0, 255, 0)  # 绿色
#         alpha = 0.4  # 透明度
#
#         # 只在管道掩膜内叠加绿色
#         mask_combined = np.logical_and(scale_only_inv > 0, mask_pipe > 0)
#         overlay[mask_combined] = green_color
#
#         # 混合图像（透明叠加）
#         cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)
#
#         # 保存调试图像
#         cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_01_gray.jpg"), gray)
#         cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_02_pipe_mask.jpg"), mask_pipe)
#         cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_03_scale_mask.jpg"), mask_scale)
#         cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_04_scale_only.jpg"), scale_only)
#         cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_05_debug.jpg"), debug_img)
#
#
#         # 判定等级和评分
#         score, level = self.grade_defect(loss_ratio)
#
#         print(f"📊 管道结构结垢率: {loss_ratio:.2%}, 等级: {level}, 评分: {score}")
#
#         return {
#             "image": img_path,
#             "blockage_ratio": loss_ratio,
#             "level": level,
#             "score": score
#         }
#
#     def grade_defect(self, ratio):
#         if ratio <= 0.15:
#             return 0.5, 1
#         elif ratio <= 0.25:
#             return 2, 2
#         elif ratio <= 0.5:
#             return 5, 3
#         else:
#             return 10, 4
#
#     def read_photo(img_path):
#         # 读取图像（以 BGR 格式）
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"图像未找到: {img_path}")
#
#         # 将 BGR 转为 RGB（matplotlib 正确显示颜色）
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         # 显示图像
#         plt.figure(figsize=(8, 6))
#         plt.imshow(img_rgb)
#         plt.title("result")
#         plt.axis("off")
#         plt.show()
#
#
