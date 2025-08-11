import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path

class DiseaseCK:
    def __init__(self, input_dir="TEST_PHOTO", output_dir="CK_debug_output"):
        """Initialize the detector with input and output directories.

        Args:
            input_dir (str): Directory containing input images. Defaults to 'TEST_PHOTO'.
            output_dir (str): Directory to save debug images and results. Defaults to 'CK_debug_output'.
        """
        self.input_dir = str(Path(input_dir))
        self.output_dir = str(Path(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_image(self, img_path):
        """Load image with support for UTF-8 paths."""
        img_path = str(Path(img_path))
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"❌ Unable to load image: {img_path}")
            return img
        except Exception as e:
            print(f"❌ Failed to load image {img_path}: {str(e)}")
            return None

    def _save_debug_image(self, image, name, img_name):
        """Save debug image to the specified directory."""
        save_path = os.path.join(self.output_dir, f"{img_name}_{name}.jpg")
        cv2.imwrite(save_path, image)
        return save_path

    def detect_deformation(self, img_path):
        """Detect pipe misalignment in the given image.

        Args:
            img_path (str): Path to the input image.

        Returns:
            tuple: (result dictionary, result image path, result text path)
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            raise FileNotFoundError(f"图像未找到: {img_path}")

        # 1. Original image
        self._save_debug_image(img, "01_original", img_name)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, "02_gray", img_name)

        # 3. Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._save_debug_image(blurred, "03_blurred", img_name)

        # 4. Binary thresholding
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        self._save_debug_image(binary, "04_binary", img_name)

        # 5. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("❌ 没有检测到任何轮廓")

        # Draw all contours for debugging
        all_contours_img = img.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 2)
        self._save_debug_image(all_contours_img, "05_all_contours", img_name)

        # 6. Select top two contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if len(contours) < 2:
            raise ValueError("❌ 检测到轮廓不足两个，无法计算错口")

        # Draw top contours
        top_contours_img = img.copy()
        cv2.drawContours(top_contours_img, contours, -1, (0, 255, 0), 2)
        for i, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(top_contours_img, f"Contour {i + 1}", (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        self._save_debug_image(top_contours_img, "06_top_contours", img_name)

        # 7. Fit ellipses
        ellipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
        if len(ellipses) < 2:
            raise ValueError("❌ 轮廓点不足，无法拟合两个椭圆")

        # Draw ellipses
        ellipse_img = img.copy()
        for i, ellipse in enumerate(ellipses):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Outer: green, Inner: red
            cv2.ellipse(ellipse_img, ellipse, color, 2)
            (cx, cy), (w, h), angle = ellipse
            text = f"Ellipse {i + 1}: ({cx:.1f},{cy:.1f}) {w:.1f}x{h:.1f} {angle:.1f}°"
            cv2.putText(ellipse_img, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self._save_debug_image(ellipse_img, "07_ellipses", img_name)

        # 8. Calculate parameters
        (cx1, cy1), (w1, h1), _ = ellipses[0]  # Outer ellipse
        (cx2, cy2), (w2, h2), _ = ellipses[1]  # Inner ellipse

        major_axis1, minor_axis1 = max(w1, h1), min(w1, h1)
        major_axis2, minor_axis2 = max(w2, h2), min(w2, h2)
        wall_thickness = min(abs(major_axis1 - major_axis2), abs(minor_axis1 - minor_axis2)) / 2
        misalignment = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        misalignment_ratio = misalignment / wall_thickness if wall_thickness > 0 else 0

        # 9. Visualize final result
        vis_img = img.copy()
        cv2.ellipse(vis_img, ellipses[0], (0, 255, 0), 2)  # Outer ellipse: green
        cv2.ellipse(vis_img, ellipses[1], (0, 0, 255), 2)  # Inner ellipse: red
        cv2.line(vis_img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)  # Center line

        # Add text information
        info_text = [
            f"Outer Ellipse: {w1:.1f}x{h1:.1f}px",
            f"Inner Ellipse: {w2:.1f}x{h2:.1f}px",
            f"Wall Thickness: {wall_thickness:.1f}px",
            f"Center Offset: {misalignment:.1f}px",
            f"Misalignment Ratio: {misalignment_ratio:.2f}x wall",
            f"Grade: {self.grade_misalignment(misalignment_ratio)}"
        ]
        for i, text in enumerate(info_text):
            cv2.putText(vis_img, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save result image
        result_img_path = os.path.join(self.output_dir, f"{img_name}_result.jpg")
        cv2.imwrite(result_img_path, vis_img)

        # Save report to text file
        report_text = ["检测报告:"]
        report_text.extend(info_text)
        result_txt_path = os.path.join(self.output_dir, f"{img_name}_result.txt")
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_text))

        print("\n🔍 调试信息:")
        print(f" - 外椭圆尺寸: {w1:.1f}x{h1:.1f}")
        print(f" - 内椭圆尺寸: {w2:.1f}x{h2:.1f}")
        print(f" - 计算壁厚: {wall_thickness:.1f} 像素")
        print(f" - 中心偏移距离: {misalignment:.1f} 像素")
        print(f"📐 检测结果 - 错口率: {misalignment_ratio:.2f}倍管壁厚")

        return {
            "image": img_path,
            "outer_ellipse": (w1, h1),
            "inner_ellipse": (w2, h2),
            "wall_thickness": wall_thickness,
            "misalignment": misalignment,
            "misalignment_ratio": misalignment_ratio,
            "level": self.grade_misalignment(misalignment_ratio),
            "result_image": result_img_path,
            "result_text": result_txt_path,
            "debug_images": [
                f"{img_name}_01_original.jpg",
                f"{img_name}_02_gray.jpg",
                f"{img_name}_03_blurred.jpg",
                f"{img_name}_04_binary.jpg",
                f"{img_name}_05_all_contours.jpg",
                f"{img_name}_06_top_contours.jpg",
                f"{img_name}_07_ellipses.jpg",
                f"{img_name}_result.jpg"
            ]
        }

    def grade_misalignment(self, ratio):
        """根据错口等级判断"""
        if ratio < 1.5:
            return "Ⅰ级（正常或轻微错口）"
        elif ratio < 2.0:
            return "Ⅱ级（轻中度错口）"
        else:
            return "Ⅲ级（严重错口）"

    def show_debug_images(self, img_name):
        """显示所有调试图像"""
        plt.figure(figsize=(15, 10))

        debug_images = [
            "01_original", "02_gray", "03_blurred", "04_binary",
            "05_all_contours", "06_top_contours", "07_ellipses", "result"
        ]

        for i, name in enumerate(debug_images, 1):
            img_path = os.path.join(self.output_dir, f"{img_name}_{name}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 4, i)
                plt.imshow(img_rgb)
                plt.title(name.replace("_", " ").title())
                plt.axis("off")

        plt.tight_layout()
        plt.show()

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

                result = self.detect_deformation(img_path)
                print(f"✅ 生成结果图像: {result['result_image']}")
                print(f"✅ 生成报告文本: {result['result_text']}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")

# 示例用法
if __name__ == "__main__":
    detector = DiseaseCK(input_dir="TEST_PHOTO")
    detector.process_directory()