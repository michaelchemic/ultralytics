import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


class PipeDeformationDetector:
    def __init__(self, debug_dir="CK_debug_output"):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    def _save_debug_image(self, image, name, img_name):
        """保存调试图像到指定目录"""
        save_path = os.path.join(self.debug_dir, f"{img_name}_{name}.jpg")
        cv2.imwrite(save_path, image)
        return save_path

    def detect_deformation(self, img_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图像未找到: {img_path}")

        # 1. 原始图像
        self._save_debug_image(img, "01_original", img_name)

        # 2. 灰度化处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, "02_gray", img_name)

        # 3. 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._save_debug_image(blurred, "03_blurred", img_name)

        # 4. 二值化处理
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        self._save_debug_image(binary, "04_binary", img_name)

        # 5. 轮廓检测
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("❌ 没有检测到任何轮廓")

        # 绘制所有轮廓（调试用）
        all_contours_img = img.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 2)
        self._save_debug_image(all_contours_img, "05_all_contours", img_name)

        # 6. 筛选最大两个轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        if len(contours) < 2:
            raise ValueError("❌ 检测到轮廓不足两个，无法计算错口")

        # 绘制筛选后的轮廓（调试用）
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

        # 7. 椭圆拟合
        ellipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
        if len(ellipses) < 2:
            raise ValueError("❌ 轮廓点不足，无法拟合两个椭圆")

        # 绘制椭圆拟合结果（调试用）
        ellipse_img = img.copy()
        for i, ellipse in enumerate(ellipses):
            color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 外椭圆绿色，内椭圆红色
            cv2.ellipse(ellipse_img, ellipse, color, 2)

            # 显示椭圆参数
            (cx, cy), (w, h), angle = ellipse
            text = f"Ellipse {i + 1}: ({cx:.1f},{cy:.1f}) {w:.1f}x{h:.1f} {angle:.1f}°"
            cv2.putText(ellipse_img, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self._save_debug_image(ellipse_img, "07_ellipses", img_name)

        # 8. 计算参数
        (cx1, cy1), (w1, h1), _ = ellipses[0]  # 外椭圆
        (cx2, cy2), (w2, h2), _ = ellipses[1]  # 内椭圆

        major_axis1, minor_axis1 = max(w1, h1), min(w1, h1)
        major_axis2, minor_axis2 = max(w2, h2), min(w2, h2)
        wall_thickness = min(abs(major_axis1 - major_axis2), abs(minor_axis1 - minor_axis2)) / 2

        misalignment = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        misalignment_ratio = misalignment / wall_thickness

        # 9. 最终可视化
        vis_img = img.copy()
        cv2.ellipse(vis_img, ellipses[0], (0, 255, 0), 2)  # 外椭圆绿色
        cv2.ellipse(vis_img, ellipses[1], (0, 0, 255), 2)  # 内椭圆红色

        # 绘制中心线
        cv2.line(vis_img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)

        # 添加详细参数信息
        info_text = [
            f"Outer Ellipse: {w1:.1f}x{h1:.1f}",
            f"Inner Ellipse: {w2:.1f}x{h2:.1f}",
            f"Wall Thickness: {wall_thickness:.1f}px",
            f"Center Offset: {misalignment:.1f}px",
            f"Misalignment Ratio: {misalignment_ratio:.2f}x wall",
            f"Grade: {self.grade_misalignment(misalignment_ratio)}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(vis_img, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        result_path = self._save_debug_image(vis_img, "08_final_result", img_name)

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
            "result_image": result_path,
            "debug_images": [
                f"{img_name}_01_original.jpg",
                f"{img_name}_02_gray.jpg",
                f"{img_name}_03_blurred.jpg",
                f"{img_name}_04_binary.jpg",
                f"{img_name}_05_all_contours.jpg",
                f"{img_name}_06_top_contours.jpg",
                f"{img_name}_07_ellipses.jpg",
                f"{img_name}_08_final_result.jpg"
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
            "05_all_contours", "06_top_contours", "07_ellipses", "08_final_result"
        ]

        for i, name in enumerate(debug_images, 1):
            img_path = os.path.join(self.debug_dir, f"{img_name}_{name}.jpg")
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 4, i)
                plt.imshow(img_rgb)
                plt.title(name.replace("_", " ").title())
                plt.axis("off")

        plt.tight_layout()
        plt.show()


# # 示例运行
# if __name__ == '__main__':
#     detector = PipeDeformationDetector()
#
#     # 替换为您的图像路径
#     image_path = "TEST_PHOTO/CK1.png"
#     img_name = os.path.splitext(os.path.basename(image_path))[0]
#
#     try:
#         result = detector.detect_deformation(image_path)
#         print("\n📊 最终结果:")
#         for k, v in result.items():
#             if k != "debug_images":
#                 print(f"{k}: {v}")
#
#         # 显示最终结果
#         detector.show_debug_images(img_name)
#
#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {str(e)}")