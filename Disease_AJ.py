import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class PipeJointDetector:
    """
    A class to detect and analyze pipe joint defects (AJ) with enhanced debugging.
    """

    def __init__(self, output_dir="AJ_debug_output"):
        self.output_dir = output_dir
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

    def _save_debug_image(self, image, step_name, img_name):
        """Save debug image with timestamp and step info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"{img_name}_{step_name}_{timestamp}.jpg")
        cv2.imwrite(save_path, image)
        return save_path

    def _auto_estimate_pipe_diameter(self, img_path):
        """
        Automatically estimates the main pipe diameter with debug outputs.
        Returns the circle parameters (x, y, radius) or None if not found.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Step 1: Original grayscale
        self._save_debug_image(img, "01_original_gray", img_name)

        # Step 2: Apply blur
        img_blurred = cv2.medianBlur(img, 5)
        self._save_debug_image(img_blurred, "02_blurred", img_name)

        # Step 3: Edge detection (for debugging)
        edges = cv2.Canny(img_blurred, 50, 150)
        self._save_debug_image(edges, "03_edges", img_name)

        # Step 4: Hough Circle Transform
        circles = cv2.HoughCircles(
            img_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=int(img.shape[0] * 0.2),
            maxRadius=int(img.shape[0] * 0.6)
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_circle = max(circles[0], key=lambda item: item[2])

            # Visualize the detected circle
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
        """
        Enhanced version with debug outputs to measure branch pipe intrusion.
        """
        # Step 1: Convert to HSV and extract dark regions (potential branch pipe)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        self._save_debug_image(mask, "05_branch_mask", img_name)

        # Step 2: Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)
        self._save_debug_image(contour_img, "06_all_contours", img_name)

        # Step 3: Filter contours by area and position
        min_area = (img.shape[0] * img.shape[1]) * 0.01  # At least 1% of image area
        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                valid_contours.append(cnt)

        if not valid_contours:
            print("🟠 未检测到有效的支管轮廓")
            return 0

        # Step 4: Find the most likely branch pipe contour
        branch_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(branch_contour)

        # Visualize the selected branch
        branch_img = img.copy()
        cv2.drawContours(branch_img, [branch_contour], -1, (0, 0, 255), 2)
        cv2.rectangle(branch_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        self._save_debug_image(branch_img, "07_selected_branch", img_name)

        # Step 5: Calculate intrusion length (simplified for demo)
        intrusion_px = w  # Using width as proxy for intrusion length
        print(f"📏 测量支管突出长度: {intrusion_px:.1f}px")
        return intrusion_px

    def detect_defect(self, img_path):
        """
        Detects the AJ defect with detailed debug outputs.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n🔍 开始分析图像: {img_name}")

        # Step 1: Load and verify image
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "无法加载图像"}

        self._save_debug_image(img, "00_original_color", img_name)

        # Step 2: Detect main pipe
        circle_params = self._auto_estimate_pipe_diameter(img_path)
        if circle_params is None:
            return {"error": "无法识别主管道"}

        # Step 3: Measure branch intrusion
        intrusion_px = self._measure_intrusion_length(img, img_name)
        intrusion_ratio = (intrusion_px / self.pipe_diameter) * 100

        # Step 4: Classify defect
        conclusion = self._classify_defect(intrusion_px, intrusion_ratio)

        # Step 5: Generate debug report
        debug_report = {
            "pipe_diameter_px": self.pipe_diameter,
            "branch_intrusion_px": intrusion_px,
            "intrusion_ratio_percent": intrusion_ratio,
            "debug_images": [
                f for f in os.listdir(self.output_dir)
                if f.startswith(img_name) and f.endswith(".jpg")
            ]
        }
        conclusion.update(debug_report)

        return conclusion

    def _classify_defect(self, intrusion_px, intrusion_ratio):
        """Classify the defect based on measurements"""
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

    def visualize(self, img_path, result, save_name="result.jpg"):
        """
        Enhanced visualization with debug information.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)

        # Draw main pipe circle
        circle_params = self._auto_estimate_pipe_diameter(img_path)
        if circle_params:
            (x, y, r) = circle_params
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.putText(img, f"Main Pipe (D={self.pipe_diameter}px)",
                        (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add analysis results
        text_y = 30
        for key, value in result.items():
            if key not in ["debug_images"]:  # Skip debug images list
                text = f"{key}: {value}" if not isinstance(value, float) else f"{key}: {value:.2f}"
                cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                text_y += 25

        # Save and display
        final_path = os.path.join(self.output_dir, save_name)
        cv2.imwrite(final_path, img)

        # Display all debug images in a grid
        self._display_debug_images(img_name)

        return final_path

    def _display_debug_images(self, img_name):
        """Display all debug images in a grid layout"""
        debug_files = [f for f in os.listdir(self.output_dir) if f.startswith(img_name)]
        if not debug_files:
            return

        # Sort files by step number
        debug_files.sort()

        plt.figure(figsize=(15, 10))
        for i, filename in enumerate(debug_files[:8]):  # Show up to 8 images
            img = cv2.imread(os.path.join(self.output_dir, filename))
            if img is not None:
                plt.subplot(2, 4, i + 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(filename.split('_')[1:3])
                plt.axis('off')

        plt.tight_layout()
        plt.show()


# 示例用法
if __name__ == "__main__":
    detector = PipeJointDetector()
    image_path = "TEST_PHOTO/AJ_20.png"  # 替换为您的图像路径

    try:
        analysis_result = detector.detect_defect(image_path)
        print("\n📊 检测报告:")
        for key, value in analysis_result.items():
            if key != "debug_images":
                print(f"{key}: {value}")

        detector.visualize(image_path, analysis_result)

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")