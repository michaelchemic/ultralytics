import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from pathlib import Path

class DiseaseBX:
    def __init__(self, input_dir="TEST_PHOTO", output_dir="BX_debug_output"):
        """Initialize the detector with input and debug output directories."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_debug_image(self, img, name, cmap=None):
        """Save debug image to the specified directory."""
        save_path = os.path.join(self.output_dir, name)
        if cmap == 'gray':
            plt.imsave(save_path, img, cmap='gray')
        else:
            if len(img.shape) == 2:  # Convert single-channel image to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(save_path, img)

    def detect_deformation(self, img_path):
        """Detect pipe deformation in the given image."""
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Save original image
        self._save_debug_image(img, f"{img_name}_00_original.jpg")

        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')

        # 2. Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._save_debug_image(blurred, f"{img_name}_02_blurred.jpg", 'gray')

        # 3. Binary thresholding
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        self._save_debug_image(binary, f"{img_name}_03_binary.jpg", 'gray')

        # 4. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("❌ No contours detected")

        # Draw all contours for debugging
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        self._save_debug_image(contour_img, f"{img_name}_04_all_contours.jpg")

        # 5. Select largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            raise ValueError("❌ Insufficient contour points for ellipse fitting")

        # Draw largest contour
        max_contour_img = img.copy()
        cv2.drawContours(max_contour_img, [contour], -1, (0, 255, 0), 2)
        self._save_debug_image(max_contour_img, f"{img_name}_05_max_contour.jpg")

        # 6. Fit ellipse
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (w, h), angle = ellipse
        major_axis = max(w, h)
        minor_axis = min(w, h)
        deformation_ratio = (major_axis - minor_axis) / major_axis

        # Grade deformation
        level = self.grade_deformation(deformation_ratio)
        print(f"📐 Detection Result - Deformation Ratio: {deformation_ratio:.2%}, Level: {level}")

        return {
            "image": img_path,
            "deformation_ratio": deformation_ratio,
            "level": level,
            "ellipse": ellipse,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "angle": angle,
            "debug_images": [
                f"{img_name}_00_original.jpg",
                f"{img_name}_01_gray.jpg",
                f"{img_name}_02_blurred.jpg",
                f"{img_name}_03_binary.jpg",
                f"{img_name}_04_all_contours.jpg",
                f"{img_name}_05_max_contour.jpg"
            ]
        }

    def grade_deformation(self, ratio):
        """Grade deformation based on ratio."""
        if ratio < 0.05:
            return "Level I (Normal)"
        elif ratio < 0.15:
            return "Level II (Minor Deformation)"
        elif ratio < 0.25:
            return "Level III (Moderate Deformation)"
        else:
            return "Level IV (Severe Deformation)"

    def visualize(self, img_path, result):
        """Visualize the deformation detection results and save report."""
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found for visualization: {img_path}")

        ellipse = result["ellipse"]
        (cx, cy), (w, h), angle = ellipse
        major_axis = result["major_axis"]
        minor_axis = result["minor_axis"]

        # Draw ellipse
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # Draw major axis (green)
        major_length = major_axis / 2
        angle_rad = np.deg2rad(angle + 90)
        dx_major = int(major_length * np.cos(angle_rad))
        dy_major = int(major_length * np.sin(angle_rad))
        cv2.line(img,
                 (int(cx - dx_major), int(cy - dy_major)),
                 (int(cx + dx_major), int(cy + dy_major)),
                 (0, 255, 0), 2)

        # Draw minor axis (red)
        minor_length = minor_axis / 2
        dx_minor = int(minor_length * np.cos(angle_rad + np.pi / 2))
        dy_minor = int(minor_length * np.sin(angle_rad + np.pi / 2))
        cv2.line(img,
                 (int(cx - dx_minor), int(cy - dy_minor)),
                 (int(cx + dx_minor), int(cy + dy_minor)),
                 (0, 0, 255), 2)

        # Add text information
        info = [
            f"Deformation: {result['deformation_ratio']:.2%}",
            f"Level: {result['level']}",
            f"Major Axis: {major_axis:.1f}px",
            f"Minor Axis: {minor_axis:.1f}px",
            f"Angle: {angle:.1f}°"
        ]
        for i, text in enumerate(info):
            cv2.putText(img, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save result image
        result_img_path = os.path.join(self.output_dir, f"{img_name}_result.jpg")
        cv2.imwrite(result_img_path, img)

        # Save report to text file
        report_text = ["检测报告:"]
        report_text.extend(info)
        result_txt_path = os.path.join(self.output_dir, f"{img_name}_result.txt")
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_text))

        return result_img_path, result_txt_path

    def process_directory(self):
        """Process all JPG and PNG images in the input directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))

        if not image_files:
            print(f"❌ 输入目录 {self.input_dir} 中未找到图像文件")
            return

        for img_path in image_files:
            try:
                print(f"\n📷 处理图像: {img_path}")
                analysis_result = self.detect_deformation(img_path)

                result_img_path, result_txt_path = self.visualize(img_path, analysis_result)
                print(f"✅ 生成结果图像: {result_img_path}")
                print(f"✅ 生成报告文本: {result_txt_path}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")

    def show_debug_images(self, result):
        """Display all debug images."""
        plt.figure(figsize=(15, 10))

        images = []
        for debug_img in result['debug_images']:
            img_path = os.path.join(self.output_dir, debug_img)
            title = debug_img.split('_')[1].replace('.jpg', '')
            img = cv2.imread(img_path)
            if img is None:
                continue
            if "gray" in title or "blurred" in title or "binary" in title:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append((title, img, 'gray'))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append((title, img))

        for i, (title, img, *args) in enumerate(images):
            plt.subplot(3, 3, i + 1)
            if args and args[0] == 'gray':
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

# 示例用法
if __name__ == "__main__":
    detector = DiseaseBX(input_dir="TEST_PHOTO")
    detector.process_directory()
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
#
# class DiseaseBX:
#     def __init__(self, debug_dir="BX_debug_output"):
#         """Initialize the detector with a debug output directory."""
#         self.debug_dir = debug_dir
#         os.makedirs(debug_dir, exist_ok=True)
#
#     def _save_debug_image(self, img, name, cmap=None):
#         """Save debug image to the specified directory."""
#         if cmap == 'gray':
#             plt.imsave(os.path.join(self.debug_dir, name), img, cmap='gray')
#         else:
#             if len(img.shape) == 2:  # Convert single-channel image to BGR
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             cv2.imwrite(os.path.join(self.debug_dir, name), img)
#
#     def detect_deformation(self, img_path):
#         """Detect pipe deformation in the given image."""
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"Image not found: {img_path}")
#
#         # Save original image
#         self._save_debug_image(img, f"{img_name}_00_original.jpg")
#
#         # 1. Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')
#
#         # 2. Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         self._save_debug_image(blurred, f"{img_name}_02_blurred.jpg", 'gray')
#
#         # 3. Binary thresholding
#         _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
#         self._save_debug_image(binary, f"{img_name}_03_binary.jpg", 'gray')
#
#         # 4. Find contours
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             raise ValueError("❌ No contours detected")
#
#         # Draw all contours for debugging
#         contour_img = img.copy()
#         cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
#         self._save_debug_image(contour_img, f"{img_name}_04_all_contours.jpg")
#
#         # 5. Select largest contour
#         contour = max(contours, key=cv2.contourArea)
#         if len(contour) < 5:
#             raise ValueError("❌ Insufficient contour points for ellipse fitting")
#
#         # Draw largest contour
#         max_contour_img = img.copy()
#         cv2.drawContours(max_contour_img, [contour], -1, (0, 255, 0), 2)
#         self._save_debug_image(max_contour_img, f"{img_name}_05_max_contour.jpg")
#
#         # 6. Fit ellipse
#         ellipse = cv2.fitEllipse(contour)
#         (cx, cy), (w, h), angle = ellipse
#         major_axis = max(w, h)
#         minor_axis = min(w, h)
#         deformation_ratio = (major_axis - minor_axis) / major_axis
#
#         # 7. Visualize results
#         vis_img = img.copy()
#         cv2.ellipse(vis_img, ellipse, (0, 255, 0), 2)  # Draw ellipse
#
#         # Draw major axis (green)
#         major_length = major_axis / 2
#         angle_rad = np.deg2rad(angle + 90)
#         dx_major = int(major_length * np.cos(angle_rad))
#         dy_major = int(major_length * np.sin(angle_rad))
#         cv2.line(vis_img,
#                  (int(cx - dx_major), int(cy - dy_major)),
#                  (int(cx + dx_major), int(cy + dy_major)),
#                  (0, 255, 0), 2)
#
#         # Draw minor axis (red)
#         minor_length = minor_axis / 2
#         dx_minor = int(minor_length * np.cos(angle_rad + np.pi / 2))
#         dy_minor = int(minor_length * np.sin(angle_rad + np.pi / 2))
#         cv2.line(vis_img,
#                  (int(cx - dx_minor), int(cy - dy_minor)),
#                  (int(cx + dx_minor), int(cy + dy_minor)),
#                  (0, 0, 255), 2)
#
#         # Add text information
#         info = [
#             f"Deformation: {deformation_ratio:.2%}",
#             f"Major Axis: {major_axis:.1f}px",
#             f"Minor Axis: {minor_axis:.1f}px",
#             f"Angle: {angle:.1f}°"
#         ]
#         for i, text in enumerate(info):
#             cv2.putText(vis_img, text, (10, 30 + 30 * i),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         # Save final result
#         result_path = os.path.join(self.debug_dir, f"{img_name}_06_result.jpg")
#         cv2.imwrite(result_path, vis_img)
#
#         # Grade deformation
#         level = self.grade_deformation(deformation_ratio)
#         print(f"📐 Detection Result - Deformation Ratio: {deformation_ratio:.2%}, Level: {level}")
#
#         return {
#             "image": img_path,
#             "deformation_ratio": deformation_ratio,
#             "level": level,
#             "result_image": result_path,
#             "debug_images": [
#                 f"{img_name}_00_original.jpg",
#                 f"{img_name}_01_gray.jpg",
#                 f"{img_name}_02_blurred.jpg",
#                 f"{img_name}_03_binary.jpg",
#                 f"{img_name}_04_all_contours.jpg",
#                 f"{img_name}_05_max_contour.jpg",
#                 f"{img_name}_06_result.jpg"
#             ]
#         }
#
#     def grade_deformation(self, ratio):
#         """Grade deformation based on ratio."""
#         if ratio < 0.05:
#             return "Level I (Normal)"
#         elif ratio < 0.15:
#             return "Level II (Minor Deformation)"
#         elif ratio < 0.25:
#             return "Level III (Moderate Deformation)"
#         else:
#             return "Level IV (Severe Deformation)"
#
#     def show_debug_images(self, result):
#         """Display all debug images."""
#         plt.figure(figsize=(15, 10))
#
#         images = [
#             ("Original", cv2.cvtColor(cv2.imread(os.path.join(self.debug_dir, result['debug_images'][0])), cv2.COLOR_BGR2RGB)),
#             ("Gray", cv2.imread(os.path.join(self.debug_dir, result['debug_images'][1]), cv2.IMREAD_GRAYSCALE), 'gray'),
#             ("Blurred", cv2.imread(os.path.join(self.debug_dir, result['debug_images'][2]), cv2.IMREAD_GRAYSCALE), 'gray'),
#             ("Binary", cv2.imread(os.path.join(self.debug_dir, result['debug_images'][3]), cv2.IMREAD_GRAYSCALE), 'gray'),
#             ("All Contours", cv2.cvtColor(cv2.imread(os.path.join(self.debug_dir, result['debug_images'][4])), cv2.COLOR_BGR2RGB)),
#             ("Max Contour", cv2.cvtColor(cv2.imread(os.path.join(self.debug_dir, result['debug_images'][5])), cv2.COLOR_BGR2RGB)),
#             ("Result", cv2.cvtColor(cv2.imread(result['result_image']), cv2.COLOR_BGR2RGB))
#         ]
#
#         for i, (title, img, *args) in enumerate(images):
#             plt.subplot(3, 3, i + 1)
#             if args and args[0] == 'gray':
#                 plt.imshow(img, cmap='gray')
#             else:
#                 plt.imshow(img)
#             plt.title(title)
#             plt.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#
