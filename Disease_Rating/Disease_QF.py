import cv2
import numpy as np
import os
import glob
from pathlib import Path

class DiseaseQF:
    """A class to detect water depth in pipe images and calculate water depth ratio."""

    def __init__(self, input_dir="TEST_PHOTO", output_dir="QF_debug_output", pipe_diameter_pixels=None):
        """Initialize the detector with input/output directories and optional pipe diameter.

        Args:
            input_dir (str): Directory containing input images. Defaults to 'TEST_PHOTO'.
            output_dir (str): Directory to save debug images and results. Defaults to 'QF_debug_output'.
            pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
        """
        self.input_dir = str(Path(input_dir))
        self.output_dir = str(Path(output_dir))
        self.pipe_diameter = pipe_diameter_pixels
        self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
        self.upper_black = np.array([10, 255, 10])  # Upper bound for black color in HSV
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

    def auto_estimate_diameter(self, img_path, method="auto"):
        """Automatically estimate the pipe diameter using Hough transform or projection method.

        Args:
            img_path (str): Path to the input image.
            method (str): Method to use ('hough', 'projection', or 'auto'). Defaults to 'auto'.

        Returns:
            int or None: Estimated diameter in pixels, or None if estimation fails.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path, grayscale=True)
        if img is None:
            print("❌ 无法加载图像，请检查文件路径")
            return None

        height, width = img.shape
        estimated_diameter = None

        # Method 1: Hough Circle Transform
        if method in ("hough", "auto"):
            img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
            self._save_debug_image(img_blurred, f"{img_name}_01_blurred_hough.jpg")
            circles = cv2.HoughCircles(
                img_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=width // 8,
                param1=100, param2=20, minRadius=50, maxRadius=width // 2
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(circles[0]) > 0:
                    estimated_diameter = 2 * circles[0][0][2]
                    print(f"[Hough] 估算管道直径: {estimated_diameter} px, 中心: ({circles[0][0][0]}, {circles[0][0][1]})")
                    hough_img = self._load_image(img_path)  # Load color image for visualization
                    cv2.circle(hough_img, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (0, 255, 0), 2)
                    self._save_debug_image(hough_img, f"{img_name}_02_hough_circle.jpg")
                    self.pipe_diameter = estimated_diameter
                    return estimated_diameter

        # Method 2: Projection Method
        if method in ("projection", "auto"):
            edges = cv2.Canny(img, 50, 150)
            self._save_debug_image(edges, f"{img_name}_01_canny_projection.jpg")
            projection = np.sum(edges, axis=1)
            threshold = np.max(projection) * 0.3
            rows = np.where(projection > threshold)[0]
            if len(rows) >= 2:
                top, bottom = min(rows), max(rows)
                estimated_diameter = bottom - top
                print(f"[Projection] 估算管道直径: {estimated_diameter} px")
                proj_img = self._load_image(img_path)  # Load color image for visualization
                cv2.line(proj_img, (0, top), (width, top), (0, 255, 0), 2)
                cv2.line(proj_img, (0, bottom), (width, bottom), (0, 255, 0), 2)
                self._save_debug_image(proj_img, f"{img_name}_02_projection_lines.jpg")
                self.pipe_diameter = estimated_diameter
                return estimated_diameter

        print("❌ 无法自动估算管道直径，请手动设置 pipe_diameter 或检查图像")
        return None

    def detect_water_depth(self, img_path):
        """Detect water depth in the pipe image, save result image and text report.

        Args:
            img_path (str): Path to the input image.

        Returns:
            tuple: (result dictionary, result image path, result text path)
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            raise ValueError("❌ 无法加载图像，请检查文件路径")

        # Estimate pipe diameter if not set
        if self.pipe_diameter is None:
            self.auto_estimate_diameter(img_path)

        if self.pipe_diameter is None:
            raise ValueError("❌ Pipe diameter not set and could not be estimated")

        height, width = img.shape[:2]
        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Convert to HSV and create water mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        water_mask_full = cv2.inRange(hsv, self.lower_black, self.upper_black)
        self._save_debug_image(water_mask_full, f"{img_name}_03_water_mask.jpg")

        # Create full circle mask
        circle_mask_full = np.zeros_like(water_mask_full, dtype=np.uint8)
        cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
        self._save_debug_image(circle_mask_full, f"{img_name}_04_circle_mask.jpg")

        # Find water level (top of water area)
        water_contours, _ = cv2.findContours(water_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_contours_img = cv2.cvtColor(water_mask_full, cv2.COLOR_GRAY2BGR)
        if water_contours:
            water_contour = max(water_contours, key=cv2.contourArea)
            water_level_y = np.min(water_contour[:, :, 1])  # Topmost point of water
            cv2.drawContours(debug_contours_img, [water_contour], -1, (0, 255, 0), 2)
            cv2.putText(debug_contours_img, f"Water Level: {water_level_y} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            water_level_y = height  # Default to bottom if no water detected
        self._save_debug_image(debug_contours_img, f"{img_name}_05_water_contours.jpg")
        print(f"🌊 检测到的水面高度: {water_level_y} px")

        # Pipe bottom y-coordinate
        pipe_bottom_y = center_y + radius

        # Actual water depth (pixels) = Pipe bottom - Water level
        water_depth_pixels = pipe_bottom_y - water_level_y
        water_depth_pixels = max(0, min(water_depth_pixels, self.pipe_diameter))

        # Water depth percentage (% of diameter)
        water_depth_ratio_percent = (water_depth_pixels / self.pipe_diameter) * 100
        water_depth_ratio_percent = max(0, min(water_depth_ratio_percent, 100.0))

        print(f"✅ 管道底部 y: {pipe_bottom_y}, 水深 (像素): {water_depth_pixels}, 水深比例 (直径%): {water_depth_ratio_percent:.2f}%")

        # Generate result dictionary
        result = self._generate_result(water_depth_ratio_percent)
        result['Water Depth (px)'] = water_depth_pixels
        result['Pipe Diameter (px)'] = self.pipe_diameter

        # Visualize result
        vis_img = img.copy()
        cv2.circle(vis_img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow pipe circle
        cv2.line(vis_img, (0, int(water_level_y)), (width, int(water_level_y)), (0, 255, 0), 2)  # Green water level
        cv2.line(vis_img, (center_x, int(pipe_bottom_y)), (center_x, int(water_level_y)), (0, 0, 255), 2)  # Red depth line

        # Add text annotations
        info = [
            f"Water Depth Ratio: {result['水深占比(直径%)']:.1f}%",
            f"Water Depth: {water_depth_pixels}px",
            f"Pipe Diameter: {self.pipe_diameter}px",
            f"Defect Level: {result['缺陷等级']}",
            f"Score: {result['分值']}"
        ]
        for i, text in enumerate(info):
            cv2.putText(vis_img, text, (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save result image
        result_img_path = os.path.join(self.output_dir, f"{img_name}_result.jpg")
        cv2.imwrite(result_img_path, vis_img)

        # Save report to text file
        report_text = ["检测报告:"]
        report_text.extend(info)
        result_txt_path = os.path.join(self.output_dir, f"{img_name}_result.txt")
        with open(result_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_text))

        # Add debug images to result
        result['debug_images'] = [
            f"{img_name}_01_blurred_hough.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_01_blurred_hough.jpg")) else None,
            f"{img_name}_02_hough_circle.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_02_hough_circle.jpg")) else None,
            f"{img_name}_01_canny_projection.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_01_canny_projection.jpg")) else None,
            f"{img_name}_02_projection_lines.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_02_projection_lines.jpg")) else None,
            f"{img_name}_03_water_mask.jpg",
            f"{img_name}_04_circle_mask.jpg",
            f"{img_name}_05_water_contours.jpg",
            f"{img_name}_result.jpg"
        ]
        result['debug_images'] = [img for img in result['debug_images'] if img and os.path.exists(os.path.join(self.output_dir, img))]

        return result, result_img_path, result_txt_path

    def _generate_result(self, water_depth_ratio):
        """Generate a result dictionary based on the water depth ratio (% of pipe diameter)."""
        result = {
            '缺陷名称': '起伏',
            '缺陷代码': 'QF',
            '水深占比(直径%)': round(water_depth_ratio, 2),
            '缺陷等级': 0,
            '缺陷描述': '水深正常',
            '分值': 0
        }

        if 10 <= water_depth_ratio < 20:
            result.update({
                '缺陷等级': 1,
                '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（10%～20%）',
                '分值': 0.5
            })
        elif 20 <= water_depth_ratio < 30:
            result.update({
                '缺陷等级': 2,
                '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（20%～30%）',
                '分值': 2
            })
        elif 30 <= water_depth_ratio < 40:
            result.update({
                '缺陷等级': 3,
                '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（30%～40%）',
                '分值': 5
            })
        elif water_depth_ratio >= 40:
            result.update({
                '缺陷等级': 4,
                '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（大于40%）',
                '分值': 10
            })

        return result

    def calculate_water_depth_pixels(self, img_path):
        """Calculate the water depth in pixels."""
        _, water_depth_pixels, _ = self.detect_water_depth(img_path)
        return water_depth_pixels

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

                result, result_img_path, result_txt_path = self.detect_water_depth(img_path)
                print(f"✅ 生成结果图像: {result_img_path}")
                print(f"✅ 生成报告文本: {result_txt_path}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")

# Example usage
if __name__ == "__main__":
    detector = DiseaseQF(input_dir="TEST_PHOTO")
    detector.process_directory()

# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# class DiseaseQF:
#     """A class to detect water depth in pipe images and calculate water depth ratio."""
#
#     def __init__(self, pipe_diameter_pixels=None, debug_dir="QF_debug_output"):
#         """Initialize the detector with an optional pipe diameter in pixels.
#
#         Args:
#             pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
#             debug_dir (str): Directory for saving debug images.
#         """
#         self.pipe_diameter = pipe_diameter_pixels
#         self.debug_dir = debug_dir
#         os.makedirs(debug_dir, exist_ok=True)
#         self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
#         self.upper_black = np.array([10, 255, 10])  # Upper bound for black color in HSV
#
#     def auto_estimate_diameter(self, img_path, method="auto"):
#         """Automatically estimate the pipe diameter using Hough transform or projection method.
#
#         Args:
#             img_path (str): Path to the input image.
#             method (str): Method to use ('hough', 'projection', or 'auto'). Defaults to 'auto'.
#
#         Returns:
#             int or None: Estimated diameter in pixels, or None if estimation fails.
#         """
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             print("❌ 无法加载图像，请检查文件路径")
#             return None
#
#         height, width = img.shape
#         estimated_diameter = None
#
#         # Method 1: Hough Circle Transform (optimized parameters)
#         if method in ("hough", "auto"):
#             img_blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Use Gaussian blur for better edge detection
#             circles = cv2.HoughCircles(
#                 img_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=width // 8,  # Adjusted minDist using image width
#                 param1=100, param2=20, minRadius=50, maxRadius=width // 2  # Adjusted ranges
#             )
#             if circles is not None:
#                 circles = np.uint16(np.around(circles))
#                 if len(circles[0]) > 0:
#                     estimated_diameter = 2 * circles[0][0][2]  # Diameter = 2 * radius
#                     print(
#                         f"[Hough] 估算管道直径: {estimated_diameter} px, 中心: ({circles[0][0][0]}, {circles[0][0][1]})")
#                     self.pipe_diameter = estimated_diameter
#                     return estimated_diameter
#
#         # Method 2: Projection Method
#         if method in ("projection", "auto"):
#             edges = cv2.Canny(img, 50, 150)
#             projection = np.sum(edges, axis=1)
#             threshold = np.max(projection) * 0.3  # Adjusted threshold for better detection
#             rows = np.where(projection > threshold)[0]
#             if len(rows) >= 2:
#                 top, bottom = min(rows), max(rows)
#                 estimated_diameter = bottom - top
#                 print(f"[Projection] 估算管道直径: {estimated_diameter} px")
#                 self.pipe_diameter = estimated_diameter
#                 return estimated_diameter
#
#         print("❌ 无法自动估算管道直径，请手动设置 pipe_diameter 或检查图像")
#         return None
#
#     def detect_water_depth(self, img_path):
#         """Detect water depth in the pipe image and calculate its ratio."""
#         if self.pipe_diameter is None:
#             raise ValueError("请先设置或自动估算管道内径 pipe_diameter")
#
#         # Load image
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ 无法加载图像，请检查文件路径")
#
#         height, width = img.shape[:2]
#         center_y = height // 2
#         center_x = width // 2
#         radius = self.pipe_diameter // 2
#
#         # Convert to HSV and create water mask (assuming black area is water)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         water_mask_full = cv2.inRange(hsv, self.lower_black, self.upper_black)
#         cv2.imwrite(os.path.join(self.debug_dir, "debug_water_mask.jpg"), water_mask_full)
#
#         # Create full circle mask
#         circle_mask_full = np.zeros_like(water_mask_full, dtype=np.uint8)
#         cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
#         cv2.imwrite(os.path.join(self.debug_dir, "debug_circle_mask.jpg"), circle_mask_full)
#
#         # Find water level (top of water area)
#         water_contours, _ = cv2.findContours(water_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         debug_contours_img = cv2.cvtColor(water_mask_full, cv2.COLOR_GRAY2BGR)
#         if water_contours:
#             water_contour = max(water_contours, key=cv2.contourArea)
#             water_level_y = np.min(water_contour[:, :, 1])  # Topmost point of water
#             cv2.drawContours(debug_contours_img, [water_contour], -1, (0, 255, 0), 2)  # Green contour
#             cv2.putText(debug_contours_img, f"Water Level: {water_level_y} px", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         else:
#             water_level_y = height  # Default to bottom if no water detected
#         cv2.imwrite(os.path.join(self.debug_dir, "debug_water_contours.jpg"), debug_contours_img)
#         print(f"🌊 检测到的水面高度: {water_level_y} px")
#
#         # Pipe bottom y-coordinate
#         pipe_bottom_y = center_y + radius
#
#         # Actual water depth (pixels) = Pipe bottom - Water level
#         water_depth_pixels = pipe_bottom_y - water_level_y
#         water_depth_pixels = max(0, min(water_depth_pixels, self.pipe_diameter))  # Ensure within diameter
#
#         # Water depth percentage (% of diameter)
#         water_depth_ratio_percent = (water_depth_pixels / self.pipe_diameter) * 100
#         water_depth_ratio_percent = max(0, min(water_depth_ratio_percent, 100.0))
#
#         print(
#             f"✅ 管道底部 y: {pipe_bottom_y}, 水深 (像素): {water_depth_pixels}, 水深比例 (直径%): {water_depth_ratio_percent:.2f}%")
#
#         # Debug image for water depth region
#         debug_depth_region = img.copy()
#         cv2.line(debug_depth_region, (0, int(pipe_bottom_y)), (width, int(pipe_bottom_y)), (0, 0, 255), 2)  # Red line for pipe bottom
#         cv2.line(debug_depth_region, (0, int(water_level_y)), (width, int(water_level_y)), (0, 255, 0), 2)  # Green line for water level
#         cv2.rectangle(debug_depth_region, (center_x - radius, int(water_level_y)),
#                       (center_x + radius, int(pipe_bottom_y)), (255, 0, 0), 2)  # Blue rectangle for water depth
#         cv2.putText(debug_depth_region, f"Water Depth: {water_depth_pixels} px", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imwrite(os.path.join(self.debug_dir, "debug_water_depth_region.jpg"), debug_depth_region)
#
#         return self._generate_result(water_depth_ratio_percent), water_depth_pixels
#
#     def _generate_result(self, water_depth_ratio):
#         """Generate a result dictionary based on the water depth ratio (% of pipe diameter)."""
#         result = {
#             '缺陷名称': '起伏',
#             '缺陷代码': 'QF',
#             '水深占比(直径%)': round(water_depth_ratio, 2),  # Modified key to reflect diameter percentage
#             '缺陷等级': 0,
#             '缺陷描述': '水深正常',
#             '分值': 0
#         }
#
#         # Adjust thresholds to fit diameter percentage
#         if 10 <= water_depth_ratio < 20:
#             result.update({
#                 '缺陷等级': 1,
#                 '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（10%～20%）',
#                 '分值': 0.5
#             })
#         elif 20 <= water_depth_ratio < 30:
#             result.update({
#                 '缺陷等级': 2,
#                 '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（20%～30%）',
#                 '分值': 2
#             })
#         elif 30 <= water_depth_ratio < 40:
#             result.update({
#                 '缺陷等级': 3,
#                 '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（30%～40%）',
#                 '分值': 5
#             })
#         elif water_depth_ratio >= 40:
#             result.update({
#                 '缺陷等级': 4,
#                 '缺陷描述': f'水深占管径的 {water_depth_ratio:.1f}%（大于40%）',
#                 '分值': 10
#             })
#
#         return result
#
#     def show_result(self, result_img_path):
#         """Display the detection result image."""
#         img = cv2.imread(result_img_path)
#         if img is None:
#             raise FileNotFoundError(f"找不到图像：{result_img_path}")
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.figure(figsize=(10, 8))
#         plt.imshow(img_rgb)
#         plt.title("Water Depth Detection Result")
#         plt.axis("off")
#         plt.show()
#
#     def visualize(self, img_path, save_path=None):
#         """Visualize the detection results on the image."""
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ 无法加载图像，请检查文件路径")
#
#         # detect_water_depth now returns a different result format
#         result, water_depth_pixels = self.detect_water_depth(img_path)
#
#         height, width = img.shape[:2]
#         center_y = height // 2
#         center_x = width // 2
#         radius = self.pipe_diameter // 2
#
#         # Get water level y-coordinate from the detect method
#         pipe_bottom_y = center_y + radius
#         water_level_y = pipe_bottom_y - water_depth_pixels  # Recalculate water level y
#
#         # Draw pipe outer circle
#         cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle
#
#         # Draw water level line (green)
#         cv2.line(img, (0, int(water_level_y)), (width, int(water_level_y)), (0, 255, 0), 2)
#
#         # Draw water depth line (vertical from bottom to water level in red)
#         cv2.line(img, (center_x, int(pipe_bottom_y)), (center_x, int(water_level_y)), (0, 0, 255), 2)
#
#         # Add pipe diameter text
#         cv2.putText(img, f"Pipe Diameter: {self.pipe_diameter}px", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#
#         # Add water depth percentage text
#         depth_text = f"Water Depth: {result['水深占比(直径%)']:.1f}%"
#         cv2.putText(img, depth_text, (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#         # Save or display the result
#         if save_path:
#             cv2.imwrite(save_path, img)
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()
#
#         return result
#
#     def calculate_water_depth_pixels(self, img_path):
#         """Calculate the water depth in pixels.
#
#         This method is renamed for clarity, as it now returns pixels, not area.
#         """
#         _, water_depth_pixels = self.detect_water_depth(img_path)
#         return water_depth_pixels
#
#
#
