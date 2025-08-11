import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

class DiseaseFZ:
    """A class to detect floating debris in pipe images and calculate its ratio and thickness."""

    def __init__(self, input_dir="TEST_PHOTO", output_dir="FZ_debug_output", pipe_diameter_pixels=None):
        """Initialize the detector with input/output directories and optional pipe diameter.

        Args:
            input_dir (str): Directory containing input images. Defaults to 'TEST_PHOTO'.
            output_dir (str): Directory to save debug images and results. Defaults to 'FZ_debug_output'.
            pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
        """
        self.input_dir = str(Path(input_dir))
        self.output_dir = str(Path(output_dir))
        self.pipe_diameter = pipe_diameter_pixels
        self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
        self.upper_black = np.array([180, 255, 50])  # Upper bound for black color in HSV
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

    def _save_debug_image(self, img, name, cmap=None):
        """Save debug image to the specified directory."""
        save_path = os.path.join(self.output_dir, name)
        if cmap == 'gray':
            plt.imsave(save_path, img, cmap='gray')
        else:
            if len(img.shape) == 2:  # Convert single-channel image to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(save_path, img)

    def auto_estimate_diameter(self, img_path, method="auto"):
        """Automatically estimate the pipe diameter using Hough transform or projection method.

        Args:
            img_path (str): Path to the input image.
            method (str): Method to use ('hough', 'projection', or 'auto'). Defaults to 'auto'.

        Returns:
            int or None: Estimated diameter in pixels, or None if estimation fails.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            raise ValueError("❌ Unable to load image, please check file path")
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(img, f"{img_name}_00_original.jpg")
        self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')

        estimated_diameter = None

        # Method 1: Hough Circle Transform
        if method in ("hough", "auto"):
            img_blurred = cv2.medianBlur(gray, 5)
            self._save_debug_image(img_blurred, f"{img_name}_02_blurred_hough.jpg", 'gray')
            circles = cv2.HoughCircles(
                img_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30, minRadius=30, maxRadius=min(height, width) // 2
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(circles[0]) > 0:
                    estimated_diameter = 2 * circles[0][0][2]  # Diameter = 2 * radius
                    print(f"[Hough] Estimated pipe diameter: {estimated_diameter} px")
                    hough_img = img.copy()
                    cv2.circle(hough_img, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (0, 255, 0), 2)
                    self._save_debug_image(hough_img, f"{img_name}_03_hough_circle.jpg")
                    self.pipe_diameter = estimated_diameter
                    return estimated_diameter

        # Method 2: Projection Method
        if method in ("projection", "auto"):
            edges = cv2.Canny(gray, 50, 150)
            self._save_debug_image(edges, f"{img_name}_02_canny_projection.jpg", 'gray')
            projection = np.sum(edges, axis=1)
            threshold = np.max(projection) * 0.5
            rows = np.where(projection > threshold)[0]
            if len(rows) >= 2:
                top, bottom = min(rows), max(rows)
                estimated_diameter = bottom - top
                print(f"[Projection] Estimated pipe diameter: {estimated_diameter} px")
                proj_img = img.copy()
                cv2.line(proj_img, (0, top), (img.shape[1], top), (0, 255, 0), 2)
                cv2.line(proj_img, (0, bottom), (img.shape[1], bottom), (0, 255, 0), 2)
                self._save_debug_image(proj_img, f"{img_name}_03_projection_lines.jpg")
                self.pipe_diameter = estimated_diameter
                return estimated_diameter

        print("❌ Failed to estimate pipe diameter automatically. Please set pipe_diameter manually or check the image.")
        return None

    def detect_debris(self, img_path):
        """Detect floating debris in the image, save result image and text report.

        Args:
            img_path (str): Path to the input image.

        Returns:
            tuple: (result dictionary, result image path, result text path)
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = self._load_image(img_path)
        if img is None:
            raise ValueError("❌ Unable to load image, please check file path")

        # Estimate pipe diameter if not set
        if self.pipe_diameter is None:
            self.auto_estimate_diameter(img_path)

        if self.pipe_diameter is None:
            raise ValueError("❌ Pipe diameter not set and could not be estimated")

        height, width = img.shape[:2]
        if self.pipe_diameter > min(height, width):
            raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")

        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Create debris mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        bottom_region = mask[center_y:, :]
        debris_mask = cv2.bitwise_not(bottom_region)
        kernel = np.ones((5, 5), np.uint8)
        debris_mask = cv2.erode(debris_mask, kernel, iterations=2)
        self._save_debug_image(debris_mask, f"{img_name}_04_eroded_debris_mask.jpg", 'gray')

        # Create full circle mask and take lower half
        circle_mask_full = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
        circle_mask_lower = circle_mask_full[center_y:, :]
        self._save_debug_image(circle_mask_lower, f"{img_name}_05_circle_mask_lower.jpg", 'gray')

        # Intersect debris mask with circle mask
        debris_inside_circle = cv2.bitwise_and(debris_mask, debris_mask, mask=circle_mask_lower)
        self._save_debug_image(debris_inside_circle, f"{img_name}_06_debris_inside_circle.jpg", 'gray')

        # Calculate areas
        half_circle_area = cv2.countNonZero(circle_mask_lower)
        debris_area = cv2.countNonZero(debris_inside_circle)
        print(f"✅ Circle area: {half_circle_area*2} px, Lower half circle area: {half_circle_area} px, Debris area: {debris_area} px")

        # Debris ratio (%)
        debris_ratio = (debris_area / (half_circle_area*2)) * 100 if half_circle_area > 0 else 0
        debris_ratio = min(debris_ratio, 100.0)

        # Calculate maximum debris thickness
        max_thickness = 0
        if debris_area > 0:
            projection = np.sum(debris_inside_circle, axis=1)
            non_zero_rows = np.where(projection > 0)[0]
            if len(non_zero_rows) > 0:
                max_thickness = len(non_zero_rows)  # Vertical extent of debris in pixels
        print(f"Max debris thickness: {max_thickness} px")

        # Generate result dictionary
        result = self._generate_result(debris_ratio)
        result['Max Thickness (px)'] = max_thickness

        # Visualize results
        vis_img = img.copy()

        # Draw pipe outer circle
        cv2.circle(vis_img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle

        # Draw debris contours
        contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img[center_y:, :], contours, -1, (0, 255, 255), 2)  # Cyan contours

        # Add text annotations
        info = [
            f"Debris Ratio: {result['Debris Area Ratio (%)']:.1f}%",
            f"Max Thickness: {max_thickness}px",
            f"Pipe Diameter: {self.pipe_diameter}px",
            f"Defect Level: {result['Defect Level']}",
            f"Score: {result['Score']}"
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
            f"{img_name}_00_original.jpg",
            f"{img_name}_01_gray.jpg",
            f"{img_name}_02_blurred_hough.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_02_blurred_hough.jpg")) else None,
            f"{img_name}_03_hough_circle.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_03_hough_circle.jpg")) else None,
            f"{img_name}_02_canny_projection.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_02_canny_projection.jpg")) else None,
            f"{img_name}_03_projection_lines.jpg" if os.path.exists(os.path.join(self.output_dir, f"{img_name}_03_projection_lines.jpg")) else None,
            f"{img_name}_04_eroded_debris_mask.jpg",
            f"{img_name}_05_circle_mask_lower.jpg",
            f"{img_name}_06_debris_inside_circle.jpg",
            f"{img_name}_result.jpg"
        ]
        result['debug_images'] = [img for img in result['debug_images'] if img and os.path.exists(os.path.join(self.output_dir, img))]

        return result, result_img_path, result_txt_path

    def _generate_result(self, debris_ratio):
        """Generate a result dictionary based on the floating debris area ratio."""
        result = {
            'Defect Name': 'Floating Debris',
            'Defect Code': 'FZ',
            'Debris Area Ratio (%)': round(debris_ratio, 2),
            'Defect Level': 0,
            'Defect Description': 'No floating debris',
            'Score': '-'
        }

        if 0 < debris_ratio <= 30:
            result.update({
                'Defect Level': 1,
                'Defect Description': f'Sparse floating debris, occupying up to 30% of water surface',
                'Score': '-'
            })
        elif 30 < debris_ratio <= 60:
            result.update({
                'Defect Level': 2,
                'Defect Description': f'Moderate floating debris, occupying 30%-60% of water surface',
                'Score': '-'
            })
        elif debris_ratio > 60:
            result.update({
                'Defect Level': 3,
                'Defect Description': f'Heavy floating debris, occupying over 60% of water surface',
                'Score': '-'
            })

        if result['Defect Level'] > 0:
            result['Defect Description'] += ' (This defect is logged but not scored)'
        return result

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

                result, result_img_path, result_txt_path = self.detect_debris(img_path)
                print(f"✅ 生成结果图像: {result_img_path}")
                print(f"✅ 生成报告文本: {result_txt_path}")

            except Exception as e:
                print(f"❌ 处理图像 {img_path} 时发生错误: {str(e)}")

# Example usage
if __name__ == "__main__":
    detector = DiseaseFZ(input_dir="TEST_PHOTO")
    detector.process_directory()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# class DiseaseFZ:
#     """A class to detect floating debris in pipe images and calculate its ratio and thickness."""
#
#     def __init__(self, pipe_diameter_pixels=None, debug_dir="FZ_debug_output"):
#         """Initialize the detector with an optional pipe diameter and debug directory.
#
#         Args:
#             pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
#             debug_dir (str): Directory to save debug images. Defaults to 'FZ_debug_output'.
#         """
#         self.pipe_diameter = pipe_diameter_pixels
#         self.debug_dir = debug_dir
#         self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
#         self.upper_black = np.array([180, 255, 50])  # Upper bound for black color in HSV
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
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ Unable to load image, please check file path")
#         height, width = img.shape[:2]
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self._save_debug_image(img, f"{img_name}_00_original.jpg")
#         self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')
#
#         estimated_diameter = None
#
#         # Method 1: Hough Circle Transform
#         if method in ("hough", "auto"):
#             img_blurred = cv2.medianBlur(gray, 5)
#             self._save_debug_image(img_blurred, f"{img_name}_02_blurred_hough.jpg", 'gray')
#             circles = cv2.HoughCircles(
#                 img_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
#                 param1=50, param2=30, minRadius=30, maxRadius=min(height, width) // 2
#             )
#             if circles is not None:
#                 circles = np.uint16(np.around(circles))
#                 if len(circles[0]) > 0:
#                     estimated_diameter = 2 * circles[0][0][2]  # Diameter = 2 * radius
#                     print(f"[Hough] Estimated pipe diameter: {estimated_diameter} px")
#                     hough_img = img.copy()
#                     cv2.circle(hough_img, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (0, 255, 0), 2)
#                     self._save_debug_image(hough_img, f"{img_name}_03_hough_circle.jpg")
#                     self.pipe_diameter = estimated_diameter
#                     return estimated_diameter
#
#         # Method 2: Projection Method
#         if method in ("projection", "auto"):
#             edges = cv2.Canny(gray, 50, 150)
#             self._save_debug_image(edges, f"{img_name}_02_canny_projection.jpg", 'gray')
#             projection = np.sum(edges, axis=1)
#             threshold = np.max(projection) * 0.5
#             rows = np.where(projection > threshold)[0]
#             if len(rows) >= 2:
#                 top, bottom = min(rows), max(rows)
#                 estimated_diameter = bottom - top
#                 print(f"[Projection] Estimated pipe diameter: {estimated_diameter} px")
#                 proj_img = img.copy()
#                 cv2.line(proj_img, (0, top), (img.shape[1], top), (0, 255, 0), 2)
#                 cv2.line(proj_img, (0, bottom), (img.shape[1], bottom), (0, 255, 0), 2)
#                 self._save_debug_image(proj_img, f"{img_name}_03_projection_lines.jpg")
#                 self.pipe_diameter = estimated_diameter
#                 return estimated_diameter
#
#         print("❌ Failed to estimate pipe diameter automatically. Please set pipe_diameter manually or check the image.")
#         return None
#
#     def process_image(self, img_path):
#         """Process the image to detect floating debris and calculate its ratio and thickness."""
#         if self.pipe_diameter is None:
#             raise ValueError("Please set or estimate pipe_diameter first")
#
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ Unable to load image, please check file path")
#
#         height, width = img.shape[:2]
#         if self.pipe_diameter > min(height, width):
#             raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")
#
#         center_y = height // 2
#         center_x = width // 2
#         radius = self.pipe_diameter // 2
#
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')
#
#         # Create debris mask
#         bottom_gray = gray[center_y:, :]
#         _, original_mask = cv2.threshold(bottom_gray, 50, 255, cv2.THRESH_BINARY_INV)
#         debris_mask = cv2.bitwise_not(original_mask)
#         self._save_debug_image(debris_mask, f"{img_name}_02_debris_mask.jpg", 'gray')
#
#         # Create full circle mask and take lower half
#         circle_mask_full = np.zeros_like(gray, dtype=np.uint8)
#         cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
#         circle_mask_lower = circle_mask_full[center_y:, :]
#         self._save_debug_image(circle_mask_lower, f"{img_name}_03_circle_mask_lower.jpg", 'gray')
#
#         # Intersect debris mask with circle mask
#         debris_inside_circle = cv2.bitwise_and(debris_mask, debris_mask, mask=circle_mask_lower)
#         self._save_debug_image(debris_inside_circle, f"{img_name}_04_debris_inside_circle.jpg", 'gray')
#
#         # Calculate areas
#         half_circle_area = cv2.countNonZero(circle_mask_lower)
#         debris_area = cv2.countNonZero(debris_inside_circle)
#         print(f"✅ Circle area: {half_circle_area*2} px, Lower half circle area: {half_circle_area} px, Debris area: {debris_area} px")
#
#         # Debris ratio (%)
#         debris_ratio = (debris_area / (half_circle_area*2)) * 100 if half_circle_area > 0 else 0
#         debris_ratio = min(debris_ratio, 100.0)
#
#         # Calculate maximum debris thickness
#         max_thickness = 0
#         if debris_area > 0:
#             projection = np.sum(debris_inside_circle, axis=1)
#             non_zero_rows = np.where(projection > 0)[0]
#             if len(non_zero_rows) > 0:
#                 max_thickness = len(non_zero_rows)  # Vertical extent of debris in pixels
#         print(f"Max debris thickness: {max_thickness} px")
#
#         # Visualize debug image
#         debug_vis = img.copy()
#
#         # Lower half circle (red translucent)
#         lower_circle_mask_colored = np.zeros_like(img)
#         lower_circle_mask_colored[center_y:, :] = cv2.merge([
#             np.zeros_like(circle_mask_lower),
#             np.zeros_like(circle_mask_lower),
#             circle_mask_lower  # Red channel
#         ])
#         debug_vis = cv2.addWeighted(debug_vis, 1.0, lower_circle_mask_colored, 0.3, 0)
#
#         # Add text for analysis area
#         cv2.putText(debug_vis, "Analysis area (lower half circle)", (center_x - 150, center_y + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#
#         # Debris area (green highlight)
#         debris_colored = np.zeros_like(img)
#         debris_colored[center_y:, :] = cv2.merge([
#             np.zeros_like(debris_inside_circle),
#             debris_inside_circle,  # Green
#             np.zeros_like(debris_inside_circle)
#         ])
#         debug_vis = cv2.addWeighted(debug_vis, 1.0, debris_colored, 0.6, 0)
#
#         # Add text for debris area
#         cv2.putText(debug_vis, "Floating Debris (FZ)", (center_x - 150, center_y + 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Draw circle boundary
#         cv2.circle(debug_vis, (center_x, center_y), radius, (255, 255, 0), 2)
#         cv2.putText(debug_vis, "Boundary", (center_x + radius + 10, center_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#
#         # Add debris ratio text
#         text = f"Debris percentage: {debris_ratio:.1f}%"
#         cv2.putText(debug_vis, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
#
#         # Add max thickness text
#         cv2.putText(debug_vis, f"Max Thickness: {max_thickness}px", (50, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#
#         # Save visualization
#         self._save_debug_image(debug_vis, f"{img_name}_05_result.jpg")
#
#         return self._generate_result(debris_ratio), max_thickness
#
#     def _generate_result(self, debris_ratio):
#         """Generate a result dictionary based on the floating debris area ratio."""
#         result = {
#             'Defect Name': 'Floating Debris',
#             'Defect Code': 'FZ',
#             'Debris Area Ratio (%)': round(debris_ratio, 2),
#             'Defect Level': 0,
#             'Defect Description': 'No floating debris',
#             'Score': '-'
#         }
#
#         if 0 < debris_ratio <= 30:
#             result.update({
#                 'Defect Level': 1,
#                 'Defect Description': f'Sparse floating debris, occupying up to 30% of water surface',
#                 'Score': '-'
#             })
#         elif 30 < debris_ratio <= 60:
#             result.update({
#                 'Defect Level': 2,
#                 'Defect Description': f'Moderate floating debris, occupying 30%-60% of water surface',
#                 'Score': '-'
#             })
#         elif debris_ratio > 60:
#             result.update({
#                 'Defect Level': 3,
#                 'Defect Description': f'Heavy floating debris, occupying over 60% of water surface',
#                 'Score': '-'
#             })
#
#         if result['Defect Level'] > 0:
#             result['Defect Description'] += ' (This defect is logged but not scored)'
#         return result
#
#     def visualize(self, img_path, save_path=None):
#         """Visualize the detection results on the image.
#
#         Args:
#             img_path (str): Path to the input image.
#             save_path (str, optional): Path to save the output image. Defaults to None.
#
#         Returns:
#             dict: Result dictionary containing debris analysis.
#         """
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ Unable to load image, please check file path")
#         height, width = img.shape[:2]
#         if self.pipe_diameter > min(height, width):
#             raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")
#         result, max_thickness = self.process_image(img_path)
#
#         center_y = height // 2
#         center_x = width // 2
#         radius = self.pipe_diameter // 2
#
#         # Draw pipe outer circle
#         cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle
#
#         # Draw debris contours
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
#         self._save_debug_image(mask, f"{img_name}_06_hsv_mask.jpg", 'gray')
#         bottom_region = mask[center_y:, :]
#         debris_mask = cv2.bitwise_not(bottom_region)
#         kernel = np.ones((5, 5), np.uint8)
#         debris_mask = cv2.erode(debris_mask, kernel, iterations=2)
#         self._save_debug_image(debris_mask, f"{img_name}_07_eroded_debris_mask.jpg", 'gray')
#         contours, _ = cv2.findContours(debris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(img[center_y:, :], contours, -1, (0, 255, 255), 2)  # Cyan contours
#
#         # Add text annotations
#         cv2.putText(img, f"Pipe Diameter: {self.pipe_diameter}px", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#         cv2.putText(img, f"Debris Ratio: {result['Debris Area Ratio (%)']:.1f}%", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#         cv2.putText(img, f"Max Thickness: {max_thickness}px", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#
#         # Save or display the result
#         if save_path:
#             cv2.imwrite(save_path, img)
#         self._save_debug_image(img, f"{img_name}_08_final_visualization.jpg")
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()
#
#         return result
#
#     def calculate_sediment_area(self, img_path):
#         """Calculate the floating debris area in pixels.
#
#         Args:
#             img_path (str): Path to the input image.
#
#         Returns:
#             float: Estimated debris area in pixels.
#         """
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise ValueError("❌ Unable to load image, please check file path")
#
#         height, width = img.shape[:2]
#         center_y = height // 2
#         center_x = width // 2
#         radius = self.pipe_diameter // 2 if self.pipe_diameter else None
#
#         if radius is None:
#             raise ValueError("Pipe diameter must be set before calculating debris area")
#         if self.pipe_diameter > min(height, width):
#             raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")
#
#         # Create debris mask
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
#         bottom_region = mask[center_y:, :]
#         debris_mask = cv2.bitwise_not(bottom_region)
#         kernel = np.ones((5, 5), np.uint8)
#         debris_mask = cv2.erode(debris_mask, kernel, iterations=2)
#         self._save_debug_image(debris_mask, f"{img_name}_09_debris_area_mask.jpg", 'gray')
#
#         # Create circle mask for lower half
#         circle_mask_full = np.zeros_like(mask, dtype=np.uint8)
#         cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
#         circle_mask_lower = circle_mask_full[center_y:, :]
#         debris_inside_circle = cv2.bitwise_and(debris_mask, debris_mask, mask=circle_mask_lower)
#         self._save_debug_image(debris_inside_circle, f"{img_name}_10_debris_area_result.jpg", 'gray')
#
#         # Calculate area
#         debris_area = cv2.countNonZero(debris_inside_circle)
#         print(f"Estimated debris area: {debris_area} px²")
#         return float(debris_area)
#
