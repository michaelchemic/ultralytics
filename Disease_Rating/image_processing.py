# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime
#
#
# class PipeJointDetector:
#     """
#     A class to detect and analyze pipe joint defects (AJ) with enhanced debugging.
#     """
#
#     def __init__(self, output_dir="AJ_debug_output"):
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#
#         self.pipe_diameter = None
#         self.defect_standards = {
#             "AJ": {
#                 "phenomena": [
#                     "接口位突出, 但主管未受损伤",
#                     "接口位突出, 且主管受损出现裂痕",
#                     "接口位突出, 且主管受损出现破裂",
#                     "支管未插入, 且主管受损出现破裂"
#                 ],
#                 "level_a": "支管未伸入主管内",
#                 "level_b": "支管伸入主管内的长度等于主管直径10%",
#                 "level_c": "支管伸入主管内的长度等于主管直径20%",
#                 "measure_desc": "支管突出长度为管径的{:.1f}%"
#             }
#         }
#
#     def _save_debug_image(self, image, step_name, img_name):
#         """Save debug image with timestamp and step info"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         save_path = os.path.join(self.output_dir, f"{img_name}_{step_name}_{timestamp}.jpg")
#         cv2.imwrite(save_path, image)
#         return save_path
#
#     def _auto_estimate_pipe_diameter(self, img_path):
#         """
#         Automatically estimates the main pipe diameter with debug outputs.
#         Returns the circle parameters (x, y, radius) or None if not found.
#         """
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             return None
#
#         # Step 1: Original grayscale
#         self._save_debug_image(img, "01_original_gray", img_name)
#
#         # Step 2: Apply blur
#         img_blurred = cv2.medianBlur(img, 5)
#         self._save_debug_image(img_blurred, "02_blurred", img_name)
#
#         # Step 3: Edge detection (for debugging)
#         edges = cv2.Canny(img_blurred, 50, 150)
#         self._save_debug_image(edges, "03_edges", img_name)
#
#         # Step 4: Hough Circle Transform
#         circles = cv2.HoughCircles(
#             img_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
#             param1=100, param2=30, minRadius=int(img.shape[0] * 0.2),
#             maxRadius=int(img.shape[0] * 0.6)
#         )
#
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             best_circle = max(circles[0], key=lambda item: item[2])
#
#             # Visualize the detected circle
#             debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             cv2.circle(debug_img, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
#             cv2.putText(debug_img, f"Radius: {best_circle[2]}px", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             self._save_debug_image(debug_img, "04_circle_detected", img_name)
#
#             self.pipe_diameter = best_circle[2] * 2
#             print(f"🔵 检测到主管道 - 半径: {best_circle[2]}px, 直径: {self.pipe_diameter}px")
#             return best_circle
#
#         print("🟠 未检测到主管道圆形")
#         return None
#
#     def _measure_intrusion_length(self, img, img_name):
#         """
#         Enhanced version with debug outputs to measure branch pipe intrusion.
#         """
#         # Step 1: Convert to HSV and extract dark regions (potential branch pipe)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         lower_black = np.array([0, 0, 0])
#         upper_black = np.array([180, 255, 100])
#         mask = cv2.inRange(hsv, lower_black, upper_black)
#         self._save_debug_image(mask, "05_branch_mask", img_name)
#
#         # Step 2: Find contours in the mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contour_img = img.copy()
#         cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)
#         self._save_debug_image(contour_img, "06_all_contours", img_name)
#
#         # Step 3: Filter contours by area and position
#         min_area = (img.shape[0] * img.shape[1]) * 0.01  # At least 1% of image area
#         valid_contours = []
#
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > min_area:
#                 valid_contours.append(cnt)
#
#         if not valid_contours:
#             print("🟠 未检测到有效的支管轮廓")
#             return 0
#
#         # Step 4: Find the most likely branch pipe contour
#         branch_contour = max(valid_contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(branch_contour)
#
#         # Visualize the selected branch
#         branch_img = img.copy()
#         cv2.drawContours(branch_img, [branch_contour], -1, (0, 0, 255), 2)
#         cv2.rectangle(branch_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         self._save_debug_image(branch_img, "07_selected_branch", img_name)
#
#         # Step 5: Calculate intrusion length (simplified for demo)
#         intrusion_px = w  # Using width as proxy for intrusion length
#         print(f"📏 测量支管突出长度: {intrusion_px:.1f}px")
#         return intrusion_px
#
#     def detect_defect(self, img_path):
#         """
#         Detects the AJ defect with detailed debug outputs.
#         """
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         print(f"\n🔍 开始分析图像: {img_name}")
#
#         # Step 1: Load and verify image
#         img = cv2.imread(img_path)
#         if img is None:
#             return {"error": "无法加载图像"}
#
#         self._save_debug_image(img, "00_original_color", img_name)
#
#         # Step 2: Detect main pipe
#         circle_params = self._auto_estimate_pipe_diameter(img_path)
#         if circle_params is None:
#             return {"error": "无法识别主管道"}
#
#         # Step 3: Measure branch intrusion
#         intrusion_px = self._measure_intrusion_length(img, img_name)
#         intrusion_ratio = (intrusion_px / self.pipe_diameter) * 100
#
#         # Step 4: Classify defect
#         conclusion = self._classify_defect(intrusion_px, intrusion_ratio)
#
#         # Step 5: Generate debug report
#         debug_report = {
#             "pipe_diameter_px": self.pipe_diameter,
#             "branch_intrusion_px": intrusion_px,
#             "intrusion_ratio_percent": intrusion_ratio,
#             "debug_images": [
#                 f for f in os.listdir(self.output_dir)
#                 if f.startswith(img_name) and f.endswith(".jpg")
#             ]
#         }
#         conclusion.update(debug_report)
#
#         return conclusion
#
#     def _classify_defect(self, intrusion_px, intrusion_ratio):
#         """Classify the defect based on measurements"""
#         conclusion = {}
#
#         if intrusion_px <= 0:
#             conclusion["code"] = "AJ"
#             conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][3]
#             conclusion["description"] = self.defect_standards["AJ"]["level_a"]
#             conclusion["location_size"] = "支管未插入"
#         elif 0 < intrusion_ratio <= 10:
#             conclusion["code"] = "AJ"
#             conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
#             conclusion["description"] = self.defect_standards["AJ"]["level_b"]
#             conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
#         elif 10 < intrusion_ratio < 20:
#             conclusion["code"] = "AJ"
#             conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
#             conclusion["description"] = f"支管伸入长度介于10%-20%之间"
#             conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
#         elif intrusion_ratio >= 20:
#             conclusion["code"] = "AJ"
#             conclusion["phenomenon"] = self.defect_standards["AJ"]["phenomena"][0]
#             conclusion["description"] = self.defect_standards["AJ"]["level_c"]
#             conclusion["location_size"] = self.defect_standards["AJ"]["measure_desc"].format(intrusion_ratio)
#         else:
#             conclusion["conclusion"] = "未检测到明显的支管暗接缺陷。"
#
#         return conclusion
#
#     def visualize(self, img_path, result, save_name="result.jpg"):
#         """
#         Enhanced visualization with debug information.
#         """
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#
#         # Draw main pipe circle
#         circle_params = self._auto_estimate_pipe_diameter(img_path)
#         if circle_params:
#             (x, y, r) = circle_params
#             cv2.circle(img, (x, y), r, (0, 255, 0), 2)
#             cv2.putText(img, f"Main Pipe (D={self.pipe_diameter}px)",
#                         (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#         # Add analysis results
#         text_y = 30
#         for key, value in result.items():
#             if key not in ["debug_images"]:  # Skip debug images list
#                 text = f"{key}: {value}" if not isinstance(value, float) else f"{key}: {value:.2f}"
#                 cv2.putText(img, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 text_y += 25
#
#         # Save and display
#         final_path = os.path.join(self.output_dir, save_name)
#         cv2.imwrite(final_path, img)
#
#         # Display all debug images in a grid
#         self._display_debug_images(img_name)
#
#         return final_path
#
#     def _display_debug_images(self, img_name):
#         """Display all debug images in a grid layout"""
#         debug_files = [f for f in os.listdir(self.output_dir) if f.startswith(img_name)]
#         if not debug_files:
#             return
#
#         # Sort files by step number
#         debug_files.sort()
#
#         plt.figure(figsize=(15, 10))
#         for i, filename in enumerate(debug_files[:8]):  # Show up to 8 images
#             img = cv2.imread(os.path.join(self.output_dir, filename))
#             if img is not None:
#                 plt.subplot(2, 4, i + 1)
#                 plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 plt.title(filename.split('_')[1:3])
#                 plt.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#
# # 示例用法
# if __name__ == "__main__":
#     detector = PipeJointDetector()
#     image_path = "TEST_PHOTO/AJ_20.png"  # 替换为您的图像路径
#
#     try:
#         analysis_result = detector.detect_defect(image_path)
#         print("\n📊 检测报告:")
#         for key, value in analysis_result.items():
#             if key != "debug_images":
#                 print(f"{key}: {value}")
#
#         detector.visualize(image_path, analysis_result)
#
#     except Exception as e:
#         print(f"❌ 处理过程中发生错误: {str(e)}")
#
#
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
#
# class PipeDeformationDetector:
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
# if __name__ == '__main__':
#     """Example usage of PipeDeformationDetector."""
#     detector = PipeDeformationDetector()
#     result = detector.detect_deformation("TEST_PHOTO/BX_1.png")
#
#     print("\n📊 Detection Report:")
#     for k, v in result.items():
#         if k != 'debug_images':
#             print(f"{k:>18}: {v}")
#
#     detector.show_debug_images(result)
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# class PipeSedimentDetector:
#     """A class to detect sediment in pipe images and calculate sediment ratio and thickness."""
#
#     def __init__(self, pipe_diameter_pixels=None, debug_dir="CJ_debug_output"):
#         """Initialize the detector with an optional pipe diameter and debug directory.
#
#         Args:
#             pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
#             debug_dir (str): Directory to save debug images. Defaults to 'CJ_debug_output'.
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
#         """Process the image to detect sediment and calculate its ratio and thickness."""
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
#         # Create sediment mask
#         bottom_gray = gray[center_y:, :]
#         _, original_mask = cv2.threshold(bottom_gray, 50, 255, cv2.THRESH_BINARY_INV)
#         sediment_mask = cv2.bitwise_not(original_mask)
#         self._save_debug_image(sediment_mask, f"{img_name}_02_sediment_mask.jpg", 'gray')
#
#         # Create full circle mask and take lower half
#         circle_mask_full = np.zeros_like(gray, dtype=np.uint8)
#         cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
#         circle_mask_lower = circle_mask_full[center_y:, :]
#         self._save_debug_image(circle_mask_lower, f"{img_name}_03_circle_mask_lower.jpg", 'gray')
#
#         # Intersect sediment mask with circle mask
#         sediment_inside_circle = cv2.bitwise_and(sediment_mask, sediment_mask, mask=circle_mask_lower)
#         self._save_debug_image(sediment_inside_circle, f"{img_name}_04_sediment_inside_circle.jpg", 'gray')
#
#         # Calculate areas
#         half_circle_area = cv2.countNonZero(circle_mask_lower)
#         sediment_area = cv2.countNonZero(sediment_inside_circle)
#         print(f"✅ Circle area: {half_circle_area*2} px, Lower half circle area: {half_circle_area} px, Sediment area: {sediment_area} px")
#
#         # Sediment ratio (%)
#         sediment_ratio = (sediment_area / (half_circle_area*2)) * 100 if half_circle_area > 0 else 0
#         sediment_ratio = min(sediment_ratio, 100.0)
#
#         # Calculate maximum sediment thickness
#         max_thickness = 0
#         if sediment_area > 0:
#             projection = np.sum(sediment_inside_circle, axis=1)
#             non_zero_rows = np.where(projection > 0)[0]
#             if len(non_zero_rows) > 0:
#                 max_thickness = len(non_zero_rows)  # Vertical extent of sediment in pixels
#         print(f"Max sediment thickness: {max_thickness} px")
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
#         # Sediment area (green highlight)
#         sediment_colored = np.zeros_like(img)
#         sediment_colored[center_y:, :] = cv2.merge([
#             np.zeros_like(sediment_inside_circle),
#             sediment_inside_circle,  # Green
#             np.zeros_like(sediment_inside_circle)
#         ])
#         debug_vis = cv2.addWeighted(debug_vis, 1.0, sediment_colored, 0.6, 0)
#
#         # Add text for sediment area
#         cv2.putText(debug_vis, "Sediment (YJ)", (center_x - 150, center_y + 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Draw circle boundary
#         cv2.circle(debug_vis, (center_x, center_y), radius, (255, 255, 0), 2)
#         cv2.putText(debug_vis, "Boundary", (center_x + radius + 10, center_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#
#         # Add sediment ratio text
#         text = f"Sediment percentage: {sediment_ratio:.1f}%"
#         cv2.putText(debug_vis, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
#
#         # Save visualization
#         self._save_debug_image(debug_vis, f"{img_name}_05_result.jpg")
#
#         return self._generate_result(sediment_ratio), max_thickness
#
#     def _generate_result(self, sediment_ratio):
#         """Generate a result dictionary based on the sediment area ratio (% of lower half-circle)."""
#         result = {
#             'Defect Name': 'Sediment',
#             'Defect Code': 'CJ',
#             'Sediment Area Ratio (%)': round(sediment_ratio, 2),
#             'Defect Level': 0,
#             'Defect Description': 'No sediment defect',
#             'Score': 0
#         }
#
#         if 20 <= sediment_ratio < 30:
#             result.update({
#                 'Defect Level': 1,
#                 'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (20%-30%)',
#                 'Score': 0.5
#             })
#         elif 30 <= sediment_ratio < 40:
#             result.update({
#                 'Defect Level': 2,
#                 'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (30%-40%)',
#                 'Score': 2
#             })
#         elif 40 <= sediment_ratio < 50:
#             result.update({
#                 'Defect Level': 3,
#                 'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (40%-50%)',
#                 'Score': 5
#             })
#         elif sediment_ratio >= 50:
#             result.update({
#                 'Defect Level': 4,
#                 'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (>50%)',
#                 'Score': 10
#             })
#
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
#             dict: Result dictionary containing sediment analysis.
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
#         if self.pipe_diameter:
#             cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle
#
#         # Draw sediment contours
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
#         self._save_debug_image(mask, f"{img_name}_06_hsv_mask.jpg", 'gray')
#         bottom_region = mask[center_y:, :]
#         sediment_mask = cv2.bitwise_not(bottom_region)
#         kernel = np.ones((5, 5), np.uint8)
#         sediment_mask = cv2.erode(sediment_mask, kernel, iterations=2)
#         self._save_debug_image(sediment_mask, f"{img_name}_07_eroded_sediment_mask.jpg", 'gray')
#         contours, _ = cv2.findContours(sediment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(img[center_y:, :], contours, -1, (0, 255, 255), 2)  # Cyan contours
#
#         # Add text annotations
#         cv2.putText(img, f"Pipe Diameter: {self.pipe_diameter}px", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
#         cv2.putText(img, f"Sediment Ratio: {result['Sediment Area Ratio (%)']:.1f}%", (10, 30),
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
#         """Calculate the sediment area in pixels.
#
#         Args:
#             img_path (str): Path to the input image.
#
#         Returns:
#             float: Estimated sediment area in pixels.
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
#             raise ValueError("Pipe diameter must be set before calculating sediment area")
#         if self.pipe_diameter > min(height, width):
#             raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")
#
#         # Create sediment mask
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
#         bottom_region = mask[center_y:, :]
#         sediment_mask = cv2.bitwise_not(bottom_region)
#         kernel = np.ones((5, 5), np.uint8)
#         sediment_mask = cv2.erode(sediment_mask, kernel, iterations=2)
#         self._save_debug_image(sediment_mask, f"{img_name}_09_sediment_area_mask.jpg", 'gray')
#
#         # Create circle mask for lower half
#         circle_mask_full = np.zeros_like(mask, dtype=np.uint8)
#         cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
#         circle_mask_lower = circle_mask_full[center_y:, :]
#         sediment_inside_circle = cv2.bitwise_and(sediment_mask, sediment_mask, mask=circle_mask_lower)
#         self._save_debug_image(sediment_inside_circle, f"{img_name}_10_sediment_area_result.jpg", 'gray')
#
#         # Calculate area
#         sediment_area = cv2.countNonZero(sediment_inside_circle)
#         print(f"Estimated sediment area: {sediment_area} px²")
#         return float(sediment_area)
#
#
# if __name__ == "__main__":
#     """Example usage of PipeSedimentDetector."""
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Pipe Sediment Detection")
#     parser.add_argument("--image", default="Test_PHOTO/CJ1.jpg", help="Path to the input image")
#     parser.add_argument("--output", default="result_with_diameter.jpg", help="Path to save the output image")
#     args = parser.parse_args()
#
#     detector = PipeSedimentDetector()
#
#     # Automatically estimate pipe diameter
#     try:
#         detector.auto_estimate_diameter(args.image, method="auto")
#         if detector.pipe_diameter is None:
#             detector.pipe_diameter = 342  # Fallback to manual setting
#             print("Automatic estimation failed! Using manual diameter: 342 px")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Process the image
#     try:
#         result, max_thickness = detector.process_image(args.image)
#         print("\nDetection Result:")
#         for k, v in result.items():
#             print(f"{k}: {v}")
#         print(f"Max Thickness: {max_thickness} px")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Visualize results
#     try:
#         detector.visualize(args.image, args.output)
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Calculate sediment area
#     try:
#         area = detector.calculate_sediment_area(args.image)
#         print(f"Estimated sediment area: {area} px²")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
#
# class PipeDeformationDetector:
#     def __init__(self, debug_dir="CK_debug_output"):
#         self.debug_dir = debug_dir
#         os.makedirs(debug_dir, exist_ok=True)
#
#     def _save_debug_image(self, image, name, img_name):
#         """保存调试图像到指定目录"""
#         save_path = os.path.join(self.debug_dir, f"{img_name}_{name}.jpg")
#         cv2.imwrite(save_path, image)
#         return save_path
#
#     def detect_deformation(self, img_path):
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"图像未找到: {img_path}")
#
#         # 1. 原始图像
#         self._save_debug_image(img, "01_original", img_name)
#
#         # 2. 灰度化处理
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         self._save_debug_image(gray, "02_gray", img_name)
#
#         # 3. 高斯模糊
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         self._save_debug_image(blurred, "03_blurred", img_name)
#
#         # 4. 二值化处理
#         _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
#         self._save_debug_image(binary, "04_binary", img_name)
#
#         # 5. 轮廓检测
#         contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             raise ValueError("❌ 没有检测到任何轮廓")
#
#         # 绘制所有轮廓（调试用）
#         all_contours_img = img.copy()
#         cv2.drawContours(all_contours_img, contours, -1, (0, 255, 255), 2)
#         self._save_debug_image(all_contours_img, "05_all_contours", img_name)
#
#         # 6. 筛选最大两个轮廓
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
#         if len(contours) < 2:
#             raise ValueError("❌ 检测到轮廓不足两个，无法计算错口")
#
#         # 绘制筛选后的轮廓（调试用）
#         top_contours_img = img.copy()
#         cv2.drawContours(top_contours_img, contours, -1, (0, 255, 0), 2)
#         for i, cnt in enumerate(contours):
#             M = cv2.moments(cnt)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 cv2.putText(top_contours_img, f"Contour {i + 1}", (cX, cY),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         self._save_debug_image(top_contours_img, "06_top_contours", img_name)
#
#         # 7. 椭圆拟合
#         ellipses = [cv2.fitEllipse(c) for c in contours if len(c) >= 5]
#         if len(ellipses) < 2:
#             raise ValueError("❌ 轮廓点不足，无法拟合两个椭圆")
#
#         # 绘制椭圆拟合结果（调试用）
#         ellipse_img = img.copy()
#         for i, ellipse in enumerate(ellipses):
#             color = (0, 255, 0) if i == 0 else (0, 0, 255)  # 外椭圆绿色，内椭圆红色
#             cv2.ellipse(ellipse_img, ellipse, color, 2)
#
#             # 显示椭圆参数
#             (cx, cy), (w, h), angle = ellipse
#             text = f"Ellipse {i + 1}: ({cx:.1f},{cy:.1f}) {w:.1f}x{h:.1f} {angle:.1f}°"
#             cv2.putText(ellipse_img, text, (10, 30 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         self._save_debug_image(ellipse_img, "07_ellipses", img_name)
#
#         # 8. 计算参数
#         (cx1, cy1), (w1, h1), _ = ellipses[0]  # 外椭圆
#         (cx2, cy2), (w2, h2), _ = ellipses[1]  # 内椭圆
#
#         major_axis1, minor_axis1 = max(w1, h1), min(w1, h1)
#         major_axis2, minor_axis2 = max(w2, h2), min(w2, h2)
#         wall_thickness = min(abs(major_axis1 - major_axis2), abs(minor_axis1 - minor_axis2)) / 2
#
#         misalignment = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
#         misalignment_ratio = misalignment / wall_thickness
#
#         # 9. 最终可视化
#         vis_img = img.copy()
#         cv2.ellipse(vis_img, ellipses[0], (0, 255, 0), 2)  # 外椭圆绿色
#         cv2.ellipse(vis_img, ellipses[1], (0, 0, 255), 2)  # 内椭圆红色
#
#         # 绘制中心线
#         cv2.line(vis_img, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)
#
#         # 添加详细参数信息
#         info_text = [
#             f"Outer Ellipse: {w1:.1f}x{h1:.1f}",
#             f"Inner Ellipse: {w2:.1f}x{h2:.1f}",
#             f"Wall Thickness: {wall_thickness:.1f}px",
#             f"Center Offset: {misalignment:.1f}px",
#             f"Misalignment Ratio: {misalignment_ratio:.2f}x wall",
#             f"Grade: {self.grade_misalignment(misalignment_ratio)}"
#         ]
#
#         for i, text in enumerate(info_text):
#             cv2.putText(vis_img, text, (10, 30 + i * 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#         result_path = self._save_debug_image(vis_img, "08_final_result", img_name)
#
#         print("\n🔍 调试信息:")
#         print(f" - 外椭圆尺寸: {w1:.1f}x{h1:.1f}")
#         print(f" - 内椭圆尺寸: {w2:.1f}x{h2:.1f}")
#         print(f" - 计算壁厚: {wall_thickness:.1f} 像素")
#         print(f" - 中心偏移距离: {misalignment:.1f} 像素")
#         print(f"📐 检测结果 - 错口率: {misalignment_ratio:.2f}倍管壁厚")
#
#         return {
#             "image": img_path,
#             "outer_ellipse": (w1, h1),
#             "inner_ellipse": (w2, h2),
#             "wall_thickness": wall_thickness,
#             "misalignment": misalignment,
#             "misalignment_ratio": misalignment_ratio,
#             "level": self.grade_misalignment(misalignment_ratio),
#             "result_image": result_path,
#             "debug_images": [
#                 f"{img_name}_01_original.jpg",
#                 f"{img_name}_02_gray.jpg",
#                 f"{img_name}_03_blurred.jpg",
#                 f"{img_name}_04_binary.jpg",
#                 f"{img_name}_05_all_contours.jpg",
#                 f"{img_name}_06_top_contours.jpg",
#                 f"{img_name}_07_ellipses.jpg",
#                 f"{img_name}_08_final_result.jpg"
#             ]
#         }
#
#     def grade_misalignment(self, ratio):
#         """根据错口等级判断"""
#         if ratio < 1.5:
#             return "Ⅰ级（正常或轻微错口）"
#         elif ratio < 2.0:
#             return "Ⅱ级（轻中度错口）"
#         else:
#             return "Ⅲ级（严重错口）"
#
#     def show_debug_images(self, img_name):
#         """显示所有调试图像"""
#         plt.figure(figsize=(15, 10))
#
#         debug_images = [
#             "01_original", "02_gray", "03_blurred", "04_binary",
#             "05_all_contours", "06_top_contours", "07_ellipses", "08_final_result"
#         ]
#
#         for i, name in enumerate(debug_images, 1):
#             img_path = os.path.join(self.debug_dir, f"{img_name}_{name}.jpg")
#             img = cv2.imread(img_path)
#             if img is not None:
#                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 plt.subplot(2, 4, i)
#                 plt.imshow(img_rgb)
#                 plt.title(name.replace("_", " ").title())
#                 plt.axis("off")
#
#         plt.tight_layout()
#         plt.show()
#
#
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
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# class PipeSedimentDetector:
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
#
# if __name__ == "__main__":
#     """Example usage of PipeSedimentDetector."""
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Pipe Floating Debris Detection")
#     parser.add_argument("--image", default="Test_PHOTO/FZ.png", help="Path to the input image")
#     parser.add_argument("--output", default="result_with_diameter.jpg", help="Path to save the output image")
#     args = parser.parse_args()
#
#     detector = PipeSedimentDetector()
#
#     # Automatically estimate pipe diameter
#     try:
#         detector.auto_estimate_diameter(args.image, method="auto")
#         if detector.pipe_diameter is None:
#             detector.pipe_diameter = 342  # Fallback to manual setting
#             print("Automatic estimation failed! Using manual diameter: 342 px")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Process the image
#     try:
#         result, max_thickness = detector.process_image(args.image)
#         print("\nDetection Result:")
#         for k, v in result.items():
#             print(f"{k}: {v}")
#         print(f"Max Thickness: {max_thickness} px")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Visualize results
#     try:
#         detector.visualize(args.image, args.output)
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
#
#     # Calculate debris area
#     try:
#         area = detector.calculate_sediment_area(args.image)
#         print(f"Estimated debris area: {area} px²")
#     except ValueError as e:
#         print(f"Error: {e}")
#         exit(1)
# import cv2
# import numpy as np
# import os
# import subprocess
# import matplotlib.pyplot as plt
#
# class PipeScaleDetector:
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
# if __name__ == '__main__':
#     detector = PipeScaleDetector()
#     result = detector.calculate_blockage_ratio("TEST_PHOTO/JG.png")  # 替换为你的图像路径
#     print("\n最终结果:")
#     for k, v in result.items():
#         print(f"{k}: {v}")
#
#     PipeScaleDetector.read_photo("JG_debug_output/JG_05_debug.jpg") # 替换为你的图像路径
#
#
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# class PipeWaterDepthDetector:
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
# if __name__ == "__main__":
#     # Initialize the detector
#     detector = PipeWaterDepthDetector()
#
#     # Automatically estimate pipe diameter
#     detector.auto_estimate_diameter("Test_PHOTO/QF.png", method="auto")
#     if detector.pipe_diameter is None:
#         detector.pipe_diameter = 342  # Fallback to manual setting if auto-estimation fails
#         print("自动估算失败！！！使用手动设置直径: 342 px")
#
#     # Process the image for water depth analysis
#     result, water_depth_pixels = detector.detect_water_depth("Test_PHOTO/QF.png")
#     print("检测结果:")
#     for k, v in result.items():
#         print(f"{k}: {v}")
#     print(f"检测到的水深 (像素): {water_depth_pixels} px")
#
#     # Visualize the results
#     detector.visualize("Test_PHOTO/QF.png", "QF_debug_output/result_water_depth_diameter_ratio.jpg")
#
#     # Test water depth pixels calculation
#     depth_pixels = detector.calculate_water_depth_pixels("Test_PHOTO/QF.png")
#     print(f"估算水深: {depth_pixels} px")
#
#
