import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class PipeWaterDepthDetector:
    """A class to detect water depth in pipe images and calculate water depth ratio."""

    def __init__(self, pipe_diameter_pixels=None, debug_dir="QF_debug_output"):
        """Initialize the detector with an optional pipe diameter in pixels.

        Args:
            pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
            debug_dir (str): Directory for saving debug images.
        """
        self.pipe_diameter = pipe_diameter_pixels
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
        self.upper_black = np.array([10, 255, 10])  # Upper bound for black color in HSV

    def auto_estimate_diameter(self, img_path, method="auto"):
        """Automatically estimate the pipe diameter using Hough transform or projection method.

        Args:
            img_path (str): Path to the input image.
            method (str): Method to use ('hough', 'projection', or 'auto'). Defaults to 'auto'.

        Returns:
            int or None: Estimated diameter in pixels, or None if estimation fails.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("❌ 无法加载图像，请检查文件路径")
            return None

        height, width = img.shape
        estimated_diameter = None

        # Method 1: Hough Circle Transform (optimized parameters)
        if method in ("hough", "auto"):
            img_blurred = cv2.GaussianBlur(img, (5, 5), 0)  # Use Gaussian blur for better edge detection
            circles = cv2.HoughCircles(
                img_blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=width // 8,  # Adjusted minDist using image width
                param1=100, param2=20, minRadius=50, maxRadius=width // 2  # Adjusted ranges
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if len(circles[0]) > 0:
                    estimated_diameter = 2 * circles[0][0][2]  # Diameter = 2 * radius
                    print(
                        f"[Hough] 估算管道直径: {estimated_diameter} px, 中心: ({circles[0][0][0]}, {circles[0][0][1]})")
                    self.pipe_diameter = estimated_diameter
                    return estimated_diameter

        # Method 2: Projection Method
        if method in ("projection", "auto"):
            edges = cv2.Canny(img, 50, 150)
            projection = np.sum(edges, axis=1)
            threshold = np.max(projection) * 0.3  # Adjusted threshold for better detection
            rows = np.where(projection > threshold)[0]
            if len(rows) >= 2:
                top, bottom = min(rows), max(rows)
                estimated_diameter = bottom - top
                print(f"[Projection] 估算管道直径: {estimated_diameter} px")
                self.pipe_diameter = estimated_diameter
                return estimated_diameter

        print("❌ 无法自动估算管道直径，请手动设置 pipe_diameter 或检查图像")
        return None

    def detect_water_depth(self, img_path):
        """Detect water depth in the pipe image and calculate its ratio."""
        if self.pipe_diameter is None:
            raise ValueError("请先设置或自动估算管道内径 pipe_diameter")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ 无法加载图像，请检查文件路径")

        height, width = img.shape[:2]
        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Convert to HSV and create water mask (assuming black area is water)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        water_mask_full = cv2.inRange(hsv, self.lower_black, self.upper_black)
        cv2.imwrite(os.path.join(self.debug_dir, "debug_water_mask.jpg"), water_mask_full)

        # Create full circle mask
        circle_mask_full = np.zeros_like(water_mask_full, dtype=np.uint8)
        cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
        cv2.imwrite(os.path.join(self.debug_dir, "debug_circle_mask.jpg"), circle_mask_full)

        # Find water level (top of water area)
        water_contours, _ = cv2.findContours(water_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_contours_img = cv2.cvtColor(water_mask_full, cv2.COLOR_GRAY2BGR)
        if water_contours:
            water_contour = max(water_contours, key=cv2.contourArea)
            water_level_y = np.min(water_contour[:, :, 1])  # Topmost point of water
            cv2.drawContours(debug_contours_img, [water_contour], -1, (0, 255, 0), 2)  # Green contour
            cv2.putText(debug_contours_img, f"Water Level: {water_level_y} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            water_level_y = height  # Default to bottom if no water detected
        cv2.imwrite(os.path.join(self.debug_dir, "debug_water_contours.jpg"), debug_contours_img)
        print(f"🌊 检测到的水面高度: {water_level_y} px")

        # Pipe bottom y-coordinate
        pipe_bottom_y = center_y + radius

        # Actual water depth (pixels) = Pipe bottom - Water level
        water_depth_pixels = pipe_bottom_y - water_level_y
        water_depth_pixels = max(0, min(water_depth_pixels, self.pipe_diameter))  # Ensure within diameter

        # Water depth percentage (% of diameter)
        water_depth_ratio_percent = (water_depth_pixels / self.pipe_diameter) * 100
        water_depth_ratio_percent = max(0, min(water_depth_ratio_percent, 100.0))

        print(
            f"✅ 管道底部 y: {pipe_bottom_y}, 水深 (像素): {water_depth_pixels}, 水深比例 (直径%): {water_depth_ratio_percent:.2f}%")

        # Debug image for water depth region
        debug_depth_region = img.copy()
        cv2.line(debug_depth_region, (0, int(pipe_bottom_y)), (width, int(pipe_bottom_y)), (0, 0, 255), 2)  # Red line for pipe bottom
        cv2.line(debug_depth_region, (0, int(water_level_y)), (width, int(water_level_y)), (0, 255, 0), 2)  # Green line for water level
        cv2.rectangle(debug_depth_region, (center_x - radius, int(water_level_y)),
                      (center_x + radius, int(pipe_bottom_y)), (255, 0, 0), 2)  # Blue rectangle for water depth
        cv2.putText(debug_depth_region, f"Water Depth: {water_depth_pixels} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(self.debug_dir, "debug_water_depth_region.jpg"), debug_depth_region)

        return self._generate_result(water_depth_ratio_percent), water_depth_pixels

    def _generate_result(self, water_depth_ratio):
        """Generate a result dictionary based on the water depth ratio (% of pipe diameter)."""
        result = {
            '缺陷名称': '起伏',
            '缺陷代码': 'QF',
            '水深占比(直径%)': round(water_depth_ratio, 2),  # Modified key to reflect diameter percentage
            '缺陷等级': 0,
            '缺陷描述': '水深正常',
            '分值': 0
        }

        # Adjust thresholds to fit diameter percentage
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

    def show_result(self, result_img_path):
        """Display the detection result image."""
        img = cv2.imread(result_img_path)
        if img is None:
            raise FileNotFoundError(f"找不到图像：{result_img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title("Water Depth Detection Result")
        plt.axis("off")
        plt.show()

    def visualize(self, img_path, save_path=None):
        """Visualize the detection results on the image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ 无法加载图像，请检查文件路径")

        # detect_water_depth now returns a different result format
        result, water_depth_pixels = self.detect_water_depth(img_path)

        height, width = img.shape[:2]
        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Get water level y-coordinate from the detect method
        pipe_bottom_y = center_y + radius
        water_level_y = pipe_bottom_y - water_depth_pixels  # Recalculate water level y

        # Draw pipe outer circle
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle

        # Draw water level line (green)
        cv2.line(img, (0, int(water_level_y)), (width, int(water_level_y)), (0, 255, 0), 2)

        # Draw water depth line (vertical from bottom to water level in red)
        cv2.line(img, (center_x, int(pipe_bottom_y)), (center_x, int(water_level_y)), (0, 0, 255), 2)

        # Add pipe diameter text
        cv2.putText(img, f"Pipe Diameter: {self.pipe_diameter}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Add water depth percentage text
        depth_text = f"Water Depth: {result['水深占比(直径%)']:.1f}%"
        cv2.putText(img, depth_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save or display the result
        if save_path:
            cv2.imwrite(save_path, img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return result

    def calculate_water_depth_pixels(self, img_path):
        """Calculate the water depth in pixels.

        This method is renamed for clarity, as it now returns pixels, not area.
        """
        _, water_depth_pixels = self.detect_water_depth(img_path)
        return water_depth_pixels

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

