import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class PipeSedimentDetector:
    """A class to detect sediment in pipe images and calculate sediment ratio and thickness."""

    def __init__(self, pipe_diameter_pixels=None, debug_dir="CJ_debug_output"):
        """Initialize the detector with an optional pipe diameter and debug directory.

        Args:
            pipe_diameter_pixels (int, optional): The diameter of the pipe in pixels. If None, it will be estimated.
            debug_dir (str): Directory to save debug images. Defaults to 'CJ_debug_output'.
        """
        self.pipe_diameter = pipe_diameter_pixels
        self.debug_dir = debug_dir
        self.lower_black = np.array([0, 0, 0])  # Lower bound for black color in HSV
        self.upper_black = np.array([180, 255, 50])  # Upper bound for black color in HSV
        os.makedirs(debug_dir, exist_ok=True)

    def _save_debug_image(self, img, name, cmap=None):
        """Save debug image to the specified directory."""
        if cmap == 'gray':
            plt.imsave(os.path.join(self.debug_dir, name), img, cmap='gray')
        else:
            if len(img.shape) == 2:  # Convert single-channel image to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(self.debug_dir, name), img)

    def auto_estimate_diameter(self, img_path, method="auto"):
        """Automatically estimate the pipe diameter using Hough transform or projection method.

        Args:
            img_path (str): Path to the input image.
            method (str): Method to use ('hough', 'projection', or 'auto'). Defaults to 'auto'.

        Returns:
            int or None: Estimated diameter in pixels, or None if estimation fails.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
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

    def process_image(self, img_path):
        """Process the image to detect sediment and calculate its ratio and thickness."""
        if self.pipe_diameter is None:
            raise ValueError("Please set or estimate pipe_diameter first")

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ Unable to load image, please check file path")

        height, width = img.shape[:2]
        if self.pipe_diameter > min(height, width):
            raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")

        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, f"{img_name}_01_gray.jpg", 'gray')

        # Create sediment mask
        bottom_gray = gray[center_y:, :]
        _, original_mask = cv2.threshold(bottom_gray, 50, 255, cv2.THRESH_BINARY_INV)
        sediment_mask = cv2.bitwise_not(original_mask)
        self._save_debug_image(sediment_mask, f"{img_name}_02_sediment_mask.jpg", 'gray')

        # Create full circle mask and take lower half
        circle_mask_full = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
        circle_mask_lower = circle_mask_full[center_y:, :]
        self._save_debug_image(circle_mask_lower, f"{img_name}_03_circle_mask_lower.jpg", 'gray')

        # Intersect sediment mask with circle mask
        sediment_inside_circle = cv2.bitwise_and(sediment_mask, sediment_mask, mask=circle_mask_lower)
        self._save_debug_image(sediment_inside_circle, f"{img_name}_04_sediment_inside_circle.jpg", 'gray')

        # Calculate areas
        half_circle_area = cv2.countNonZero(circle_mask_lower)
        sediment_area = cv2.countNonZero(sediment_inside_circle)
        print(f"✅ Circle area: {half_circle_area*2} px, Lower half circle area: {half_circle_area} px, Sediment area: {sediment_area} px")

        # Sediment ratio (%)
        sediment_ratio = (sediment_area / (half_circle_area*2)) * 100 if half_circle_area > 0 else 0
        sediment_ratio = min(sediment_ratio, 100.0)

        # Calculate maximum sediment thickness
        max_thickness = 0
        if sediment_area > 0:
            projection = np.sum(sediment_inside_circle, axis=1)
            non_zero_rows = np.where(projection > 0)[0]
            if len(non_zero_rows) > 0:
                max_thickness = len(non_zero_rows)  # Vertical extent of sediment in pixels
        print(f"Max sediment thickness: {max_thickness} px")

        # Visualize debug image
        debug_vis = img.copy()

        # Lower half circle (red translucent)
        lower_circle_mask_colored = np.zeros_like(img)
        lower_circle_mask_colored[center_y:, :] = cv2.merge([
            np.zeros_like(circle_mask_lower),
            np.zeros_like(circle_mask_lower),
            circle_mask_lower  # Red channel
        ])
        debug_vis = cv2.addWeighted(debug_vis, 1.0, lower_circle_mask_colored, 0.3, 0)

        # Add text for analysis area
        cv2.putText(debug_vis, "Analysis area (lower half circle)", (center_x - 150, center_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Sediment area (green highlight)
        sediment_colored = np.zeros_like(img)
        sediment_colored[center_y:, :] = cv2.merge([
            np.zeros_like(sediment_inside_circle),
            sediment_inside_circle,  # Green
            np.zeros_like(sediment_inside_circle)
        ])
        debug_vis = cv2.addWeighted(debug_vis, 1.0, sediment_colored, 0.6, 0)

        # Add text for sediment area
        cv2.putText(debug_vis, "Sediment (YJ)", (center_x - 150, center_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw circle boundary
        cv2.circle(debug_vis, (center_x, center_y), radius, (255, 255, 0), 2)
        cv2.putText(debug_vis, "Boundary", (center_x + radius + 10, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Add sediment ratio text
        text = f"Sediment percentage: {sediment_ratio:.1f}%"
        cv2.putText(debug_vis, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Save visualization
        self._save_debug_image(debug_vis, f"{img_name}_05_result.jpg")

        return self._generate_result(sediment_ratio), max_thickness

    def _generate_result(self, sediment_ratio):
        """Generate a result dictionary based on the sediment area ratio (% of lower half-circle)."""
        result = {
            'Defect Name': 'Sediment',
            'Defect Code': 'CJ',
            'Sediment Area Ratio (%)': round(sediment_ratio, 2),
            'Defect Level': 0,
            'Defect Description': 'No sediment defect',
            'Score': 0
        }

        if 20 <= sediment_ratio < 30:
            result.update({
                'Defect Level': 1,
                'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (20%-30%)',
                'Score': 0.5
            })
        elif 30 <= sediment_ratio < 40:
            result.update({
                'Defect Level': 2,
                'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (30%-40%)',
                'Score': 2
            })
        elif 40 <= sediment_ratio < 50:
            result.update({
                'Defect Level': 3,
                'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (40%-50%)',
                'Score': 5
            })
        elif sediment_ratio >= 50:
            result.update({
                'Defect Level': 4,
                'Defect Description': f'Sediment area occupies {sediment_ratio:.1f}% of lower half circle (>50%)',
                'Score': 10
            })

        return result

    def visualize(self, img_path, save_path=None):
        """Visualize the detection results on the image.

        Args:
            img_path (str): Path to the input image.
            save_path (str, optional): Path to save the output image. Defaults to None.

        Returns:
            dict: Result dictionary containing sediment analysis.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ Unable to load image, please check file path")
        height, width = img.shape[:2]
        if self.pipe_diameter > min(height, width):
            raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")
        result, max_thickness = self.process_image(img_path)

        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2

        # Draw pipe outer circle
        if self.pipe_diameter:
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 0), 2)  # Yellow circle

        # Draw sediment contours
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        self._save_debug_image(mask, f"{img_name}_06_hsv_mask.jpg", 'gray')
        bottom_region = mask[center_y:, :]
        sediment_mask = cv2.bitwise_not(bottom_region)
        kernel = np.ones((5, 5), np.uint8)
        sediment_mask = cv2.erode(sediment_mask, kernel, iterations=2)
        self._save_debug_image(sediment_mask, f"{img_name}_07_eroded_sediment_mask.jpg", 'gray')
        contours, _ = cv2.findContours(sediment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img[center_y:, :], contours, -1, (0, 255, 255), 2)  # Cyan contours

        # Add text annotations
        cv2.putText(img, f"Pipe Diameter: {self.pipe_diameter}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(img, f"Sediment Ratio: {result['Sediment Area Ratio (%)']:.1f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Max Thickness: {max_thickness}px", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Save or display the result
        if save_path:
            cv2.imwrite(save_path, img)
        self._save_debug_image(img, f"{img_name}_08_final_visualization.jpg")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return result

    def calculate_sediment_area(self, img_path):
        """Calculate the sediment area in pixels.

        Args:
            img_path (str): Path to the input image.

        Returns:
            float: Estimated sediment area in pixels.
        """
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("❌ Unable to load image, please check file path")

        height, width = img.shape[:2]
        center_y = height // 2
        center_x = width // 2
        radius = self.pipe_diameter // 2 if self.pipe_diameter else None

        if radius is None:
            raise ValueError("Pipe diameter must be set before calculating sediment area")
        if self.pipe_diameter > min(height, width):
            raise ValueError(f"Pipe diameter ({self.pipe_diameter}px) exceeds image dimensions ({width}x{height})")

        # Create sediment mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        bottom_region = mask[center_y:, :]
        sediment_mask = cv2.bitwise_not(bottom_region)
        kernel = np.ones((5, 5), np.uint8)
        sediment_mask = cv2.erode(sediment_mask, kernel, iterations=2)
        self._save_debug_image(sediment_mask, f"{img_name}_09_sediment_area_mask.jpg", 'gray')

        # Create circle mask for lower half
        circle_mask_full = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(circle_mask_full, (center_x, center_y), radius, 255, -1)
        circle_mask_lower = circle_mask_full[center_y:, :]
        sediment_inside_circle = cv2.bitwise_and(sediment_mask, sediment_mask, mask=circle_mask_lower)
        self._save_debug_image(sediment_inside_circle, f"{img_name}_10_sediment_area_result.jpg", 'gray')

        # Calculate area
        sediment_area = cv2.countNonZero(sediment_inside_circle)
        print(f"Estimated sediment area: {sediment_area} px²")
        return float(sediment_area)


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