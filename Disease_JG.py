import cv2
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt

class PipeScaleDetector:
    def __init__(self, debug_dir="JG_debug_output"):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    def detect_pipe(self, img_gray):
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

    def calculate_blockage_ratio(self, img_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("无法加载图像")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cx, cy, r = self.detect_pipe(blurred)

        # 创建管道圆形掩膜
        mask_pipe = np.zeros_like(gray)
        cv2.circle(mask_pipe, (cx, cy), r, 255, -1)

        # 阈值分割出结垢区域（暗区域）
        _, mask_scale = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        scale_only = cv2.bitwise_and(mask_scale, mask_pipe)

        # 取反（仅限管道内区域）
        scale_only_inv = np.zeros_like(scale_only)
        scale_only_inv[mask_pipe > 0] = 255 - scale_only[mask_pipe > 0]

        # 仅统计管道圆内区域的像素
        pipe_area = np.count_nonzero(mask_pipe)
        scale_area = np.count_nonzero(scale_only_inv)
        loss_ratio = scale_area / pipe_area

        # 绘制调试图像
        debug_img = img.copy()
        cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Blockage: {loss_ratio:.2%}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 创建透明绿色图层
        overlay = debug_img.copy()
        green_color = (0, 255, 0)  # 绿色
        alpha = 0.4  # 透明度

        # 只在管道掩膜内叠加绿色
        mask_combined = np.logical_and(scale_only_inv > 0, mask_pipe > 0)
        overlay[mask_combined] = green_color

        # 混合图像（透明叠加）
        cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)

        # 保存调试图像
        cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_01_gray.jpg"), gray)
        cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_02_pipe_mask.jpg"), mask_pipe)
        cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_03_scale_mask.jpg"), mask_scale)
        cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_04_scale_only.jpg"), scale_only)
        cv2.imwrite(os.path.join(self.debug_dir, f"{img_name}_05_debug.jpg"), debug_img)


        # 判定等级和评分
        score, level = self.grade_defect(loss_ratio)

        print(f"📊 管道结构结垢率: {loss_ratio:.2%}, 等级: {level}, 评分: {score}")

        return {
            "image": img_path,
            "blockage_ratio": loss_ratio,
            "level": level,
            "score": score
        }

    def grade_defect(self, ratio):
        if ratio <= 0.15:
            return 0.5, 1
        elif ratio <= 0.25:
            return 2, 2
        elif ratio <= 0.5:
            return 5, 3
        else:
            return 10, 4

    def read_photo(img_path):
        # 读取图像（以 BGR 格式）
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图像未找到: {img_path}")

        # 将 BGR 转为 RGB（matplotlib 正确显示颜色）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title("result")
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    detector = PipeScaleDetector()
    result = detector.calculate_blockage_ratio("TEST_PHOTO/JG.png")  # 替换为你的图像路径
    print("\n最终结果:")
    for k, v in result.items():
        print(f"{k}: {v}")

    PipeScaleDetector.read_photo("JG_debug_output/JG_05_debug.jpg") # 替换为你的图像路径



