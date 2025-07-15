import os
import cv2
import sys
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

# 关闭 tkinter 主窗口
root = tk.Tk()
root.withdraw()

# ========== 模型路径配置 ==========
model_path = 'runs/train/yolo_obb_sewer10/weights/best.pt'  # 修改为你的模型路径
img_size = 640  # 推理输入尺寸，必须和训练一致

# ========== 选择输入文件夹 ==========
input_folder = filedialog.askdirectory(title="选择包含图片的输入文件夹")
if not input_folder:
    print("未选择输入文件夹，已取消。")
    sys.exit()

# ========== 选择输出文件夹 ==========
output_folder = filedialog.askdirectory(title="选择输出结果保存文件夹")
if not output_folder:
    print("未选择输出文件夹，已取消。")
    sys.exit()

# ========== 初始化模型 ==========
model = YOLO(model_path)

# ========== 遍历图片 ==========
image_paths = list(Path(input_folder).glob('*.jpg')) + list(Path(input_folder).glob('*.png'))

if not image_paths:
    messagebox.showinfo("提示", "未在选中的文件夹中找到 .jpg 或 .png 图片。")
    sys.exit()

print(f"共检测到 {len(image_paths)} 张图片，开始处理...")

for idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"跳过无法读取的文件：{img_path}")
        continue

    results = model(img, imgsz=img_size, device='cpu')[0]

    for box in results.obb:
        if not hasattr(box, 'xyxy') or box.xyxy is None:
            continue

        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls_id]} {conf:.2f}"

        # 旋转框点集
        points = box.xyxyxyxy.cpu().numpy().reshape(-1, 2).astype(int)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    out_path = os.path.join(output_folder, img_path.name)
    cv2.imwrite(out_path, img)
    print(f"[{idx}/{len(image_paths)}] 已处理: {img_path.name}")

# ========== 弹出完成提示 ==========
messagebox.showinfo("完成", f"图片处理完成，结果保存在：\n{output_folder}")

# ========== 打开输出文件夹 ==========
if sys.platform == 'win32':
    os.startfile(output_folder)
elif sys.platform == 'darwin':
    subprocess.call(['open', output_folder])
else:
    subprocess.call(['xdg-open', output_folder])
