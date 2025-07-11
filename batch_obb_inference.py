import os
import cv2
from ultralytics import YOLO
from pathlib import Path

# ========== 配置部分 ==========
model_path = 'runs/train/yolo_obb_sewer10/weights/best.pt'  # 你训练好的模型路径
input_folder = 'dataset/images/val'  # 存放原始图片的文件夹
output_folder = 'results/'  # 存放标注后图片的文件夹
img_size = 640  # 模型输入尺寸（必须和训练时一致）

# ========== 初始化 ==========
model = YOLO(model_path)
os.makedirs(output_folder, exist_ok=True)

# ========== 遍历并推理 ==========
image_paths = list(Path(input_folder).glob('*.jpg')) + list(Path(input_folder).glob('*.png'))

print(f"共检测到 {len(image_paths)} 张图片，开始处理...")

for idx, img_path in enumerate(image_paths, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"跳过无法读取的文件：{img_path}")
        continue

    # 使用模型进行推理
    results = model(img, imgsz=img_size, device='cpu')[0]

    # 绘制结果
    for box in results.obb:
        xyxy = box.xyxy.numpy().astype(int)[0]
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls_id]} {conf:.2f}"

        # 画旋转框
        points = box.xyxy.numpy().reshape(-1, 2).astype(int)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 保存结果
    out_path = os.path.join(output_folder, img_path.name)
    cv2.imwrite(out_path, img)
    print(f"[{idx}/{len(image_paths)}] 已处理: {img_path.name}")

print("全部处理完成！输出保存在：", output_folder)
