import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 🟢 解决 OpenMP 冲突

from ultralytics import YOLO

# 1. 加载模型（使用轻量级旋转框预训练模型 yolov8n-obb）
model = YOLO('yolov8n-obb.pt')

# 2. 开始训练
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=4,                   # CPU下建议用小批量
    name='yolo_obb_sewer',
    project='runs/train',
    verbose=True,
    resume=False,
    device='cpu'               # ← 没有GPU就用 'cpu'
)
