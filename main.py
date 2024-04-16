from ultralytics import YOLO
import os

# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）
if __name__ == '__main__':
    # Use the model
    results = model.train(data="ultralytics/datasets/sample_data.yaml", epochs=20, batch=-1, imgsz=320,
                          device=0)  # 训练模型
