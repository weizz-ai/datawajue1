from ultralytics import YOLO # type: ignore


model = YOLO('yolov8n-cls.pt')  # 使用 YOLOv8 nano 版本的分类模型

# 训练分类模型
model.train(
    data= 'F:\\datawajue\\data.yaml',   # 训练数据路径
    epochs=300,         # 训练的轮数
    imgsz=624,         # 图片尺寸
    batch=16,          # 批次大小
    lr0=0.001          # 学习率
)
# 评估模型
metrics = model.val()
# 导出模型
model.export(format='onnx')  # 导出为 ONNX 格式

