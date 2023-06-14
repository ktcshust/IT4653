import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path

# Đường dẫn đến file CSV chứa thông tin nhãn
csv_path = "./data/labels.csv"

# Đường dẫn đến thư mục chứa ảnh
image_dir = "./data/resized"

# Load dữ liệu từ file CSV
df = pd.read_csv(csv_path)

# Chia train/test dataset
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

# Hàm chuyển đổi ảnh thành tensor
def image_to_tensor(image):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

# Lớp YOLOv5
class YOLOv5:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        
    def predict(self, image):
        results = self.model(image)
        labels = results.pandas().xyxy[0].class_name.tolist()
        return labels

# Khởi tạo đối tượng YOLOv5
yolov5 = YOLOv5('yolov5s.pt')

# Hàm dự đoán nhãn của ảnh
def predict_labels(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_to_tensor(image)
    labels = yolov5.predict(image_tensor)
    return labels

# Dự đoán nhãn cho tập test
for index, row in test_df.iterrows():
    image_path = row['id']
    image_path = Path(image_dir) / Path(image_path)
    labels = predict_labels(str(image_path))
    print(f"Image: {image_path}, Labels: {labels}")
