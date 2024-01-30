from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.faster_r_cnn_model import get_detection_model
from PIL import ImageDraw,Image
import matplotlib.pyplot as plt
from utils.dataset import MyDataset 
from config import root_dir, train_data_file, test_data_file
import torch
import math

# 加载模型
num_classes = 4
model = get_detection_model(num_classes)
model.load_state_dict(torch.load('text_detection_model_address_type2.pth'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# 创建数据集和数据加载器
transform = transforms.Compose([
    # transforms.Resize((1080, 1080)),
    transforms.ToTensor()
])

dataset = MyDataset(root_dir, test_data_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 遍历数据集
for images, targets, _ in data_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)
    # print('>>>>>>>',outputs)

    # 可视化第一个图像和检测框
    if len(images) > 0:
        image = images[0].permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype('uint8')
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        if len(outputs) > 0:
            boxes = outputs[0]['boxes']
            scores = outputs[0]['scores']
            # print('yyy',boxes)
            # 获取得分最高的三个框
            _, indices = scores.sort(descending=True)
            top_indices = indices[:4]  # 取前三个
            top_boxes = boxes[top_indices]
            for box in top_boxes:
                box = [math.ceil(b) for b in box.tolist()]
                draw.rectangle(box, outline="green", width=3)

            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.show()
