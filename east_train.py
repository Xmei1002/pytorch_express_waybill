
from torch.utils.data import DataLoader
from models.east_model import EAST  
from utils.dataset import MyDataset 
from config import root_dir, train_data_file, test_data_file
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.east_model import EAST
from utils.dataset import MyDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def collate_fn(batch):
    images, transcriptions, boxes = zip(*batch)
    images = torch.stack(images, 0)  # 合并图像张量
    return images, transcriptions, boxes

# np.set_printoptions(threshold=np.inf)
#可视化geo_map
def visualize_geo_map(geo_map):
    # geo_map 的通道: [0] 和 [1] 是中心点坐标偏移，[2] 是宽度，[3] 是高度，[4] 是旋转角度
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Center X Offset', 'Center Y Offset', 'Width', 'Height', 'Angle']

    for i in range(5):
        ax = axes[i]
        ax.imshow(geo_map[i,...], cmap='jet')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def generate_target_maps(points_list, image_size):
    _, _, h, w = image_size
    score_map = np.zeros((h, w), dtype=np.uint8)  # 使用 uint8 类型
    geo_map = np.zeros((5, h, w), dtype=np.float32)  # 几何图可以保持为 float32

    for points in points_list:
        points_array = np.array([[p.item() for p in point] for point in points], dtype=np.float32)
        if np.any(points_array[:, 0] >= w) or np.any(points_array[:, 1] >= h):
            continue  # 如果坐标超出范围，则跳过此文本区域

        # 计算矩形框的四个角点
        rect = cv2.minAreaRect(points_array)
        box = cv2.boxPoints(rect).astype(np.int32)
        box = np.int32(box)  # 将 box 的类型转换为整数

        cv2.fillPoly(score_map, [box], 1)  # 使用255作为填充值
        plt.imshow(score_map, cmap='gray')

        # 计算矩形框的中心点，宽度，高度和旋转角度
        center_x, center_y = rect[0]
        width, height = rect[1]
        angle = rect[2]

        # 确保分数图中的像素被正确标记
        for i in range(h):
            for j in range(w):
                if score_map[i, j] > 0:  # 仅处理分数图中标记为文本的像素
                    geo_map[0, i, j] = center_x - j
                    geo_map[1, i, j] = center_y - i
                    geo_map[2, i, j] = width
                    geo_map[3, i, j] = height
                    geo_map[4, i, j] = angle

    # visualize_geo_map(geo_map)
    return score_map, geo_map

# 损失计算
def east_loss_function(predicted_scores, predicted_geos, target_scores, target_geos, criterion1, criterion2):
    score_loss = criterion1(predicted_scores, target_scores)
    geo_loss = criterion2(predicted_geos, target_geos)
    total_loss = score_loss + geo_loss
    # print(score_loss,geo_loss)
    return total_loss


# 加载数据集
def load_data(root_dir, data_file, transform):
    dataset = MyDataset(root_dir, data_file, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)

# 训练函数
def train(model, device, train_loader, optimizer, criterion1, criterion2, epoch, print_freq=10):
    model.train()
    total_loss = 0
    for i, (img, _, boxes_list) in enumerate(train_loader):
        img = img.to(device)
        # 处理数据并生成目标图
        # print('boxes_list',boxes_list)
        target_score_map, target_geo_map = generate_target_maps(boxes_list, img.shape)

        target_score_map = torch.tensor(target_score_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 添加两个维度
        target_geo_map = torch.tensor(target_geo_map, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度

        optimizer.zero_grad()
        predicted_scores, predicted_geos = model(img)  # 获取模型的两个输出       
        predicted_scores = F.interpolate(predicted_scores, size=(1080, 1080), mode='bilinear')
        predicted_geos = F.interpolate(predicted_geos, size=(1080, 1080), mode='bilinear')
        
        loss = east_loss_function(predicted_scores, predicted_geos, target_score_map, target_geo_map, criterion1, criterion2)  # 计算损失
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % print_freq == 0:
            print(f'Epoch [{epoch+1}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((1080, 1080)),
        transforms.ToTensor(),
    ])

    train_loader = load_data(root_dir, train_data_file, transform)
    model = EAST().to(device)


    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion1 = nn.BCEWithLogitsLoss()  # 分数图的损失函数
    criterion2 = nn.MSELoss()  # 几何图的损失函数

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        avg_loss = train(model, device, train_loader, optimizer, criterion1, criterion2, epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}')

    # 保存模型
    torch.save(model.state_dict(), 'east_model.pth')

if __name__ == "__main__":
    main()
