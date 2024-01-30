from torch.utils.data import DataLoader
from utils.dataset import MyDataset 
from config import root_dir, train_data_file, test_data_file
from models.faster_r_cnn_model import get_detection_model
import torch
import torchvision.transforms as transforms
from torch.optim import SGD , Adam # 两种优化器
import matplotlib.pyplot as plt

# 参数设置
num_classes = 4  # 包括背景在内的类别数
num_epochs = 60  # 训练周期
learning_rate = 1e-5  # 学习率 0.00001
batch_size = 1  # 批大小  
weight_decay = 0.005 # Adam 权重衰减参数 0.005

def iou(box1, box2):
    """计算两个边界框的交并比"""
    # 计算交集
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    # 计算并集
    union_area = ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
                  (box2[2] - box2[0]) * (box2[3] - box2[1]) -
                  inter_area)

    return inter_area / union_area if union_area != 0 else 0

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """根据预测和真实目标计算TP, FP, FN, 准确率, 召回率, F1分数"""
    TP = FP = FN = 0

    for i in range(len(predictions)):
        pred_boxes = predictions[i]["boxes"]
        true_boxes = targets[i]["boxes"]

        for p_box in pred_boxes:
            # 检查是否与任何真实框匹配
            match = False
            for t_box in true_boxes:
                if iou(p_box, t_box) > iou_threshold:
                    match = True
                    break

            if match:
                TP += 1
            else:
                FP += 1

        for t_box in true_boxes:
            # 检查是否被任何预测框匹配
            match = False
            for p_box in pred_boxes:
                if iou(p_box, t_box) > iou_threshold:
                    match = True
                    break

            if not match:
                FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def my_collate_fn(batch):
    images = []
    targets = []
    transcriptions = []
    for sample in batch:
        if sample[0] is None:  # 如果图像加载失败，则跳过
            continue
        image, target, transcription = sample
        images.append(image)
        targets.append(target)
        transcriptions.append(transcription)
    return images, targets,transcriptions

# 数据转换
transform = transforms.Compose([
    # transforms.Resize((1080, 1080)),  
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
])

# 创建数据集和数据加载器
dataset = MyDataset(root_dir, train_data_file, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# 添加验证集
val_dataset = MyDataset(root_dir, test_data_file, transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# 初始化损失记录器
train_losses = []
val_losses = []

# 创建模型
model = get_detection_model(num_classes)

# 定义优化器
# optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)

# Adam 优化器
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

# GPU支持
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
model.to(device)

# 训练循环
print('批次：',batch_size,'学习率：',learning_rate,'权重衰减参数：',weight_decay)
for epoch in range(num_epochs):
    # 初始化指标
    train_precision = train_recall = train_f1 = 0
    val_precision = val_recall = val_f1 = 0
    num_train_batches = num_val_batches = 0
    model.train()
    total_loss = 0
    for images, targets, _ in data_loader:  # 忽略 transcriptions
        # print(targets)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        # print('>>>>>>>>>>>>>',loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    avg_train_loss = total_loss / len(data_loader)
    train_losses.append(avg_train_loss)

    # 在验证集上
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets, _ in val_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 性能指标
            predictions = model(images)
            p, r, f1 = calculate_metrics(predictions, targets)
            val_precision += p
            val_recall += r
            val_f1 += f1
            num_val_batches += 1
            # 损失率
            losses = model(images, targets)
            for loss_name, loss in loss_dict.items():
                # print(f"{loss_name}: {loss}")  # 打印每个损失分量的详细信息
                total_val_loss += loss.item()

    # 计算平均性能指标
    val_precision /= num_val_batches
    val_recall /= num_val_batches
    val_f1 /= num_val_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}]: 准确率: {val_precision}, 召回率: {val_recall}, F1: {val_f1}')

    # # 计算平均损失
    # avg_val_loss = total_val_loss / len(val_data_loader)
    # val_losses.append(avg_val_loss)
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'text_detection_model_epoch_{epoch+1}.pth')

# 保存模型
torch.save(model.state_dict(), 'text_detection_model.pth')

print("Training completed")