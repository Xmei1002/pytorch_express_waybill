import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.ocr_model import CRNN  # CRNN的模型
from utils.dataset import MyDataset  # 您提供的数据集类
from torch.utils.tensorboard import SummaryWriter
from config import root_dir, train_data_file, test_data_file
import torch.nn.functional as F
from torch.nn import CTCLoss
from utils.charset import char_set  # 字符集
import torchvision.transforms.functional as TF

batch_size = 32
learning_rate = 0.0001  # 学习率

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

def train_crnn():
    # 创建字符到索引的映射
    char_to_index = {char: index for index, char in enumerate(char_set)}
    # 定义图像变换
    transform = transforms.Compose([
        # transforms.Resize((1080, 1080)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    ])

    # 数据加载
    train_dataset = MyDataset(root_dir, train_data_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,collate_fn=my_collate_fn)

    validate_dataset = MyDataset(root_dir, test_data_file, transform=transform)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,collate_fn=my_collate_fn)
    # 初始化CRNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(img_channel=1, num_classes=len(char_set) + 1, hidden_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss = CTCLoss(blank=0, zero_infinity=True)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        for i, (images, points_lists, transcriptions_lists) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            for image, points, transcriptions in zip(images, points_lists, transcriptions_lists):
                boxes = points['boxes'].tolist()
                for box, transcription in zip(boxes, transcriptions):
                        # 裁剪图像区域
                        box = torch.tensor(box).round().int()
                        min_x, min_y, max_x, max_y = box
                        min_x, max_x = max(min_x, 0), min(max_x, image.size(2))
                        min_y, max_y = max(min_y, 0), min(max_y, image.size(1))
                        cropped_image = image[:, min_y:max_y, min_x:max_x]
                        # 转换标签
                        targets = torch.tensor([char_to_index[char] for char in transcription], dtype=torch.long)
                        # print(f'>>>>>{targets},>>>>>{transcription}')
                        cropped_image = cropped_image.to(device)
                        output = model(cropped_image.unsqueeze(0))
                        # 这里假设每次处理一个裁剪后的图像，因此batch size为1
                        input_lengths = torch.tensor([output.size(0)], dtype=torch.long)
                        # 相应地调整target_lengths
                        target_lengths = torch.tensor([len(targets)], dtype=torch.long)
                        # 计算CTC损失
                        loss = ctc_loss(output.log_softmax(2), targets, input_lengths, target_lengths)
                        # 反向传播和优化
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        total_batches += 1

        average_loss = total_loss / total_batches

        # 验证过程
        model.eval()
        total_val_loss = 0
        total_val_batches = 0

        with torch.no_grad():
            for images, points_lists, transcriptions_lists in validate_loader:
                images = [img.to(device) for img in images]

                for image, points, transcriptions in zip(images, points_lists, transcriptions_lists):
                    boxes = points['boxes'].tolist()
                    for box, transcription in zip(boxes, transcriptions):
                        # 裁剪图像区域
                        box = torch.tensor(box).round().int()
                        min_x, min_y, max_x, max_y = box
                        min_x, max_x = max(min_x, 0), min(max_x, image.size(2))
                        min_y, max_y = max(min_y, 0), min(max_y, image.size(1))
                        cropped_image = image[:, min_y:max_y, min_x:max_x]

                        # 转换标签
                        targets = torch.tensor([char_to_index[char] for char in transcription], dtype=torch.long)
                        cropped_image = cropped_image.to(device)
                        output = model(cropped_image.unsqueeze(0))

                        # 这里假设每次处理一个裁剪后的图像，因此batch size为1
                        input_lengths = torch.tensor([output.size(0)], dtype=torch.long)
                        target_lengths = torch.tensor([len(targets)], dtype=torch.long)

                        # 计算CTC损失
                        loss = ctc_loss(output.log_softmax(2), targets, input_lengths, target_lengths)
                        total_val_loss += loss.item()
                        total_val_batches += 1

            average_val_loss = total_val_loss / total_val_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_loss}, Val Loss: {average_val_loss}')
        # 保存模型
        
    torch.save(model.state_dict(), f"ocr_model.pth")

if __name__ == "__main__":
    train_crnn()
