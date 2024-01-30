import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.ocr_model import CRNN  # 您的CRNN模型定义
from utils.dataset import MyDataset  # 您的数据集类
from config import root_dir, train_data_file, test_data_file
from utils.charset import char_set  # 您的字符集

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


def validate_crnn(model, data_loader, device, char_to_index):
    model.eval()  # 将模型设置为评估模式
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # 在评估过程中不计算梯度
        for images, points_lists, transcriptions_list in data_loader:
            images = [img.to(device) for img in images]
            for image, points, transcriptions in zip(images, points_lists, transcriptions_list):
                boxes = points['boxes'].tolist()
                for box, transcription in zip(boxes, transcriptions):
                    # 裁剪图像区域
                    box = torch.tensor(box).round().int()
                    min_x, min_y, max_x, max_y = box
                    min_x, max_x = max(min_x, 0), min(max_x, image.size(2))
                    min_y, max_y = max(min_y, 0), min(max_y, image.size(1))
                    cropped_image = image[:, min_y:max_y, min_x:max_x]
                    # 预测
                    output = model(cropped_image.unsqueeze(0))
                    predicted_text = convert_output_to_text(output, char_to_index)
                    print('>>>',predicted_text,transcription)
                    # 比较预测结果和真实标签
                    correct_predictions += (predicted_text == transcription)
                    total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
 

def convert_output_to_text(output, char_to_index):
    # 应用softmax获取概率分布
    probs = torch.softmax(output, dim=2)
    # 在每个时间步选择概率最高的类别
    _, indices = probs.max(2)
    indices = indices.squeeze(1).cpu().numpy()

    # 转换索引到字符
    char_list = []
    prev_index = None
    for idx in indices:
        if idx != prev_index:  # 移除连续重复
            char_list.append(char_set[idx])  # 将索引转换为字符
        prev_index = idx

    # 移除CTC空白字符
    text = ''.join([c for c in char_list if c != char_set[0]]) # 替换'<空白字符>'为您的空白字符标识
    return text

def main():
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(img_channel=3, num_classes=len(char_set) + 1, hidden_size=128)
    model.load_state_dict(torch.load('ocr_model.pth'))
    model.to(device)

    # 准备验证数据集
    transform = transforms.Compose([transforms.ToTensor()])
    validate_dataset = MyDataset(root_dir, test_data_file, transform=transform)
    validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False,collate_fn=my_collate_fn)
    char_to_index = {char: index for index, char in enumerate(char_set)}

    # 验证模型
    validate_crnn(model, validate_loader, device, char_to_index)

if __name__ == "__main__":
    main()
