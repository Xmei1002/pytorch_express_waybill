import os
import json
from PIL import Image, UnidentifiedImageError,ImageFile
from torch.utils.data import Dataset
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, root_dir, data_file, transform=None):
        self.transform = transform
        self.data = []
        data_file = os.path.join(root_dir, data_file)

        # 加载指定文件的数据
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                image_path, transcriptions_str = line.strip().split('\t')
                image_path = os.path.join(root_dir, 'train_imgs', image_path)
                transcriptions = json.loads(transcriptions_str)
                self.data.append((image_path, transcriptions))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image_path, transcription_list = self.data[idx]
            # print('~~~~~',image_path,transcription_list)
            # 尝试加载图片并获取原始尺寸
            try:
                # print('图片位置：',image_path)
                image = Image.open(image_path).convert('RGB')
                original_size = image.size  # 获取原始尺寸 (width, height)
                # print('原始图片信息',image,image.size)
            except UnidentifiedImageError:
                print(f"加载图片失败: {image_path}")
                return None

            # 应用图像变换
            if self.transform:
                image = self.transform(image)
                new_size = (image.shape[2], image.shape[1])  # 获取新尺寸 (width, height)
                # print('转换后图片信息',image,new_size)
                # print(idx)
            # 提取每个特征的文本和调整坐标
            boxes = []
            labels = []
            transcriptions = []
            for item in transcription_list:
                transcription = item['transcription']
                transcriptions.append(transcription)
                points = self.adjust_coordinates(original_size, new_size, item['points'])
                
                # 假设 points 是 [top_left, top_right, bottom_right, bottom_left]
                x_coordinates = [p[0] for p in points]
                y_coordinates = [p[1] for p in points]
                boxes.append([min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)])
                labels.append(1)  # 所有文本分配同一个类别标签


            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels}
            # print(image,target,transcriptions)
            # 返回图像、目标字典和 transcriptions
            return image, target, transcriptions
        
        except IOError:
            print(f"Error loading image {image_path}")
            return None, None, None

    @staticmethod
    def adjust_coordinates(original_size, new_size, points):
        """
        调整坐标点以匹配新的图像尺寸。
        :param original_size: 原始图像的尺寸 (width, height)
        :param new_size: 新图像的尺寸 (width, height)
        :param points: 原始图像中的坐标点列表
        :return: 调整后的坐标点列表
        """
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        adjusted_points = [[int(x * scale_x), int(y * scale_y)] for [x, y] in points]
        return adjusted_points
