import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class EAST(nn.Module):
    def __init__(self):
        super(EAST, self).__init__()
        # 特征提取器
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, stride=2),
            ConvBlock(64, 256)  # 假设最后一层输出通道为256
        )
        # 特征合并（示例）
        self.merge = nn.Sequential(
            ConvBlock(256, 256)
        )

        # 输出层
        self.output_score = nn.Conv2d(256, 1, 1)  # 分数图
        self.output_geo = nn.Conv2d(256, 5, 1)  # 几何图

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 特征合并
        x = self.merge(x)
        # 输出层
        score = self.output_score(x)
        geo = self.output_geo(x)
        return score, geo

# 测试模型
if __name__ == "__main__":
    model = EAST()
    print(model)
    test_input = torch.rand(1, 3, 1080, 1080)
    score, geo = model(test_input)
    print("Score Map Size:", score.size())
    print("Geometry Map Size:", geo.size())
