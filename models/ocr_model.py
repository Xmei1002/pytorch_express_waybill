import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_channel, num_classes, hidden_size):
        super(CRNN, self).__init__()

        # 调整CNN部分，减少池化步骤
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第一次池化
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 第二次池化
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 使用自适应池化以处理不同尺寸的特征图
            nn.AdaptiveAvgPool2d((1, None))
        )

        # RNN部分
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN层
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "卷积层输出的高度必须为1"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)

        # RNN层
        output, _ = self.lstm(conv)

        # 线性层
        output = self.linear(output)
        output = output.permute(1, 0, 2)  # [T, N, C]

        return output
