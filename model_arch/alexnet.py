import torch
import torch.nn as nn
import numpy as np
import random


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

class post(nn.Module):
    def __init__(self):
        super(post, self).__init__()

    def forward(self, x):
        # 确保x在正确的设备上
        x = x.to(device)

        # 找到每个样本的最大值及其索引
        m, midx = x.max(-1, keepdim=True)

        # 复制m以便之后修改x时不会改变m
        m_copy = m.clone()

        # 遍历每个样本
        for i in range(x.size(0)):
            # 遍历除了最大值索引之外的所有类别
            for j in range(x.size(1)):
                if j != midx[i].item():
                    x[i][j] = (random.uniform(0.1, 0.3)) * m_copy[i][0].item()  # 使用较小的随机数范围

        # 生成噪声并添加到x上
        noise = torch.from_numpy(np.random.normal(0, 1, size=x.shape).astype(np.float32)).to(device)
        x = x + noise

        # 应用softmax确保输出是概率分布
        return nn.functional.softmax(x, dim=-1)

class AlexNet_Militrary(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Militrary, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.post = post()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.post(x)
        return x

class AlexNet_ImageNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet_ImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.post = post()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.post(x)
        return x


def alexnet(dataset , pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if 'Military' in dataset:
        model = AlexNet_Militrary(**kwargs)
    else:
        model = AlexNet_ImageNet()
    return model
