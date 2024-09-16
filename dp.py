import torch
import os
import onnx
import onnxruntime
import numpy as np
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from collections import namedtuple
import warnings
import argparse
#from model_arch import *
from helper import *
from Confirguration import args_parser
import random
import sys
import time
import net1
from output_authorization import *
sys.path.append("/home/Newdisk/zhaoxiaoming/steal")
from data_loader import data_loader

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



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

if __name__ == '__main__':
    # model_path = pytorch_2_onnx()
    #
    # onnx_check(model_path)
    #
    # onnx_inference(model_path)
    model = net1.resnet50('imagenet_100')
    print(model)
    model = model.to(device)
    model.load_state_dict(torch.load("/data/Newdisk/zhaoxiaoming/steal/sjs/differential_privacy_docker/original_model/ImageNet_100/ResNet50_imagenet_100_64.26%.pth", map_location=device))
    train_loader, val_loader = data_loader(
        root="/data/Newdisk/zhaoxiaoming/datasets/Imagenet_100", arch="resnet50")

#    for image,label in val_loader:
#        image=image.to(device)
#        label=label.to(device)
#        out=model(image)
		
    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    #
    for data in val_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # print(outputs.shape)
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
    
        correct += (predicted == labels).sum()
        for idx, probs in enumerate(outputs):
            # 使用detach()来创建一个不需要梯度的Tensor副本
            probs_numpy = probs.detach().cpu().numpy()
            # print(f"Sample {idx}: Predicted class {torch.argmax(probs).item()}, Confidences: {probs_numpy}")
    end_time = time.time()  # 结束运行时间
    print("总样本数量：", total)
    print("测试准确率：", correct / total)
    print('cost %f second' % (end_time - start_time))

    print('****** test finished ******')
    torch.save(model.state_dict(), '/data/Newdisk/zhaoxiaoming/steal/sjs/differential_privacy_docker/differential_privacy/resnet50_imagenet_100_dp.pth')

