# Author:Alvin
# Time:2022/3/19 19:49
import os

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from MNIST_RESNET import Net

from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
model = Net()
model.load_state_dict(torch.load('minist1.pkl'))
model.eval()
img0 = Image.open('dataset/detect/平板黑白9.png')
img0 = img0.resize((28, 28), Image.ANTIALIAS)
img0 = img0.convert("L")  # 图像转换黑白
plt.imshow(img0)
plt.show()
img0 = transform(img0)  # 这里跟训练数据中的transform一样
img0 = Variable(img0.unsqueeze(0))  # 添加维度(eg:3->4)
# print(img0)
output1 = model(Variable(img0))  # Variable可求导变量
print(output1)
_, pred = torch.max(output1.data, dim=1)
print(f'检测值为： {pred.numpy()[0]}')


