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
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
model = Net()
model.load_state_dict(torch.load('ministceshi.pkl'))
model.eval()
# img = cv2.imread('dataset/detect/8.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# img_bin = cv2.resize(img_bin, (28, 28))
# img_gray = cv2.resize(img_gray, (28, 28))

# img_bin = Image.fromarray(cv2.cvtColor(img_bin,cv2.COLOR_BGR2RGB))
# cv2.imshow("img_bin", img_gray)
# cv2.waitKey(0)

img0 = Image.open('dataset/detect/2手写.png')
img0 = img0.resize((28, 28), Image.ANTIALIAS)
img0 = img0.convert("L")  # 图像转灰度图

threshold = 130

table = []
for i in range(256):
    if i > threshold:
        table.append(0)
    else:
        table.append(1)

# 图片二值化
img0 = img0.point(table, '1')


print(np.shape(img0))
plt.imshow(img0)
plt.show()
img0 = transform(img0)  # 这里跟训练数据中的transform一样
img0 = Variable(img0.unsqueeze(0))  # 添加维度(eg:3->4)
img0 = img0.permute(1, 0, 2, 3)   # 交换维度
print(np.shape(img0))
# print(img0)
output1 = model(Variable(img0))  # Variable可求导变量
print(output1)
_, pred = torch.max(output1.data, dim=1)
print(f'检测值为： {pred.item()}')
