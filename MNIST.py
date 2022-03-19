# Author:Alvin
# Time:2021/9/2 22:31
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(
    root='dataset/MNIST/raw',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size
)
test_dataset = datasets.MNIST(
    root='dataset/MNIST/raw',
    train=False,
    download=True,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=batch_size
)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        # self.l1 = torch.nn.Linear(784, 1024)
        # self.l2 = torch.nn.Linear(1024, 800)
        # self.l3 = torch.nn.Linear(800, 512)
        # self.l4 = torch.nn.Linear(512, 256)
        # self.l5 = torch.nn.Linear(256, 128)
        # self.l6 = torch.nn.Linear(128, 64)
        # self.l7 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        # x = F.relu(self.l6(x))
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
        return self.l5(x)


model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0  # 损失值
    for batch_idx, data in enumerate(train_loader):  # 把下标和数据打包为元组
        inputs, target = data  # data中的训练数据及标签
        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
        outputs = model(inputs)
        loss = criterion(outputs, target) # 计算loss
        loss.backward() #反向传播
        optimizer.step() #参数更新
        running_loss += loss.item() #  loss叠加
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def t1est():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('正确率：%d%%' % (100 * correct / total))


# def detect():
#     with torch.no_grad():
#         img = Image.open('dataset/detect/7.jpg')
#         reIm = img.resize((28, 28), Image.ANTIALIAS)
#         img_arr = np.array(reIm.convert('L'))
#         threshlod = 50
#         for i in range(28):
#             for j in range(28):
#                 img_arr[i][j] = 255 - img_arr[i][j]
#                 if (img_arr[i][j] < threshlod):
#                     img_arr[i][j] = 0
#                 else:img_arr[i][j] = 255
#         num_arr = img_arr.reshape([1, 784])
#         num_arr = num_arr.astype(np.float32)
#         img_ready = np.multiply(num_arr, 1.0 / 255.0)
#         img_ready = torch.tensor(img_ready)
#         output = model(img_ready)
#
#         print(output)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        t1est()
