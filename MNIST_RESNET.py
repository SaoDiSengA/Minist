# Author:Alvin
# Time:2021/9/3 16:17
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import cv2
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


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        # self.conv3 = torch.nn.Conv2d(
        #     channels, channels, kernel_size=3, padding=1
        # )
        # self.conv4 = torch.nn.Conv2d(
        #     channels, channels, kernel_size=3, padding=1
        # )

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x))
        y = F.leaky_relu(self.conv2(y) + x)
        # y = F.relu(self.conv2(y))
        # y = F.relu(self.conv3(y))
        # y = F.relu(self.conv4(y) + x)
        return y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.leaky_relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.leaky_relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)    # 全连接时需要进行 view操作
        x = self.fc(x)
        return x


model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def t1est():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('正确率：%.2f%% [%d %d]' % (100 * correct / total, correct, total))


def detect():
    with torch.no_grad():
        img = Image.open('dataset/detect/0.jpg')
        reIm = img.resize((28, 28), Image.ANTIALIAS)
        img_arr = np.array(reIm.convert('L'))
        threshlod = 50
        for i in range(28):
            for j in range(28):
                img_arr[i][j] = 255 - img_arr[i][j]
                if (img_arr[i][j] < threshlod):
                    img_arr[i][j] = 0
                else:img_arr[i][j] = 255
        # num_arr = img_arr.reshape([1, 784])
        num_arr = img_arr.astype(np.float32)
        img_ready = np.multiply(num_arr, 1.0 / 255.0)
        img_ready = torch.tensor(img_ready)
        output = model(img_ready)
        print(output)

if __name__ == '__main__':
        for epoch in range(5):
            train(epoch)
            t1est()
