# Author:Alvin
# Time:2021/9/3 12:59
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

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
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(20, 30, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(30, 20, kernel_size=3)
        # self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=3)
        # self.conv2 = torch.nn.Conv2d(5, 10, kernel_size=3)
        # self.conv3 = torch.nn.Conv2d(10, 30, kernel_size=5)
        # self.conv4 = torch.nn.Conv2d(30, 10, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(20, 20)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.pooling(self.conv2(x)))
        # x = F.relu(self.pooling(self.conv3(x)))
        # x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
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
    print('正确率：%d%% [%d %d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        t1est()
