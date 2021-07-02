# -*- coding = utf-8 -*-
# @Time : 2021/4/28 15:34
# @Author : 戎昱
# @File : myDataset.py
# @Software : PyCharm
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os

path = "data/train"  # 图片集路径
classes = [i for i in os.listdir(path)]
files = os.listdir(path)
train = open("train.txt", 'w')
val = open("val.txt", 'w')
for i in classes:
    s = 0
    for imgname in os.listdir(os.path.join(path, i)):

        if s % 7 != 0:  # 7：1划分训练集测试集
            name = os.path.join(path, i) + '\\' + imgname + ' ' + str(classes.index(i)) + '\n'  # 我是win10,是\\,ubuntu注意！
            train.write(name)
        else:
            name = os.path.join(path, i) + '\\' + imgname + ' ' + str(classes.index(i)) + '\n'
            val.write(name)
        s += 1

val.close()
train.close()


class MyDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 打开txt，读取内容
        imgs = []
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除本行string字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，words[0]是图片信息，words[1]是label

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path
        img = Image.open(fn).convert('RGB')  # from PIL import Image

        if self.transform is not None:  # 是否进行transform
            img = self.transform(img)
        return img, label  # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imgs)


'''标准化、图片变换'''
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

train_data = MyDataset(datatxt='train.txt', transform=train_transforms)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)


class ReID_Net(nn.Module):
    def __init__(self):
        super(ReID_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=7,dilation=2)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(55696*2,10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 35)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,55696*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)

use_cuda = True if torch.cuda.is_available() else False
# """ 训练时:"""
for data, label in train_loader:
    # print(data.size())
    # print(label)
    pass

def judgement(model, data, device):
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        pred = output.max(1)[1]
    return pred

if __name__ == "__main__":
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    model = ReID_Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 30 + 1):
        if epoch>20:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.train()
        pid = os.getpid()
        for batch_idx, (data, target) in enumerate(train_loader):
            # 优化器梯度置0
            # print(data.size())
            optimizer.zero_grad()
            # 输入特征预测值
            output = model(data.to(device))
            # print(output.size())
            # print(target.size())
            # 预测值与标准值计算损失
            loss = F.nll_loss(output, target.to(device))
            # 计算梯度
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 每10步打印一下日志
            if batch_idx % 10 == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(pid, epoch, batch_idx * len(data),
                                                                                   len(train_loader.dataset),
                                                                                   100. * batch_idx / len(train_loader),
                                                                                   loss.item()))
    torch.save(model.state_dict(), 'myModel.pth')
    print('save success!')

