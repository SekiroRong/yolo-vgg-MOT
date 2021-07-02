# -*- coding = utf-8 -*-
# @Time : 2021/4/27 0:45
# @Author : 戎昱
# @File : prac.py
# @Software : PyCharm
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import os
from utlis import train
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def train(model, device, dataloader_kwargs, data):
    #手动设置随机种子
    torch.manual_seed(1)
    #加载训练数据
    train_loader = torch.utils.data.DataLoader(data,batch_size=128, shuffle=True, num_workers=1,**dataloader_kwargs)
    #使用随机梯度下降进行优化
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #开始训练，训练epoches次
    for epoch in range(1, 20 + 1):
        train_epoch(epoch, model, device, train_loader, optimizer)


def train_epoch(epoch, model, device, data_loader, optimizer):
    #模型转换为训练模式
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        #优化器梯度置0
        optimizer.zero_grad()
        #输入特征预测值
        output = model(data.to(device))
        #预测值与标准值计算损失
        loss = F.nll_loss(output, target.to(device))
        #计算梯度
        loss.backward()
        #更新梯度
        optimizer.step()
        #每10步打印一下日志
        if batch_idx % 10 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                                                                               100. * batch_idx / len(data_loader), loss.item()))
simple_transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = ImageFolder('./data/train',simple_transform)
test_data = ImageFolder('./data/train',simple_transform)
# print(train_data.class_to_idx)
# print(train_data.classes)

class ReID_Net(nn.Module):
    def __init__(self):
        super(ReID_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=7,dilation=2)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(55696*2,1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 26)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,55696*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

use_cuda = True if torch.cuda.is_available() else False

if __name__ == "__main__":
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}
    model = ReID_Net().to(device)
    train(model, device,dataloader_kwargs,train_data)

