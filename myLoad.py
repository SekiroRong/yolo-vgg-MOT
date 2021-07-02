# -*- coding = utf-8 -*-
# @Time : 2021/4/28 18:04
# @Author : 戎昱
# @File : myLoad.py
# @Software : PyCharm
from  myDataset import ReID_Net
import torch

def Load():
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ReID_Net().to(device)
    PATH = 'myModel.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model, device

