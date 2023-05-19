# -*- codeing = utf-8 -*-
# @Time : 4/5/2023 下午9:10
# @Author : 姚天灿
# @File : test.py
# @Software : PyCharm

import torch
import torch.nn as nn

x = torch.randn((1,1,28,28))
y = torch.tensor([1.0])
model = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.AdaptiveAvgPool2d(1))
out = model(x).squeeze()
loss = nn.MSELoss()(out, y)
loss.backward()