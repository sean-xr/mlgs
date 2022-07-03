from typing import Callable, Union, Tuple, List, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from models import SmoothClassifier
import numpy as np
# logits = torch.randn(4, 5)
# prediction = torch.argmax(logits, dim=1)
# print(prediction.shape)
# print(prediction)
# X = torch.randn(3,4,2,2)
# print('X',X.shape)
# print('X.reshape', (X.reshape(3,-1)).shape)
# sc = np.array([[.5,-0.8,1.2],[1.7,2.0,0.7]])
# ssc = torch.from_numpy(sc)
# ones = torch.ones_like(ssc)
# zeros = torch.zeros_like(ssc)
# print('ssc',ssc)
# mm1 = torch.clamp(ssc,min=0,max=1)
# print('mm1',mm1)
# mm2 = torch.minimum(ones,torch.maximum(ssc,zeros))
# print('mm2',mm2)
# print(ssc.shape)
# #sssc1 = ssc.reshape(ssc.shape[0],-1)
# sssc2 = ssc.reshape(1,-1)
# #print('sssc1',sssc1)
# print('ssc2',sssc2)
# norm = 2
# x = torch.randn(5,4,6,8)
# grad_norm = torch.norm(x, p=2, dim=(1,2,3))
# grad_norm1 = grad_norm.reshape(-1,1)
# grad_norm2 = x.reshape(5,-1)
# final = grad_norm2 / grad_norm1
# print(final.shape)
# print(grad_norm2.shape)
# y = torch.zeros(5)
# print('y=',y)
# classes = torch.arange(10)
# print('classes=', classes)
# C = 3
# N = 5
# this_batch_size = 5
# inputs = torch.randn([1,C,N,N])
# x = inputs.repeat(this_batch_size,1,1,1)
# print(x.shape)
class_counts = torch.arange(1,7)
top2 = torch.topk(class_counts, 2)
print(top2)