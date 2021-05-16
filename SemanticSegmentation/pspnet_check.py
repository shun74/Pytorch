import torch
from DataLoader import *
from NetworkModule import *
# from utils.pspnet import PSPNet


net = PSPNet(n_classes=21)
print(net)

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 475, 475)

outputs = net(dummy_img)
print(outputs)