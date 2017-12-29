#!/usr/bin/env python3
import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mimage

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision

LOAD_PATH = "net.weights"

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# Neural net training
resnet = torchvision.models.resnet50(pretrained=False)
resnet.fc = nn.Linear(2048, 9)
resnet.load_state_dict(torch.load(LOAD_PATH))

x_batch = Variable(torch.stack(x_frames))
y_batch = Variable(torch.stack(y_grounds))
coords_pred = resnet.forward(x_batch)

loss = l2_loss(coords_pred, y_batch)

y_grounds.append(torch.from_numpy(np.array(ground_truth)).type(dtype))
x_frames.append(torchvision.transforms.ToTensor()(cam_img).type(dtype))





