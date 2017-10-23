#!/usr/bin/env python3

import torchvision.models as models

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

model = load_model_from_path("xmls/fetch/main.xml")

vgg16 = models.vgg16_bn(pretrained=True)
vgg16.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 256),
    nn.ReLU(True),
    nn.Linear(256, 64)
    nn.ReLU(True),
    nn.Linear(64, 3)
    )

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def l2_loss(y_pred, y):
    """L2 norm (half norm with no sqrt, copied from tensorflow source"""
    return torch.sum((y_pred-y)**2) / 2

for t in range(500):
    coords_pred = vgg16.forward(camera_pixels)
    loss = l2_loss(coords_pred, real_coords)

    print(t, loss.data[0])
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()


#TODO: randomize the shape or size of the rocks
# Looked it up, and the way to do it is just modify the xml file and reload.
# not too bad

# PARAMS TO RANDOMIZE

#  map
# - 0.5 of the time have the arena map
# - 0.5 of the time have an open map

# rock
# - xyz coordinate of rock within some xy range and always at some z height
# - shape
# - size

#  camera
# - easy camera modder

# texture
# - easy texture modder


# TODO: need to think of some method for when to sample and randomize.  
# modding and sampling textures is super fast. so maybe do like a batch of 10
# and then a batchbatch of 10 of those (so 100 frames).
# 
# But really I need to worry about what can fit in memory in terms of batch size




sim = MjSim(model)
viewer = MjViewer(sim)
modder = TextureModder(sim)

t = 0

while True:
    for name in sim.model.geom_names:
        modder.rand_all(name)

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
