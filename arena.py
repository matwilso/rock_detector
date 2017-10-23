#!/usr/bin/env python3
from collections import namedtuple
import random
import numpy as np

import torchvision.models as models

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import CameraModder, LightModder, MaterialModder, TextureModder
import os

# Define struct like objects
Coord = namedtuple("Coord", "x y z")
Range3D = namedtuple("Range3D", "min max")


#model = load_model_from_path("xmls/nasa/minimal.xml")
model = load_model_from_path("xmls/nasa/box.xml")

# Neural net training
#vgg16 = models.vgg16_bn(pretrained=True)
#vgg16.classifier = nn.Sequential(
#    nn.Linear(512 * 7 * 7, 256),
#    nn.ReLU(True),
#    nn.Linear(256, 64),
#    nn.ReLU(True),
#    nn.Linear(64, 3)
#    )
#
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
#
#def l2_loss(y_pred, y):
#    """L2 norm (half norm with no sqrt, copied from tensorflow source"""
#    return torch.sum((y_pred-y)**2) / 2
#
#for t in range(500):
#    coords_pred = vgg16.forward(camera_pixels)
#    loss = l2_loss(coords_pred, real_coords)
#
#    print(t, loss.data[0])
#    optimizer.zero_grad()
#
#    loss.backward()
#    optimizer.step()


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

# > The MJB is a stand-alone file and does not refer to any other files. It also 
# loads faster, especially when the XML contains meshes that require processing 
# by the compiler. **So we recommend saving commonly used models as MJB and 
# loading them when needed for simulation.**


sim = MjSim(model)
viewer = MjViewer(sim)
tex_modder = TextureModder(sim)
cam_modder = CameraModder(sim)
light_modder = LightModder(sim)
#matmodder = MaterialModder(sim)
#for name in sim.model.geom_names:
#    matmodder.rand_all(name)


# TODO: set these to actual from xml file
min_coord = Coord(0, 0, 0)
max_coord = Coord(5, 5, 5)
range3d = Range3D(min_coord, max_coord)


def mod_textures():
    tex_modder.randomize()
    tex_modder.rand_all('skybox')

# TODO: figure out how to set lights
def mod_lights():
    for name in sim.model.light_names:
        x = random.random() * (range3d.max.x - range3d.min.x) + range3d.min.x
        y = random.random() * (range3d.max.y - range3d.min.y) + range3d.min.y
        z = random.random() * (range3d.max.z - range3d.min.z) + range3d.min.z
        model.light_pos[0] = np.array([-1.29, 3.0, 1.0])
        light_modder.set_pos('light1', (1, 2, 5))
        light_modder.set_active(name, 0)
        light_modder.set_active('light1', 1)

def mod_camera():
    pass

t = 0
while True:
    mod_textures()
    mod_lights()

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
