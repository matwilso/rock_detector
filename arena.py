#!/usr/bin/env python3
from collections import namedtuple
import random
import numpy as np
import quaternion

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
Range = namedtuple("Range", "min max")
Range3D = namedtuple("Range3D", "x y z")


#model = load_model_from_path("xmls/nasa/minimal.xml")

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


# TODO: also need to train on a lot of examples that don't have any rocks
# and find some way to represent that it doesn't see anything. maybe a true
# or false flag? That actually might be good.

# What about detecting multiple objects? 
# One thing I could do is just have it detect 3 rocks all the time.  It would give
# 4 number for each: xyz and radius. But how do I backprop on that? They are all 
# rocks.  I could do rock on the right, rock in the middle, and rock on the left.
# I think the net could handle that. Only 1 way to find out.
# On some runs I would randomly remove the rock. The radius should be 0. Maybe
# I could modify the gradient in these cases and not backprop on any xyz.
# Yeah that would be easy.


# TODO: consider switching to ResNet-50 because it is faster and better (jcjohnson)

#matmodder = MaterialModder(sim)
#for name in sim.model.geom_names:
#    matmodder.rand_all(name)


# TODO: set these to actual from xml file
# farthest back camera should be is camera y = [-2.00, 6.00]



# TODO: need to get the ranges to randomize for the camera
# one issue that I have is what if the rock goes out of the frame.  Would that mess
# things up if we try to backprop on something we can't see?  I guess just limit it
# to a range where this near or strictly impossible.

rx = Range(-4.29, -0.1) 
ry = Range(0, 5)
rz = Range(-3.79, 3.59)
range3d = Range3D(rx, ry, rz)

rroll = Range(-90, -90)
rpitch = Range(70, 85)
ryaw = Range(90, 90)


def sample(num_range, as_int=False):
    samp = random.uniform(num_range.max, num_range.min)
    if as_int:
        return int(samp)
    else:
        return samp

def mod_textures():
    tex_modder.randomize()
    tex_modder.rand_all('skybox')

# These ones don't work. They just never that in the documentation
def mod_lights():
    for name in sim.model.light_names:
        x = sample(range3d.x)
        y = sample(range3d.y)
        z = sample(range3d.z)
        light_modder.set_pos(name, (x, y, z))

def mod_camera():
    roll = sample(rroll) * np.pi / 180
    pitch = sample(rpitch) * np.pi / 180
    yaw = sample(ryaw) * np.pi / 180
    print(roll, pitch, yaw)

    # xyz are mixed up because of camera coordinate frames
    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    cam_modder.set_quat('camera1', quat.normalized().components)

model = load_model_from_path("xmls/nasa/box.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
tex_modder = TextureModder(sim)
cam_modder = CameraModder(sim)
light_modder = LightModder(sim)

t = 0
while True:
    mod_textures()
    mod_lights()
    mod_camera()
    sim.step()  # NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK 

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
