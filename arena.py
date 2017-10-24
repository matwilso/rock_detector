#!/usr/bin/env python3
from collections import namedtuple
import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt

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
# I think to be safe, it is about 16, maybe 32 frames.  So I could just do a 
# minibatch that shows the same scene of rocks, just randomized angles

# > The MJB is a stand-alone file and does not refer to any other files. It also 
# loads faster, especially when the XML contains meshes that require processing 
# by the compiler. **So we recommend saving commonly used models as MJB and 
# loading them when needed for simulation.**

# I wonder if there is anyway I could do this.  Somehow merge models? But then
# again, my underlying model is just a box geom, so will be fast 


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
# then I gotta figure out where to stick the regression head

#matmodder = MaterialModder(sim)
#for name in sim.model.geom_names:
#    matmodder.rand_all(name)


# TODO: set these to actual from xml file
# farthest back camera should be is camera y = [-2.00, 6.00]


# TODO: need to get the ranges to randomize for the camera
# one issue that I have is what if the rock goes out of the frame.  Would that mess
# things up if we try to backprop on something we can't see?  I guess just limit it
# to a range where this near or strictly impossible.


# TODO: find a bunch of rock stls on the internet: https://free3d.com/3d-models/rock

# 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
# could be about 2 days for full convergence


# x is left and right
# y is back and forth
# TODO: refactor these
acx = -2.19
xoff = 0.5
leftx = -4.29
rightx = -0.1
biny = -3.79
digy = 3.59
afz = 0.0
zlow = 0.2
zhigh = 1.0

sz_len = 1.5
obs_len = 2.94
dig_len = 2.94
sz_endy = biny + sz_len # start zone end
cam_ydelta = 0.75 
obs_sy = sz_endy
obs_endy = obs_sy + obs_len


# These need to be placed into the xml
rock_rx = Range(acx, acx) 
rock_ry = Range(obs_sy, obs_endy)
rock_rz = Range(afz, afz)
rock_range3d = Range3D(rock_rx, rock_ry, rock_rz)


light_rx = Range(leftx, rightx) 
light_ry = Range(biny, digy)
light_rz = Range(afz, afz + zhigh)
light_range3d = Range3D(light_rx, light_ry, light_rz)

light_rroll = Range(-180, 180)
light_rpitch = Range(-180, 180)
light_ryaw = Range(-180, 180)
light_angle3 = Range3D(light_rroll, light_rpitch, light_ryaw)

cam_rx = Range(acx - xoff, acx + xoff) # center of arena +/- 0.5
cam_ry = Range(biny, biny+cam_ydelta)
cam_rz = Range(afz + zlow, afz + zhigh)
cam_range3d = Range3D(cam_rx, cam_ry, cam_rz)

cam_rroll = Range(-80, -100)
cam_rpitch = Range(80, 90)
cam_ryaw = Range(88, 92)
cam_angle3 = Range3D(cam_rroll, cam_rpitch, cam_ryaw)
cam_rfovy = Range(35, 55)

rvariance = Range(0.0, 0.0001)


def sample(num_range, as_int=False):
    """Sample a float in the num_range"""
    samp = random.uniform(num_range.max, num_range.min)
    if as_int:
        return int(samp)
    else:
        return samp

def sample_xyz(range3d):
    x = sample(range3d.x)
    y = sample(range3d.y)
    z = sample(range3d.z)
    return (x, y, z)


def sample_quat(angle3):
    roll = sample(angle3.x) * np.pi / 180
    pitch = sample(angle3.y) * np.pi / 180
    yaw = sample(angle3.z) * np.pi / 180

    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components

def mod_textures():
    """Randomize all the textures in the scene, including the skybox"""
    tex_modder.randomize()
    tex_modder.rand_all('skybox')

def mod_lights():
    # TODO: - set direction
    #       - set active
    #       - set specular
    #       - set ambient
    #       - set diffuse
    #       - set castshadow 
    for name in sim.model.light_names:
        light_modder.set_pos(name, sample_xyz(light_range3d))

def mod_camera():
    """Randomize pos, direction, and fov of camera"""

    cam_modder.set_pos('camera1', sample_xyz(cam_range3d))
    cam_modder.set_quat('camera1', sample_quat(cam_angle3))

    fovy = sample(cam_rfovy)
    cam_modder.set_fovy('camera1', fovy)


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

    #
    cam_img = sim.render(224, 224, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
    variance = sample(rvariance)
    cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=variance) * 255).astype(np.uint8)
    plt.imshow(cam_img)
    plt.show()

    # TODO: - add some random noise (type and amount) 

    floor_offset = model.body_pos[model.body_name2id('floor')]
    r1_pos = floor_offset + model.body_pos[model.body_name2id('rock1')]
    r2_pos = floor_offset + model.body_pos[model.body_name2id('rock2')]
    r3_pos = floor_offset + model.body_pos[model.body_name2id('rock3')]
    cam_pos = model.cam_pos[0]

    r1_diff = r1_pos - cam_pos
    r2_diff = r2_pos - cam_pos
    r3_diff = r3_pos - cam_pos
    r1_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r1_diff[0], r1_diff[1], r1_diff[2])
    r2_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r2_diff[0], r2_diff[1], r2_diff[2])
    r3_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r3_diff[0], r3_diff[1], r3_diff[2])

    quat = np.quaternion(*model.cam_quat[0])
    rpy = quaternion.as_euler_angles(quat) * 180 / np.pi

    viewer.add_marker(pos=r1_pos, label=r1_text)
    #viewer.add_marker(pos=r2_pos, label=r2_text)
    #viewer.add_marker(pos=r3_pos, label=r3_text)
    viewer.add_marker(pos=cam_pos, label="CAM: {}".format(rpy))

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break
