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
# TODO: just make these into numpy arrays so ops are defined
class Range(namedtuple('Range', 'min max')):
    def __add__(self, other):
        if isinstance(other, Range):
            return Range(self.min + other.min, self.max + other.max)
        else:
            return Range(self.min + other, self.max + other)

    def __mul__(self, other):
        return Range(self.min * other, self.max * other)

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



# What is going to be my network output and how am I going to compute a loss on it?
# I am thinking x and y coordinates of rocks and their height above ground (z-height)

#TODO: randomize the shape or size of the rocks
# Alex says there is a way to do it live
# I should also modify the height of the competition arena walls from like
# 0 to what I think would be their max height. This can be done live in 
# some way similar to the CameraModder

# PARAMS TO RANDOMIZE

#  map
# - 0.5 of the time have the arena map
# - 0.5 of the time have an open map

# rock
# - xyz coordinate of rock within some xy range and always at some z height
# - shape
# - size


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

# TODO: find a bunch of rock stls on the internet: https://free3d.com/3d-models/rock

# TODO: add some distractor objects, like smaller rocks.  And just the bigger 
# rocks are the only thing that matters

# TODO: gonna need a lot of finetuning and making sure things are good before I train 

# NOTE: 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
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
zlow = 0.3
zhigh = 1.0

sz_len = 1.5
obs_len = 2.94
dig_len = 2.94
sz_endy = biny + sz_len # start zone end
cam_ydelta = 1.5 
obs_sy = sz_endy
obs_endy = obs_sy + obs_len

rock_rx = Range(acx, acx) 
rock_ry = Range(obs_sy+1.0, obs_endy)
rock_rz = Range(afz, afz)
rock_range3d = Range3D(rock_rx, rock_ry, rock_rz)

light_rx = Range(leftx, rightx) 
light_ry = Range(biny, digy)
light_rz = Range(afz, afz + zhigh)
light_range3d = Range3D(light_rx, light_ry, light_rz)

light_dir3 = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))

cam_rx = Range(acx - xoff, acx + xoff) # center of arena +/- 0.5
cam_ry = Range(biny+0.2, sz_endy)
cam_rz = Range(afz + zlow, afz + zhigh)
cam_range3d = Range3D(cam_rx, cam_ry, cam_rz)

cam_rroll = Range(-85, -95) # think this might actually be yaw
cam_rpitch = Range(65, 90)
cam_ryaw = Range(88, 92) # this might actually be pitch, based on coordinate frames
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
    """Randomize pos, direction, and lights"""
    for i, name in enumerate(sim.model.light_names):
        # random sample 50% of any given light being on 
        light_modder.set_active(name, random.uniform(0, 1) > 0.5)

        # Pretty sure light_dir is just the xyz of a quat with w = 0.
        # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
        dir_xyz = np.quaternion(0, *sample_xyz(light_dir3)).normalized().components.tolist()[1:]
        light_modder.set_pos(name, sample_xyz(light_range3d))
        light_modder.set_dir(name, dir_xyz)
        light_modder.set_specular(name, sample_xyz(light_dir3))
        light_modder.set_diffuse(name, sample_xyz(light_dir3))
        # TODO: add marker for visualizing light
        # TODO: also consider adding more lights and turning them on and off

def mod_camera():
    """Randomize pos, direction, and fov of camera"""

    cam_modder.set_pos('camera1', sample_xyz(cam_range3d))
    cam_modder.set_quat('camera1', sample_quat(cam_angle3))

    fovy = sample(cam_rfovy)
    cam_modder.set_fovy('camera1', fovy)



# Range to a tripled Range3D 
rto3d = lambda r : Range3D(*((r,)*3))

# TODO: add 3 more rocks maybe that are from meshses. else later try to 
# be able to change something to a mesh

rock_r1dim = Range(0.05, 0.2)
rock_size_range = rto3d(rock_r1dim)
rock_rtypes = Range(3, 7+1) 
rock_mesh_scaleup = 100

#rock_rx = Range(acx, acx) 
#rock_ry = Range(obs_sy, obs_endy)
#rock_rz = Range(afz, afz)
#rock_range3d = Range3D(rock_rx, rock_ry, rock_rz)

#<body name="floor" pos="-2.19 -0.1 -0.05">

rock_ry = Range(obs_sy + 1.5, obs_endy)

rock_lanex = 0.4
outer_extra = 0.5
rock_buffx = 0.2

left_rx = Range(-3*rock_lanex - outer_extra, -rock_lanex - rock_buffx)
left_rz = Range(afz, afz)
left_rock_range = Range3D(left_rx, rock_ry, left_rz)

mid_rx = Range(-rock_lanex, rock_lanex)
mid_rz = Range(afz, afz)
mid_rock_range = Range3D(mid_rx, rock_ry, mid_rz)

right_rx = Range(rock_buffx+rock_lanex, 3*rock_lanex + outer_extra)
right_rz = Range(afz, afz)
right_rock_range = Range3D(right_rx, rock_ry, right_rz)

rocks_active = {}

def mod_rocks():
    rock_body_ids = []
    rock_geom_ids = []
    for name in model.geom_names:
        if name[:4] != "rock":
            continue 
        
        geom_id = model.geom_name2id(name)
        body_id = model.body_name2id(name)
        rock_geom_ids.append(geom_id)
        rock_body_ids.append(body_id)


        geom_type =  sample(rock_rtypes, as_int=True)
        model.geom_type[geom_id] = geom_type       
        model.geom_size[geom_id] = np.array(sample_xyz(rock_size_range))

#        if geom_type == 7:
#            this_rock_range = 
#        model.geom_size[geom_id] = np.array(sample_xyz(rock_size_range if geom_type != 7 else rock_size_range*rock_mesh_scaleup))
#
        if random.uniform(0, 1) < 0.05:
            #model.geom_rgba[geom_id] = np.array([1, 1, 1, 0])
            rocks_active[name] = False
        else:
            #model.geom_rgba[geom_id] = np.array([1, 1, 1, 1])
            rocks_active[name] = True

        model.body_quat[body_id] = random_quat()

    #random.shuffle(rock_body_ids)
    model.body_pos[rock_body_ids[0]] = np.array(sample_xyz(left_rock_range))
    model.body_pos[rock_body_ids[1]] = np.array(sample_xyz(mid_rock_range))
    model.body_pos[rock_body_ids[2]] = np.array(sample_xyz(right_rock_range))
    #print("1", model.geom_type[rock_geom_ids[0]])
    #print("2", model.geom_type[rock_geom_ids[1]])
    #print("3", model.geom_type[rock_geom_ids[2]])



def random_quat():
    quat_random = np.quaternion(*(np.random.randn(4))).normalized()
    return quat_random.components

def jitter_quat(quat, amount):
    jitter = amount * np.random.randn(4)
    quat_jittered = np.quaternion(*(quat + jitter)).normalized()
    return quat_jittered.components

def mod_arena():
    return
    for name in model.geom_names:
        if name[-4:] != "wall":
            continue 

        geom_id = model.geom_name2id(name)
        body_id = model.body_name2id(name)

        jitter_x = Range(-0.2, 0.2)
        jitter_y = Range(-0.2, 0.2)
        jitter_z = Range(-2.0, 0.0)
        jitter3D = Range3D(jitter_x, jitter_y, jitter_z)

        model.body_pos[body_id] = start_body_pos[body_id] + sample_xyz(jitter3D)
        model.body_quat[body_id] = jitter_quat(model.body_quat[body_id], 0.001)


model = load_model_from_path("xmls/nasa/box.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
tex_modder = TextureModder(sim)
cam_modder = CameraModder(sim)
light_modder = LightModder(sim)

start_geo_size = model.geom_size.copy()
start_body_pos = model.body_pos.copy()
start_body_quat = model.body_quat.copy()

t = 0
while True:
    mod_textures()
    mod_lights()
    mod_camera()
    mod_rocks()
    mod_arena()
    sim.step()  # NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK 

    #
    cam_img = sim.render(224, 224, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
    variance = sample(rvariance)
    cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=variance) * 255).astype(np.uint8)
    #plt.imshow(cam_img)
    #plt.show()

    floor_offset = model.body_pos[model.body_name2id('floor')]
    r1_pos = floor_offset + model.body_pos[model.body_name2id('rock1')]
    r2_pos = floor_offset + model.body_pos[model.body_name2id('rock2')]
    r3_pos = floor_offset + model.body_pos[model.body_name2id('rock3')]
    cam_pos = model.cam_pos[0]

    r1_diff = r1_pos - cam_pos
    r2_diff = r2_pos - cam_pos
    r3_diff = r3_pos - cam_pos
    r1_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r1_diff[0], r1_diff[1], r1_diff[2])
    #r2_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r2_diff[0], r2_diff[1], r2_diff[2])
    #r3_text = "x: {0:.2f} y: {1:.2f} z:{2:.2f}".format(r3_diff[0], r3_diff[1], r3_diff[2])

    quat = np.quaternion(*model.cam_quat[0])
    rpy = quaternion.as_euler_angles(quat) * 180 / np.pi
    #viewer.add_marker(pos=r1_pos, label=r1_text)
    #viewer.add_marker(pos=r2_pos, label=r2_text)
    #viewer.add_marker(pos=r3_pos, label=r3_text)
    viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, rpy))

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break


# TODO: set the arena center bin is 0,0

# NOTES (for PR):
# could have a jitter method where it just moves a bit from the current location (pass in jitter)

