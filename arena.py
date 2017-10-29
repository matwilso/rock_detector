#!/usr/bin/env python3

# BOOKMARK
# (1) need to figure out rock randomization blender maybe

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

# TODO: need to add some distractor field of view objects like the arms that were
# in the shot during the comp.  We may not have any such things, but it would be
# good to be robust to them

# TODO: in the autonomy sequence, we should have it rotate slightly towards the middle, 
# depending on which side it is on.  So just set a nav goal to rotate. Pretty easy

# OBJECT TYPE THINGS
def Range(min, max):
    return np.array([min, max])

def Range3D(x, y, z):
    return np.array([x,y,z])

def rto3d(r):
    return Range3D(r, r, r)


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

image_noise_rvariance = Range(0.0, 0.0001)

# Rock placement range parameters
rock_lanex = 0.4  # width parameters of x range
outer_extra = 0.5 # how much farther rocks should go out on the right and left lanes
rock_buffx = 0.2  # distacne between rock lanes

# How far into the obstacle zone the rocks should start.  
rock_start_offset = 0.2  

rock_ry = Range(obs_sy + rock_start_offset, obs_endy)
rock_rz = Range(afz, afz + 0.2)

# Position dependent ranges
left_rx = Range(-3*rock_lanex - outer_extra, -rock_lanex - rock_buffx)
mid_rx = Range(-rock_lanex, rock_lanex)
right_rx = Range(rock_buffx+rock_lanex, 3*rock_lanex + outer_extra)

# Form full 3D sample range
left_rock_range = Range3D(left_rx, rock_ry, rock_rz)
mid_rock_range = Range3D(mid_rx, rock_ry, rock_rz)
right_rock_range = Range3D(right_rx, rock_ry, rock_rz)
rock_ranges = [left_rock_range, mid_rock_range, right_rock_range]

# Rock size and type (only matters if we are using things besides meshes)
rock_r1dim = Range(0.2, 0.2)  # how large 1 dim of the rock is
rock_size_range = rto3d(rock_r1dim) 
rock_rtypes = Range(7, 7)   # can sample 3 to 8 for different geom shapes


# UTILS

def sample(num_range, as_int=False):
    """Sample a float in the num_range"""
    samp = random.uniform(num_range[0], num_range[1])
    if as_int:
        return int(samp)
    else:
        return samp

def sample_xyz(range3d):
    """Sample 3 floats in the 3 num_ranges"""
    x = sample(range3d[0])
    y = sample(range3d[1])
    z = sample(range3d[2])
    return (x, y, z)

def sample_quat(angle3):
    """Sample a quaterion from a range of euler angles in degrees"""
    roll = sample(angle3[0]) * np.pi / 180
    pitch = sample(angle3[1]) * np.pi / 180
    yaw = sample(angle3[2]) * np.pi / 180

    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components

def random_quat():
    """Sample a completely random quaternion"""
    quat_random = np.quaternion(*(np.random.randn(4))).normalized()
    return quat_random.components

def jitter_quat(quat, amount):
    """Jitter a given quaternion by amount"""
    jitter = amount * np.random.randn(4)
    quat_jittered = np.quaternion(*(quat + jitter)).normalized()
    return quat_jittered.components

# MODDERS

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

def mod_camera():
    """Randomize pos, direction, and fov of camera"""

    cam_modder.set_pos('camera1', sample_xyz(cam_range3d))
    cam_modder.set_quat('camera1', sample_quat(cam_angle3))

    fovy = sample(cam_rfovy)
    cam_modder.set_fovy('camera1', fovy)

def mod_arena():
    """
    Randomize the x, y, and orientation of the walls slights.
    Also drastically randomize the height of the walls, in many cases they won't
    be seen at all. This will allow the model to generalize to scenarios without
    walls, or where the walls and geometry is slightly different than the sim 
    model
    """

    for name in model.geom_names:
        if name[-4:] != "wall":
            continue 

        geom_id = model.geom_name2id(name)
        body_id = model.body_name2id(name)

        jitter_x = Range(-0.2, 0.2)
        jitter_y = Range(-0.2, 0.2)
        jitter_z = Range(-1.0, 0.0)
        jitter3D = Range3D(jitter_x, jitter_y, jitter_z)

        model.body_pos[body_id] = start_body_pos[body_id] + sample_xyz(jitter3D)
        model.body_quat[body_id] = jitter_quat(start_body_quat[body_id], 0.005)


def mod_rocks():
    """
    Randomize the rocks so that the model will generalize to competition rocks
    This involves:
        - Random positions
        - Random orientations
        - Rotating positions of meshes
        - [TODO]:  completely randomizing the meshes every n runs (e.g., with blender)
    """
    rock_body_ids = {}
    rock_geom_ids = {}
    rock_mesh_ids = {}
    max_heights = {} 
    rocks_active = {} # which of the rocks are currently visible (will be used for training)

    for name in model.geom_names:
        if name[:4] != "rock":
            continue 
        
        geom_id = model.geom_name2id(name)
        body_id = model.body_name2id(name)
        mesh_id = model.geom_dataid[geom_id]
        rock_geom_ids[name] = geom_id
        rock_body_ids[name] = body_id
        rock_mesh_ids[name] = mesh_id

        geom_type =  sample(rock_rtypes, as_int=True)
        model.geom_type[geom_id] = geom_type       

        #this_range = rock_size_range if geom_type != 7 else rock_size_range*rock_mesh_scale
        model.geom_size[geom_id] = sample_xyz(rock_size_range)

        rocks_active[name] = True
        #if random.uniform(0, 1) < 0.01:
        #    #model.geom_rgba[geom_id] = np.array([1, 1, 1, 0])
        #    rocks_active[name] = False
        #else:
        #    #model.geom_rgba[geom_id] = np.array([1, 1, 1, 1])
        #    rocks_active[name] = True

        rot_quat = random_quat()

        # Rotate the rock and get  z value of the highest point in the rotated rock mesh
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        mesh_verts = model.mesh_vert[vert_adr : vert_adr+vert_num]
        rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
        model.geom_quat[geom_id] = rot_quat  
        max_height = np.max(rots[:,2])
        max_heights[name] = max_height

    rock_mod_cache = [] 

    # Randomize the positions of the rocks. 
    shuffle_names = list(rock_body_ids.keys())
    random.shuffle(shuffle_names)

    for i in range(len(shuffle_names)):
        name = shuffle_names[i]
        active = rocks_active[name]
        model.body_pos[rock_body_ids[name]] = np.array(sample_xyz(rock_ranges[i]))
        z_height = max_heights[name] + model.body_pos[rock_body_ids[name]][2]

        rock_mod_cache.append((name, active, z_height))

    return rock_mod_cache


model = load_model_from_path("xmls/nasa/box.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
tex_modder = TextureModder(sim)
cam_modder = CameraModder(sim)
light_modder = LightModder(sim)

# Get start state of params to slightlt jitter later
start_geo_size = model.geom_size.copy()
start_body_pos = model.body_pos.copy()
start_body_quat = model.body_quat.copy()


def preproc_img(img):
    crop = img[24:-24, 80:-80, :]
    down_sample = crop[::3, ::5, :]
    return down_sample 

def display_image(cam_img):
    practice_img = preproc_img(plt.imread("assets/practice.jpg"))
    round1_img = preproc_img(plt.imread("assets/round1.jpg"))
    fig = plt.figure()

    subp = fig.add_subplot(1,3,1)
    imgplot = plt.imshow(practice_img)
    subp.set_title('Practice Round')

    subp = fig.add_subplot(1,3,2)
    imgplot = plt.imshow(cam_img)
    subp.set_title('Sim')

    subp = fig.add_subplot(1,3,3)
    imgplot = plt.imshow(round1_img)
    subp.set_title('Round 1')

    plt.show()


t = 0
while True:

    # Randomize (mod) all relevant parameters
    mod_textures()
    mod_lights()
    mod_camera()
    rock_mod_cache = mod_rocks()
    mod_arena()
    sim.step()  # NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK 


    # Grab an image from the camera at (224, 244, 3) to feed into CNN
    cam_img = sim.render(1280, 720, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
    image_noise_variance = sample(image_noise_rvariance) 
    cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)
    cam_img = preproc_img(cam_img)

    display_image(cam_img)

    floor_offset = model.body_pos[model.body_name2id('floor')]
    cam_pos = model.cam_pos[0]

    r1_pos = floor_offset + model.body_pos[model.body_name2id('rock1')]
    r2_pos = floor_offset + model.body_pos[model.body_name2id('rock2')]
    r3_pos = floor_offset + model.body_pos[model.body_name2id('rock3')]

    r1_diff = r1_pos - cam_pos
    r2_diff = r2_pos - cam_pos
    r3_diff = r3_pos - cam_pos
    for slot in rock_mod_cache:
        name = slot[0]
        active = slot[1]
        z_height = slot[2]

        pos = floor_offset + model.body_pos[model.body_name2id(name)]
        diff = pos - cam_pos

        text = "x: {0:.2f} y: {1:.2f} height:{2:.2f}".format(diff[0], diff[1], z_height)
        if active:
            viewer.add_marker(pos=pos, label=text, rgba=np.zeros(4))


    quat = np.quaternion(*model.cam_quat[0])
    rpy = quaternion.as_euler_angles(quat) * 180 / np.pi
    viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, rpy))

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break



# TODO: set the arena center bin is 0,0

# NOTES (for PR):
# could have a jitter method where it just moves a bit from the current location (pass in jitter)

