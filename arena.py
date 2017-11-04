#!/usr/bin/env python3

import argparse
from itertools import count
import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt

import tensorflow as tf

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import CameraModder, LightModder, MaterialModder, TextureModder
import os


# from utils import *


parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_epochs', type=int, default=100,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=16,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--visualize', type=bool, default=False,
    help="Don't train, just interact with the mujoco sim and visualize everything")

parser.add_argument(
    '--super_batch', type=int, default=1,
    help='Number of batches before generating new rocks')

parser.add_argument(
    '--save_path', type=str, default="net.weights",
    help='File to save weights to')

parser.add_argument(
    '--save_every', type=int, default=1,
    help='Number of batches before saving weights')

parser.add_argument(
    '--load_path', type=str, default="net.weights",
    help='File to load weights from')

parser.add_argument(
    '--dtype', type=str, default="cpu",
    help='cpu or gpu')

parser.add_argument(
    '--blender_path', type=str, default="blender", help='Path to blender executable')

parser.add_argument(
    '--os', type=str, default="none",
    help='none (don\'t override any defaults) or mac or ubuntu')

# What about detecting multiple objects? 
# One thing I could do is just have it detect 3 rocks all the time.  It would give
# 4 number for each: xyz and radius. But how do I backprop on that? They are all 
# rocks.  I could do rock on the right, rock in the middle, and rock on the left.
# I think the net could handle that. Only 1 way to find out.
# On some runs I would randomly remove the rock. The radius should be 0. Maybe
# I could modify the gradient in these cases and not backprop on any xyz.
# Yeah that would be easy.

# TODO: add some distractor objects, like smaller rocks.  And just the bigger 
# rocks are the only thing that matters

# TODO: need to add some distractor field of view objects like the arms that were
# in the shot during the comp.  We may not have any such things, but it would be
# good to be robust to them

# NOTE: 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
# could be about 2 days for full convergence

# TODO: better preproc on the image if necessary (i.e., research this)

# TODO: Randomize the floor, either by giving it rotation or adding the
# ocean meshes

# TODO: I should try to run this with and without the floor randomization and see if it
# still works as well.  My guess is that with floor it will still be fine

# TODO: set the arena center bin is 0,0


# OBJECT TYPE THINGS
def Range(min, max):
    if min < max:
        return np.array([min, max])
    else:
        return np.array([max, min])

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

light_rx = Range(leftx, rightx) 
light_ry = Range(biny, digy)
light_rz = Range(afz, afz + zhigh)
light_range3d = Range3D(light_rx, light_ry, light_rz)

light_dir3 = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))

cam_rx = Range(acx - xoff, acx + xoff) # center of arena +/- 0.5
cam_ry = Range(biny+0.2, sz_endy)
cam_rz = Range(afz + zlow, afz + zhigh)
cam_range3d = Range3D(cam_rx, cam_ry, cam_rz)

cam_rroll = Range(-95, -85) # think this might actually be yaw
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
mid_start_offset = 0.4 # bit more for middle rock

rock_ry = Range(obs_sy + rock_start_offset, obs_endy)
mid_ry = Range(obs_sy + mid_start_offset, obs_endy)
rock_rz = Range(afz - 0.02, afz + 0.2)

# Position dependent ranges
left_rx = Range(-3*rock_lanex - outer_extra, -rock_lanex - rock_buffx)
mid_rx = Range(-rock_lanex, rock_lanex)
right_rx = Range(rock_buffx+rock_lanex, 3*rock_lanex + outer_extra)

# Form full 3D sample range
left_rock_range = Range3D(left_rx, rock_ry, rock_rz)
mid_rock_range = Range3D(mid_rx, mid_ry, rock_rz)
right_rock_range = Range3D(right_rx, rock_ry, rock_rz)
rock_ranges = [left_rock_range, mid_rock_range, right_rock_range]

# Rock size and type (only matters if we are using things besides meshes)
#rock_r1dim = Range(0.2, 0.2)  # how large 1 dim of the rock is
#rock_size_range = rto3d(rock_r1dim) 
#rock_rtypes = Range(3, 8)   # can sample 3 to 8 for different geom shapes

dirt_rx = Range(0.0, 0.3)
dirt_ry = Range(0.0, 0.3)
dirt_rz = Range(-0.05, 0.05)
dirt_range3d = Range3D(dirt_rx, dirt_ry, dirt_rz)

dirt_rroll = Range(-180, 180)  # yaw
dirt_rpitch = Range(-90, -90)
dirt_ryaw = Range(0, 0)  # roll
dirt_angle3 = Range3D(dirt_rroll, dirt_rpitch, dirt_ryaw)


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


# TODO: need to get the height of this mesh to calculate rock height off

# Not currently used
def mod_dirt():
    """Randomize position and rotation of dirt"""
    geom_id = model.geom_name2id("dirt")
    body_id = model.body_name2id("dirt")
    mesh_id = model.geom_dataid[geom_id]

    model.body_pos[body_id] = start_body_pos[body_id]  + sample_xyz(dirt_range3d)
    model.geom_quat[geom_id] = sample_quat(dirt_angle3)
    
    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    mesh_verts = model.mesh_vert[vert_adr : vert_adr+vert_num]

    rot_quat = model.geom_quat[geom_id]
    rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)

    mesh_abs_pos = floor_offset + model.body_pos[body_id] + rots

    xy_indexes = mesh_abs_pos[:, 0:2]
    z_heights = mesh_abs_pos[:, 2]

    # NOTE: this is not the best method.  It could be that the dirt is placed in a way
    # that the max height is not the effective max_height of a rock mesh, since it
    # could be buried.  What would be a better way to do this? 
    # I could do some subtraction of the rock mesh with the dirt mesh, but this 
    # becomes quite complicated because the indexing does not line up
    # This is a decent simple method for now

    def dirt_height_xy(xy):
        # Min squared distance
        z_index = np.argmin( np.sum(np.square(xy_indexes - xy), axis=1) - 0.5*z_heights )

        viewer.add_marker(pos=mesh_abs_pos[z_index, :], label="o", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))

        #print(np.max(mesh_abs_pos, axis=0))

        height = z_heights[z_index]
        viewer.add_marker(pos=np.concatenate([xy, np.array([height])]), label="x", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
        viewer.add_marker(pos=np.concatenate([xy, np.array([height])]), label="x")
        if height < 0 or height > 0.3:
            height = 0 
        return height

    def mean_height(xy):
        return np.maximum(0, np.mean(z_heights[z_heights > 0]))

    return mean_height


def mod_rocks():
    """
    Randomize the rocks so that the model will generalize to competition rocks
    This modifications currently being done are:
        - Randomizing positions
        - Randomizing orientations
        - Shuffling the 3 rock meshes so that they can be on the left, middle, or right
        - Generating new random rock meshes every n runs (with Blender)
    """
    rock_body_ids = {}
    rock_geom_ids = {}
    rock_mesh_ids = {}
    max_height_idxs = {} 
    rot_cache = {}
    #max_height_xys = {}

    #dirt_height_xy = mod_dirt()

    for name in model.geom_names:
        if name[:4] != "rock":
            continue 
        
        geom_id = model.geom_name2id(name)
        body_id = model.body_name2id(name)
        mesh_id = model.geom_dataid[geom_id]
        rock_geom_ids[name] = geom_id
        rock_body_ids[name] = body_id
        rock_mesh_ids[name] = mesh_id

        # Rotate the rock and get z value of the highest point in the rotated rock mesh
        rot_quat = random_quat()
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        mesh_verts = model.mesh_vert[vert_adr : vert_adr+vert_num]
        rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
        model.geom_quat[geom_id] = rot_quat  
        max_height_idx = np.argmax(rots[:,2])
        max_height_idxs[name] =  max_height_idx
        rot_cache[name] = rots

    rock_mod_cache = [] 

    # Randomize the positions of the rocks. 
    shuffle_names = list(rock_body_ids.keys())
    random.shuffle(shuffle_names)

    for i in range(len(shuffle_names)):
        name = shuffle_names[i]
        rots = rot_cache[name]
        model.body_pos[rock_body_ids[name]] = np.array(sample_xyz(rock_ranges[i]))

        max_height_idx = max_height_idxs[name]
        xyz_for_max_z = rots[max_height_idx]

        global_xyz = floor_offset + xyz_for_max_z + model.body_pos[rock_body_ids[name]]
        gxy = global_xyz[0:2]
        max_height = global_xyz[2] 
        viewer.add_marker(pos=global_xyz, label="m", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))

        #dirt_z = dirt_height_xy(gxy)
        dirt_z = 0
        #print(name, dirt_z)


        z_height = max_height - dirt_z

        rock_mod_cache.append((name, z_height))

    return rock_mod_cache


def randrocks():
    """Generate a new set of 3 random rock meshes using a Blender script"""
    import subprocess
    subprocess.call([FLAGS.blender_path, "--background", "--python", "randrock.py"])

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

def main():
    global model, sim, tex_modder, cam_modder, light_modder
    x_batch = []
    y_batch = []
    x_frames = []
    y_grounds = []
    batch_count = 0
    last_batch_count = 0
    
    for i_step in count(1):
        # Randomize (mod) all relevant parameters
        mod_textures()
        mod_lights()
        mod_camera()
        rock_mod_cache = mod_rocks()
        mod_arena()
        sim.step()  # NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK 
    
        # Get angle of camera and display it 
        quat = np.quaternion(*model.cam_quat[0])
        rpy = quaternion.as_euler_angles(quat) * 180 / np.pi
        cam_pos = model.cam_pos[0]
        viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, rpy))
    
        # Grab an image from the camera at (224, 244, 3) to feed into CNN
        cam_img = sim.render(1280, 720, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
        image_noise_variance = sample(image_noise_rvariance) 
        cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)
        cam_img = preproc_img(cam_img)
    
        #display_image(cam_img)
    
        r1_pos = floor_offset + model.body_pos[model.body_name2id('rock1')]
        r2_pos = floor_offset + model.body_pos[model.body_name2id('rock2')]
        r3_pos = floor_offset + model.body_pos[model.body_name2id('rock3')]
    
        r1_diff = r1_pos - cam_pos
        r2_diff = r2_pos - cam_pos
        r3_diff = r3_pos - cam_pos
    
        ground_truth = []
        for slot in rock_mod_cache:
            name = slot[0]
            z_height = slot[1]
    
            pos = floor_offset + model.body_pos[model.body_name2id(name)]
            diff = pos - cam_pos
            #text = "x: {0:.2f} y: {1:.2f} height:{2:.2f}".format(diff[0], diff[1], z_height)
            text = "height:{0:.2f}".format(z_height)
            viewer.add_marker(pos=pos, label=text, rgba=np.zeros(4))
    
            ground_truth += [diff[0], diff[1], z_height]
        
        # If visualizing, skip network training
        if FLAGS.visualize:
            viewer.render()
            continue

        # Ground truth measurements and camera frames to torch Tensors
    
        # Batch is full, do a network update
        if i_step % FLAGS.batch_size == 0:
            batch_count += 1
            # Stack all the Tensors for this batch, do a forward pass, and compute
            # the loss
            x_batch = None
            y_batch = None
            coords_pred = resnet.forward(x_batch)
            loss = l2_loss(coords_pred, y_batch)
            print(i_step, y_batch.data, coords_pred.data, loss.data[0])

            # Backward pass and update weights 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Reset batch lists
            del x_frames[:]
            del y_grounds[:]
    
        # Save weights every x batches
        if batch_count != last_batch_count and batch_count % FLAGS.save_every == 0:
            print("saving weights to {}".format(FLAGS.save_path))
            print("done saving")

        # If super batch, generate new rocks and reload model
        if batch_count != last_batch_count and batch_count % FLAGS.super_batch == 0:
            randrocks()

            model = load_model_from_path("xmls/nasa/box.xml")
            sim = MjSim(model)
            tex_modder = TextureModder(sim)
            cam_modder = CameraModder(sim)
            light_modder = LightModder(sim)

        last_batch_count = batch_count
    

if __name__ == "__main__":
    # Parse command line arguments
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.os == "mac":
        FLAGS.dtype = "cpu"
        FLAGS.blender_path = "/Applications/blender.app/Contents/MacOS/blender"
    if FLAGS.os == "ubuntu":
        FLAGS.dtype = "gpu"
        FLAGS.blender_path = "blender"
    # Set torch type for either CPU or CUDA (this allows ops to be run on GPU)
    if FLAGS.dtype == "cpu":
        pass
    else:
        pass

    # Mujoco setup
    #randrocks()
    model = load_model_from_path("xmls/nasa/box.xml")
    sim = MjSim(model)
    tex_modder = TextureModder(sim)
    cam_modder = CameraModder(sim)
    light_modder = LightModder(sim)
    if FLAGS.visualize:
        viewer = MjViewer(sim)
    else:
        # If we are not visualizing, create a fake visualizer that does nothing
        # This was just so that the rest of the code stays clean and doesn't have
        # to do a bunch of checks. 
        class FakeViewer(object):
            def __init__(self):
                pass
            def add_marker(self, **kwargs):
                pass
        viewer = FakeViewer()
    
    # Get start state of params to slightly jitter later
    start_geo_size = model.geom_size.copy()
    start_geom_quat = model.geom_quat.copy()
    start_body_pos = model.body_pos.copy()
    start_body_quat = model.body_quat.copy()
    floor_offset = model.body_pos[model.body_name2id('floor')]


    # Neural network setup
    resnet = tf.keras.applications.ResNet50(include_top=True, weights=None)
    resnet.layers.pop()
    resnet.layers.pop()
    predictions = tf.keras.layers.Dense(9, activation=None, name='predictions')
    inp = resnet.input
    out = predictions(resnet.layers[-1].output)
    resnet = tf.keras.models.Model(inp, out, name="resnet50")
    #resnet.summary()

    #flatten = tf.keras.layers.Flatten()(resnet.layers[-1].output)
    #predictions = tf.keras.layers.Dense(9, name='predictions')(flatten)
    #inp2 = resnet.input
    #resnet = tf.keras.models.Model(inp2, predictions)

    main()
