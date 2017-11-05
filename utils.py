import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt



"""

Holds various low level utils for the rock detector training

"""

# IMAGE UTILS
def preproc_image(img):
    crop = img[24:-24, 80:-80, :]
    down_sample = crop[::3, ::5, :]
    return down_sample 

def display_image(cam_img, label):
    practice_img = preproc_image(plt.imread("assets/practice.jpg"))
    round1_img = preproc_image(plt.imread("assets/round1.jpg"))
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

    fig.text(0, 0, label)
    plt.show()

# OBJECT TYPE THINGS
def Range(min, max):
    """Return 1d numpy array of with min and max"""
    if min < max:
        return np.array([min, max])
    else:
        return np.array([max, min])

def Range3D(x, y, z):
    """Return numpy 1d array of with min and max"""
    return np.array([x,y,z])

def rto3d(r):
    return Range3D(r, r, r)


# UTIL FUNCTIONS FOR RANDOMIZATION
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



