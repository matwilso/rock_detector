import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt



"""

Holds various low level utils for the rock detector training

"""
# OBJECT TYPE THINGS
def Range(min, max):
    """Return 1d numpy array of with min and max"""
    if min <= max:
        return np.array([min, max])
    else:
        print("WARNING: min {} was greater than max {}".format(min, max))
        return np.array([max, min])

def Range3D(x, y, z):
    """Return numpy 1d array of with min and max"""
    return np.array([x,y,z])

def rto3d(r):
    return Range3D(r, r, r)


def str2bool(v):
    """For argparse"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# PRINTING
def print_rocks(arr):
    rocks = np.split(arr, 3)
    for i, r in enumerate(rocks):
        print("rock{0:d} x: {1:.2f}, y: {2:.2f}, h: {3:.2f}".format(i+1, r[0], r[1], r[2]))

# IMAGE UTILS
def preproc_image(img):
    """Chop off edges and then downsample to (224, 224, 3)"""
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.shape == (224, 224, 3):
        return img

    hdiv, hcrop = img.shape[0] // 224, (img.shape[0] % 224) // 2
    wdiv, wcrop = img.shape[1] // 224, (img.shape[1] % 224) // 2

    crop = img[hcrop:-hcrop, wcrop:-wcrop, :]
    down_sample = crop[::hdiv, ::wdiv, :]

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



# UTIL FUNCTIONS FOR RANDOMIZATION
def sample(num_range, as_int=False):
    """Sample a float in the num_range"""
    samp = random.uniform(num_range[0], num_range[1])
    if as_int:
        return int(samp)
    else:
        return samp

def sample_from_list(choices):
    return choices[sample([0,len(choices)], as_int=True)]

def sample_geom_type(reject=[]):
    """Sample a mujoco geom type (range 3-6 capsule-box)"""

    types = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]
    while True:
        samp = random.randrange(3,7)
        rejected = False
        for r in reject:
            if types.index(r) == samp:
                rejected = True
                break
        if rejected:
            continue
        else:
            return samp

def sample_xyz(range3d):
    """Sample 3 floats in the 3 num_ranges"""
    x = sample(range3d[0])
    y = sample(range3d[1])
    z = sample(range3d[2])
    return (x, y, z)

def sample_light_dir():
    """Sample a random direction for a light. I don't quite understand light dirs so
    this might be wrong"""
    # Pretty sure light_dir is just the xyz of a quat with w = 0.
    # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
    LIGHT_DIR = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))
    return np.quaternion(0, *sample_xyz(LIGHT_DIR)).normalized().components.tolist()[1:]

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

# PLOTTING
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.


    Usage:
    ```
    filtered = x[~is_outlier(x)]
    ```

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


