import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import CameraModder, LightModder, MaterialModder, TextureModder

from utils import preproc_image, display_image
from utils import Range, Range3D, rto3d # object type things
from utils import sample, sample_xyz, sample_quat, random_quat, jitter_quat # for randomizing

#from collections import OrderedDict

# TODO [p1]: add some distractor objects, like smaller rocks.  And just the bigger 
# rocks are the only thing that matters

# TODO: need to add some distractor field of view objects like the arms that were
# in the shot during the comp.  We may not have any such things, but it would be
# good to be robust to them

# TODO: Randomize the floor, either by giving it rotation or adding the
# ocean meshes

# TODO: set the arena center bin is 0,0

# TODO: do i even really need heights?  Well one thing it does, is if a 
# non-rock (just a dirt bump) is detected, you can check the height and it will 
# be low.



# MODDING PARAMETERS
# x is left and right
# y is back and forth
ACX = -2.19
XOFF = 0.5
LEFTX = -4.29
RIGHTX = -0.1
BINY = -3.79
DIGY = 3.59
AFZ = 0.0
ZLOW = 0.3
ZHIGH = 1.0

SZ_LEN = 1.5
OBS_LEN = 2.94
DIG_LEN = 2.94
SZ_ENDY = BINY + SZ_LEN # start zone end
CAM_YDELTA = 1.5 
OBS_SY = SZ_ENDY
OBS_ENDY = OBS_SY + OBS_LEN

LIGHT_RX = Range(LEFTX, RIGHTX) 
LIGHT_RY = Range(BINY, DIGY)
LIGHT_RZ = Range(AFZ, AFZ + ZHIGH)
LIGHT_RANGE3D = Range3D(LIGHT_RX, LIGHT_RY, LIGHT_RZ)
LIGHT_DIR3 = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))

CAM_RX = Range(ACX - XOFF, ACX + XOFF) # center of arena +/- 0.5
CAM_RY = Range(BINY+0.2, SZ_ENDY)
CAM_RZ = Range(AFZ + ZLOW, AFZ + ZHIGH)
CAM_RANGE3D = Range3D(CAM_RX, CAM_RY, CAM_RZ)

CAM_RROLL = Range(-95, -85) # think this might actually be yaw
CAM_RPITCH = Range(65, 90)
CAM_RYAW = Range(88, 92) # this might actually be pitch, based on coordinate frames
CAM_ANGLE3 = Range3D(CAM_RROLL, CAM_RPITCH, CAM_RYAW)
CAM_RFOVY = Range(35, 55)

IMAGE_NOISE_RVARIANCE = Range(0.0, 0.0001)

# Rock placement range parameters
ROCK_LANEX = 0.4  # width parameters of x range
OUTER_EXTRA = 0.5 # how much farther rocks should go out on the right and left lanes
ROCK_BUFFX = 0.2  # distacne between rock lanes

# How far into the obstacle zone the rocks should start.  
ROCK_START_OFFSET = 0.2  
MID_START_OFFSET = 0.4 # bit more for middle rock

ROCK_RY = Range(OBS_SY + ROCK_START_OFFSET, OBS_ENDY)
MID_RY = Range(OBS_SY + MID_START_OFFSET, OBS_ENDY)
ROCK_RZ = Range(AFZ - 0.02, AFZ + 0.2)

# Position dependent ranges
LEFT_RX = Range(-3*ROCK_LANEX - OUTER_EXTRA, -ROCK_LANEX - ROCK_BUFFX)
MID_RX = Range(-ROCK_LANEX, ROCK_LANEX)
RIGHT_RX = Range(ROCK_BUFFX+ROCK_LANEX, 3*ROCK_LANEX + OUTER_EXTRA)

# Form full 3D sample range
LEFT_ROCK_RANGE = Range3D(LEFT_RX, ROCK_RY, ROCK_RZ)
MID_ROCK_RANGE = Range3D(MID_RX, MID_RY, ROCK_RZ)
RIGHT_ROCK_RANGE = Range3D(RIGHT_RX, ROCK_RY, ROCK_RZ)
ROCK_RANGES = [LEFT_ROCK_RANGE, MID_ROCK_RANGE, RIGHT_ROCK_RANGE]

DIRT_RX = Range(0.0, 0.3)
DIRT_RY = Range(0.0, 0.3)
DIRT_RZ = Range(-0.05, 0.05)
DIRT_RANGE3D = Range3D(DIRT_RX, DIRT_RY, DIRT_RZ)

DIRT_RROLL = Range(-180, 180)  # yaw
DIRT_RPITCH = Range(-90, -90)
DIRT_RYAW = Range(0, 0)  # roll
DIRT_ANGLE3 = Range3D(DIRT_RROLL, DIRT_RPITCH, DIRT_RYAW)

class ArenaModder(object):
    def __init__(self, filepath, blender_path=None, visualize=False):
        self._init(filepath, blender_path, visualize)

    def _init(self, filepath, blender_path=None, visualize=False):
        self.filepath = filepath
        self.blender_path = blender_path 
        self.model = load_model_from_path(filepath)
        self.sim = MjSim(self.model)
        self.visualize = visualize

        # Get start state of params to slightly jitter later
        self.start_geo_size = self.model.geom_size.copy()
        self.start_geom_quat = self.model.geom_quat.copy()
        self.start_body_pos = self.model.body_pos.copy()
        self.start_body_quat = self.model.body_quat.copy()
        self.floor_offset = self.model.body_pos[self.model.body_name2id('floor')]

        self.tex_modder = TextureModder(self.sim)
        self.cam_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)

        if not hasattr(self, 'viewer'):
            self.viewer = MjViewer(self.sim) if self.visualize else None
        else:
            self.viewer.update_sim(self.sim) if self.visualize else None


    def step(self):
        """
        Advances simulator a step (NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK)
        """
        self.sim.step() 

        if self.visualize:
            # Get angle of camera and display it 
            quat = np.quaternion(*self.model.cam_quat[0])
            rpy = quaternion.as_euler_angles(quat) * 180 / np.pi
            cam_pos = self.model.cam_pos[0]
            self.viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, rpy))

            self.viewer.render()




    def get_cam_frame(self, display=False, ground_truth=None):
        """Grab an image from the camera at (224, 244, 3) to feed into CNN"""

        cam_img = self.sim.render(1280, 720, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
        image_noise_variance = sample(IMAGE_NOISE_RVARIANCE) 
        cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)
        cam_img = preproc_image(cam_img)
        if display:
            label = str(ground_truth[3:6])
            display_image(cam_img, label)

        return cam_img

    def mod_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        self.tex_modder.randomize()
        self.tex_modder.rand_all('skybox')
    
    def mod_lights(self):
        """Randomize pos, direction, and lights"""
        for i, name in enumerate(self.model.light_names):
            # random sample 50% of any given light being on 
            self.light_modder.set_active(name, random.uniform(0, 1) > 0.5)
    
            # Pretty sure light_dir is just the xyz of a quat with w = 0.
            # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
            dir_xyz = np.quaternion(0, *sample_xyz(LIGHT_DIR3)).normalized().components.tolist()[1:]
            self.light_modder.set_pos(name, sample_xyz(LIGHT_RANGE3D))
            self.light_modder.set_dir(name, dir_xyz)
            self.light_modder.set_specular(name, sample_xyz(LIGHT_DIR3))
            self.light_modder.set_diffuse(name, sample_xyz(LIGHT_DIR3))
    
    def mod_camera(self):
        """Randomize pos, direction, and fov of camera"""
    
        self.cam_modder.set_pos('camera1', sample_xyz(CAM_RANGE3D))
        self.cam_modder.set_quat('camera1', sample_quat(CAM_ANGLE3))
    
        fovy = sample(CAM_RFOVY)
        self.cam_modder.set_fovy('camera1', fovy)
    
    def mod_walls(self):
        """
        Randomize the x, y, and orientation of the walls slights.
        Also drastically randomize the height of the walls, in many cases they won't
        be seen at all. This will allow the model to generalize to scenarios without
        walls, or where the walls and geometry is slightly different than the sim 
        model
        """
    
        for name in self.model.geom_names:
            if name[-4:] != "wall":
                continue 
    
            geom_id = self.model.geom_name2id(name)
            body_id = self.model.body_name2id(name)
    
            jitter_x = Range(-0.2, 0.2)
            jitter_y = Range(-0.2, 0.2)
            jitter_z = Range(-1.0, 0.0)
            jitter3D = Range3D(jitter_x, jitter_y, jitter_z)
    
            self.model.body_pos[body_id] = self.start_body_pos[body_id] + sample_xyz(jitter3D)
            self.model.body_quat[body_id] = jitter_quat(self.start_body_quat[body_id], 0.005)
    
    
    # TODO: need to get the height of this mesh to calculate rock height off
    
    # Not currently used
    def mod_dirt(self):
        """Randomize position and rotation of dirt"""
        geom_id = self.model.geom_name2id("dirt")
        body_id = self.model.body_name2id("dirt")
        mesh_id = self.model.geom_dataid[geom_id]
    
        self.model.body_pos[body_id] = self.start_body_pos[body_id]  + sample_xyz(DIRT_RANGE3D)
        self.model.geom_quat[geom_id] = sample_quat(DIRT_ANGLE3)
        
        vert_adr = self.model.mesh_vertadr[mesh_id]
        vert_num = self.model.mesh_vertnum[mesh_id]
        mesh_verts = self.model.mesh_vert[vert_adr : vert_adr+vert_num]
    
        rot_quat = self.model.geom_quat[geom_id]
        rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
    
        mesh_abs_pos = floor_offset + self.model.body_pos[body_id] + rots
    
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
            #print(np.max(mesh_abs_pos, axis=0))
    
            height = z_heights[z_index]

            if height < 0 or height > 0.3:
                height = 0 

            if self.visualize:
                self.viewer.add_marker(pos=mesh_abs_pos[z_index, :], label="o", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
                self.viewer.add_marker(pos=np.concatenate([xy, np.array([height])]), label="x", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([1.0, 0.0, 0.0, 1.0]))
                self.viewer.add_marker(pos=np.concatenate([xy, np.array([height])]), label="x")

            return height
    
        def mean_height(xy):
            return np.maximum(0, np.mean(z_heights[z_heights > 0]))
    
        return mean_height
    
    
    def mod_rocks(self):
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
    
        for name in self.model.geom_names:
            if name[:4] != "rock":
                continue 
            
            geom_id = self.model.geom_name2id(name)
            body_id = self.model.body_name2id(name)
            mesh_id = self.model.geom_dataid[geom_id]
            rock_geom_ids[name] = geom_id
            rock_body_ids[name] = body_id
            rock_mesh_ids[name] = mesh_id
    
            # Rotate the rock and get z value of the highest point in the rotated rock mesh
            rot_quat = random_quat()
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            mesh_verts = self.model.mesh_vert[vert_adr : vert_adr+vert_num]
            rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
            self.model.geom_quat[geom_id] = rot_quat  
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
            self.model.body_pos[rock_body_ids[name]] = np.array(sample_xyz(ROCK_RANGES[i]))
    
            max_height_idx = max_height_idxs[name]
            xyz_for_max_z = rots[max_height_idx]
    
            global_xyz = self.floor_offset + xyz_for_max_z + self.model.body_pos[rock_body_ids[name]]
            gxy = global_xyz[0:2]
            max_height = global_xyz[2] 

            if self.visualize:
                self.viewer.add_marker(pos=global_xyz, label="m", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    
            #dirt_z = dirt_height_xy(gxy)
            dirt_z = 0
            #print(name, dirt_z)
    
            z_height = max_height - dirt_z
            rock_mod_cache.append((name, z_height))
    
        return self._get_ground_truth(rock_mod_cache)

    def _get_ground_truth(self, rock_mod_cache):
        """
        Pass in rock_mod_cache returned from mod_rocks

        Return 1d numpy array of 9 elements for positions of all 3 rocks including:
            - rock x dist from cam
            - rock y dist from cam
            - rock z height from arena floor
        """
        cam_pos = self.model.cam_pos[0]

        r1_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock1')]
        r2_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock2')]
        r3_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock3')]
    
        r1_diff = r1_pos - cam_pos
        r2_diff = r2_pos - cam_pos
        r3_diff = r3_pos - cam_pos
    
        ground_truth = np.zeros(9, dtype=np.float32)
        for i, slot in enumerate(rock_mod_cache):
            name = slot[0]
            z_height = slot[1]
    
            pos = self.floor_offset + self.model.body_pos[self.model.body_name2id(name)]
            diff = pos - cam_pos
            #text = "x: {0:.2f} y: {1:.2f} height:{2:.2f}".format(diff[0], diff[1], z_height)
            text = "height:{0:.2f}".format(z_height)
            if self.visualize:
                self.viewer.add_marker(pos=pos, label=text, rgba=np.zeros(4))
    
            ground_truth[3*i+0] = diff[0]
            ground_truth[3*i+1] = diff[1]
            ground_truth[3*i+2] = z_height

        ##print(ground_truth)
        return ground_truth

    
    
    def randrocks(self):
        """Generate a new set of 3 random rock meshes using a Blender script"""
        if self.blender_path is None:
            raise Exception('You must install Blender and include the path to its exectubale in the constructor to use this method')

        import subprocess
        subprocess.call([self.blender_path, "--background", "--python", "randrock.py"])
        self._init(self.filepath, self.blender_path, self.visualize)




