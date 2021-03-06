import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjSimPool, MjViewer 
from mujoco_py.modder import BaseModder, CameraModder, LightModder, MaterialModder, TextureModder

from utils import preproc_image, display_image
from utils import Range, Range3D, rto3d # object type things
from utils import sample, sample_from_list, sample_xyz, sample_light_dir, sample_quat, sample_geom_type, random_quat, jitter_quat 

# TODO: set the arena center bin is 0,0

# gid = geom_id
# bid = body_id

# MODDING PARAMETERS
# x is left and right
# y is back and forth
ACX = -2.19
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
# start zone end
SZ_ENDY = BINY + SZ_LEN 
OBS_SY = SZ_ENDY
OBS_ENDY = OBS_SY + OBS_LEN

class ArenaModder(BaseModder):
    """
    Object to handle randomization of all relevant properties of Mujoco sim

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get start state of params to slightly jitter later
        self.start_geo_size = self.model.geom_size.copy()
        self.start_geom_quat = self.model.geom_quat.copy()
        self.start_body_pos = self.model.body_pos.copy()
        self.start_body_quat = self.model.body_quat.copy()
        self.start_matid = self.model.geom_matid.copy()
        self.floor_offset = self.model.body_pos[self.model.body_name2id('floor')]
        self.rock_mod_cache = None

        self.tex_modder = TextureModder(self.sim)
        self.cam_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)
        
        # Set these both externally to visualize
        self.visualize = False
        self.viewer = None

    def whiten_materials(self):
        self.tex_modder.whiten_materials()

    def randomize(self):
        self.mod_textures()
        self.mod_lights()
        self.mod_camera()
        self.mod_walls()
        self.mod_extras()
        self.mod_rocks()

    def mod_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        for name in self.sim.model.geom_names:
            if name != "billboard" and "light_disc" not in name:
                self.tex_modder.rand_all(name)
                ##texture_sizes = [32, 64, 128, 256, 512, 1024]
                ##gid = self.model.geom_name2id(name)
                ##mid = self.model.geom_matid[gid]
                ##tid = self.model.mat_texid[mid]
                ##self.model.tex_width[tid] = sample_from_list(texture_sizes)
                ##self.model.tex_height[tid] = 6 * self.model.tex_width[tid]
        self.tex_modder.rand_all('skybox')

    
    def mod_lights(self):
        """Randomize pos, direction, and lights"""
        # light stuff
        LIGHT_RX = Range(LEFTX, RIGHTX) 
        LIGHT_RY = Range(BINY, DIGY)
        LIGHT_RZ = Range(AFZ, AFZ + ZHIGH)
        LIGHT_RANGE3D = Range3D(LIGHT_RX, LIGHT_RY, LIGHT_RZ)
        LIGHT_UNIF = Range3D(Range(0,1), Range(0,1), Range(0,1))

        for i, name in enumerate(self.model.light_names):
            lid = self.model.light_name2id(name)
            # random sample 80% of any given light being on 
            self.light_modder.set_active(name, sample([0,1]) < 0.8)
            #self.light_modder.set_active(name, 0)
            dir_xyz = sample_light_dir()
            self.light_modder.set_pos(name, sample_xyz(LIGHT_RANGE3D))
            self.light_modder.set_dir(name, dir_xyz)
            self.light_modder.set_specular(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_diffuse(name, sample_xyz(LIGHT_UNIF))
            #self.light_modder.set_ambient(name, sample_xyz(LIGHT_UNIF))
            #self.model.light_directional[lid] = sample([0,1]) < 0.01
    
    def mod_camera(self):
        """Randomize pos, direction, and fov of camera"""
        # Params
        XOFF = 1.0
        CAM_RX = Range(ACX - XOFF, ACX + XOFF) # center of arena +/- 0.5
        CAM_RY = Range(BINY+0.2, SZ_ENDY)
        CAM_RZ = Range(AFZ + ZLOW, AFZ + ZHIGH)
        CAM_RANGE3D = Range3D(CAM_RX, CAM_RY, CAM_RZ)
        CAM_RYAW = Range(-95, -85)
        CAM_RPITCH = Range(65, 90)
        CAM_RROLL = Range(85, 95) # this might actually be pitch?
        CAM_ANGLE3 = Range3D(CAM_RYAW, CAM_RPITCH, CAM_RROLL)

        # "The horizontal field of view is computed automatically given the 
        # window size and the vertical field of view." - Mujoco
        # This range was calculated using: themetalmuncher.github.io/fov-calc/
        # ZED has 110° hfov --> 78° vfov, Logitech C920 has 78° hfov ---> 49° vfov
        # These were rounded all the way down to 40° and up to 80°. It starts
        # to look a bit bad in the upper range, but I think it will help 
        # generalization.
        CAM_RFOVY = Range(40, 80)

        # Actual mods
        self.cam_modder.set_pos('camera1', sample_xyz(CAM_RANGE3D))
        self.cam_modder.set_quat('camera1', sample_quat(CAM_ANGLE3))
    
        fovy = sample(CAM_RFOVY)
        self.cam_modder.set_fovy('camera1', fovy)

    def _set_visible(self, prefix, range_top, visible):
        """Helper function to set visibility of several objects"""
        if not visible:
            if range_top == 0:
                name = prefix
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 0.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 0.0
        else:
            if range_top == 0:
                name = prefix
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 1.0

            for i in range(range_top):
                name = "{}{}".format(prefix, i)
                gid = self.model.geom_name2id(name)
                self.model.geom_rgba[gid][-1] = 1.0
    
    def mod_extra_distractors(self, visible=True):
        """mod rocks and tools on the side of the arena"""
        # TODO: I might consider changing these to look like rocks instead of 
        # just random shapes.  It just looks weird to me right now.  Ok for now,
        # but it seems a bit off.
        N = 20
        self._set_visible("side_obj", N, visible)
        if not visible:
            return

        Z_JITTER = 0.05
        OBJ_XRANGE = Range(0.01, 0.1)
        OBJ_YRANGE = Range(0.01, 0.1)
        OBJ_ZRANGE = Range(0.01, 0.1)
        OBJ_SIZE_RANGE = Range3D(OBJ_XRANGE, OBJ_YRANGE, OBJ_ZRANGE)

        floor_gid = self.model.geom_name2id("floor")

        left_body_id = self.model.body_name2id("left_wall")
        left_geom_id = self.model.geom_name2id("left_wall")
        right_body_id = self.model.body_name2id("right_wall")
        right_geom_id = self.model.geom_name2id("right_wall")

        left_center = self.model.body_pos[left_body_id]
        left_geo = self.model.geom_size[left_geom_id]
        left_height = left_center[2] + left_geo[2]
        left_xrange = Range(left_center[0]-left_geo[0], left_center[0]+left_geo[0])
        left_yrange = Range(left_center[1]-left_geo[1], left_center[1]+left_geo[1])
        left_zrange = 0.02+Range(left_height-Z_JITTER, left_height+Z_JITTER)
        left_range = Range3D(left_xrange, left_yrange, left_zrange)

        right_center = self.model.body_pos[right_body_id]
        right_geo = self.model.geom_size[right_geom_id]
        right_height = right_center[2] + right_geo[2]
        right_xrange = Range(right_center[0]-right_geo[0], right_center[0]+right_geo[0])
        right_yrange = Range(right_center[1]-right_geo[1], right_center[1]+right_geo[1])
        right_zrange = 0.02+Range(right_height-Z_JITTER, right_height+Z_JITTER)
        right_range = Range3D(right_xrange, right_yrange, right_zrange)

        for i in range(N):
            name = "side_obj{}".format(i)
            obj_bid = self.model.body_name2id(name)
            obj_gid = self.model.geom_name2id(name)
            self.model.geom_quat[obj_gid] = random_quat() 
            self.model.geom_size[obj_gid] = sample_xyz(OBJ_SIZE_RANGE)
            self.model.geom_type[obj_gid] = sample_geom_type()

            # 50% chance of invisible 
            if sample([0,1]) < 0.5:
                self.model.geom_rgba[obj_gid][-1] = 0.0
            else:
                self.model.geom_rgba[obj_gid][-1] = 1.0
            ## 50% chance of same color as floor and rocks 
            if sample([0,1]) < 0.5:
                self.model.geom_matid[obj_gid] = self.model.geom_matid[floor_gid]
            else:
                self.model.geom_matid[obj_gid] = self.start_matid[obj_gid]

            # 10 always on the left, 10 always on the right
            if i < 10:
                self.model.body_pos[obj_bid] = sample_xyz(left_range)
            else:
                self.model.body_pos[obj_bid] = sample_xyz(right_range)

    def mod_extra_judges(self, visible=True):
        """mod NASA judges around the perimeter of the arena"""
        # TODO: might want to add regions on the sides of the arena, but these
        # may be covered by the distractors already
        N = 5
        self._set_visible("judge", N, visible)
        if not visible:
            return

        JUDGE_XRANGE = Range(0.1, 0.2)
        JUDGE_YRANGE = Range(0.1, 0.2)
        JUDGE_ZRANGE = Range(0.75, 1.0)
        JUDGE_SIZE_RANGE = Range3D(JUDGE_XRANGE, JUDGE_YRANGE, JUDGE_ZRANGE)

        digwall_bid = self.model.body_name2id("dig_wall")
        digwall_gid = self.model.geom_name2id("dig_wall")


        digwall_center = self.model.body_pos[digwall_bid]
        digwall_geo = self.model.geom_size[digwall_gid]
        digwall_xrange = Range(-1.0 + digwall_center[0]-digwall_geo[0], 1.0 + digwall_center[0]+digwall_geo[0])
        digwall_yrange = Range(digwall_center[1]+0.5, digwall_center[1]+1.5)
        digwall_zrange = JUDGE_ZRANGE - 0.75
        digwall_range = Range3D(digwall_xrange, digwall_yrange, digwall_zrange)

        for i in range(N):
            name = "judge{}".format(i)
            judge_bid = self.model.body_name2id(name)
            judge_gid = self.model.geom_name2id(name)

            #self.model.geom_quat[judge_gid] = jitter_quat(self.start_geom_quat[judge_gid], 0.05)
            self.model.geom_quat[judge_gid] = random_quat()
            self.model.geom_size[judge_gid] = sample_xyz(JUDGE_SIZE_RANGE)
            self.model.geom_type[judge_gid] = sample_geom_type()
            if self.model.geom_type[judge_gid] == 3 or self.model.geom_type[judge_gid] == 5:
                self.model.geom_size[judge_gid][1] = self.model.geom_size[judge_gid][2]

            self.model.body_pos[judge_bid] = sample_xyz(digwall_range)

            # 50% chance of invisible 
            self.model.geom_rgba[judge_gid][-1] = sample([0,1]) < 0.5

    def mod_extra_robot_parts(self, visible=True):
        """add distractor parts of robots in the lower area of the camera frame"""
        N = 3
        self._set_visible("robot_part", N, visible)
        if not visible:
            return 
        # Project difference into camera coordinate frame
        cam_pos = self.model.cam_pos[0]
        cam_quat = np.quaternion(*self.model.cam_quat[0])
        lower_range  = Range3D([0.0, 0.0], [-0.2, -0.3], [-0.2, -0.3])
        lower_size = Range3D([0.2, 0.6], [0.01, 0.15], [0.01, 0.15])
        lower_angle = Range3D([-85.0, -95.0], [-180, 180], [-85, -95])

        upper_range  = Range3D([-0.6, 0.6], [-0.05, 0.05], [-0.05, 0.05])
        upper_size = Range3D([0.005, 0.05], [0.005, 0.05], [0.01, 0.3])
        upper_angle = Range3D([-85.0, -95.0], [-180, 180], [-85, -95])

        name = "robot_part0"
        lower_bid = self.model.body_name2id(name)
        lower_gid = self.model.geom_name2id(name)

        lower_pos = cam_pos + quaternion.rotate_vectors(cam_quat, sample_xyz(lower_range))
        self.model.body_pos[lower_bid] = lower_pos
        self.model.geom_size[lower_gid] = sample_xyz(lower_size)
        self.model.geom_quat[lower_gid] = sample_quat(lower_angle)
        self.model.geom_type[lower_gid] = sample_geom_type(reject=["capsule"])

        if self.model.geom_type[lower_gid] == 5:
            self.model.geom_size[lower_gid][0] = self.model.geom_size[lower_gid][2]

        for i in range(1,10):
            name = "robot_part{}".format(i)
            upper_bid = self.model.body_name2id(name)
            upper_gid = self.model.geom_name2id(name)
            upper_pos = lower_pos + sample_xyz(upper_range)
            self.model.body_pos[upper_bid] = upper_pos
            self.model.geom_size[upper_gid] = sample_xyz(upper_size)
            self.model.geom_type[upper_gid] = sample_geom_type()

            # 50% of the time, choose random angle instead reasonable angle
            if sample([0,1]) < 0.5:
                self.model.geom_quat[upper_gid] = sample_quat(upper_angle)
            else:
                self.model.geom_quat[upper_gid] = random_quat()


    def mod_extra_arena_structure(self, visible=True):
        """add randomized structure of the arena in the background with 
        pillars and a crossbar"""
        N = 16
        self._set_visible("arena_structure", N, visible)
        if not visible:
            return 

        STARTY = DIGY + 3.0
        ENDY = DIGY + 5.0
        XBAR_SIZE = Range3D([10.0, 20.0], [0.05, 0.1], [0.2, 0.5])
        XBAR_RANGE = Range3D([0.0, 0.0], [STARTY, ENDY], [3.0, 5.0])
        STRUCTURE_SIZE = Range3D([0.05, 0.1], [0.05, 0.1], [3.0, 6.0])
        STRUCTURE_RANGE = Range3D([np.nan, np.nan], [STARTY, ENDY], [0.0, 0.0])
        x_range = np.linspace(ACX - 10.0, ACX + 10.0, 15)

        # crossbar
        name = "arena_structure{}".format(N-1)
        xbar_bid = self.model.body_name2id(name)
        xbar_gid = self.model.geom_name2id(name)

        self.model.geom_quat[xbar_gid] = jitter_quat(self.start_geom_quat[xbar_gid], 0.01)
        self.model.geom_size[xbar_gid] = sample_xyz(XBAR_SIZE)
        self.model.body_pos[xbar_bid] = sample_xyz(XBAR_RANGE)

        ### 10% chance of invisible 
        ##if sample([0,1]) < 0.1:
        ##    self.model.geom_rgba[xbar_gid][-1] = 0.0
        ##else:
        ##    self.model.geom_rgba[xbar_gid][-1] = 1.0

        for i in range(N-1):
            name = "arena_structure{}".format(i)
            arena_structure_bid = self.model.body_name2id(name)
            arena_structure_gid = self.model.geom_name2id(name) 
            STRUCTURE_RANGE[0] = Range(x_range[i]-0.1, x_range[i]+0.1)

            self.model.geom_quat[arena_structure_gid] = jitter_quat(self.start_geom_quat[arena_structure_gid], 0.01)
            self.model.geom_size[arena_structure_gid] = sample_xyz(STRUCTURE_SIZE)
            self.model.body_pos[arena_structure_bid] = sample_xyz(STRUCTURE_RANGE)

            self.model.geom_matid[arena_structure_gid] = self.model.geom_matid[xbar_gid]

            # 10% chance of invisible 
            if sample([0,1]) < 0.1:
                self.model.geom_rgba[arena_structure_gid][-1] = 0.0
            else:
                self.model.geom_rgba[arena_structure_gid][-1] = 1.0


    def mod_extra_arena_background(self, visible=True, N=10):
        """ """
        self._set_visible("background", N, visible)
        if not visible:
            return

        BACKGROUND_XRANGE = Range(0.1, 0.5)
        BACKGROUND_YRANGE = Range(0.1, 0.5)
        BACKGROUND_ZRANGE = Range(0.1, 1.0)
        BACKGROUND_SIZE_RANGE = Range3D(BACKGROUND_XRANGE, BACKGROUND_YRANGE, BACKGROUND_ZRANGE)

        digwall_bid = self.model.body_name2id("dig_wall")
        digwall_gid = self.model.geom_name2id("dig_wall")
        c = digwall_center = self.model.body_pos[digwall_bid]

        background_range = Range3D([c[0] - 4.0, c[0] + 4.0], [c[1] + 8.0, c[1] + 12.0], [0.0, 5.0])

        for i in range(N):
            name = "background{}".format(i)
            background_bid = self.model.body_name2id(name)
            background_gid = self.model.geom_name2id(name)

            self.model.geom_quat[background_gid] = random_quat()
            self.model.geom_size[background_gid] = sample_xyz(BACKGROUND_SIZE_RANGE)
            self.model.geom_type[background_gid] = sample_geom_type()

            self.model.body_pos[background_bid] = sample_xyz(background_range)


    def mod_extra_lights(self, visible=True):
        """"""
        #visible = False
        MODE_TARGETBODY = 3
        STARTY = DIGY + 3.0
        ENDY = DIGY + 5.0
        LIGHT_RANGE = Range3D([ACX - 2.0, ACX + 2.0], [STARTY, ENDY], [3.0, 5.0])
        any_washed = False

        for i in range(3):
            name = "extra_light{}".format(i)
            lid = self.model.light_name2id(name)
            self.model.light_active[lid] = visible
            if not visible:
                return

            self.model.light_pos[lid] = sample_xyz(LIGHT_RANGE)
            self.model.light_dir[lid] = sample_light_dir()
            self.model.light_directional[lid] = 0

            ### 3% chance of directional light, washing out colors
            ### (P(at least 1 will be triggered) ~= 0.1) = 1 - 3root(p)
            washout = sample([0,1]) < 0.03
            any_washed = any_washed or washout
            self.model.light_directional[lid] = washout

        if any_washed:
            self.mod_extra_light_discs(visible = visible and sample([0,1]) < 0.9)
        else:
            self.mod_extra_light_discs(visible=False)


    def mod_extra_light_discs(self, visible=True):
        """"""
        N = 10
        self._set_visible("light_disc", N, visible)
        if not visible:
            return

        Z_JITTER = 0.05
        DISC_XRANGE = Range(0.1, 4.0)
        DISC_YRANGE = Range(0.1, 4.0)
        DISC_ZRANGE = Range(0.1, 4.0)
        DISC_SIZE_RANGE = Range3D(DISC_XRANGE, DISC_YRANGE, DISC_ZRANGE)
        OUTR = 20.0
        INR = 10.0

        floor_bid = self.model.body_name2id("floor")
        c = self.model.body_pos[floor_bid]
        disc_xrange = Range(c[0] - OUTR, c[0] + OUTR)
        disc_yrange = Range(c[1], c[1] + OUTR)
        disc_zrange = Range(-5.0, 10.0)
        disc_range = Range3D(disc_xrange, disc_yrange, disc_zrange)

        for i in range(N):
            name = "light_disc{}".format(i)
            disc_bid = self.model.body_name2id(name)
            disc_gid = self.model.geom_name2id(name)
            disc_mid = self.model.geom_matid[disc_gid]

            self.model.geom_quat[disc_gid] = random_quat() 
            self.model.geom_size[disc_gid] = sample_xyz(DISC_SIZE_RANGE)
            self.model.geom_type[disc_gid] = sample_geom_type()

            # keep trying to place the disc until it lands outside of blocking stuff
            # (they will land in a u shape, because they are sampled with a 
            # constrain to not land in the middle)
            while True:
                xyz = sample_xyz(disc_range)

                if ((xyz[0] > (c[0] - INR) and xyz[0] < (c[0] + INR)) and
                   (xyz[1] > (c[1] - INR) and xyz[1] < (c[1] + INR))):
                    continue
                else:
                    self.model.geom_pos[disc_gid] = xyz
                    break

            # 50% chance of invisible 
            if sample([0,1]) < 0.5:
                self.model.geom_rgba[disc_gid][-1] = 0.0
            else:
                self.model.geom_rgba[disc_gid][-1] = 1.0

    def mod_extras(self, visible=True):
        """
        Randomize extra properties of the world such as the extra rocks on the side
        of the arena wal

        The motivation for these mods are that it seems likely that these distractor 
        objects, judges, etc, in the real images could degrade the performance 
        of the detector. 

        Artifacts:
        - Rocks and tools on edges of bin
        - NASA judges around the perimeter
        - Parts of the robot in the frame
        - Background arena structure and crowd 
        - Bright light around edges of arena
        """
        self.mod_extra_distractors(visible)
        self.mod_extra_robot_parts(visible)
        self.mod_extra_lights(visible)
        # 10% of the time, hide the other distractor pieces
        self.mod_extra_judges(visible = visible and sample([0,1]) < 0.9)
        self.mod_extra_arena_structure(visible = visible and sample([0,1]) < 0.9)
        self.mod_extra_arena_background(visible = visible and sample([0,1]) < 0.9)
    
    def mod_walls(self):
        """
        Randomize the x, y, and orientation of the walls slights.
        Also drastically randomize the height of the walls. In many cases they won't
        be seen at all. This will allow the model to generalize to scenarios without
        walls, or where the walls and geometry is slightly different than the sim 
        model
        """

        wall_names = ["left_wall", "right_wall", "bin_wall", "dig_wall"]
        for name in wall_names:
            geom_id = self.model.geom_name2id(name)
            body_id = self.model.body_name2id(name)
    
            jitter_x = Range(-0.2, 0.2)
            jitter_y = Range(-0.2, 0.2)
            jitter_z = Range(-0.75, 0.0)
            jitter3D = Range3D(jitter_x, jitter_y, jitter_z)
    
            self.model.body_pos[body_id] = self.start_body_pos[body_id] + sample_xyz(jitter3D)
            self.model.body_quat[body_id] = jitter_quat(self.start_body_quat[body_id], 0.005)
            if sample([0,1]) < 0.05:
                self.model.body_pos[body_id][2] = -2.0
    
    
    def mod_dirt(self):
        """Randomize position and rotation of dirt"""
        # TODO: 50% chance to flip the dirt pile upsidedown, then we will have 
        # to account for this in calculations

        # dirt stuff
        DIRT_RX = Range(0.0, 0.3)
        DIRT_RY = Range(0.0, 0.3)
        DIRT_RZ = Range(-0.05, 0.03)
        DIRT_RANGE3D = Range3D(DIRT_RX, DIRT_RY, DIRT_RZ)
        DIRT_RYAW = Range(-180, 180) 
        DIRT_RPITCH = Range(-90.5, -89.5)
        DIRT_RROLL = Range(-0.5, 0.5)  
        DIRT_ANGLE3 = Range3D(DIRT_RYAW, DIRT_RPITCH, DIRT_RROLL)
        dirt_bid = self.model.body_name2id("dirt")
        dirt_gid = self.model.geom_name2id("dirt")
        dirt_mid = self.model.geom_dataid[dirt_gid]
    
        # randomize position and yaw of dirt
        self.model.body_pos[dirt_bid] = self.start_body_pos[dirt_bid]  + sample_xyz(DIRT_RANGE3D)
        self.model.geom_quat[dirt_gid] = sample_quat(DIRT_ANGLE3)
        
        vert_adr = self.model.mesh_vertadr[dirt_mid]
        vert_num = self.model.mesh_vertnum[dirt_mid]
        mesh_verts = self.model.mesh_vert[vert_adr : vert_adr+vert_num]
    
        rot_quat = self.model.geom_quat[dirt_gid]
        rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
    
        mesh_abs_pos = self.floor_offset + self.model.body_pos[dirt_bid] + rots
    
        #xy_indexes = mesh_abs_pos[:, 0:2]
        #z_heights = mesh_abs_pos[:, 2]
    

        # Return a function that the user can call to get the approximate 
        # height of an xy location
        def local_mean_height(xy):
            """
            Take an xy coordinate, and the approximate z height of the mesh at that
            location. It works decently.

            Uses a weighted average mean of all points within a threshold of xy.
            """
            # grab all mesh points within threshold euclidean distance of xy
            gt0_xyz = mesh_abs_pos[mesh_abs_pos[:,2] > 0.01]
            eudists = np.sum(np.square(gt0_xyz[:,0:2] - xy), axis=1)
            indices =  eudists < 0.1
            close_xyz = gt0_xyz[indices]

            # if there are any nearby points above 0
            if np.count_nonzero(close_xyz[:, 2]) > 0:
                # weights for weighted sum. closer points to xy have higher weight 
                weights = 1 / (eudists[indices]) 
                weights = np.expand_dims(weights / np.sum(weights), axis=1)
                pos = np.sum(close_xyz * weights, axis=0)

                # show an "o" and a marker where the height is
                if self.visualize:
                    self.viewer.add_marker(pos=pos, label="o", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 1.0, 0.0, 1.0]))
                
                # approximate z height of ground
                return pos[2]
            else:
                return 0

        def always_zero(xy):
            return 0
        
        dirt_height_xy = local_mean_height
        #dirt_height_xy = always_zero
        return dirt_height_xy
    
    
    def mod_rocks(self):
        """
        Randomize the rocks so that the model will generalize to competition rocks
        This randomizations currently being done are:
            - Positions (within guesses of competition regions)
            - Orientations
            - Shuffling the 3 rock meshes so that they can be on the left, middle, or right
            - Generating new random rock meshes every n runs (with Blender)
        """
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

        # actual mods
        rock_body_ids = {}
        rock_geom_ids = {}
        rock_mesh_ids = {}
        max_height_idxs = {} 
        rot_cache = {}
        #max_height_xys = {}
    
        dirt_height_xy = self.mod_dirt()
    
        for name in self.model.geom_names:
            if name[:4] != "rock":
                continue 
            
            geom_id = self.model.geom_name2id(name)
            body_id = self.model.body_name2id(name)
            mesh_id = self.model.geom_dataid[geom_id]
            rock_geom_ids[name] = geom_id
            rock_body_ids[name] = body_id
            rock_mesh_ids[name] = mesh_id
    
            # Rotate the rock and get the z value of the highest point in the 
            # rotated rock mesh
            rot_quat = random_quat()
            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            mesh_verts = self.model.mesh_vert[vert_adr : vert_adr+vert_num]
            rots = quaternion.rotate_vectors(np.quaternion(*rot_quat).normalized(), mesh_verts)
            self.model.geom_quat[geom_id] = rot_quat  
            max_height_idx = np.argmax(rots[:,2])
            max_height_idxs[name] =  max_height_idx
            rot_cache[name] = rots
    
    
        # Shuffle the positions of the rocks (l or m or r)
        shuffle_names = list(rock_body_ids.keys())
        random.shuffle(shuffle_names)
        self.rock_mod_cache = [] 
        for i in range(len(shuffle_names)):
            name = shuffle_names[i]
            rots = rot_cache[name]
            self.model.body_pos[rock_body_ids[name]] = np.array(sample_xyz(ROCK_RANGES[i]))
    
            max_height_idx = max_height_idxs[name]
            xyz_for_max_z = rots[max_height_idx]
    
            # xyz coords in global frame
            global_xyz = self.floor_offset + xyz_for_max_z + self.model.body_pos[rock_body_ids[name]]
            gxy = global_xyz[0:2]
            max_height = global_xyz[2] 

            if self.visualize:
                self.viewer.add_marker(pos=global_xyz, label="m", size=np.array([0.01, 0.01, 0.01]), rgba=np.array([0.0, 0.0, 1.0, 1.0]))
    
            dirt_z = dirt_height_xy(gxy)
            #dirt_z = 0
            #print(name, dirt_z)
    
            z_height = max_height - dirt_z
            self.rock_mod_cache.append((name, z_height))

    def get_ground_truth(self):
        """
        self.rock_mod_cache set from self.mod_rocks

        Return 1d numpy array of 9 elements for positions of all 3 rocks including:
            - rock x dist from cam
            - rock y dist from cam
            - rock z height from arena floor
        """
        cam_pos = self.model.cam_pos[0]

        #line_pos = self.floor_offset + np.array([0.0, 0.75, 0.0])
        #self.viewer.add_marker(pos=line_pos)

        ##r0_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock0')]
        ##r1_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock1')]
        ##r2_pos = self.floor_offset + self.model.body_pos[self.model.body_name2id('rock2')]
        ##r0_diff = r0_pos - cam_pos
        ##r1_diff = r1_pos - cam_pos
        ##r2_diff = r2_pos - cam_pos
    
        ground_truth = np.zeros(9, dtype=np.float32)
        for i, slot in enumerate(self.rock_mod_cache):
            name = slot[0]
            z_height = slot[1]
    
            pos = self.floor_offset + self.model.body_pos[self.model.body_name2id(name)]
            diff = pos - cam_pos

            # Project difference into camera coordinate frame
            cam_angle = quaternion.as_euler_angles(np.quaternion(*self.model.cam_quat[0]))[0]
            cam_angle += np.pi/2
            in_cam_frame = np.zeros_like(diff)
            x = diff[1]
            y = -diff[0]
            in_cam_frame[0] = x * np.cos(cam_angle) + y * np.sin(cam_angle)
            in_cam_frame[1] = -x * np.sin(cam_angle) + y * np.cos(cam_angle)
            in_cam_frame[2] = z_height
            # simple check that change of frame is mathematically valid
            assert(np.isclose(np.sum(np.square(diff[:2])), np.sum(np.square(in_cam_frame[:2]))))
            # swap positions to match ROS standard coordinates
            ground_truth[3*i+0] = in_cam_frame[0]
            ground_truth[3*i+1] = in_cam_frame[1]
            ground_truth[3*i+2] = in_cam_frame[2]
            text = "{0} x: {1:.2f} y: {2:.2f} height:{3:.2f}".format(name, ground_truth[3*i+0], ground_truth[3*i+1], z_height)
            ##text = "x: {0:.2f} y: {1:.2f} height:{2:.2f}".format(ground_truth[3*i+0], ground_truth[3*i+1], z_height)
            ##text = "height:{0:.2f}".format(z_height)
            if self.visualize:
                self.viewer.add_marker(pos=pos, label=text, rgba=np.zeros(4))

        return ground_truth
