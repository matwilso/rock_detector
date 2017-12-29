import random
import numpy as np
import quaternion
import skimage
import matplotlib.pyplot as plt
from arena_modder import ArenaModder

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjSimPool, MjViewer 

from utils import preproc_image, display_image
from utils import Range, Range3D, rto3d # object type things
from utils import sample, sample_from_list, sample_xyz, sample_quat, sample_geom_type, random_quat, jitter_quat 

class SimManager():
    """"""
    def __init__(self, filepath, blender_path=None, visualize=False, num_sims=1):
        self.filepath = filepath
        self.blender_path = blender_path 
        self.visualize = visualize

        self.base_model = load_model_from_path(filepath)

        if num_sims > 1:
            self.pool = MjRenderPool(self.base_model, device_ids=num_sims, modder=ArenaModder)
        else:
            self.sim = MjSim(self.model)
            self.modder = ArenaModder(self.sim)
            if self.visualize:
                self.modder.visualize = True
                self.viewer = MjViewer(self.sim)
                self.modder.viewer = self.viewer
            else:
                self.viewer = None 


    def forward(self):
        """
        Advances simulator a step (NECESSARY TO MAKE CAMERA AND LIGHT MODDING WORK)
        """
        if self.simpool:
            self.pool.forward()
        else:
            self.sim.forward() 
            if self.visualize:
                # Get angle of camera and display it 
                quat = np.quaternion(*self.model.cam_quat[0])
                ypr = quaternion.as_euler_angles(quat) * 180 / np.pi
                cam_pos = self.model.cam_pos[0]
                #self.viewer.add_marker(pos=cam_pos, label="CAM: {}{}".format(cam_pos, ypr))
                self.viewer.add_marker(pos=cam_pos, label="CAM: {}".format(ypr))
                self.viewer.render()

    def _pool_get_cam_frame(self):
        cam_imgs = self.pool.render(640, 360, camera_name='camera1', randomize=True)[:, ::-1, :, :] # Rendered images are upside-down.
        image_noise_variance = sample(IMAGE_NOISE_RVARIANCE) 
        cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)
        cam_img = preproc_image(cam_img)
        if display:
            label = str(ground_truth[3:6])
            display_image(cam_img, label)
        return cam_img


    def _get_cam_frame(self, display, ground_truth):
        cam_img = self.sim.render(640, 360, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.
        image_noise_variance = sample(IMAGE_NOISE_RVARIANCE) 
        cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=image_noise_variance) * 255).astype(np.uint8)
        cam_img = preproc_image(cam_img)
        if display:
            label = str(ground_truth[3:6])
            display_image(cam_img, label)
        return cam_img

    def get_cam_frame(self, display=False, ground_truth=None):
        """Grab an image from the camera (224, 244, 3) to feed into CNN"""
        if self.num_sims > 1:
            self._pool_get_cam_frame()
        else:
            return self._get_cam_frame(display, ground_truth)
        return cam_img

    def get_ground_truths():
        if self.num_sims > 1:
            for sim in self.

        else:
            return self.modder.get_ground_truths()

    def randrocks(self):
        """Generate a new set of 3 random rock meshes using a Blender script"""
        if self.blender_path is None:
            raise Exception('You must install Blender and include the path to its exectubale in the constructor to use this method')

        import subprocess
        subprocess.call([self.blender_path, "--background", "--python", "randrock.py"])

