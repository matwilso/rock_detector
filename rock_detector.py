#!/usr/bin/env python3

import argparse
import itertools
import tensorflow as tf
from arena_modder import ArenaModder

from utils import display_image


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
    '--super_batch', type=int, default=1000,
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

# NOTE: 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
# could be about 2 days for full convergence

# TODO: better preproc on the image if necessary (i.e., research this)

# TODO: I should try to run this with and without the floor randomization and see if it
# still works as well.  My guess is that with floor it will still be fine


def arena_sampler():
    """
    Randomize all relevant parameters, return an image and the ground truth
    labels for the rocks in the image
    """
    for i in itertools.count(1):
        # Randomize (mod) all relevant parameters
        arena_modder.mod_textures()
        arena_modder.mod_lights()
        arena_modder.mod_camera()
        arena_modder.mod_walls()
        rock_ground_truth = arena_modder.mod_rocks()
    
        arena_modder.step()
        
        cam_img = arena_modder.get_cam_frame()
        tf_cam_img = tf.image.convert_image_dtype(cam_img, tf.float32)

        #cam_img = arena_modder.get_cam_frame(display=True, ground_truth=rock_ground_truth)
    
        yield {"input_1" : tf_cam_img, "ground_truth": rock_ground_truth}

        # If super batch, generate new rocks and reload model
        if i % FLAGS.super_batch == 0:
            arena_modder.randrocks()

def main():
    x_batch = []
    y_batch = []
    x_frames = []
    y_grounds = []
    batch_count = 0
    last_batch_count = 0




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

    # Arena modder setup
    arena_modder = ArenaModder("xmls/nasa/box.xml", blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)




    main()
