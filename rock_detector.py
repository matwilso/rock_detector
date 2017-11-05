#!/usr/bin/env python3

import argparse
from itertools import count
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

# NOTE: 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
# could be about 2 days for full convergence

# TODO: better preproc on the image if necessary (i.e., research this)

# TODO: I should try to run this with and without the floor randomization and see if it
# still works as well.  My guess is that with floor it will still be fine


def sample_arena(arena_modder):
    """
    Randomize all relevant parameters, return an image and the ground truth
    labels for the rocks in the image
    """
    # Randomize (mod) all relevant parameters
    arena_modder.mod_textures()
    arena_modder.mod_lights()
    arena_modder.mod_camera()
    arena_modder.mod_walls()
    rock_ground_truth = arena_modder.mod_rocks()

    arena_modder.step()
    
    cam_img = arena_modder.get_cam_frame()
    #cam_img = arena_modder.get_cam_frame(display=True, ground_truth=rock_ground_truth)

    return cam_img, rock_ground_truth


def main():
    #global model, sim, tex_modder, cam_modder, light_modder
    x_batch = []
    y_batch = []
    x_frames = []
    y_grounds = []
    batch_count = 0
    last_batch_count = 0
    
    for i_step in count(1):
        # cam_img =  input to neural net
        # rock_ground_truth = compare to output of neural net to train
        cam_img, rock_ground_truth = sample_arena(arena_modder)
        
        # Wrap camera frame and ground truth measurements in tensorflow Tensors
        # TODO:
    
        # Batch is full, do a network update
        if i_step % FLAGS.batch_size == 0:
            batch_count += 1
            # Stack all the Tensors for this batch, do a forward pass, and compute
            # the loss
            x_batch = None
            y_batch = None
            #coords_pred = resnet.forward(x_batch)
            #loss = l2_loss(coords_pred, y_batch)
            #print(i_step, y_batch.data, coords_pred.data, loss.data[0])

            ## Backward pass and update weights 
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            # Reset batch lists
            del x_frames[:]
            del y_grounds[:]
    
        # Save weights every x batches
        if batch_count != last_batch_count and batch_count % FLAGS.save_every == 0:
            print("saving weights to {}".format(FLAGS.save_path))
            print("done saving")

        # If super batch, generate new rocks and reload model
        if batch_count != last_batch_count and batch_count % FLAGS.super_batch == 0:
            pass
            #arena_modder.randrocks()
            
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

    # Arena modder setup
    arena_modder = ArenaModder("xmls/nasa/box.xml", blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)

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
