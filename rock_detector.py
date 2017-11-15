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
        #cam_img = arena_modder.get_cam_frame(display=True, ground_truth=rock_ground_truth)
    
        yield (cam_img, rock_ground_truth)

        # If super batch, generate new rocks and reload model
        if i % FLAGS.super_batch == 0:
            arena_modder.randrocks()


def dataset_input_fn():
    dataset = tf.data.Dataset.from_generator(arena_sampler, (tf.float32, tf.float32), \
            (tf.TensorShape([224, 224, 3]), tf.TensorShape([9])))

    dataset = dataset.batch(FLAGS.batch_size)
    print(dataset)

    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labelsi.
    features, labels = iterator.get_next()
    print(features)
    print(labels)
    return {'input_1' : features}, labels


def main():
    x_batch = []
    y_batch = []
    x_frames = []
    y_grounds = []
    batch_count = 0
    last_batch_count = 0

    #value = dataset.make_one_shot_iterator().get_next()

    est_vgg16.train(input_fn=dataset_input_fn, steps=10)


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

    conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    keras_vgg16 = tf.keras.models.Sequential()
    keras_vgg16.add(conv_section)
    keras_vgg16.add(tf.keras.layers.Flatten())
    keras_vgg16.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(None, 512*7*7), name="fc1"))
    keras_vgg16.add(tf.keras.layers.Dense(64, activation="relu"))
    keras_vgg16.add(tf.keras.layers.Dense(9, activation="linear", name="classifier"))
    keras_vgg16.summary()

    keras_vgg16.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss='mean_squared_error',
                          metric='accuracy')

    est_vgg16 = tf.keras.estimator.model_to_estimator(keras_model=keras_vgg16)


    main()
