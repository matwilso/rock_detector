#!/usr/bin/env python3
import os
import argparse
import itertools
import numpy as np
import tensorflow as tf
from arena_modder import ArenaModder
from utils import display_image


# TODO: verify that the numbers are coming up at the right times
# TODO: add a seperated loss to view middle loss compared to the edge losses
# TODO: multithread optimize so that we can feed the neural net faster and
# we don't have to wait while simulating
# TODO: merge this together with high-level api


# NOTE: 2^2 * 1k images should get decent convergence (about ~4k, ~64k should be bomb)
# could be about 2 days for full convergence

# TODO: better preproc on the image if necessary (i.e., research this)

# TODO: I should try to run this with and without the floor randomization and see if it
# still works as well.  My guess is that with floor it will still be fine

def arena_sampler():
    """
    Generator to randomize all relevant parameters, return an image and the 
    ground truth labels for the rocks in the image
    """
    global arena_modder
    for i in itertools.count(1):
        # Randomize (mod) all relevant parameters
        arena_modder.mod_textures()
        arena_modder.mod_lights()
        arena_modder.mod_camera()
        arena_modder.mod_walls()
        rock_ground_truth = arena_modder.mod_rocks()
        arena_modder.step()
        
        # Grab cam frame and convert pixel value range from (0, 255) to (-0.5, 0.5)
        cam_img = arena_modder.get_cam_frame()
        cam_img = (cam_img.astype(np.float32) - 127.5) / 255

        yield cam_img, rock_ground_truth

        # If super batch, generate new rocks and reload model
        if i % FLAGS.super_batch == 0:
            arena_modder.randrocks()

def main():
    x_batch = []
    y_batch = []
    x_frames = []
    y_grounds = []
    sampler = arena_sampler()

    for i in itertools.count(1):
    #for i in range(50):
        batch_imgs = []
        batch_ground_truths = []
        for _ in range(FLAGS.batch_size):
            cam_img, rock_ground_truth = next(sampler)

            batch_imgs.append(cam_img)
            batch_ground_truths.append(rock_ground_truth)

        cam_imgs = np.stack(batch_imgs)
        ground_truths = np.stack(batch_ground_truths)
        
        _, curr_loss, curr_pred_output, summary  = sess.run([train, loss, pred_output, merged], {img_input : cam_imgs, real_output : ground_truths})

        if i % FLAGS.log_every == 0:
            test_writer.add_summary(summary, i)
        if i % FLAGS.save_every == 0:
            save_path = saver.save(sess, FLAGS.save_path)
            print("Model saved in file: %s" % FLAGS.save_path)

        #print(ground_truths)
        #print(curr_pred_output)
        #print(curr_loss)

def just_visualize():
    """Don't do any training, just loop through all the samples"""
    sampler = arena_sampler()
    for i in itertools.count(1):
        next(sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/rock_detector'),
        help="Directory to log data for TensorBoard to")
    parser.add_argument(
        '--log_every', type=int, default=10,
        help='Number of batches to run before logging data')
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
        '--super_batch', type=int, default=9223372036854775807,
        help='Number of batches before generating new rocks')
    parser.add_argument(
        '--save_path', type=str, default="weights/model.ckpt",
        help='File to save weights to')
    parser.add_argument(
        '--save_every', type=int, default=100,
        help='Number of batches before saving weights')
    parser.add_argument(
        '--load_path', type=str, default="weights/model.ckpt",
        help='File to load weights from')
    parser.add_argument(
        '--dtype', type=str, default="cpu",
        help='cpu or gpu')
    parser.add_argument(
        '--blender_path', type=str, default="blender", help='Path to blender executable')
    parser.add_argument(
        '--os', type=str, default="none",
        help='none (don\'t override any defaults) or mac or ubuntu')
    parser.add_argument(
        '--clean_log', type=bool, default=False,
        help="Delete previous tensorboard logs and start over")


    # Parse command line arguments
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.os == "mac":
        FLAGS.blender_path = "/Applications/blender.app/Contents/MacOS/blender"
    if FLAGS.os == "ubuntu":
        FLAGS.blender_path = "blender"

    # Arena modder setup
    arena_modder = ArenaModder("xmls/nasa/box.xml", blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)

    if FLAGS.visualize:
        just_visualize()
        exit(0)

    if FLAGS.clean_log:
        # Delete and remake log dir if it already exists 
        if tf.gfile.Exists(FLAGS.log_dir):
          tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    #with tf.device('/device:GPU:0'):
    # Neural network setup
    conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224,224,3))
    keras_vgg16 = tf.keras.models.Sequential()
    keras_vgg16.add(conv_section)
    keras_vgg16.add(tf.keras.layers.Flatten())
    keras_vgg16.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(None, 512*7*7), name="fc1"))
    keras_vgg16.add(tf.keras.layers.Dense(64, activation="relu", name="fc2"))
    keras_vgg16.add(tf.keras.layers.Dense(9, activation="linear", name="predictions"))

    keras_vgg16.summary()
    #keras_vgg16.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    #                      loss='mean_squared_error',
    #                      metric='accuracy')

    pred_output = keras_vgg16.output
    img_input = keras_vgg16.input
    
    real_output = tf.placeholder(tf.float32, shape=(None, 9), name="real_output")
    
    # loss (sum of squares)
    loss = tf.reduce_sum(tf.square(real_output - pred_output)) 
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-4) # 1e-4 suggested from dom rand paper
    train = optimizer.minimize(loss)


    tf.summary.scalar('ground', real_output[0, 3])
    tf.summary.scalar('pred', pred_output[0, 3])
    tf.summary.image('input', img_input, 10)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss_histogram', loss)
    merged = tf.summary.merge_all()

    sess = tf.Session()

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    if FLAGS.load_path != "0":
        saver.restore(sess,  FLAGS.load_path)
        print("Model restored.")
    else:
        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

    main()
