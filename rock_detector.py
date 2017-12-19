#!/usr/bin/env python3

import os
import argparse
import itertools
import numpy as np
import tensorflow as tf
from arena_modder import ArenaModder
from utils import display_image, preproc_image, print_rocks
import matplotlib.pyplot as plt
from threading import Thread, Event
from queue import Queue

# TODO: add distractor objects to the training, like artifacts of the robot.
# ...I think this project will actually be pretty tricky to get right. The 
# manifold of training is just so small

# TODO: create some verification images with ground truths that you know,
# using Gazebo or Bullet or (preferably in the real world)

# TODO: thicken the walls of the arena and add some distractors to the tops

# TODO: Randomize the floor adding the ocean meshes

# TODO: add billboard in background with sampled images, including of
# the actual competition area

# TODO: add a seperated loss to view middle loss compared to the edge losses
# TODO: multithread optimize so that we can feed the neural net faster and
# we don't have to wait while simulating

# TODO: better preproc (stats wise) on the image if necessary (i.e., research this)

# TODO: I should try to run this with and without the floor randomization and 
# see if it still works as well.  My guess is that with floor it will still be 
# fine

# TODO: do some calculation to detect if the rocks are out of the frame of
# view and to have some empty signal that the network can use

# TODO: sim pool so I can simulate multiple things at once, because that is my biggest bottleneck

# TODO: merge this together with high-level api

## NOTE: For a simpler task, they used 2^2 * 1k images. to get decent convergence (about ~4k, ~64k should be bomb). could be about 2 days for full convergence

data_queue = Queue()
super_batch_event = Event()


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/rock_debug'),
        help="Directory to log data for TensorBoard to")
    parser.add_argument(
        '--log_every', type=int, default=100,
        help='Number of batches to run before logging data')
    parser.add_argument(
        '--train_epochs', type=int, default=100,
        help='The number of epochs to use for training.')
    parser.add_argument(
        '--epochs_per_eval', type=int, default=1,
        help='The number of training epochs to run between evaluations.')
    parser.add_argument(
        '--batch_size', type=int, default=24,
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
        help='mac or ubuntu or none (don\'t override any defaults)')
    parser.add_argument(
        '--clean_log', type=bool, default=False,
        help="Delete previous tensorboard logs and start over")
    parser.add_argument(
        '--eval', type=bool, default=False,
        help="Evaluate on test images")

    # Parse command line arguments
    args, _ = parser.parse_known_args()
    return args


def evaluate():
    practice_img = preproc_image(plt.imread("assets/practice.jpg"))
    round1_img = preproc_image(plt.imread("assets/round1.jpg"))

    test_imgs = np.stack([practice_img, round1_img])

    curr_pred_output  = sess.run([pred_output], {img_input : test_imgs})[0]
    print("practice")
    print_rocks(curr_pred_output[0])
    print("round1")
    print_rocks(curr_pred_output[1])

def arena_sampler(arena_modder):
    """
    Generator to randomize all relevant parameters, return an image and the 
    ground truth labels for the rocks in the image
    """
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

        #display_image(cam_img, rock_ground_truth)


        #for r in rock_ground_truth:
        #    print('{0:.2f}'.format(r), end=', ')
        #print()
        #import matplotlib.pyplot as plt
        #plt.imshow(cam_img)
        #plt.show()

        cam_img = (cam_img.astype(np.float32) - 127.5) / 255
        yield cam_img, rock_ground_truth

def generate_data():
    # Arena modder setup
    arena_modder = ArenaModder("xmls/nasa/box.xml", blender_path=args.blender_path, visualize=args.visualize)

    for i in itertools.count(1):
        sampler = arena_sampler(arena_modder)
        (cam_img, rock_ground_truth) = next(sampler)
        data_queue.put((cam_img, rock_ground_truth))
        
        #if data_queue.qsize() > 
        if (data_queue.qsize() != 1):
            print(data_queue.qsize())

        if super_batch_event.is_set():
            arena_modder.randrocks()
            return

def train_loop():
    for i in itertools.count(1):
    #for i in range(50):
        batch_imgs = []
        batch_ground_truths = []

        for _ in range(args.batch_size):
            cam_img, rock_ground_truth = data_queue.get()
            batch_imgs.append(cam_img)
            batch_ground_truths.append(rock_ground_truth)

        cam_imgs = np.stack(batch_imgs)
        ground_truths = np.stack(batch_ground_truths)
        
        _, curr_loss, curr_pred_output, summary  = sess.run([train, loss, pred_output, merged], {img_input : cam_imgs, real_output : ground_truths})

        if i % args.log_every == 0:
            test_writer.add_summary(summary, i)
        if i % args.save_every == 0:
            save_path = saver.save(sess, args.save_path)
            print("Model saved in file: %s" % args.save_path)

        if i % args.super_batch == 0:
            super_batch_event.set()
            return 
        #print(ground_truths)
        #print(curr_pred_output)
        #print(curr_loss)

def main():
    if args.eval:
        evaluate()
    else:
        #import ipdb; ipdb.set_trace()
        # start the chain thread
        train_thread = Thread(target=train_loop, daemon=True)
        train_thread.start()
        while True:
            data_thread = Thread(target=generate_data, daemon=True)
            data_thread.start()
            data_thread.join()
            if super_batch_event.is_set():
                return

        train_thread.join()

def just_visualize():
    """Don't do any training, just loop through all the samples"""
    # Arena modder setup
    arena_modder = ArenaModder("xmls/nasa/box.xml", blender_path=args.blender_path, visualize=args.visualize)
    sampler = arena_sampler(arena_modder)
    for i in itertools.count(1):
        next(sampler)

if __name__ == "__main__":
    args = parse_command_line()
    if args.os == "mac":
        args.blender_path = "/Applications/blender.app/Contents/MacOS/blender"
    if args.os == "ubuntu":
        args.blender_path = "blender"

    if args.visualize:
        just_visualize()
        exit(0)

    if args.clean_log:
        # Delete and remake log dir if it already exists 
        if tf.gfile.Exists(args.logdir):
          tf.gfile.DeleteRecursively(args.logdir)
    tf.gfile.MakeDirs(args.logdir)

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

    train_writer = tf.summary.FileWriter(args.logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(args.logdir + '/test')
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    if args.load_path != "0":
        saver.restore(sess,  args.load_path)
        print("Model restored from "+args.load_path)
    else:
        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

    main()
