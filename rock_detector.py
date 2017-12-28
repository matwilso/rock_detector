#!/usr/bin/env python3
import os
import os.path
import pickle
import argparse
import itertools
import yaml
import numpy as np
import tensorflow as tf
from arena_modder import ArenaModder
from utils import display_image, preproc_image, print_rocks, str2bool
import matplotlib.pyplot as plt
from threading import Thread, Event
from queue import Queue

# TODO: create some verification images with ground truths that you know,
# using Gazebo or Bullet or (preferably) in the real world

# TODO: add billboard in background with sampled images, including of
# the actual competition area

# TODO: setup random seeding for the training to make sure experiments are consistent

# TODO: sim pool so I can simulate multiple things at once, because that is my biggest bottleneck

# TODO: better preproc (stats wise) on the image if necessary (i.e., research this)

# TODO: do some calculation to detect if the rocks are out of the frame of
# view and to have some empty signal that the network can use

# TODO: add a seperated loss to view middle loss compared to the edge losses

# TODO: I should try to run this with and without the floor randomization and 
# see if it still works as well.  My guess is that without floor rando, it won't 
# work well irl

# TODO: merge this together with high-level api

# TODO: add some notes on what this detector is robust to

# TODO: at some point, go and clean up the organization of this repo

# TODO: write code to automatically run several ablation tests (no walls, etc)

# TODO: make blender optional.  Generate fake rocks by overlapping a bunch of Mujoco
# primitives together and blending it each round

# NOTE: For a simpler task, they used 2^2 * 1k images. to get decent convergence (about ~4k, ~64k should be bomb). could be about 2 days for full convergence

# Throw data in to feed network 
data_queue = Queue()
# If either of the threads crashes, we want to bring down the other
crash_event = Event()
# When we have reached a super_batch, we should gen new rocks (with how I had
# to do it, this requires restarting the progam (see launcher.sh))
super_batch_event = Event()

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/rock_detector/'),
        help='Directory to log data for TensorBoard to')
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
        '--visualize', type=str2bool, default=False,
        help="Don't train, just interact with the mujoco sim and visualize everything")
    parser.add_argument(
        '--super_batch', type=int, default=9223372036854775807,
        help='Number of batches before generating new rocks')
    parser.add_argument(
        '--save_path', type=str, default='weights/model.ckpt',
        help='File to save weights to')
    parser.add_argument(
        '--save_every', type=int, default=100,
        help='Number of batches before saving weights')
    parser.add_argument(
        '--eval_every', type=int, default=100,
        help='Number of batches before evaluating model')
    parser.add_argument(
        '--load_path', type=str, default='weights/model.ckpt',
        help='File to load weights from')
    parser.add_argument(
        '--dtype', type=str, default='cpu',
        help='cpu or gpu')
    parser.add_argument(
        '--blender_path', type=str, default='blender', help='Path to blender executable')
    parser.add_argument(
        '--os', type=str, default='none',
        help='mac or ubuntu or none (don\'t override any defaults)')
    parser.add_argument(
        '--clean_log', type=str2bool, default=False,
        help='Delete previous tensorboard logs and start over')
    parser.add_argument(
        '--eval', type=str2bool, default=True,
        help='Evaluate on test images')
    parser.add_argument(
        '--freeze_conv', type=str2bool, default=False,
        help='Evaluate on test images')
    parser.add_argument(
        '--threaded', type=str2bool, default=True,
        help='Multithread image generation and model training')

    # Parse command line arguments
    FLAGS, _ = parser.parse_known_args()
    print(FLAGS)
    return FLAGS


def evaluate():
    """Evaluate the accuracy of the model on real images to make sure it generalizes"""
    data = yaml.load(open('assets/real_data.yaml', 'r'))
    data_path = 'assets/data/'

    test_imgs = []
    test_truths = []
    for line in data:
        ground_truth = line[0]
        filepath = line[1]
        test_imgs.append(preproc_image(plt.imread(data_path+filepath)))
        test_truths.append(np.array(ground_truth))

    test_batch_imgs = np.stack(test_imgs)
    test_batch_truths = np.stack(test_truths)

    # Split by batch size so we fit in GPU memory. Issue is how do you 
    # add 2 summaries together?
    ##batch_split_imgs = [test_imgs[i:i+FLAGS.batch_size] for i in range(0, len(test_imgs),FLAGS.batch_size)]
    ##batch_split_truths = [test_truths[i:i+FLAGS.batch_size] for i in range(0, len(test_truths),FLAGS.batch_size)]

    ##for i in range(len(batch_split_imgs)):
    ##    test_batch_imgs = np.stack(batch_split_imgs[0])
    ##    test_batch_truths = np.stack(batch_split_truths[0])

    val_loss, predictions, summary = sess.run([loss, pred_tf, loss_summary], {img_input : test_batch_imgs, real_output : test_batch_truths})
    test_writer.add_summary(summary)

    # Add plots
    matplotx, matploty, matploth = summary_plots(test_batch_truths, predictions)
    psumm = sess.run([plot_summary], {x_plot: matplotx, y_plot: matploty, h_plot: matploth})[0]
    test_writer.add_summary(psumm)

    print('practice')
    print_rocks(predictions[0])
    print('round1')
    print_rocks(predictions[1])

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
        arena_modder.mod_extras()
        rock_ground_truth = arena_modder.mod_rocks()
        arena_modder.step()
        
        # Grab cam frame and convert pixel value range from (0, 255) to (-0.5, 0.5)
        cam_img = arena_modder.get_cam_frame()
        ##camid = arena_modder.cam_modder.get_camid('camera1')
        ##cam_fovy = arena_modder.model.cam_fovy[camid]
        ##display_image(cam_img, "{} fovy={}".format(rock_ground_truth, cam_fovy))

        ##for r in rock_ground_truth:
        ##    print('{0:.2f}'.format(r), end=', ')
        ##print()
        ##plt.imshow(cam_img)
        ##plt.show()

        cam_img = (cam_img.astype(np.float32) - 127.5) / 255
        yield cam_img, rock_ground_truth

def generate_data():
    """Loop to generate data in separate thread to feed the training loop"""
    try:
        # Arena modder setup
        arena_modder = ArenaModder('xmls/nasa/box.xml', blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)

        for i in itertools.count(1):
            sampler = arena_sampler(arena_modder)
            (cam_img, rock_ground_truth) = next(sampler)
            data_queue.put((cam_img, rock_ground_truth))
            
            if (data_queue.qsize() != 1):
                print('queue_size = {}'.format(data_queue.qsize()))

            if super_batch_event.is_set():
                arena_modder.randrocks()
                return
    except Exception as e:
        print("CRASH EVENT: {}".format(e))
        crash_event.set()
        return 

def summary_plots(grounds, preds):
    """
    Create plots showing both ground truths and predictions of rock
    positions. Doing this natively (with Tensorflow tools) is currently 
    impossible, so I hacked together this matplotlib plotting
    """
    ground_summary_hist.append(grounds[0])
    pred_summary_hist.append(preds[0])
    with open(FLAGS.logdir+'summary.pkl', 'wb') as f:
        pickle.dump((ground_summary_hist, pred_summary_hist), f)

    glxs = [g[0] for g in ground_summary_hist]
    glys = [g[1] for g in ground_summary_hist]
    glhs = [g[2] for g in ground_summary_hist]
    gmxs = [g[3] for g in ground_summary_hist]
    gmys = [g[4] for g in ground_summary_hist]
    gmhs = [g[5] for g in ground_summary_hist]
    grxs = [g[6] for g in ground_summary_hist]
    grys = [g[7] for g in ground_summary_hist]
    grhs = [g[8] for g in ground_summary_hist]

    plxs = [p[0] for p in pred_summary_hist]
    plys = [p[1] for p in pred_summary_hist]
    plhs = [p[2] for p in pred_summary_hist]
    pmxs = [p[3] for p in pred_summary_hist]
    pmys = [p[4] for p in pred_summary_hist]
    pmhs = [p[5] for p in pred_summary_hist]
    prxs = [p[6] for p in pred_summary_hist]
    prys = [p[7] for p in pred_summary_hist]
    prhs = [p[8] for p in pred_summary_hist]

    # x-coords for plotting
    ts = 100*np.arange(len(ground_summary_hist))
    fig = plt.figure()
    plt.title('X')
    plt.plot(ts, gmxs, label='Ground')
    plt.plot(ts, pmxs, label='Pred')
    plt.legend(loc='best')
    fig.canvas.draw()
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    X = X.reshape(1, 480, 640, 3)
    plt.clf()

    plt.title('Y')
    plt.plot(ts, glys, label='Left Ground')
    plt.plot(ts, plys, label='Left Pred')
    plt.plot(ts, gmys, label='Mid Ground')
    plt.plot(ts, pmys, label='Mid Pred')
    plt.plot(ts, grys, label='Right Ground')
    plt.plot(ts, prys, label='Right Pred')
    plt.legend(loc='best')
    fig.canvas.draw()
    Y = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    Y = Y.reshape(1, 480, 640, 3)
    plt.clf()

    plt.title('H')
    plt.plot(ts, gmhs, label='Ground')
    plt.plot(ts, pmhs, label='Pred')
    plt.legend(loc='best')
    fig.canvas.draw()
    H = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    H = H.reshape(1, 480, 640, 3)
    plt.close()

    return X, Y, H

def train_loop():
    if not FLAGS.threaded:
        # Arena modder setup
        arena_modder = ArenaModder('xmls/nasa/box.xml', blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)
        sampler = arena_sampler(arena_modder)

    for i in itertools.count(1):
        batch_imgs = []
        batch_ground_truths = []

        if FLAGS.threaded:
            for _ in range(FLAGS.batch_size):
                cam_img, rock_ground_truth = data_queue.get()
                batch_imgs.append(cam_img)
                batch_ground_truths.append(rock_ground_truth)
        else:
            for _ in range(FLAGS.batch_size):
                cam_img, rock_ground_truth = next(sampler)
                batch_imgs.append(cam_img)
                batch_ground_truths.append(rock_ground_truth)

        cam_imgs = np.stack(batch_imgs)
        ground_truths = np.stack(batch_ground_truths)

        if i % FLAGS.log_every != 0:
            _, curr_loss, pred_output = sess.run([train_op, loss, pred_tf], {img_input : cam_imgs, real_output : ground_truths})
        else:
            # add special stuff for logging to tensorboard
            _, curr_loss, pred_output, tsumm  = sess.run([train_op, loss, pred_tf, train_summary], {img_input : cam_imgs, real_output : ground_truths})
            train_writer.add_summary(tsumm, i)
            matplotx, matploty, matploth = summary_plots(ground_truths, pred_output)
            psumm = sess.run([plot_summary], {x_plot: matplotx, y_plot: matploty, h_plot: matploth})[0]
            train_writer.add_summary(psumm, i)

        if i % FLAGS.save_every == 0:
            save_path = saver.save(sess, FLAGS.save_path)
            print('Model saved in file: %s' % FLAGS.save_path)
        if FLAGS.eval and i % FLAGS.eval_every == 0:
            evaluate()
        if i % FLAGS.super_batch == 0:
            super_batch_event.set()
            if not FLAGS.threaded:
                arena_modder.randrocks()
            return 
        if crash_event.is_set():
            return
        ##print(ground_truths)
        ##print(pred_output)
        ##print(curr_loss)

def main():
    if FLAGS.eval and FLAGS.eval_every == -1:
        evaluate()
    else:
        if FLAGS.threaded:
            # start the chain thread
            train_thread = Thread(target=train_loop, daemon=True)
            train_thread.start()
            while True:
                data_thread = Thread(target=generate_data, daemon=True)
                data_thread.start()
                data_thread.join()
                if super_batch_event.is_set():
                    return
                if crash_event.is_set():
                    exit(1)
            train_thread.join()
        else:
            train_loop()

    if crash_event.is_set():
        exit(1)


def just_visualize():
    """Don't do any training, just loop through all the samples"""
    # Arena modder setup
    arena_modder = ArenaModder('xmls/nasa/box.xml', blender_path=FLAGS.blender_path, visualize=FLAGS.visualize)
    sampler = arena_sampler(arena_modder)
    for i in itertools.count(1):
        next(sampler)

        if i % FLAGS.super_batch == 0:
            arena_modder.randrocks()
            return 

if __name__ == '__main__':
    FLAGS = parse_command_line()
    if FLAGS.os == 'mac':
        FLAGS.blender_path = '/Applications/blender.app/Contents/MacOS/blender'
    if FLAGS.os == 'ubuntu':
        FLAGS.blender_path = 'blender'

    if FLAGS.visualize:
        just_visualize()
        exit(0)

    # try to load summary histories from file
    if os.path.isfile(FLAGS.logdir+'summary.pkl'):
        try:
            with open(FLAGS.logdir+'summary.pkl', 'rb') as f:
                ground_summary_hist, pred_summary_hist = pickle.load(f)
        except:
            ground_summary_hist = []
            pred_summary_hist = []
    else:
        ground_summary_hist = []
        pred_summary_hist = []

    if FLAGS.clean_log:
        ground_summary_hist = []
        pred_summary_hist = []
        # Delete and remake log dir if it already exists 
        if tf.gfile.Exists(FLAGS.logdir):
          tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)

    #with tf.device('/device:GPU:0'):
    # Neural network setup
    conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224,224,3))
    keras_vgg16 = tf.keras.models.Sequential()
    keras_vgg16.add(conv_section)
    keras_vgg16.add(tf.keras.layers.Flatten())
    keras_vgg16.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(None, 512*7*7), name='fc1'))
    keras_vgg16.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
    keras_vgg16.add(tf.keras.layers.Dense(9, activation='linear', name='predictions'))

    keras_vgg16.summary()
    #keras_vgg16.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    #                      loss='mean_squared_error',
    #                      metric='accuracy')

    pred_tf = keras_vgg16.output
    img_input = keras_vgg16.input
    
    real_output = tf.placeholder(tf.float32, shape=(None, 9), name='real_output')
    
    # loss (sum of squares)
    loss = tf.reduce_sum(tf.square(real_output - pred_tf)) 
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-4) # 1e-4 suggested from dom rand paper

    # only train last layers
    if FLAGS.freeze_conv: 
        last_layers = tf.trainable_variables('(?![block.*conv.*])')
        train_op = optimizer.minimize(loss, var_list=last_layers)
    else:
        train_op = optimizer.minimize(loss)

    x_plot = tf.placeholder(tf.uint8, shape=(None, 480, 640, 3), name='x_plot')
    y_plot = tf.placeholder(tf.uint8, shape=(None, 480, 640, 3), name='y_plot')
    h_plot = tf.placeholder(tf.uint8, shape=(None, 480, 640, 3), name='h_plot')

    input_summary = tf.summary.image('input', img_input, 10)
    loss_summary = tf.summary.scalar('loss', loss)
    loss_hist = tf.summary.histogram('loss_histogram', loss)

    train_summary = tf.summary.merge([input_summary, loss_summary, loss_hist])
    x_summary_plot = tf.summary.image('plot_x', x_plot, 1)
    y_summary_plot = tf.summary.image('plot_y', y_plot, 1)
    h_summary_plot = tf.summary.image('plot_h', h_plot, 1)
    plot_summary = tf.summary.merge([x_summary_plot, y_summary_plot, h_summary_plot])
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    if FLAGS.load_path != '0':
        saver.restore(sess,  FLAGS.load_path)
        print('Model restored from '+FLAGS.load_path)
    else:
        # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)

    main()
