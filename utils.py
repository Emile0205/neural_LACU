import os
from time import time
import numpy as np
import tensorflow as tf
import functools


def apply_conv(x, filters=32, kernel_size=3, he_init = True):
    initializer = tf.keras.initializers.VarianceScaling()

    return tf.compat.v1.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                            padding='SAME', kernel_initializer=initializer)

############## the activation function #####################################################
def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)

############### batch normalization ######################################################
def bn(x):
    return tf.compat.v1.layers.batch_normalization(x,
                                                  momentum=0.9,
                                                  center=True,
                                                  scale=True,
                                                  epsilon=1e-5,
                                                  training=True)

##################### All the required building blocks for architecture ##########
def stable_norm(x, ord):
    x = tf.contrib.layers.flatten(x)
    alpha = tf.reduce_max(tf.abs(x) + 1e-5, axis=1)
    result = alpha * tf.norm(x / alpha[:, None], ord=ord, axis=1)
    return result

def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        temp = tf.add_n([x[:,::2,::2,:], x[:,0::2,::2,:], x[:,::2,0::2,:], x[:,0::2,0::2,:]]) / 4.
        return temp

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.compat.v1.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))

def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)

def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)
################### resnet block ##########################################
def resblock(x, filters, resample=None, normalize=False):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(conv_meanpool, filters=filters)
        conv_shortcut = functools.partial(conv_meanpool, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == 'up':
        conv_1 = functools.partial(upsample_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = functools.partial(upsample_conv, filters=filters,
                                          kernel_size=1, he_init=False)
    elif resample == None:
        conv_1 = functools.partial(apply_conv, filters=filters)
        conv_2 = functools.partial(apply_conv, filters=filters)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update

############## resnet block optimized #############################
def resblock_optimized(x, filters):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters)
        update = conv_meanpool(activation(update), filters=filters)

        skip = meanpool_conv(x, filters=128, kernel_size=1, he_init=False)
        return skip + update


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        self.logdir = logdir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H_%M", localtime())

        self.model_path = os.path.join(ProjectPath.base, self.logdir, self.timestamp)


class Timer:
    def __init__(self):
        self.curr_time = time()

    def time(self):
        diff = time() - self.curr_time
        self.curr_time = time()
        return diff
