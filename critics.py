import tensorflow as tf
from utils import *


class FCCritic():
    def __init__(self, name, img_size_x, img_size_y, img_size_z, output):
        """
        Neural network which takes a batch of images and creates a batch of scalars which represent a score for how
        real the image looks.
        Uses just several fully connected layers.
        Works for arbitrary image size and number of channels, because it flattens them first.

        :param img_size:
        :param channels: number of channels in the image (RGB = 3, Black/White = 1)
        """
        self.name = name
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.img_size_z = img_size_z
        self.output = output

        self.gaussian_noise = tf.keras.layers.GaussianNoise(0.1)
        self.gaussian_noise2 = tf.keras.layers.GaussianNoise(0.3)

    def __call__(self, image, reuse=None, mini_batch = True):
        """
        Method which performs the computation.

        :param image: Tensor of shape [batch_size, self.img_size, self.img_size, self.channels]
        :param reuse: Boolean which determines tf scope reuse.
        :return: Tensor of shape [batch_size, 1]
        """

        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            image = tf.reshape(image, [-1, self.img_size_x * self.img_size_y * self.img_size_z])

            image = tf.compat.v1.layers.dense(image, 1000, tf.nn.relu)
            #image = self.gaussian_noise(image)
            image = tf.compat.v1.layers.dense(image, 500, tf.nn.relu)
            #image = self.gaussian_noise2(image)
            image = tf.compat.v1.layers.dense(image, 250, tf.nn.relu)
            #image = self.gaussian_noise2(image)
            image = tf.compat.v1.layers.dense(image, 250, tf.nn.relu)
            intermediate_layer = image
            #image = self.gaussian_noise2(image)
            image = tf.compat.v1.layers.dense(image, 250, tf.nn.relu)
            #image = self.gaussian_noise(image)
            image = tf.compat.v1.layers.dense(image, self.output)
            return image, intermediate_layer





