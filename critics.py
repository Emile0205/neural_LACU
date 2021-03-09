import tensorflow as tf
from utils import *


class DCDiscriminator():
    def __init__(self, name, img_shape, output):
        """
        Neural network which takes a batch of images and creates a batch of scalars which represent a score for how
        real the image looks.
        Uses just several fully connected layers.
        Works for arbitrary image size and number of channels, because it flattens them first.

        :param img_size:
        :param channels: number of channels in the image (RGB = 3, Black/White = 1)
        """
        self.name = name
        self.img_shape = img_shape
        self.output = output
        
    def __call__(self, print_summary = False):
        """
        Method which performs the computation.

        :param image: Tensor of shape [batch_size, self.img_size, self.img_size, self.channels]
        :param reuse: Boolean which determines tf scope reuse.
        :return: Tensor of shape [batch_size, 1]
        """
        model = Sequential(name=self.name)

        model.add(Conv2D(16, kernel_size=3, strides = 2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        
        
        model.add(Conv2D(32, kernel_size=3, strides = 2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))       
        
        model.add(Conv2D(64, kernel_size=3, strides = 2, padding="same"))
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(GaussianNoise(0.25))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))       
        
        model.add(Flatten())
        model.add(Dense(self.output))
                         
        if print_summary == True:
            print("Input shape: " + str(self.img_shape))
            model.summary()

        return model 




