import scipy
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import scipy.io as sio


project_path = ProjectPath("log")

class dataset():
    def __init__(self, num_labelled_classes, path = './data', batch_size = 128, validation = 0.2):
        '''
        Dataset class used to retrieve data in batches.
        Variables:
            path 		- directory with data in npz filed
            batch_size		- batch size
            validation		- size of validation set extracted from labelled train set (set to 0.0 if no validation set required)
        '''
        self.batch_size = batch_size
        self.num_labelled_classes = num_labelled_classes
        self.rng = np.random.RandomState(1234)  # seed labels
        self.rng_data = np.random.RandomState(self.rng.randint(0, 2**11))  # seed shuffling

        with np.load(path + '/trainx_lab.npz') as a:
            self.trainx_lab = a['arr_0']
        with np.load(path + '/trainy_lab.npz') as a:
            self.trainy_lab = a['arr_0']
        with np.load(path + '/trainx_unlab.npz') as a:
            self.trainx_unlab = a['arr_0']
        with np.load(path + '/trainy_unlab.npz') as a:
            self.trainy_unlab = a['arr_0']

        with np.load(path + '/testx_.npz') as a:
            self.testx = a['arr_0']
        with np.load(path + '/testy_.npz') as a:
            self.testy = a['arr_0']
    
        self.img_size_x = self.trainx_lab.shape[1]
        self.img_size_y = self.trainx_lab.shape[2]
        self.img_size_z = self.trainx_lab.shape[3]
            
        temp = self.rng_data.permutation(self.trainx_lab.shape[0])
        self.trainx_lab = self.trainx_lab[temp]
        self.trainy_lab = self.trainy_lab[temp]

        temp = self.rng_data.permutation(self.trainx_unlab.shape[0])
        self.trainx_unlab = self.trainx_unlab[temp]
        self.trainy_unlab = self.trainy_unlab[temp]

        self.trainx = np.append(self.trainx_lab, self.trainx_unlab, axis = 0)
        self.trainy = np.append(self.trainy_lab, self.trainy_unlab, axis = 0)

        temp = self.rng_data.permutation(self.trainx.shape[0])
        self.trainx = self.trainx[temp]
        self.trainy = self.trainy[temp]

        temp = self.rng_data.permutation(self.testx.shape[0])
        self.testx = self.testx[temp]
        self.testy = self.testy[temp]

        del temp

        self.validx = self.trainx_lab[0:int(validation*self.trainx_lab.shape[0])]
        self.validy = self.trainy_lab[0:int(validation*self.trainx_lab.shape[0])]
        self.trainx_lab = self.trainx_lab[int(validation*self.trainx_lab.shape[0]): self.trainx_lab.shape[0]]
        self.trainy_lab = self.trainy_lab[int(validation*self.trainy_lab.shape[0]): self.trainy_lab.shape[0]]

        if self.trainx_lab.shape[0] < self.batch_size:
            print('Batch size too large for train set size. Copying train set to match') 
            while self.txs.shape[0] < self.batch_size:
                self.trainx_lab = np.append(self.trainx_lab, self.trainx_lab, axis = 0)
                self.trainy_lab = np.append(self.trainy_lab, self.trainy_lab, axis = 0)


        while self.validx.shape[0] < self.batch_size:
            self.validx = np.append(self.validx, self.validx, axis = 0)
            self.validy = np.append(self.validy, self.validy, axis = 0)

        while self.testx.shape[0] < self.batch_size:
            self.testx = np.append(self.testx, self.testx, axis = 0)
            self.testy = np.append(self.testy, self.testy, axis = 0)


        self.trainy_lab = np.eye(self.num_labelled_classes + 1)[self.trainy_lab]
        self.trainy = np.eye(self.num_labelled_classes + 1)[self.trainy]
        self.testy = np.eye(self.num_labelled_classes + 1)[self.testy]
        self.trainy_unlab = np.eye(self.num_labelled_classes + 1)[self.trainy_unlab]
        self.validy = np.eye(self.num_labelled_classes + 1)[self.validy]

        self.count_real = 0
        self.count_lab = 0
        self.count_unlab = 0

        self.trainx_lab2 = self.trainx_lab
        self.count_lab2 = 0

 
        '''
        Open set: Fashion MNIST
        TO DO: Include open sets functionally
        '''

        fashion = tf.keras.datasets.fashion_mnist
        (trainx, trainy),(testx, testy) = fashion.load_data()
        fashion_trainx, fashion_testx = trainx / 255.0, testx / 255.0

        openx = np.zeros((fashion_trainx.shape[0], 32, 32, 1))
        openy = np.zeros((fashion_trainx.shape[0]), dtype=np.int32)
        temp_count = 0
        for i in range(fashion_trainx.shape[0]):
            temp = np.pad(fashion_trainx[i], (2 , 2), 'constant')
            openx[temp_count] = np.reshape(temp, (32, 32, 1))
            openy[temp_count] = self.num_labelled_classes
            temp_count += 1

        temp = self.rng_data.permutation(openx.shape[0])
        self.openx = openx[temp]
        self.openy = openy[temp]
        self.openy = np.eye(self.num_labelled_classes + 1)[self.openy]
        del fashion
        del trainx
        del trainy
        del testx
        del testy
        del fashion_trainx
        del fashion_testx


    def next_batch_fake(self, z_size, size = 0):
        if size == 0:
            return np.random.rand(self.batch_size, z_size)
        else:
            return np.random.rand(size, z_size)

    def next_batch_real(self):
        if self.count_real + self.batch_size >= self.trainx.shape[0]:
            self.count_real = 0
            temp = self.rng_data.permutation(self.trainx.shape[0])
            self.trainx = self.trainx[temp]
            self.trainy = self.trainy[temp]

        images = self.trainx[self.count_real: self.count_real + self.batch_size]
        labels = self.trainy[self.count_real: self.count_real + self.batch_size]
        self.count_real += self.batch_size

        return images, labels

    def next_batch_labelled(self):
        if self.count_lab + self.batch_size >= self.trainx_lab.shape[0]:
            self.count_lab = 0
            temp = self.rng_data.permutation(self.trainx_lab.shape[0])
            self.trainx_lab = self.trainx_lab[temp]
            self.trainy_lab = self.trainy_lab[temp]

        images = self.trainx_lab[self.count_lab: self.count_lab + self.batch_size ]
        labels = self.trainy_lab[self.count_lab: self.count_lab + self.batch_size ]
        self.count_lab += self.batch_size 

        return images, labels

    def next_batch_unlab(self):
        if self.count_unlab + (self.batch_size) >= self.trainx_unlab.shape[0]:
            self.count_unlab = 0
            temp = self.rng_data.permutation(self.trainx_unlab.shape[0])
            self.trainx_unlab = self.trainx_unlab[temp]
            self.trainy_unlab = self.trainy_unlab[temp]

        images_unlab = self.trainx_unlab[self.count_unlab: self.count_unlab + (self.batch_size)]
        labels_unlab = self.trainy_unlab[self.count_unlab: self.count_unlab + (self.batch_size)]
        self.count_unlab += (self.batch_size)

        temp = self.rng_data.permutation(images_unlab.shape[0])
        images = images_unlab[temp]
        labels = labels_unlab[temp]

        del temp
        return images, labels


