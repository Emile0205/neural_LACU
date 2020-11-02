import os
import sys
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from miscelaneous import *

def pre_process_set(total_num_classes, num_lab_classes, num_unlab_classes,
                    set_lab_classes = [], set_unlab_classes = []):
    '''
    Randomly defines the labelled and unlabelled classes used in the set. 

    Variables:
        total_num_classes 		- the total number of classes used in the set
        num_lab_classes 		- determines how many classes are labelled, these are chosen randomly
        num_unlab_classes 		- determines how many classes are unlabelled, chosen randomly the non-labelled classes
        set_lab_classes 		- optional array that explicitly defines the labelled classes, e.g. [0, 5, 7, 9]
        set_unlab_classes 		- optional array that explicitly defines the unlabelled classes

    Return:
        Array containing label classes, e.g. [0, 5, 7, 9]
        Array containing unlabelled classes 
    '''

    # Randomly choose unlabelled classes or set to given classes
    if set_unlab_classes == []:
        unlab_class_labels = [0 for i in range(num_unlab_classes)]
        temp_count = 0
        for i in range(num_unlab_classes):
            temp = random.randint(0, total_num_classes - 1) 
            while temp in unlab_class_labels:
                temp = random.randint(0, total_num_classes - 1)
            unlab_class_labels[temp_count] = temp
            temp_count += 1
            del temp
        del temp_count
    else:
        unlab_class_labels = set_unlab_classes

    #unlab_class_labels = [5, 6, 7, 8, 9]

    print('Unlabelled classes are set to - ' + str(unlab_class_labels)) 


    # Randomly choose the labelled classes from the remaining available classes

    if len(unlab_class_labels) > 0:
        temp = [unlab_class_labels[0] for i in range(num_lab_classes)]
    else:
        temp = [total_num_classes for i in range(num_lab_classes)]

    for i in range(len(temp) - 1, -1, -1):
        if len(unlab_class_labels) > 0:
            num = unlab_class_labels[0] 
        else:
            num = 10

        while num in unlab_class_labels or num == total_num_classes or num in temp:
            num = random.randint(0, total_num_classes)
        
        temp[i] = num

    lab_class_labels = temp
    del temp

    #lab_class_labels = [0, 1, 2, 3, 4]

    print('Labelled classes are set to - ' + str(lab_class_labels))

    if len(lab_class_labels) + len(unlab_class_labels) < total_num_classes:
        print('Please note, according to the number of classes indicated, some classes have been excluded from the set')

    print()
    return lab_class_labels, unlab_class_labels


def create_set(dataset, path, lab_classes, num_lab_samples_per_class,
               unlab_classes, num_unlab_samples_per_class):
    '''
    Creates the training and testing sets from research datasets. Note that the labels are altered so that 
    labelled classes are labelled from 0 - K, and all unlabelled classes have simulated label K + 1.
    e.g. If the labelled classes are [2, 3, 1] and the unlabelled class is [0, 4]
         then this function relabels the samples from [2, 3, 1] => [0, 1, 2] (i.e. class 2 has label 0 etc.)
         and samples from class [0, 4] => [3] (i.e. all unlabelled classes relabelled to K + 1).
         This is to provide ease of accuracy calculations and training procedures. 

    Variables:
        dataset 			- a keyword string representing the research dataset used, e.g. 'MNIST', 'CIFAR10'
        path				- the directory to which the train and test sets are saved
        lab_classes			- the classes in the set that have labels
        num_lab_samples_per_class	- the number of labelled samples per labelled class
        unlab_classes 			- the classes in the set that do not have labels (although the test set does)
        num_unlab_samples_per_class	- the number of unlabelled samples for labelled and unlabelled classes (assuming balance sets)
                                        - set to zero to include all available training samples as unlabelled samples
    '''

    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == 'MNIST':
        data = tf.keras.datasets.mnist
        x = 28
        y = 28
        z = 1
        total_num_classes = 10

    elif dataset == 'CIFAR10':
        data = tf.keras.datasets.cifar10
        x = 32
        y = 32
        z = 3
        total_num_classes = 10

    (trainx, trainy),(testx, testy) = data.load_data()
    trainx, testx = trainx / 255.0, testx / 255.0


    # Train set split into labelled and unlabelled set
    count = count_classes(trainy, total_num_classes, onehot = False)
    print('Original Train set: ' + str(trainx.shape))
    print(count)

    trainx_lab = []
    trainy_lab = []
    temp_lab_count = [0 for i in range(total_num_classes)]
    trainx_unlab = []
    trainy_unlab = []
    temp_unlab_count = [0 for i in range(total_num_classes)]

    for i in range(trainx.shape[0]):
        label = trainy[i]
        sample = trainx[i]
        # Zero padding MNIST 
        # To do: The x,y and z set above should be used to define padding
        if dataset == 'MNIST':
            sample = np.pad(sample, (2 , 2), 'constant')
            sample = np.reshape(sample, (32, 32, 1))

        if label in lab_classes:
            if temp_lab_count[label] < num_lab_samples_per_class:
                trainx_lab.append(sample)
                trainy_lab.append(lab_classes.index(trainy[i]))
                temp_lab_count[label] += 1

            elif num_unlab_samples_per_class == 0:
                trainx_unlab.append(sample)
                trainy_unlab.append(lab_classes.index(trainy[i]))
                temp_unlab_count[label] += 1

            elif temp_unlab_count[label] < num_unlab_samples_per_class:
                trainx_unlab.append(sample)
                trainy_unlab.append(lab_classes.index(trainy[i]))
                temp_unlab_count[label] += 1

        else:
            if num_unlab_samples_per_class == 0:
                trainx_unlab.append(sample)
                trainy_unlab.append(len(lab_classes))
                temp_unlab_count[label] += 1
            elif temp_unlab_count[label] < num_unlab_samples_per_class:
                trainx_unlab.append(sample)
                trainy_unlab.append(len(lab_classes))
                temp_unlab_count[label] += 1


    trainx_lab = np.asarray(trainx_lab)
    trainy_lab = np.asarray(trainy_lab)
    trainx_unlab = np.asarray(trainx_unlab)
    trainy_unlab = np.asarray(trainy_unlab)

    print('Number of samples per class in labelled train set before label change: \t\t' + str(temp_lab_count)) 
    print('Number of samples per class in unlabelled train set before label change: \t' + str(temp_unlab_count)) 
    del temp_lab_count
    del temp_unlab_count
    print('Number of samples per class in labelled set after label change: \t\t' + 
          str(count_classes(trainy_lab, total_num_classes, onehot = False))) 
    print('Number of samples per class in unlabelled set after label change: \t\t' + 
           str(count_classes(trainy_unlab, total_num_classes, onehot = False)))
    print() 
   
    np.savez(path + '/trainx_lab.npz', trainx_lab)
    np.savez(path + '/trainy_lab.npz', trainy_lab)
    np.savez(path + '/trainx_unlab.npz', trainx_unlab)
    np.savez(path + '/trainy_unlab.npz', trainy_unlab)

    # Test set
    count = count_classes(testy, 10, onehot = False)
    print('Original Test set: ' + str(testx.shape))
    print(count)

    testx_ = []
    testy_ = []
    temp_count = [0 for i in range(total_num_classes)]

    for i in range(testx.shape[0]):
        label = testy[i]
        sample = testx[i]
        # Zero padding MNIST 
        # To do: The x,y and z set above should be used to define padding
        if dataset == 'MNIST':
            sample = np.pad(sample, (2 , 2), 'constant')
            sample = np.reshape(sample, (32, 32, 1))

        if label in lab_classes:
            testx_.append(sample)
            testy_.append(lab_classes.index(testy[i]))
            temp_count[label] += 1

        else:
            testx_.append(sample)
            testy_.append(len(lab_classes))
            temp_count[label] += 1

    testx_ = np.asarray(testx_)
    testy_ = np.asarray(testy_)

    print('Number of samples per class in test set before label change: \t\t\t' + str(temp_count)) 
    print('Number of samples per class in test set after label change: \t\t\t' + 
          str(count_classes(testy_, total_num_classes, onehot = False))) 
    del temp_count

    np.savez(path + '/testx_.npz', testx_)
    np.savez(path + '/testy_.npz', testy_)
    print('Finished')



 
        
    
    

           
   





