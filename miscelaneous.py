import os
import sys
import tensorflow as tf
import numpy as np
from utils import Timer
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def count_classes(labels_, num_classes = 10, class_labels = [], onehot = True):
    '''
    Counts the number of samples per class (onehot encoded or not)

    Return:
        Array containing number of samples for the class represented by the position of the array
        e.g. [100, 50, 20, 10] has 100 samples for class 0, 50 samples for class 1, etc.
    '''
    return_count = [0 for i in range(num_classes)]
    for i in range(labels_.shape[0]):
        if onehot:
            label = int(np.argmax(labels_[i], axis = 0))
        else:
            label = int(labels_[i]) 
        if len(class_labels) == 0 and num_classes > 0:
            return_count[label] += 1
        elif (label in class_labels):
            return_count[class_labels.index(label)] += 1
    return return_count

def result(WGAN, sess, data, labels, num_labelled_classes, sa = False):
    prediction, option = sess.run([WGAN.classification, WGAN.out_reject_option],
                                   feed_dict={WGAN.lab_image_pl: data})

    option_lab = sess.run(WGAN.out_reject_option,
                          feed_dict={WGAN.lab_image_pl: WGAN.dataset.trainx_lab})

    for i in range(num_labelled_classes):
        temp = []
        for m in range(option_lab.shape[0]):
            if np.argmax(WGAN.dataset.trainy_lab[m]) == i:
                temp.append(option_lab[m])
        sorted_ = sorted(temp)
        min_ = min(sorted_[int(0.00 * len(sorted_)): len(sorted_)])
        max_ = max(sorted_[0: int(0.99* len(sorted_))])

        for m in range(prediction.shape[0]):
            if prediction[m] == i:
                if option[m] >= max_:
                    prediction[m] = num_labelled_classes 

    labels_temp = []
    for i in range(labels.shape[0]):
        labels_temp.append(np.argmax(labels[i]))

    labels_temp = np.asarray(labels_temp)

    accuracy = accuracy_score(labels_temp, prediction) 
    f1 = f1_score(labels_temp, prediction, average='macro')
    confusion = confusion_matrix(labels_temp, prediction)

    return accuracy, f1, confusion



