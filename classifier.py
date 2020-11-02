import os
import sys
import tensorflow as tf
import numpy as np
from utils import Timer
import random
import matplotlib.pyplot as plt
from miscelaneous import *



class classifier:
    max_summary_images = 10

    def __init__(self, network, dataset, steps = 1000):
        """
        Definition of the Wasserstein GAN with Gradient Penalty (WGAN-GP)

        :param network: neural network which takes a batch of images and outputs a "realness" score for each of them
        :param dataset: dataset which will be reconstructed
        :param z_size: size of the random vector used for generation
        :param steps: number of batch setps
        :param valid_decrease: number of steps the validation accuracy must decrease to stop training
        """

        self.network = network
        self.max_steps = steps
        self.current_step = 0

        self.valid_acc_count = 1
        self.valid_acc_temp = 0

        self.valid_acc_count_sa = 1
        self.valid_acc_temp_sa = 0

        self.lr = tf.compat.v1.placeholder(tf.float32, [], name="lr")
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9)
        self.dataset = dataset
        self.num_labelled_classes = self.dataset.num_labelled_classes
        self.lab_image_pl = tf.compat.v1.placeholder(tf.float32,
                                              [None, self.dataset.img_size_x, self.dataset.img_size_y, self.dataset.img_size_z],
                                              name="lab_image")
        self.unlab_image_pl = tf.compat.v1.placeholder(tf.float32,
                                              [None, self.dataset.img_size_x, self.dataset.img_size_y, self.dataset.img_size_z],
                                              name="unlab_image")
        self.labels_pl = tf.compat.v1.placeholder(tf.int32, [None, self.num_labelled_classes + 1], name='lbl_pl_K')

        self.train_acc_pl = tf.compat.v1.placeholder(tf.float32, [], name="train_acc")
        self.train_acc_unlab_pl = tf.compat.v1.placeholder(tf.float32, [], name="train_acc_unlab")
        self.train_acc_lab_pl = tf.compat.v1.placeholder(tf.float32, [], name="train_acc_lab")
        self.test_acc_pl = tf.compat.v1.placeholder(tf.float32, [], name="test_acc")
        self.valid_acc_pl = tf.compat.v1.placeholder(tf.float32, [], name="valid_acc")
        self.open_acc_pl = tf.compat.v1.placeholder(tf.float32, [], name="open_acc")
        self.test_f1_pl = tf.compat.v1.placeholder(tf.float32, [], name="test_f1")
       
        self.out_lab, _ = self.network(self.lab_image_pl, mini_batch = False)
        self.out_unlab, _ = self.network(self.unlab_image_pl, reuse = True, mini_batch = False)
        self.out_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_pl,
                                                                              logits=self.out_lab))
        self.out_cost += tf.nn.sigmoid_cross_entropy_with_logits(labels = [0.0 for i in range(self.dataset.batch_size)],
                                                                logits = self.out_lab[:, self.num_labelled_classes])
        self.out_cost += tf.nn.sigmoid_cross_entropy_with_logits(labels = [1.0 for i in range(self.dataset.batch_size)],
                                                                logits = self.out_unlab[:, self.num_labelled_classes])


        self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "classifier")  # weights for the critic
        self.optimizer = self.optimizer.minimize(self.out_cost, var_list=self.variables, name="optimizer")

        self.out_softmax = tf.nn.softmax(self.out_lab)
        self.out_reject_option = self.out_softmax[:, self.num_labelled_classes]
        self.classification = tf.cast(tf.argmax(self.out_softmax[:, 0:self.num_labelled_classes], 1), tf.int32)

        # Defining summaries for tensorflow until the end of the method
        tf.compat.v1.summary.scalar("Train accuracy", self.train_acc_pl)
        tf.compat.v1.summary.scalar("Train accuracy unlabelled", self.train_acc_unlab_pl)
        tf.compat.v1.summary.scalar("Train accuracy labelled", self.train_acc_lab_pl)
        tf.compat.v1.summary.scalar("Test accuracy", self.test_acc_pl)
        tf.compat.v1.summary.scalar("Valid accuracy", self.valid_acc_pl)
        tf.compat.v1.summary.scalar("Open set accuracy", self.open_acc_pl)
        tf.compat.v1.summary.scalar("Test f1 score", self.test_f1_pl)

        self.merged_before = tf.compat.v1.summary.merge_all()

    def call(self, model_path, main_path, fid_bol = False):
        """
        Trains the neural network by calling the .one_step() method "steps" number of times.

        :param batch_size: batch size
        :param model_path: location of the model on the filesystem
        :param main_path: location of the model on the filesystem
        :param fid_bol: True or False to calculate the FID score
        """
        model_path = model_path + '/model'
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        #saver = tf.train.Saver()
        sess = tf.compat.v1.Session(config=config)

        writer_before = tf.compat.v1.summary.FileWriter(model_path, sess.graph)
        tf.compat.v1.global_variables_initializer().run(session=sess)

        while self.current_step < self.max_steps:
            sys.stdout.flush()

            self.one_step(sess, self.current_step)
            if self.current_step % 100 == 0:
                self.add_summary(sess, self.current_step, writer_before)
                print(self.current_step)
            self.current_step += 1  

        sess.close()
        writer_before.close()
        return True
   
 
    def one_step(self, sess, step):
        """
        Performs one batch step of WGAN update, which is actually several different optimizations according to the 
        WGAN training, the semi-supervised training and the additional adversarial binary classification training.

        :param sess: Tensorflow session in which the update will be performed
        :param batch_size: batch size
        :param step: current step
        """
        # Updating variable that depreciates learning rate
        
        if step > 0:
            acc_valid, _, _ = result(self, sess, self.dataset.validx, self.dataset.validy, self.num_labelled_classes)
            if acc_valid > self.valid_acc_temp:
                self.valid_acc_temp = acc_valid
                if not self.valid_acc_count <= 1:
                    self.valid_acc_count = self.valid_acc_count - 1
            elif acc_valid <= self.valid_acc_temp:
                self.valid_acc_count += 1     
             

        data_lab, labels = self.dataset.next_batch_labelled()
        data_unlab, _ = self.dataset.next_batch_unlab()

        ## Smaller = 0.0005 start and 0.0000001 end
        ## Small = 0.0005 start and 0.0000001 end
        ## Large = 0.1 start and 0.000005 end

        lr = 0.0005/self.valid_acc_count 
        #lr = 0.0005
        if lr < 0.0000001:
            self.current_step = self.max_steps

        sess.run([self.optimizer], feed_dict={self.lab_image_pl: data_lab, 
                                              self.labels_pl: labels, 
                                              self.unlab_image_pl: data_unlab,
                                              self.lr: lr}) 

        if step % 100 == 0:
            print(lr)

    def add_summary(self, sess, step, writer_before):
        """
        Adds a summary for the specified step in Tensorboard
        Tries to reconstruct new samples from dataset

        :param writer: Tensorboard writer
        :paths ...
        """

        acc_train, f1_train, _ = result(self, sess, self.dataset.trainx, self.dataset.trainy, self.num_labelled_classes)
        acc_unlab_train, f1_unlab_train, _ = result(self, sess, self.dataset.trainx_unlab, self.dataset.trainy_unlab, self.num_labelled_classes)
        acc_lab_train, f1_lab_train, _ = result(self, sess, self.dataset.trainx_lab, self.dataset.trainy_lab, self.num_labelled_classes)
        acc_test, f1_test, confusion = result(self, sess, self.dataset.testx, self.dataset.testy, self.num_labelled_classes)
        acc_valid, f1_valid, _ = result(self, sess, self.dataset.validx, self.dataset.validy, self.num_labelled_classes)
        acc_open, f1_open, _  = result(self, sess, self.dataset.openx, self.dataset.openy, self.num_labelled_classes)

        feed_dict = {self.train_acc_pl : acc_train,
                     self.train_acc_unlab_pl: acc_unlab_train, self.train_acc_lab_pl: acc_lab_train,
                     self.test_acc_pl: acc_test, self.valid_acc_pl: acc_valid, self.open_acc_pl: acc_open,
                     self.test_f1_pl: f1_test}
        
        #acc_valid = sess.run(self.accuracy, feed_dict={self.real_image_pl: self.dataset.validx, self.k_labels_pl: self.dataset.validy})
        #acc_valid_sa = sess.run(self.accuracy_sa, feed_dict={self.real_image_pl: self.dataset.validx, self.k_labels_pl: self.dataset.validy})

        summary = sess.run(self.merged_before, feed_dict=feed_dict)
        writer_before.add_summary(summary, step)

        print('Accuracy scores - \t\t Train: ' + "%.3f" % acc_train + '\t\t Valid: ' + str("%.3f" % acc_valid) + '\t\t Test: ' + str("%.3f" % acc_test) + '\t\t Open: ' + str("%.3f" % acc_open))
        print('F1 scores - \t\t\t Train: ' + "%.3f" % f1_train + '\t\t Valid: ' + str("%.3f" % f1_valid) + '\t\t Test: ' + str("%.3f" % f1_test) + '\t\t Open: ' + str("%.3f" % f1_open))
        print(confusion)
        print()

   
         



     





