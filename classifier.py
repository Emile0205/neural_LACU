import tensorflow as tf
import numpy as np
import gc

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from miscelaneous import *
from utils import count_classes
############################################

class run:
    def __init__(self, count, discriminator, 
                 dataset, steps = 1000, 
                 load = False, main_path = ""):
        '''
        
        '''            
        
        self.count = count
        self.max_steps = steps
        self.current_step = 0
        self.dataset = dataset
        self.num_lab_classes = self.dataset.num_lab_classes
        self.main_path = main_path 
                
        self.discriminator = discriminator(print_summary = True)  
               
        self.cat_loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)         
        self.bin_loss = tf.keras.losses.BinaryCrossentropy(from_logits = True)
                                                  
        lab_label = np.zeros((self.dataset.batch_size, 1))
        self.lab_label = tf.convert_to_tensor(lab_label, dtype = tf.float32)         
        
        unlab_label = np.ones((self.dataset.batch_size, 1))
        self.unlab_label = tf.convert_to_tensor(unlab_label, dtype = tf.float32)              
                                  
        del lab_label
        del unlab_label

        
      
    def call(self):
        '''
        
        '''
        while self.current_step < self.max_steps:
            self.one_step()
            if self.current_step % 100 == 0:
                 print(self.current_step)
            self.current_step += 1 
                                                                                                      
        return True
   


    def one_step(self):       
        '''
        Generate samples.
        Samples that get classified into a labelled class must NOT be used to train the unknown class
        '''    
        lr = 0.0002
                             
        data_lab, data_lab_labels = self.dataset.next_batch_labelled()
        data_lab = tf.convert_to_tensor(data_lab, dtype = tf.float32)
        data_lab_labels = tf.convert_to_tensor(data_lab_labels, dtype = tf.float32)               
        data_unlab = self.dataset.next_batch_unlab()    
        data_unlab = tf.convert_to_tensor(data_unlab, dtype = tf.float32)  
        
        with tf.GradientTape() as tape:    
            tape.watch(data_lab)               
            d_loss = self.bin_loss(self.lab_label, self.discriminator(data_lab)[:, self.num_lab_classes])
            
            tape.watch(data_unlab)            
            d_loss += self.bin_loss(self.unlab_label, self.discriminator(data_unlab)[:, self.num_lab_classes])
            
            tape.watch(data_lab)  
            d_loss += self.cat_loss(data_lab_labels, self.discriminator(data_lab))
                
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)     
        d_grads_norm = tf.linalg.global_norm(d_grads)        
        Adam(lr, 0.5).apply_gradients(zip(d_grads, self.discriminator.trainable_weights))  

        del data_lab, data_lab_labels
        del data_unlab
        del tape
        del d_loss
        del d_grads
        del d_grads_norm
        gc.collect()                
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                 
        
        

   
         



     




