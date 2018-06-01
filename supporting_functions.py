# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:49:43 2018

@author: Hermes
"""

# DeepSnakes
# Supportig functions

import h5py as h5

def snake_data():
    #Load the snakes dataset
    # Extracting the train dataset from the h5 file.
    f = h5.File("./dataset/train_set.hdf5",'r')
    # Storing the original data "permanently".
    images_train_orig = f["images_train"].value
    labels_train_orig = f["labels_train"].value
    f.close()
    # Extracting the dev dataset from the h5 file.
    f = h5.File("./dataset/dev_set.hdf5",'r')
    # Storing the original data "permanently".
    images_dev_orig = f["images_dev"].value
    labels_dev_orig = f["labels_dev"].value
    f.close()
    return images_train_orig, labels_train_orig, images_dev_orig, labels_dev_orig

def reg_reshape_snakes(images_in,labels_in):
    # Regularizing images
    images = images_in/images_in.max()
    # Reshaping/Transposing images
    images = images.reshape((len(images),-1))
    images = images.transpose()
    # Reshaping labels
    labels = labels_in.reshape([1,len(labels_in)])
    return images, labels