# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:49:10 2017

@author: sm186047
"""

import numpy as np
import cv2
import gc


class DataGenerator(object):
    """Generates data for Keras"""
    def __init__(self, dim_x, dim_y, batch_size, shuffle, n_classes):
        # Initialization
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes

    def generate(self, labels, list_IDs, n_classes):
        """Generates batches of samples"""
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                #print("Producing")
                #print(list_IDs_temp)
                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp, n_classes)
                # print(X.shape)
                # print(y.shape)
                #print("Target Label")
                #print(y)
                gc.collect()
                yield X, y


    def __get_exploration_order(self, list_IDs):
        # Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, labels, list_IDs_temp, n_classes):
        # Generates data of batch_size samples
        #  X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, 3))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            img = cv2.imread('./data/train/'+ID + '.jpg')
            X[i, :, :] = cv2.resize(img, (self.dim_y, self.dim_x))
            X = X.astype('float32') / 255
          
            # Store class
            y[i] = labels[ID]

        return X, sparsify(y, n_classes)


def sparsify(y, n_classes):
    # Returns labels in binary NumPy array
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
