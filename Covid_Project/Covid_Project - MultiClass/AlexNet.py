# -*- coding: utf-8 -*-
"""AlexNet

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eeuFZZf7H3AeS63P-tAGZ_sHcLrbDMcT
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense
from tensorflow import keras
import numpy as np
from os import listdir
from matplotlib import image
from skimage.transform import resize
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
import json
from matplotlib.pyplot import imread
from keras.preprocessing import image

class AlexNet:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.x_train_normalization = None
        self.x_test_normalization = None
        self.y_train_one_hot = None
        self.y_test_one_hot = None
        self.x_shuffled_default = None
        self.y_shuffled_default = None
        
    def execute_alexnet_model(self):
        
        # Creating a Sequential model
        model = Sequential()

        # 1st Convolution Layer
        model.add(Conv2D(filters=96, kernel_size=(11,11), input_shape=(224, 224, 3), strides=(4,4), padding='valid'))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))
        # Max-Pooling
        model.add(MaxPooling2D((3,3), strides=(2,2), padding='valid'))


        # 2nd Convolution Layer
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))
        # Max-Pooling
        model.add(MaxPooling2D((3,3), strides=(2,2), padding='valid'))


        # 3rd Convolution Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same'))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))


        # 4th Convolution Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same'))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))


        # 5th Convolution Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))
        # Max-Pooling
        model.add(MaxPooling2D((3,3), strides=(2,2), padding='valid'))


        # Flattening before passing to the Dense layer
        model.add(Flatten())


        # 1st Dense Layer
        model.add(Dense(4096))
        # Dropout
        model.add(Dropout(0.4))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))


        # 2nd Dense Layer
        model.add(Dense(4096))
        # Dropout
        model.add(Dropout(0.4))
        # Normalization
        model.add(BatchNormalization())
        # Activation Function
        model.add(Activation('relu'))


        # Output softmax Layer
        model.add(Dense(5))
        # Activation Function
        model.add(Activation('softmax'))
        model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
       
        return model
        
    def summary(self,model):
        model.summary()


    def image_processing(self):
        y_train_values, unique = pd.factorize(self.Y_train)
        # print('y_train ', y_train_values, unique)
        y_test_values, unique = pd.factorize(self.Y_test)
#         print('y_test ', y_test_values, unique)

        y_train_one_hot = to_categorical(y_train_values)
        y_test_one_hot = to_categorical(y_test_values)
        
        self.y_train_one_hot = y_train_one_hot
        self.y_test_one_hot = y_test_one_hot


        x_train_normalization = self.X_train / 255.0
        x_test_normalization = self.X_test / 255.0
        
        self.x_train_normalization = x_train_normalization
        self.x_test_normalization = x_test_normalization

        self.x_shuffled_default, self.y_shuffled_default = shuffle(self.x_train_normalization, self.y_train_one_hot)
        

    def fit_model(self,model):
        print(self.x_shuffled_default.shape, self.y_shuffled_default.shape) 
        history = model.fit(self.x_shuffled_default, self.y_shuffled_default, epochs = 30, batch_size = 32)
        with open('alexnet.json', 'w') as file:
                json.dump(history.history, file)

    def evaluate_model(self,model):
        preds = model.evaluate(self.x_test_normalization, self.y_test_one_hot)
        print ("Loss = " + str(preds[0]))
        print ("Test Accuracy = " + str(preds[1]))

    def predict_model(self):
#         img_path = '/content/drive/MyDrive/NORMAL(1576).jpg'
#         img = image.load_img(img_path, target_size=(224, 224,3))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         print('Input image shape:', x.shape)
#         my_image = imread(img_path)
#         imshow(my_image)
#         print(model.predict(x))
        print("called")