import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import pandas as pd
import json
from keras.utils import to_categorical
from sklearn.utils import shuffle
from matplotlib.pyplot import imread
from keras.preprocessing import image
from matplotlib.pyplot import imread
from keras.preprocessing import image
import os
from os import listdir
from matplotlib import image
from skimage.transform import resize



class Inceptionv4:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.x_train_normalization = None
        self.x_test_normalization = None
        self.y_train_one_hot = None
        self.y_test_one_hot = None
        self.x_shuffled_default = None
        self.y_shuffled_default = None
    
    
    def load_data(self,path):
    
        train_files = []
        for foldername1 in listdir(path):
            filepath1 = path  + "/" + foldername1
            for filename1 in listdir(filepath1):
                train_files.append(filepath1 + "/" + filename1)

        # Original Dimensions
        image_width = 299
        image_height = 299
        channels = 3

        loaded_images = np.ndarray(shape=(len(train_files), image_height, image_width, channels),dtype=np.float32)
        print(type(loaded_images))
        loaded_class = []
        i = 0
        for foldername in listdir(path):
            filepath = path  + "/" + foldername
            print("Folder : ",filepath)
            for filename in listdir(filepath):
                # load image
                img_data = image.imread(filepath + "/" + filename)

                # store loaded image
                img_data = resize(img_data, (299, 299, 3))
                loaded_images[i] = img_data
                loaded_class.append(foldername)
                i = i + 1
                #print('> loaded %s %s' % (filename, img_data.shape))
            print('Loaded: ',i , ' images from ',filepath)

        return loaded_images,loaded_class
    
    def split_data(self,x,y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)
        print(X_train.shape)
        print(X_test.shape)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        

    def conv_block(self,x, nb_filter, nb_row, nb_col, padding = "same", strides = (1, 1), use_bias = False):
        '''Defining a Convolution block that will be used throughout the network.'''
    
        x = Conv2D(nb_filter, (nb_row, nb_col), strides = strides, padding = padding, use_bias = use_bias)(x)
        x = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(x)
        x = Activation("relu")(x)

        return x


    def stem(self,input):
        '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''
    
        # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
        x = self.conv_block(input, 32, 3, 3, strides = (2, 2), padding = "same") # 149 * 149 * 32
        x = self.conv_block(x, 32, 3, 3, padding = "same") # 147 * 147 * 32
        x = self.conv_block(x, 64, 3, 3) # 147 * 147 * 64

        x1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)
        x2 = self.conv_block(x, 96, 3, 3, strides = (2, 2), padding = "same")

        x = concatenate([x1, x2], axis = -1) # 73 * 73 * 160

        x1 = self.conv_block(x, 64, 1, 1)
        x1 = self.conv_block(x1, 96, 3, 3, padding = "same")

        x2 = self.conv_block(x, 64, 1, 1)
        x2 = self.conv_block(x2, 64, 1, 7)
        x2 = self.conv_block(x2, 64, 7, 1)
        x2 = self.conv_block(x2, 96, 3, 3, padding = "same")

        x = concatenate([x1, x2], axis = -1) # 71 * 71 * 192

        x1 = self.conv_block(x, 192, 3, 3, strides = (2, 2), padding = "same")

        x2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)

        x = concatenate([x1, x2], axis = -1) # 35 * 35 * 384

        return x

    def inception_A(self,input):
        '''Architecture of Inception_A block which is a 35 * 35 grid module.'''

        a1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
        a1 = self.conv_block(a1, 96, 1, 1)

        a2 = self.conv_block(input, 96, 1, 1)

        a3 = self.conv_block(input, 64, 1, 1)
        a3 = self.conv_block(a3, 96, 3, 3)

        a4 = self.conv_block(input, 64, 1, 1)
        a4 = self.conv_block(a4, 96, 3, 3)
        a4 = self.conv_block(a4, 96, 3, 3)

        merged = concatenate([a1, a2, a3, a4], axis = -1)

        return merged

    def inception_B(self,input):
        '''Architecture of Inception_B block which is a 17 * 17 grid module.'''
    
        b1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
        b1 = self.conv_block(b1, 128, 1, 1)

        b2 = self.conv_block(input, 384, 1, 1)

        b3 = self.conv_block(input, 192, 1, 1)
        b3 = self.conv_block(b3, 224, 1, 7)
        b3 = self.conv_block(b3, 256, 7, 1)

        b4 = self.conv_block(input, 192, 1, 1)
        b4 = self.conv_block(b4, 192, 7, 1)
        b4 = self.conv_block(b4, 224, 1, 7)
        b4 = self.conv_block(b4, 224, 7, 1)
        b4 = self.conv_block(b4, 256, 1, 7)

        merged = concatenate([b1, b2, b3, b4], axis = -1)

        return merged


    def inception_C(self,input):
        '''Architecture of Inception_C block which is a 8 * 8 grid module.'''

        c1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
        c1 = self.conv_block(c1, 256, 1, 1)

        c2 = self.conv_block(input, 256, 1, 1)

        c3 = self.conv_block(input, 384, 1, 1)
        c31 = self.conv_block(c2, 256, 1, 3)
        c32 = self.conv_block(c2, 256, 3, 1)
        c3 = concatenate([c31, c32], axis = -1)

        c4 = self.conv_block(input, 384, 1, 1)
        c4 = self.conv_block(c3, 448, 3, 1)
        c4 = self.conv_block(c3, 512, 1, 3)
        c41 = self.conv_block(c3, 256, 1, 3)
        c42 = self.conv_block(c3, 256, 3, 1)
        c4 = concatenate([c41, c42], axis = -1)

        merged = concatenate([c1, c2, c3, c4], axis = -1)

        return merged


    def reduction_A(self,input, k = 192, l = 224, m = 256, n = 384):
        '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

        ra1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)

        ra2 = self.conv_block(input, n, 3, 3, strides = (2, 2), padding = "same")

        ra3 = self.conv_block(input, k, 1, 1)
        ra3 = self.conv_block(ra3, l, 3, 3)
        ra3 = self.conv_block(ra3, m, 3, 3, strides = (2, 2), padding = "same")

        merged = concatenate([ra1, ra2, ra3], axis = -1)

        return merged


    def reduction_B(self,input):
        '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''

        rb1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)

        rb2 = self.conv_block(input, 192, 1, 1)
        rb2 = self.conv_block(rb2, 192, 3, 3, strides = (2, 2), padding = "same")

        rb3 = self.conv_block(input, 256, 1, 1)
        rb3 = self.conv_block(rb3, 256, 1, 7)
        rb3 = self.conv_block(rb3, 320, 7, 1)
        rb3 = self.conv_block(rb3, 320, 3, 3, strides = (2, 2), padding = "same")

        merged = concatenate([rb1, rb2, rb3], axis = -1)

        return merged


    def inception_v4(self):
        '''Creates the Inception_v4 network.'''

        init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering

        # Input shape is 299 * 299 * 3
        x = self.stem(init) # Output: 35 * 35 * 384

        # 4 x Inception A
        for i in range(4):
            x = self.inception_A(x)
            # Output: 35 * 35 * 384

        # Reduction A
        x = self.reduction_A(x, k = 192, l = 224, m = 256, n = 384) # Output: 17 * 17 * 1024

        # 7 x Inception B
        for i in range(7):
            x = self.inception_B(x)
            # Output: 17 * 17 * 1024

        # Reduction B
        x = self.reduction_B(x) # Output: 8 * 8 * 1536

        # 3 x Inception C
        for i in range(3):
            x = self.inception_C(x) 
            # Output: 8 * 8 * 1536

        # Average Pooling
        x = AveragePooling2D((8, 8))(x) # Output: 1536

        # Dropout
        x = Dropout(0.2)(x) # Keep dropout 0.2 as mentioned in the paper
        x = Flatten()(x) # Output: 1536

        # Output layer
        output = Dense(units = 5, activation = "softmax")(x) # Output: 1000
        model = Model(init, output, name = "Inception-v4")
        return model

    

    def execute_inceptionv4_model(self):
        inception_v4 = self.inception_v4()
        inception_v4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return inception_v4


    def summary(self,model):
        model.summary()



    def image_processing(self):
        y_train_values, unique = pd.factorize(self.Y_train)
        # print('y_train ', y_train_values, unique)
        y_test_values, unique = pd.factorize(self.Y_test)
        print('y_test ', y_test_values, unique)

        self.y_train_one_hot = to_categorical(y_train_values)
        self.y_test_one_hot = to_categorical(y_test_values)

        self.x_train_normalization = self.X_train / 255.0
        self.x_test_normalization = self.X_test / 255.0

        self.x_shuffled_default, self.y_shuffled_default = shuffle(self.x_train_normalization, self.y_train_one_hot)



    def fit_model(self,model):
        history = model.fit(self.x_shuffled_default, self.y_shuffled_default, epochs = 30, batch_size = 32)
        with open('inceptionv4.json', 'w') as file:
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





