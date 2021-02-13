import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle
import json
import os
from os import listdir
from matplotlib import image
from skimage.transform import resize

class Inceptionv3:
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
        
    def inception_module(self,x, f1, f2, f3):
        # 1x1 conv
        conv1 =  keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)
        # 3x3 conv
        conv3 = keras.layers.Conv2D(f2, (3,3), padding='same', activation='relu')(x)
        # 5x5 conv
        conv5 = keras.layers.Conv2D(f3, (5,5), padding='same', activation='relu')(x)
        # 3x3 max pooling
        pool = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
        # concatenate filters
        out = keras.layers.merge.concatenate([conv1, conv3, conv5, pool])
        return out


    def conv2d_bn(self,x,filters,num_row,num_col,padding='same',strides=(1, 1)):
   
        x = keras.layers.Conv2D(filters, (num_row, num_col),strides=strides,padding=padding)(x)
        x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
        x = keras.layers.Activation('relu')(x)
        return x


    def inc_block_a(self,x):    
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)  # 64 filters of 1*1

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)  #48 filters of 1*1
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)
        return x

    def reduction_block_a(self,x):  
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],axis=3)
        return x


    # 17 x 17 x 768
    def inc_block_b(self,x):
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1),padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)
        return x



    # mixed 8: 8 x 8 x 1280
    def reduction_block_b(self,x): 
        branch3x3 = self.conv2d_bn(x, 192, 1, 1)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3,strides=(2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self.conv2d_bn( branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3)
        return x



    def inc_block_c(self,x):        
        branch1x1 = self.conv2d_bn(x, 320, 1, 1)

        branch3x3 = self.conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2],axis=3)

        branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate( [branch1x1, branch3x3, branch3x3dbl, branch_pool],axis=3)
        return x


    def inceptionv3_model(self):
        # input image size: 299 x 299 x 3

        img_input = keras.Input(shape=(299, 299, 3))  #shape=(None, 299, 299, 3)
        classes=5
        x = self.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid') # 149 x 149 x 32
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid')  # 147 x 147 x 32
        x = self.conv2d_bn(x, 64, 3, 3) # 147 x 147 x 64

        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)   # 73  x 73 x 64
        x = self.conv2d_bn(x, 80, 1, 1, padding='valid') # 73 x 73 x 80
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid')  # 71 x 71 x 192
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)  # 35 x 35 x 192


        x=self.inc_block_a(x) #35, 35, 256
        x=self.inc_block_a(x) #35, 35, 256
        x=self.inc_block_a(x) #35, 35, 256

        x=self.reduction_block_a(x) #17, 17, 736

        x=self.inc_block_b(x) #17, 17, 768
        x=self.inc_block_b(x) #17, 17, 768
        x=self.inc_block_b(x) #17, 17, 768
        x=self.inc_block_b(x) #17, 17, 768

        x=self.reduction_block_b(x) #shape=(None, 8, 8, 1280)

        x=self.inc_block_c(x) # shape=(None, 8, 8, 2048) 
        x=self.inc_block_c(x) # shape=(None, 8, 8, 2048) 

        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x) # shape=(None, 2048)

        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x) #shape=(None, 1000) 
        # Create model.
        inputs = img_input
        model =  keras.Model(inputs, x, name='inception_v3')
        return model



    def execute_inceptionv3_model(self):
        model = self.inceptionv3_model()
        model.compile(
    loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
        return model



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
        with open('inceptionv3.json', 'w') as file:
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

