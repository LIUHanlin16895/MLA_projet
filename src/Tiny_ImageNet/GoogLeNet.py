import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
#from keras.layers import concatenate


def Inception(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
    # Input: 
    # - f1: number of filters of the 1x1 convolutional layer in the first path
    # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
    # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
    # - f4: number of filters of the 1x1 convolutional layer in the fourth path
      
    # 1st path:
    path1 = layers.Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
      
    # 2nd path
    path2 = layers.Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
    path2 = layers.Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)
      
    # 3rd path
    path3 = layers.Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
    path3 = layers.Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)
      
    # 4th path
    path4 = layers.MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
    path4 = layers.Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)
      
    output_layer = layers.concatenate([path1, path2, path3, path4], axis = 3)
      
    return output_layer

def GoogLeNet(x_train): # (32,32,3)
    # input layer 
    input_layer = layers.Input(shape = (56,56,3))
    input_tensor = layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:])(input_layer)
    # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
    X = layers.Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'same', activation = 'relu')(input_tensor)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = layers.MaxPooling2D(pool_size = (3,3), strides = 2)(X)
    
    # convolutional layer: filters = 64, strides = 1
    X = layers.Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
    
    # convolutional layer: filters = 192, kernel_size = (3,3)
    X = layers.Conv2D(filters = 192, kernel_size = (3,3), strides = 1, padding = 'same', activation = 'relu')(X)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = layers.MaxPooling2D(pool_size= (3,3), strides = 2)(X)
    
    # 1st Inception block 3a
    X = Inception(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
    
    # 2nd Inception block 3b
    X = Inception(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = layers.MaxPooling2D(pool_size= (3,3), strides = 2)(X)
    
    # 3rd Inception block  4a
    X = Inception(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)
    
    # Extra network 1:
    X1 = layers.AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X1 = layers.Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
    X1 = layers.Flatten()(X1)
    X1 = layers.Dense(1024, activation = 'relu')(X1)
    X1 = layers.Dropout(0.7)(X1)
    X1 = layers.Dense(200, activation = 'softmax')(X1)
    
    
    # 4th Inception block  4b
    X = Inception(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    
    # 5th Inception block  4c
    X = Inception(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    
    # 6th Inception block  4d
    X = Inception(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
    
    # Extra network 2:
    X2 = layers.AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X2 = layers.Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
    X2 = layers.Flatten()(X2)
    X2 = layers.Dense(1024, activation = 'relu')(X2)
    X2 = layers.Dropout(0.7)(X2)
    X2 = layers.Dense(200, activation = 'softmax')(X2)
    
    
    # 7th Inception block   4e
    X = Inception(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
    
    # max-pooling layer: pool_size = (3,3), strides = 2
    X = layers.MaxPooling2D(pool_size = (3,3), strides = 2)(X)
    
    # 8th Inception block   5a
    X = Inception(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
     
    # 9th Inception block   5b
    X = Inception(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)
    
    # Global Average pooling layer 
    X = layers.GlobalAveragePooling2D(name = 'GAPL')(X)
    
    # Dropoutlayer 
    X = layers.Dropout(0.4)(X)
    
    # output layer 
    X = layers.Dense(200, activation = 'softmax')(X)
    
    # model
    model = Model(input_layer, [X, X1, X2], name = 'GoogLeNet')
    
    return model
