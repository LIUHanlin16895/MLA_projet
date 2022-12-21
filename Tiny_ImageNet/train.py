
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import random
import scipy.ndimage
from matplotlib.pyplot import imread
from input_pipe import *
import numpy as np

train_image,train_label = data('train')
print(np.shape(train_image))
print(np.shape(train_label))

#%%
val_image,val_label = data('val')
print(np.shape(val_image))
print(np.shape(val_label))

#%%
train_cut = train_image[0:20000]  # 4 premières classes 4*500
t_cut_lab = train_label[0:20000]
val_cut = val_image[0:100]
v_cut_lab = val_label[0:100]

train_cut=np.array(train_cut)
t_cut_lab=np.array(t_cut_lab)
val_cut=np.array(val_cut)
v_cut_lab=np.array(v_cut_lab)

train_cut = (train_cut - 127.5)/127.5
val_cut = (val_cut - 127.5)/127.5 


train_cut = tf.cast(train_cut, tf.float32)
val_cut = tf.cast(val_cut, tf.float32)
t_cut_lab = tf.cast(t_cut_lab, tf.int64)
v_cut_lab = tf.cast(v_cut_lab, tf.int64)


#%%
dataset = tf.data.Dataset.from_tensor_slices((train_cut, t_cut_lab)).shuffle(20000).batch(128)
val_dataset = tf.data.Dataset.from_tensor_slices((val_cut, v_cut_lab)).batch(128)


#%% GoogLeNet
from GoogLeNet import *
model = GoogLeNet(train_cut)
model.summary()

model.compile(optimizer='adam', loss=[losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy], loss_weights=[1, 0.3, 0.3], metrics=['accuracy'])
history = model.fit(train_cut, [t_cut_lab, t_cut_lab, t_cut_lab], validation_data=(val_cut, [v_cut_lab, v_cut_lab, v_cut_lab]), batch_size=128, epochs=2)

predict_lab = model.predict(val_cut)
score = model.evaluate(val_cut,v_cut_lab,verbose=0)


#%% CNN
from tensorflow.keras.layers import Dense , Input , Dropout, Multiply, Layer, Conv2D, Activation, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential

num_classes = 1
model_CNN = Sequential()
model_CNN.add(Conv2D(32, (3, 3), padding='same',  
                 input_shape=train_cut.shape[1:]))  
model_CNN.add(Activation('relu'))
model_CNN.add(Conv2D(32, (3, 3)))
model_CNN.add(Activation('relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Conv2D(64, (3, 3), padding='same'))
model_CNN.add(Activation('relu'))
model_CNN.add(Conv2D(64, (3, 3)))
model_CNN.add(Activation('relu'))
model_CNN.add(MaxPooling2D(pool_size=(2, 2)))
model_CNN.add(Dropout(0.25))

model_CNN.add(Flatten())
model_CNN.add(Dense(512))
model_CNN.add(Activation('relu'))
model_CNN.add(Dropout(0.5))
model_CNN.add(Dense(200,activation='softmax'))
model_CNN.summary()
opt = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
model_CNN.compile(loss=loss_func,
              optimizer=opt,
              metrics=['acc'])


history = model_CNN.fit(train_cut,t_cut_lab, validation_data=(val_cut, v_cut_lab), batch_size=128, epochs=10)



#%% 

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
        
test = "F:\\SAR M2\\Machine learning\\jigu\\tiny-imagenet-200\\train\\n01443537\\images\\n01443537_63.JPEG"
file = tf.io.read_file(test)#,encoding='latin-1')
img = tf.image.decode_jpeg(file, channels=3)
img = tf.image.random_crop(img, np.array([56, 56, 3]))
img = tf.image.random_flip_left_right(img)
test_img = img.reshape(1,56,56,3)

predict_lab = model_CNN(test_img)
print(np.argmax(predict_lab))

#%% FGSM génerer l'exemple contradictoire
print(test_img)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, pred)
  gradient = tape.gradient(loss, input_image)
  # Utiliser la fonction signe sur le gradient pour créer une perturbation
  signed_grad = np.sign(gradient)
  return signed_grad

perturbation = create_adversarial_pattern(test_img, train_label[0])

#%%
epsilon = 0.1
test_img_adv = test_image + epsilon*perturbation
pred_adv = model_CNN.predict(test_img_adv)
print(pred_adv)-


#%% plot test_image et l'exemple contradictoire
plt.subplot(121)
plt.imshow(test_image)
plt.subplot(122)
plt.imshow(test_img_adv)
plt.show()




#%%
# #%matplotlib inline
# plt.rcParams['figure.figsize'] = (10, 6)

# def distort(filename):
#     """Apply image distortions"""
#     with tf.Graph().as_default():
#         file = tf.io.read_file(filename)
#         img = tf.image.decode_jpeg(file, 3)
#         img = tf.image.adjust_saturation(img, 0.5)
# #         img = tf.image.adjust_hue(img, -0.05)
#         with tf.compat.v1.Session() as sess:
#             dist_img = sess.run(img)
    
#     return dist_img

# filenames = glob.glob('../tiny-imagenet-200/test\\images\\*.JPEG')
# pick_8 = random.sample(filenames, 8)

# count = 0
# for filename in pick_8:
#     count += 1
#     plt.subplot(4, 4, count)
#     img = imread(filename)
#     plt.imshow(img)
#     plt.axis('off')
#     img_distort = distort(filename)
#     count += 1
#     plt.subplot(4, 4, count)
#     plt.imshow(img_distort)
#     plt.axis('off')

# plt.show()   