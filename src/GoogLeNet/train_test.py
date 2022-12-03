from GoogLeNet_model import GoogLeNet
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.datasets.mnist as mnist

import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split

#%%# Import les données et traitement les données
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)

#%%# 前6000个作为train， 200个validation， 1000个test
x_test_temp = x_test
y_test_temp = y_test

x_val = x_train[6000:6200,:,:]
y_val = y_train[6000:6200]

x_train = x_train[:6000,:,:]
y_train = y_train[:6000]

x_test = x_test[:1000,:,:]
y_test = y_test[:1000]

x_test_temp = x_test_temp[:1000,:,:]
y_test_temp = y_test_temp[:1000]

batchsize = 128
#dataset = tf.data.Dataset.from_tensor_slices((x_train, [y_train,y_train,y_train])).shuffle(6000).batch(batchsize)
#test_dataset = tf.data.Dataset.from_tensor_slices((x_test, [y_test,y_test,y_test])).batch(batchsize)
#%%# Train (Model original) 
model = GoogLeNet(x_train)
# model.summary()
# model.compile(optimizer='adam', loss=[losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy], loss_weights=[1, 0.3, 0.3], metrics=['accuracy'])
# history = model.fit(x_train, [y_train, y_train, y_train], validation_data=(x_val, [y_val, y_val, y_val]), batch_size=64, epochs=10)

# #%%# plot accuracy
# fig, axs = plt.subplots(2, 1, figsize=(10,10))

# axs[0].plot(history.history['loss'])
# axs[0].plot(history.history['val_loss'])
# axs[0].title.set_text('Training Loss vs Validation Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].legend(['Train','Val'])

# axs[1].plot(history.history['dense_4_accuracy'])
# axs[1].plot(history.history['val_dense_4_accuracy'])
# axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('Accuracy')
# axs[1].legend(['Train', 'Val'])

#%%#
import time as time
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
def loss_with_FGSM(input_image, input_label, epsilon=1, alpha=0.5):
    with tf.GradientTape() as tape:
      tape.watch(input_image)
      prediction = model(input_image)
      loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    adv_img = input_image + epsilon*signed_grad
    adv_img = tf.clip_by_value(adv_img, -1, 1)
    prediction2 = model(adv_img)
    loss2 = loss_object(input_label, prediction2)
    return alpha*loss + (1-alpha) * loss2

def training_loop(model, epochs, optimizer, learning_rate, loss_fn, x_train, y_train, val_images, val_labels, nombre_example=60000, batchsize=128):
	average_time_step = []
	optimizer.lr = learning_rate
	acc_collect = []
	loss_collect = []
	len = int(np.floor(nombre_example / batchsize))
	total_steps = epochs*len
	for epoch in range(epochs):
		if epoch > 20:
			optimizer.lr = learning_rate/10
		if epoch > 40:
			optimizer.lr = learning_rate/100
		start_epoch = time.time()
		step = 0
		for i,element in enumerate(x_train):
			step = step+1
			start_step = time.time()
			images  = element
			labels = [y_train[i],y_train[i] ,y_train[i] ] 
			with tf.GradientTape() as tape:
				loss_value = loss_with_FGSM(images, labels, epsilon=0.5, alpha=0.5)
			grads = tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			if step == len-1:
				logits_val = model(val_images)
				logits_val = np.argmax(logits_val, axis=-1)
				m = tf.keras.metrics.Accuracy()
				m.update_state(val_labels, logits_val)
				acc_collect.append(m.result().numpy())
				print('epoch = ',epoch,'acc train fin epoch = ',m.result().numpy())
				loss_value_temp = loss_value._copy()
				loss_value_temp = float(loss_value_temp)
				loss_collect.append(loss_value_temp)
				print('loss fin epoch = ',loss_value_temp)
	print()
	return model

optimizer = tf.keras.optimizers.Adam(decay=1e-6)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

model = training_loop(model, epochs=10, optimizer=optimizer, learning_rate=0.001, loss_fn=loss_func, x_train=x_train, y_train=y_train, val_images=x_val, val_labels=[y_val,y_val,y_val], nombre_example=6000, batchsize=128)