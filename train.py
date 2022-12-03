import tensorflow as tf # 2.3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.datasets.mnist as mnist
import time as time
from maxout_network import create_amxout

(train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
# train_image.shape = (60000, 28, 28), train_labels.shape = (60000,)

# Normalisation les données 0 - 255 en -1 - 1
train_image = (train_image - 127.5)/127.5 # 把0-255的数据范围变为-1到1之间
test_image = (test_image - 127.5)/127.5 # 把0-255的数据范围变为-1到1之间

# Augmenter la dimension du canal 增加通道维度
#train_image = tf.expand_dims(train_image, -1)
#test_image = tf.expand_dims(test_image, -1)
# train_image.shape = ([60000, 28, 28, 1]), train_labels.shape = (60000,)

# Transformation de type 类型转换
train_image = tf.cast(train_image, tf.float32)
test_image = tf.cast(test_image, tf.float32)
train_labels = tf.cast(train_labels, tf.int64)
test_labels = tf.cast(test_labels, tf.int64)

# 创建Dataset
batchsize = 128
dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels)).shuffle(60000).batch(batchsize)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels)).batch(batchsize)


optimizer = tf.keras.optimizers.Adam(decay=1e-6)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
def loss_with_FGSM(model,input_image, input_label, epsilon=1, alpha=0.5):
    with tf.GradientTape() as tape:
      tape.watch(input_image)
      prediction = model(input_image)
      loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    # Utiliser la fonction signe sur le gradient pour créer une perturbation对梯度使用sign函数，创建扰动
    signed_grad = tf.sign(gradient)
    adv_img = input_image + epsilon*signed_grad
    adv_img = tf.clip_by_value(adv_img, -1, 1)
    prediction2 = model(adv_img)
    loss2 = loss_object(input_label, prediction2)
    return alpha*loss + (1-alpha) * loss2



def training_loop(model, epochs, optimizer, learning_rate, loss_fn, train_set, val_images, val_labels, nombre_example=60000, batchsize=128):
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
		for element in dataset:
			step = step+1
			start_step = time.time()
			images  = element[0]
			labels =element[1] 
			with tf.GradientTape() as tape:
				loss_value = loss_with_FGSM(model, images, labels, epsilon=0.5, alpha=0.5)
			grads = tape.gradient(loss_value, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
			if step == len-1:
				logits_val = model(val_images, training=True)
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

maxout = create_amxout()
model = training_loop(maxout, epochs=10, optimizer=optimizer, learning_rate=0.001, loss_fn=loss_func, train_set=dataset, val_images=test_image, val_labels=test_labels, nombre_example=60000, batchsize=128)