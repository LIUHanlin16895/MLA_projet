
# MLA_projet
Reproduce the results of the article "Explaining and Harnessing Adversarial Examples". 

"We argue that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature...this view yields a simple and fast method of generating adversarial examples" -Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy

In this project, we will try to apply the FGSM method on various neural networks mentioned in the article and try to get the same data as the results in the article.

## Requirements:

* Python 3
* Colab or Jupyter notebook
* numpy
* tensorflow (tested with versions 1.2 and 1.4)

## Dataset

* Mnist
* ImageNet
## Fast Gradient Sign Method 

[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) paper describes a new way of attacking networks called Fast Gradient Sign Method (FGSM).

Adversarial examples are specific inputs created with the purpose of fooling a neural network. They are formed by applying small perturbations to examples from the dataset, such that the 
perturbed input results in the model outputting an incorrect answer with high confidence. These examples are indistinguishable to the human eye.

FGSM is a white box attack whose goal is to ensure misclassification. A white box attack is where the attacker has complete access to the model being attacked.

The fast gradient sign method works by using the gradients of the neural network to create an adversarial example. 
For an input image, the method uses the gradients of the loss with respect to the input image to create a new image that 
maximizes the loss. This new image is called the adversarial image.

We create adversarial examples by taking the gradient of the network with respect to the input image.
```
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)
  gradient = tape.gradient(loss, input_image)
  # Utiliser la fonction signe sur le gradient pour créer une perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
  
```
tape.gradient(loss, input_image) does the deriviation computation.

## Adversarial examples and different neural networks

In the following network, I will first use the FGSM method to generate adversarial examples and observe the impact of adversarial examples on the neural network. We then continue to train the network using data containing adversarial examples until it is fully resistant to adversarial example attacks.

  **Linear Classification Network**

In this network, we will...

  **Maxout network**

In this network, we will...

  **Deep Neural Network - CNN**  ([notebook-CNN-DNN](src))

In this network, we will...

## Reference

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Attacking Machine Learning with Adversarial Examples](https://openai.com/blog/adversarial-example-research/)
