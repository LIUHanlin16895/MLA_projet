* The project is still in progress and the completed neural networks will be implemented into .py format files, while the neural networks being experimented with will be experimented with in .jupyter format files.

* to do: 

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

[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) describes a new way of attacking networks: Fast Gradient Sign Method (FGSM).

Researchers have identified a serious security concern with existing neural network models : an attacker can easily fool a neural network by adding specific noise to benign samples, often undetected. The attacker uses perturbations that are not perceptible to human vision/audition, which are sufficient to cause a normally trained model to output false predictions with high confidence, a phenomenon that researchers call adversarial attacks.

Existing adversarial attacks can be classified as white-box, grey-box and black-box attacks based on the threat model. The difference between these three models lies in the information known to the attacker, and the FGSM approach is a white-box attack in which the threat model assumes that the attacker has complete knowledge of his target model, including the model architecture and parameters. The attacker can therefore create an adversarial sample directly on the target model by any means. The attacker can therefore create an adversarial sample directly on the target model by any means

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
  signed_grad = tf.sign(gradient)
  
  return signed_grad
  
```
tape.gradient(loss, input_image) does the deriviation computation.

## Adversarial examples and different neural networks

In the following network, we will first use the FGSM method to generate adversarial examples and observe the impact of adversarial examples on the neural network.  In the article, the authors propose adversarial training methods based on reminders from linear models as well as deep networks, respectively, and we will implement each of these two defences into the following networks and observe their effects.

  * Linear Classification Network
  
  1. Logistic regression (Experiment in progress)
  2. Softmax neural network (Experiment in progress)
    
  In this network, we will...
    
  * Deep Neural Network
  
  1. GoogLeNet (Experiment in progress)
  2. Maxout network (Complete)
  3. CNN ([notebook-CNN-DNN](src)) (Experiment in progress)
    
  In this network, we will...


## Reference

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Attacking Machine Learning with Adversarial Examples](https://openai.com/blog/adversarial-example-research/)
