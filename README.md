!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Attention !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Afin de modifier les fichiers de github, il faut surtout utilise le dépôt git mais pas modifier les fichiers directement dans GitHub.  

# MLA_projet
Reproduce the results of the article "Explaining and Harnessing Adversarial Examples". 

"We argue that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature...this view yields a simple and fast method of generating adversarial examples" -Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy

In this project, we will try to apply the FGSM method on various neural networks mentioned in the article and try to get the same data as the results in the article.

## Requirements:

* Python 3
* Colab or Jupyter notebook
* numpy
* tensorflow (tested with versions 1.2 and 1.4)

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
