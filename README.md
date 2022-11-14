# Corrupting Labels with Distillation
Code for paper "Label Noise Types and Their Effects on Learning".

4 different types of synthetic label noises are implemented as illustrated in the figure below.

![](/images/noisetypes.png)
*Four types of different label noise types. Starting from left: no noise, uniform noise, class-dependent noise, locally-concentrated noise, feature-dependent noise (generated with proposed algorithm)*

Requirements:
* Tensorflow 1.x
* numpy
* scikit-learn
* seaborn

In order to import pre-generated noisy labels:

```python
import numpy as np

dataset='mnist_fashion'         # alternatives = 'mnist', 'cifar10', 'cifar100'
noise_type='feature-dependent'  # alternatives = 'uniform', 'class-dependent', 'locally-concentrated'
noise_ratio=35                  # alternatives = 5,15,25,35,45,55,65,75,85

y_train_noisy = np.load('./noisylabels/{}/{}_{}_ytrain.npy'.format(dataset,noise_type,noise_ratio))
```

In order to generate noisy labels for different noise ratio than pre-generated noise ratios

```python
import numpy as np
from dataset import get_data
from noise_xy import noise_softmax

dataset='mnist_fashion'         # alternatives = 'mnist', 'cifar10', 'cifar100'
noise_type='feature-dependent'  # alternatives = 'uniform', 'class-dependent', 'locally-concentrated'
noise_ratio = 0.35              # can be any number between 0-1

_dataset = get_data(dataset)
logits = np.load('./noisylabels/{}/logits.npy'.format(dataset))

y_train_noisy, probs = noise_softmax(_dataset.x_train, _dataset.y_train_int(), logits, noise_ratio)
_dataset = get_data(dataset, y_noisy=y_train_noisy)
_dataset.get_stats()
x_train, y_train, x_test, y_test = _dataset.get_data()
```
