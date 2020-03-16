import os, sys
import gzip
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from visualizer import plot_matrix
from util import normalize, create_folders, DATASETS, RANDOM_SEED

import urllib

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve

MNIST_RESOURCES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))

def download(destination_path, urlbase, resources):
    for resource in resources:
        path = os.path.join(os.getcwd(),'{}{}'.format(destination_path,resource))
        url = '{}{}'.format(urlbase,resource)
        if not os.path.exists(path):
            print('Downloading {} ...'.format(url))
            try:
                hook = report_download_progress
                urlretrieve(url, path, reporthook=hook)
            except URLError:
                raise RuntimeError('Error downloading resource!')
        #else:
        #    print('{} already exists, skipping ...'.format(path))

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

class DatasetCls:
    def __init__(self, x_train, y_train, x_test, y_test, y_noisy = None, y_noisy_test = None,
                num_classes = None, dataset_name=None, class_names=None):

        self.x_train = x_train.astype('float32')
        self.y_train = y_train
        self.x_test = x_test.astype('float32')
        self.y_test = y_test
        self.y_noisy = y_noisy if y_noisy is not None else np.copy(y_train)
        self.y_noisy_test = y_noisy_test if y_noisy_test is not None else np.copy(y_test)
        self.num_classes = num_classes
        self.name = dataset_name

        self.num_train_samples = self.y_train.shape[0]
        self.num_test_samples = self.y_test.shape[0]
        self.num_samples = self.num_train_samples + self.num_test_samples
        self.img_shape = x_train.shape[1:]
        self.mean = self.x_train.mean(axis=0)
        self.std = np.std(self.x_train)

        self.idx = np.arange(self.y_train.shape[0])
        self.idx_clean = np.nonzero(self.y_noisy == self.y_train)[0]
        self.idx_noisy = np.nonzero(self.y_noisy != self.y_train)[0]


        self.num_noisy_samples = len(self.idx_noisy)
        self.num_clean_samples = self.num_train_samples - self.num_noisy_samples
        self.noise_ratio = (self.num_noisy_samples*100/self.num_train_samples)

        self._set_num_classes()

        self.int_to_onehot()

        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_namesclass_names = range(self.num_classes)

    def get_data(self):
        return self.x_train, self.y_noisy, self.x_test, self.y_test

    def get_stats(self):
        assert(self.y_train.shape == self.y_noisy.shape), "Shapes should be equavelent!"

        print('Number of training samples: ', self.num_train_samples)
        print('Number of test samples: ', self.num_test_samples)

        str_class_nums = 'Number of samples per class: '
        for i in range(self.num_classes):
            str_class_nums += str(self.class_names[i])+'('+str(np.sum(self.y_train_int() == i))+')'
        print(str_class_nums)
        
        print('Number of all samples: ', self.num_samples)
        print('Number of noisy samples: ', self.num_noisy_samples)
        print('Ratio of noise: ', self.noise_ratio)
        print('Image shape: ', self.img_shape)

    def get_cm_train(self):
        return confusion_matrix(self.y_train_int(),self.y_noisy_int())

    def get_cm_test(self):
        return confusion_matrix(self.y_test_int(),self.y_noisy_test())

    def get_percentage(self, percentage):
        if percentage < 1 and percentage > 0:
            _, x_clean, _, y_clean = train_test_split(self.x_train[self.idx_clean], self.y_train[self.idx_clean], test_size=percentage, random_state=RANDOM_SEED)
            _, x_noisy, _, y_noisy = train_test_split(self.x_train[self.idx_noisy], self.y_noisy[self.idx_noisy], test_size=percentage, random_state=RANDOM_SEED)
            _, x_noisy_clean, _, y_noisy_clean = train_test_split(self.x_train[self.idx_noisy], self.y_train[self.idx_noisy], test_size=percentage, random_state=RANDOM_SEED)

            assert np.array_equal(x_noisy, x_noisy_clean)

            self.x_train = np.concatenate((x_clean,x_noisy))
            self.y_train = np.concatenate((y_clean,y_noisy_clean))
            self.y_noisy = np.concatenate((y_clean,y_noisy))

            self.x_train, self.y_train, self.y_noisy = shuffle(self.x_train, self.y_train, self.y_noisy, random_state=RANDOM_SEED)

            y_train_int = np.array([np.where(r==1)[0][0] for r in self.y_train])
            y_noisy_int = np.array([np.where(r==1)[0][0] for r in self.y_noisy])

            self.idx = np.arange(self.y_train.shape[0])
            self.idx_clean = np.nonzero(y_noisy_int == y_train_int)[0]
            self.idx_noisy = np.nonzero(y_noisy_int != y_train_int)[0]

            self.num_train_samples = self.y_train.shape[0]
            self.num_test_samples = self.y_test.shape[0]
            self.num_samples = self.num_train_samples + self.num_test_samples
            self.num_noisy_samples = len(self.idx_noisy)
            self.num_clean_samples = self.num_train_samples - self.num_noisy_samples
            self.mean = self.x_train.mean(axis=0)
            self.std = np.std(self.x_train)
            self.noise_ratio = (self.num_noisy_samples*100/self.num_train_samples)

    def flow_train(self,batch_size):
        datagen = ImageDataGenerator()
        gen = datagen.flow(self.x_train, self.y_train, batch_size)
        while True:
            xy = gen.next()
            yield(xy[0], xy[1])
    
    def flow_test(self,batch_size):
        datagen = ImageDataGenerator()
        gen = datagen.flow(self.x_test, self.y_test, batch_size)
        while True:
            xy = gen.next()
            yield(xy[0], xy[1])

    def save_cm_train(self,path=None):
        if path is None:
            path  = 'noisylabels/0_{}_{}_cm_train.png'.format(int(self.noise_ratio*100), self.name)
        cm = self.get_cm_train()
        plot_matrix(cm, self.class_names, title='Noise ratio: {}'.format(self.noise_ratio))
        plt.savefig(path)
        plt.close()

    def save_cm_test(self,path=None):
        if path is None:
            path  = 'noisylabels/0_{}_{}_cm_test.png'.format(int(self.noise_ratio*100), self.name)
        cm = self.get_cm_test()
        plot_matrix(cm, self.class_names, title='Noise ratio: {}'.format(self.noise_ratio))
        plt.savefig(path)
        plt.close()

    def int_to_onehot(self):
        if len(self.y_train.shape) == 1:
            self.y_train = to_categorical(self.y_train, self.num_classes)
        if len(self.y_test.shape) == 1:
            self.y_test = to_categorical(self.y_test, self.num_classes)
        if len(self.y_noisy.shape) == 1:
            self.y_noisy = to_categorical(self.y_noisy, self.num_classes)

    def get_labels_int(self):
        return self.y_train_int(), self.y_test_int(), self.y_noisy_int()

    def x_train_img(self):
        if self.name is 'mnist' or  self.name is 'mnist_fashion':
            return self.x_train.reshape(self.num_train_samples,28,28)
        else:
            return self.x_train

    def y_train_int(self):
        if not len(self.y_train.shape) == 1:
            y_train_int = np.argmax(self.y_train, axis=1)
        else:
            y_train_int = self.y_train
        return y_train_int
    
    def y_test_int(self):
        if not len(self.y_test.shape) == 1:
            y_test_int = np.argmax(self.y_test, axis=1)
        else:
            y_test_int = self.y_test
        return y_test_int

    def y_noisy_int(self):
        if not len(self.y_noisy.shape) == 1:
            y_noisy_int = np.argmax(self.y_noisy, axis=1)
        else:
            y_noisy_int = self.y_noisy
        return y_noisy_int
    
    def y_noisy_test_int(self):
        if not len(self.y_noisy_test.shape) == 1:
            y_noisy_test_int = np.argmax(self.y_noisy_test, axis=1)
        else:
            y_noisy_test_int = self.y_noisy_test
        return y_noisy_test_int

    def shuffle(self):
        idx_perm = np.random.permutation(self.num_train_samples)
        self.x_train, self.y_train = self.x_train[idx_perm], self.y_train[idx_perm]

    def normalize_mean(self):
        self.x_train = self.x_train - self.mean
        self.x_test = self.x_test - self.mean

    def normalize_std(self):
        self.x_train = self.x_train / self.std
        self.x_test = self.x_test / self.std

    def _set_num_classes(self):
        if self.num_classes == None:
            if len(self.y_train.shape) == 1:
                self.num_classes = max(self.y_train) - min(self.y_train) + 1
            else:
                self.num_classes = self.y_train.shape[1]
                

def get_data(dataset_name='mnist', **kwargs):
    assert dataset_name in DATASETS, "dataset must be one of: mnist, mnist_fashion, cifar10, cifar100"
    if dataset_name == 'mnist':
        download('mnist/dataset/', 'http://yann.lecun.com/exdb/mnist/', MNIST_RESOURCES)
        x_train, y_train = load_mnist('mnist/dataset/', kind='train')
        x_test, y_test = load_mnist('mnist/dataset/', kind='t10k')

        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

        num_classes = 10
        class_names = np.arange(num_classes)

    elif dataset_name == 'mnist_fashion':
        download('mnist_fashion/dataset/', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', MNIST_RESOURCES)
        x_train, y_train = load_mnist('mnist_fashion/dataset/', kind='train')
        x_test, y_test = load_mnist('mnist_fashion/dataset/', kind='t10k')

        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

        num_classes = 10
        class_names = ['Tshirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']

    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.reshape(-1, 32, 32, 3) / 255.0
        x_test = x_test.reshape(-1, 32, 32, 3) / 255.0
        
        means = x_train.mean(axis=0)
        # std = np.std(X_train)
        x_train = (x_train - means)  # / std
        x_test = (x_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        x_train = x_train.reshape(-1, 32, 32, 3) / 255.0
        x_test = x_test.reshape(-1, 32, 32, 3) / 255.0
        
        means = x_train.mean(axis=0)
        # std = np.std(X_train)
        x_train = (x_train - means)  # / std
        x_test = (x_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        num_classes = 100
        class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm']
    
    else:
        return None

    return DatasetCls(x_train, y_train, x_test, y_test, dataset_name=dataset_name, num_classes=num_classes, class_names=class_names, **kwargs)



if __name__ == "__main__":
    dataset = get_data('mnist')
    dataset.get_stats()
    dataset = get_data('mnist_fashion')
    dataset.get_stats()
    download('mnist/dataset/', 'http://yann.lecun.com/exdb/mnist/', MNIST_RESOURCES)
    load_mnist('mnist/dataset/')
    np.random.seed(123)