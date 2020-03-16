import os
from shutil import rmtree
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors.nearest_centroid import NearestCentroid

VERBOSE = 0
ALPHA = 0.1
TEMPERATURE = 4
RANDOM_SEED = 0
NUM_CLASS = 10

DATASETS = ['mnist', 'mnist_fashion', 'cifar10', 'cifar100']
MODELS = ['ce', 'boot_hard', 'boot_soft', 'forward', 'd2l', 'coteaching'] # 'backward'
NOISETYPES = ['uniform', 'class-dependent', 'locally-concentrated', 'feature-dependent']

PARAMS = {'mnist':{'epochs': 2, 'batch_size':256, 'patience': 10, 'alpha':0.1, 'temperature':16},
          'mnist_fashion':{'epochs': 40, 'batch_size':256, 'patience': 10, 'alpha':0.1, 'temperature':16},
          'cifar10':{'epochs': 125, 'batch_size':128, 'patience': 23, 'alpha':0.3, 'temperature':2},
          'cifar100':{'epochs': 125, 'batch_size':256, 'patience': 23}}

def clean_empty_logs():
    paths = ['logs/mnist', 'logs/cifar10']
    for log_dir in paths:
        if os.path.isdir(log_dir):
            for dirs in os.listdir(log_dir):
                path = os.path.join(log_dir,dirs)
                if os.path.isdir(path): # 'logs/mnist/nr_45/'
                    for dirs2 in os.listdir(path):
                        path2 = os.path.join(path,dirs2)
                        if len(os.listdir(path2)) < 2:
                            rmtree(path2)

def create_folders(*folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def delete_folders(*folders):
    for folder in folders:
        if os.path.isdir(folder):
            rmtree(folder)

def seed_everything(seed=RANDOM_SEED):
    np.random.seed(seed=seed)

def normalize(arr, div=None):
    if div is None:
        max_val = np.amax(arr)
        min_val = np.amin(arr)
        div = max_val - min_val
    return arr / div

def save_model(model,path):
    model_json = model.to_json()
    with open(path+'.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(path+'.h5')

def load_model(path):
    with open(path+'.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+'.h5')
    return loaded_model

def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

def get_centroids(logits, labels, num_classes=None):
    # extract number of classes if not given
    if num_classes is None:
        num_classes = len(unique_labels(labels))
    # find centroid locations with scikit-learn
    clf = NearestCentroid()
    clf.fit(logits, labels)
    return clf.centroids_

def dm_centroid(centroids):
    num_classes = centroids.shape[1]
    cm = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        diffs = centroids - centroids[i,:]
        cm[i,:] = np.linalg.norm(diffs, axis=1)
    return cm

def get_sorted_idx(probs, labels, class_id=None):
    '''
    Returns indices of samples beloning to class_id. Indices are sorted according to probs. First one is least confidently class_id
    and last one is most confidently class_id.
    If class_id is None, then just sorts all samples according to given probability
    '''
    # indices of samples which belong to class i
    if class_id is None:
        idx_i = labels
    else:
        idx_i = np.where(labels == class_id)[0]
    # order according to probabilities of confidence. First one is least likely for class i and last one is most likely
    idx_tmp = np.argsort(probs[idx_i])
    idx_sorted = idx_i[idx_tmp]

    # make sure sorted idx indeed belongs to given class
    if class_id is not None:
        assert np.sum(labels[idx_sorted] == class_id) == len(idx_sorted)
    # make sure idx are indeed sorted
    assert np.sum(np.diff(probs[idx_sorted])<0) == 0

    return idx_sorted

def get_sorted_intersect(arr1, arr2, arr_val):
    '''
    Returns the intersection of arr1 and arr2 arrays of indices. Then intersection array of indices according to index value in arr_val
    '''
    intersects = np.intersect1d(arr1, arr2)
    idx_tmp = np.argsort(arr_val[intersects])
    intersects_sorted = intersects[idx_tmp]

    assert len(np.intersect1d(arr1, intersects_sorted)) == len(intersects_sorted)
    assert len(np.intersect1d(arr2, intersects_sorted)) == len(intersects_sorted)
    assert np.sum(np.diff(arr_val[intersects_sorted])<0) == 0

    return intersects_sorted

if __name__ == "__main__":
    clean_empty_logs()