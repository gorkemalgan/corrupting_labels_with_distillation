import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import confusion_matrix

from noise_xy import noise_softmax, noise_xy_localized
from util import NOISETYPES

def get_noisy_labels(dataset, noise_type, noise_ratio):
    assert noise_type in NOISETYPES, "invalid noise type"
    num_classes = dataset.num_classes
    probs = None
    # if noise ratio is integer, convert it to float
    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.
    # generate confusion matrix for given noise type
    if noise_type == 'none':
        y_train_noisy = dataset.y_train_int()
        y_test_noisy =  dataset.y_test_int()
    elif noise_type == 'uniform':
        P = cm_uniform(num_classes, noise_ratio)
        y_train_noisy, _ = noise_cm(dataset.y_train_int(), P)
        y_test_noisy, _ = noise_cm(dataset.y_test_int(), P)
    elif noise_type == 'class-dependent':
        path = '{}/models/teacher/npy/'.format(dataset.name)
        test_logits = np.load(path+'test_logits.npy')
        P = cm_model_prediction(dataset.x_test, dataset.y_test_int(), test_logits, noise_ratio)
        y_train_noisy, _ = noise_cm(dataset.y_train_int(), P)
        y_test_noisy, _ = noise_cm(dataset.y_test_int(), P)
    elif noise_type == 'feature-dependent':
        path = '{}/models/student/npy/'.format(dataset.name)
        train_preds = np.load(path+'train_preds.npy')
        test_preds = np.load(path+'test_preds.npy')
        y_train_noisy, probs = noise_softmax(dataset.x_train, dataset.y_train_int(), train_preds, noise_ratio)
        y_test_noisy, _ = noise_softmax(dataset.x_test, dataset.y_test_int(), test_preds, noise_ratio)
    elif noise_type == 'locally-concentrated':
        path = '{}/models/teacher/npy/'.format(dataset.name)
        train_logits = np.load(path+'train_logits.npy')
        test_logits = np.load(path+'test_logits.npy')
        y_train_noisy = noise_xy_localized(train_logits, dataset.y_train_int(), noise_ratio)
        y_test_noisy = noise_xy_localized(test_logits, dataset.y_test_int(), noise_ratio)
    
    return y_train_noisy, y_test_noisy, probs

def cm_uniform(num_classes, noise_ratio):
    # if noise ratio is integer, convert it to float
    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.
    assert (noise_ratio >= 0.) and (noise_ratio <= 1.)

    P = noise_ratio / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise_ratio) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def cm_model_prediction(x_test, y_test, logits, noise_ratio):
    # if noise ratio is integer, convert it to float
    if noise_ratio > 1 and noise_ratio < 100:
        noise_ratio = noise_ratio / 100.

    y_pred = np.argmax(logits, axis=1)
    
    # extract confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # set diagonal entries to 0 for now
    np.fill_diagonal(cm, 0)
    # find probability of each misclassification with avoiding zero division
    sums = cm.sum(axis=1)
    idx_zeros = np.where(sums == 0)[0]
    sums[idx_zeros] = 1
    cm = (cm.T / sums).T
    # weight them with noise
    cm = cm * noise_ratio
    # set diagonal entries
    np.fill_diagonal(cm, (1-noise_ratio))
    # if noise was with zero probabiilty, set the coresponding class probability to 1
    for idx in idx_zeros:
        cm[idx,idx] = 1

    assert_array_almost_equal(cm.sum(axis=1), 1, 1)

    return cm

def noise_cm(y, cm=None):
    assert_array_almost_equal(cm.sum(axis=1), 1, 1)

    y_noisy = np.copy(y)
    num_classes = cm.shape[0]

    for i in range(num_classes):
        # indices of samples belonging to class i
        idx = np.where(y == i)[0]
        # number of samples belonging to class i
        n_samples = len(idx)
        for j in range(num_classes):
            if i != j:
                # number of noisy samples according to confusion matrix
                n_noisy = int(n_samples*cm[i,j])
                if n_noisy > 0:
                    # indices of noisy samples
                    noisy_idx = np.random.choice(len(idx), n_noisy, replace=False)
                    # change their classes
                    y_noisy[idx[noisy_idx]] = j
                    # update indices
                    idx = np.delete(idx, noisy_idx)

    return y_noisy, None
