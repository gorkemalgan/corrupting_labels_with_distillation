import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from util import get_sorted_idx

def noise_softmax(x_train, y_train_int, probs, noise_ratio):
  y_noisy = np.copy(y_train_int)
  num_classes = len(unique_labels(y_train_int))
  num_noisy = int(x_train.shape[0]*noise_ratio)
  # find maximum noise ratio and change original noise ratio if necessary
  non_zeros = np.count_nonzero(probs, axis=1)
  num_multiple_guess = np.sum(non_zeros > 1)
  # class ids sorted according to their probabilities for each instance shape=(num_samples,num_classes)
  prob_preds = np.argsort(probs, axis=1)
  # first and second predicted classes for each instance shape=(num_samples)
  prob_pred1, prob_pred2 = prob_preds[:,-1], prob_preds[:,-2]
  # indices of wrong predictions for first prediction 
  idx_wrong = np.where(prob_pred1 != y_train_int)[0]
  # change mis-predicted instances to their first prediction because it is most similer to that class
  if len(idx_wrong) >= num_noisy:
    # get the probabilities of first predictions for each sample shape=(num_samples)
    prob1 = np.array([probs[i,prob_pred1[i]] for i in range(len(prob_pred1))])
    # sorted list of second prediction probabilities
    idx_sorted = np.argsort(prob1)
    # sort them according to prob1
    idx_wrong2 = get_sorted_idx(prob1, idx_wrong)
    # get elements with highest probability on second prediciton because they are closest to other classes
    idx2change = idx_wrong2[-num_noisy:]
    # change them to their most likely class which is second most probable prediction
    y_noisy[idx2change] = prob_pred1[idx2change]
  else:
    y_noisy[idx_wrong] = prob_pred1[idx_wrong]
    # remaining number of elements to be mislabeled
    num_noisy_remain = num_noisy - len(idx_wrong)
    # get the probabilities of second predictions for each sample shape=(num_samples)
    prob2 = np.array([probs[i,prob_pred2[i]] for i in range(len(prob_pred2))])
    # sorted list of second prediction probabilities
    idx_sorted = np.argsort(prob2)
    # remove already changed indices for wrong first prediction
    idx_wrong2 = np.setdiff1d(idx_sorted, idx_wrong)
    # sort them according to prob2
    idx_wrong2 = get_sorted_idx(prob2, idx_wrong2)
    # get elements with highest probability on second prediciton because they are closest to other classes
    idx2change = idx_wrong2[-num_noisy_remain:]
    # change them to their most likely class which is second most probable prediction
    y_noisy[idx2change] = prob_pred2[idx2change]
    # get indices where second prediction has zero probability
    idx_tmp = np.where(prob2[idx2change] == 0)[0]
    idx_prob0 = idx2change[idx_tmp]
    assert np.sum(prob2[idx_prob0] != 0) == 0
    # since there is no information in second prediction, to prevent all samples with zero probability on second prediction to have same class
    # we will choose a random class for that sample
    for i in idx_prob0:
        classes = np.arange(num_classes)
        classes_clipped = np.delete(classes, y_train_int[i])
        y_noisy[i] = np.random.choice(classes_clipped, 1)

  return y_noisy, probs

def noise_xy_localized(features, y_train_int, noise_ratio):
  y_noisy = np.copy(y_train_int)
  n_clusters = 20
  num_classes = len(unique_labels(y_train_int))
  for i in range(num_classes):
    idx = y_train_int == i
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features[idx])
    y_pred = kmeans.labels_

    # number of samples for clusters
    n_samples = [np.sum(y_pred==k) for k in range(n_clusters)]
    sorted_idx = np.argsort(n_samples)
    # find clusters idx whose sum is equal for noise ratio
    n_tobecorrupted = round(np.sum(idx)*noise_ratio)

    for j in range(len(sorted_idx)):
      if n_samples[sorted_idx[j]] > n_tobecorrupted:
        break
    if j > 0:
      mid = sorted_idx[j-1]
    else:
      mid = sorted_idx[0]
    idx_class=np.where(idx==True)[0]
    idx2change=idx_class[y_pred==mid]
    y_noisy[idx2change] = np.random.choice(np.delete(np.arange(num_classes),i),1)
    num_corrupted = len(idx2change)

    for k in reversed(range(j-1)):
      if n_samples[sorted_idx[k]] + num_corrupted < n_tobecorrupted:
        idx2change=idx_class[y_pred==sorted_idx[k]]
        y_noisy[idx2change] = np.random.choice(np.delete(np.arange(num_classes),i),1)
        num_corrupted += len(idx2change)

  return y_noisy