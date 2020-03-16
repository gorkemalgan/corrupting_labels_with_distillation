import util
import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy

def get_metrics(metric_type):
    if metric_type is 'distillation':
        return acc
    else:
        return 'accuracy'

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    num_classes = util.NUM_CLASS
    y_true = y_true[:, :num_classes]
    y_pred = y_pred[:, :num_classes]
    return categorical_accuracy(y_true, y_pred)