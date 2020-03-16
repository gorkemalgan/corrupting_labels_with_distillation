import numpy as np
from os.path import exists, join, isfile
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

import util
from util import VERBOSE, create_folders

def csv_logger(log_dir, **kwargs):
    return CSVLogger(log_dir+'logs.csv', **kwargs)

def model_checkpoint(log_dir, **kwargs):
    logdir = log_dir+'model/'
    create_folders(logdir)
    return ModelCheckpoint(logdir+"checkpoint.hdf5", **kwargs)

def early_stop(**kwargs):
    return EarlyStopping(**kwargs)

def learning_rate_scheduler(schedule,verbose=0):
    return LearningRateScheduler(schedule,verbose)

def lr_plateau(**kwargs):
    return ReduceLROnPlateau(**kwargs)

def tensor_board(dataset, log_dir, is_embed=False, **kwargs):
    tb_dir = log_dir+'tensorboard/'
    create_folders(tb_dir)
    with open(join(tb_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, dataset.y_test_int())
    if not is_embed:
        return TensorBoard(log_dir=tb_dir, **kwargs)
    else:
        return TensorBoard(log_dir=tb_dir,
                        embeddings_freq=1,
                        embeddings_layer_names=['features'],
                        embeddings_metadata='metadata.tsv',
                        embeddings_data=dataset.x_test,
                        **kwargs)        

class MyLogger(Callback):
    def __init__(self, log_dir, dataset):
        self.keys = None
        self.log_dir = log_dir
        _,_,self.x_test,self.y_test = dataset.get_data()
        super().__init__()

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.test_losses = []
        self.acc = []
        self.val_acc = []
        self.test_acc = []
        self.fig = plt.figure()        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        self.test_losses.append(loss)
        self.test_acc.append(acc)

        plt.figure(self.fig.number)
        
        plt.clf()
        filename = self.log_dir +'loss.png'
        plt.plot(self.losses, label="loss")
        plt.plot(self.val_losses, label="val_loss")
        plt.plot(self.test_losses, label="test_loss")
        plt.legend(loc='best')
        plt.xlabel('# iterations')
        plt.title('Loss')
        plt.savefig(filename)

        plt.clf()
        filename = self.log_dir +'accuracy.png'
        plt.plot(self.acc, label="acc")
        plt.plot(self.val_acc, label="val_acc")
        plt.plot(self.test_acc, label="test_acc")
        plt.legend(loc='best')
        plt.xlabel('# iterations')
        plt.title('Accuracy')
        plt.savefig(filename)

        # log csv
        if self.keys is None:
            self.keys = sorted(logs.keys())

        cols = ['epoch']
        cols.extend([key for key in self.keys])

        # if csv doesn't exists, initialize it
        if not isfile(self.log_dir+'log.csv'):
            row_dict = OrderedDict()
            row_dict['epoch']= [epoch]
            for col in cols[1:]:
                row_dict[col] = [logs.get(col)]
            row_dict['test_acc'] = [acc]
            row_dict['test_loss'] = [loss]
            df = pd.DataFrame(row_dict)
        else:
            df = pd.read_csv(self.log_dir+'log.csv')
            row_dict = OrderedDict()
            row_dict['epoch']= epoch
            for col in cols[1:]:
                row_dict[col] = logs.get(col)
            row_dict['test_acc'] = acc
            row_dict['test_loss'] = loss
            df = df.append([row_dict])

        df.to_csv(self.log_dir+'log.csv', index=False)
    
    def on_train_end(self, logs=None):
        plt.close(self.fig)