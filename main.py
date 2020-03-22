import os
from os.path import isfile
import shutil
from datetime import datetime
import argparse
import gc
from pathlib import Path
import numpy as np
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Lambda, concatenate, Input, Average
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from scipy.special import softmax as softmax_scipy

from dataset import get_data, DatasetCls
from noise import get_noisy_labels, cm_uniform
from models import get_model, compile_model
from loss import distillation_loss, acc_distillation
from metrics import get_metrics
from callbacks import csv_logger, model_checkpoint, early_stop, learning_rate_scheduler,lr_plateau, tensor_board, MyLogger
from visualizer import plot_cm, plot_dm, plot_matrix, plot_confused_samples, plot_confused_samples2, plot_overall
from util import clean_empty_logs, create_folders, seed_everything
from util import save_model, load_model, softmax, get_centroids, dm_centroid
from util import MODELS, DATASETS, NOISETYPES, PARAMS, RANDOM_SEED, PLOT_NOISE
import util

logcsv_cols = ['Dataset', 'Noise Type','Model Name', 'Noise Rate', 'Accuracy', 'Loss', 'Similarity']

def lr_schedule(epoch):
    learning_rate = 0.1
    return learning_rate * (0.5 ** (epoch // 20))

def train(dataset, model, epochs=50, batch_size=128, log_dir=None, callbacks=[], verbose=1):
    # prepare folders
    create_folders(log_dir+'model/', log_dir+'npy/')
    
    # get data
    x_train, y_train, x_test, y_test = dataset.get_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.5, random_state=RANDOM_SEED)

    # callbacks
    mycallbacks = []
    mycallbacks.extend(callbacks)
    if log_dir is not None:
        mycallbacks.extend([MyLogger(log_dir, dataset),    
                            learning_rate_scheduler(lr_schedule,verbose=verbose),
                            lr_plateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6, verbose=verbose)
                            ])
        plot_model(model, log_dir+'model/model.png')
    # image generator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    # train model on clean data    
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        steps_per_epoch=dataset.num_train_samples/batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        verbose=verbose,
                        callbacks=mycallbacks
                        )
    
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('val_loss:', loss, '- val_acc:', acc)
    
    del callbacks
    gc.collect()
    
    return model

def train_coteaching(dataset, model1, model2, epochs=50, batch_size=128, log_dir=None, forget_rate=0.2, num_graduals=10,exponent=0.2,learning_rate=1e-3,epoch_decay_start=30):
    def epoch_stats(dataset, model, epoch, logdir, csv_path=None):
        x_train, y_train_noisy, x_test, y_test = dataset.get_data()
        y_train_int = dataset.y_noisy_int()
        y_test_int = dataset.y_test_int()
        clean_index = dataset.idx_clean
        noisy_index = dataset.idx_noisy
        
        if csv_path is not None:
            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            print("-%s: acc_test: %.4f - acc_mix: %.4f" % (model.name, test_acc, train_acc))
            df = pd.read_csv(log_dir+csv_path)
            #row = [{'epoch':epoch, 'acc':acc_mix,'loss':loss_mix, 'val_acc':acc_test, 'val_loss':loss_test,'test_acc':test_acc,'test_loss':test_loss}]
            row = [{'epoch':epoch, 'acc':train_acc,'loss':train_loss,'test_acc':test_acc,'test_loss':test_loss}]
            df = df.append(row)
            df.to_csv(log_dir+csv_path, index=False)

            xticks = np.arange(1, epoch+1, 1.0)
            plt.figure()
            plt.plot(xticks, df['acc'], label="acc")
            #plt.plot(xticks, df['val_acc'], label="val_acc")
            plt.plot(xticks, df['test_acc'], label="test_acc")
            plt.legend(loc='best')
            plt.xlabel('# iterations')
            plt.xticks(xticks)
            plt.title('Accuracy')
            plt.savefig(log_dir+'accuracy_{}.png'.format(model.name))

            plt.clf()
            plt.plot(xticks, df['loss'], label="loss")
            #plt.plot(xticks, df['val_loss'], label="val_loss")
            plt.plot(xticks, df['test_loss'], label="test_loss")
            plt.legend(loc='best')
            plt.xlabel('# iterations')
            plt.xticks(xticks)
            plt.title('Loss')
            plt.savefig(log_dir+'loss_{}.png'.format(model.name))
            plt.close()
    create_folders(log_dir+'model/', log_dir+'npy/')
    # get data
    x_train, y_train, x_test, y_test = dataset.get_data()
    model1._name = 'model1'
    model2._name = 'model2'

    # number of batches in an epoch
    num_batch_iter = x_train.shape[0] / batch_size
    # calculate forget rates for each epoch (from origianl code)
    forget_rates = np.ones(epochs)*forget_rate
    forget_rates[:num_graduals] = np.linspace(0, forget_rate**exponent, num_graduals)
    # calculate learning rates for each epoch (from origianl code)
    learning_rates = [learning_rate] * epochs
    for i in range(epoch_decay_start, epochs):
        learning_rates[i] = float(epochs - i) / (epochs - epoch_decay_start) * learning_rate

    if log_dir is not None:
        #logcsv_cols =['epoch','acc','loss','val_acc','val_loss','test_acc','test_loss']
        logcsv_cols =['epoch','acc','loss','test_acc','test_loss']
        df = pd.DataFrame(columns=logcsv_cols)
        df.to_csv(log_dir+'log1.csv', index=False)
        df.to_csv(log_dir+'log2.csv', index=False)

    for e in range(1,epochs): 
        # if learning rate changes, recompile
        if (learning_rates[e] != learning_rates[e-1]):   
            model1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rates[e], decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
            model2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rates[e], decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
            #model2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rates[e]), metrics=['accuracy'])

        remember_rate = 1 - forget_rates[e]
        print("Epoch: %d/%d; Learning rate: %.7f; n_keep: %d" % (e+1, epochs, learning_rates[e], remember_rate*batch_size))        

        # iterate for each batch in an epoch
        for (i, (x_batch, y_batch)) in enumerate(dataset.flow_train(batch_size)):
            num_remember = int(remember_rate * len(x_batch))

            # select samples based on model 1
            y_pred = model1.predict_on_batch(x_batch)
            cross_entropy = np.sum(-y_batch*np.log(y_pred+1e-8),axis=1)
            batch_idx1= np.argsort(cross_entropy)[:num_remember]
    
            # select samples based on  model 2
            y_pred = model2.predict_on_batch(x_batch)
            cross_entropy = np.sum(-y_batch*np.log(y_pred+1e-8),axis=1)
            batch_idx2 = np.argsort(cross_entropy)[:num_remember]
            
            # training
            model1.train_on_batch(x_batch[batch_idx2,:], y_batch[batch_idx2,:])  
            model2.train_on_batch(x_batch[batch_idx1,:], y_batch[batch_idx1,:])              
            
            if i >= num_batch_iter:
                break
            
        epoch_stats(dataset,model1,e,log_dir,'log1.csv')
        epoch_stats(dataset,model2,e,log_dir,'log2.csv')   

    # choose best model
    loss1, acc1 = model1.evaluate(x_test, y_test, verbose=0)
    loss2, acc2 = model2.evaluate(x_test, y_test, verbose=0)
    if acc1 > acc2:
        os.rename(log_dir+'log1.csv', log_dir+'log.csv')
        return model1
    else:
        os.rename(log_dir+'log2.csv', log_dir+'log.csv')
        return model2

def save_model_outputs(model, _dataset, model_path):
    npy_path = model_path+'npy/'
    create_folders(npy_path, model_path+'model/')    
    model_soft = Model(model.input, model.get_layer('features').output)
    # save softmax predictions
    pred = model.predict(_dataset.x_train)[:, :_dataset.num_classes]
    pred_int = np.argmax(pred, axis=1)
    np.save(npy_path+'train_preds.npy', pred)
    np.save(npy_path+'train_preds_int.npy', pred_int)
    pred = model.predict(_dataset.x_test)[:, :_dataset.num_classes]
    pred_int = np.argmax(pred, axis=1)
    np.save(npy_path+'test_preds.npy', pred)
    np.save(npy_path+'test_preds_int.npy', pred_int)
    # save logits
    logits_train = model_soft.predict(_dataset.x_train)[:, :_dataset.num_classes]
    logits_test = model_soft.predict(_dataset.x_test)[:, :_dataset.num_classes]
    np.save(npy_path+'train_logits.npy', logits_train)
    np.save(npy_path+'test_logits.npy', logits_test)
    # save confusion matrices
    cm_train = plot_cm(model, _dataset.x_train,_dataset.y_train_int(),_dataset.class_names, model_path+'train_cm.png')
    cm_test = plot_cm(model, _dataset.x_test,_dataset.y_test_int(),_dataset.class_names, model_path+'test_cm.png')
    np.save(npy_path+'train_cm.npy', cm_train)
    np.save(npy_path+'test_cm.npy', cm_test)
    # save distance matrices
    plot_dm(model_soft, _dataset.x_train, _dataset.y_train_int(), _dataset.class_names, model_path+'train_dm.png')
    plot_dm(model_soft, _dataset.x_test, _dataset.y_test_int(), _dataset.class_names, model_path+'test_dm.png')
    # save model
    plot_model(model,model_path+'model/model.png')
    save_model(model,model_path+'model/model')
    K.clear_session()

def prep_teacher_model(dataset, verbose):
    # train model on noise free data with dropout
    model_path = '{}/models/teacher/'.format(dataset)
    if not isfile(model_path+'model/model.h5') or not isfile(model_path+'model/model.json'):
        print('Main model doesnt exist, training it...')
        # train teacher
        _dataset = get_data(dataset)
        model = get_model(_dataset, is_dropout=True)
        model = train(_dataset, model, 
                        PARAMS[dataset]['epochs'], PARAMS[dataset]['batch_size'],
                        log_dir=model_path, 
                        callbacks=[early_stop(patience=PARAMS[dataset]['patience'], monitor='val_loss', verbose=verbose)],
                        verbose=verbose)
        # save output files
        save_model_outputs(model, _dataset, model_path)

def prep_student(dataset, verbose, alpha, temperature):
    # if student is not saved beforehand, train and save it
    model_path = '{}/models/student/'.format(dataset)
    if not isfile(model_path+'model/model.h5') or not isfile(model_path+'model/model.json'):
        print('Student model doesnt exist, training it...')
        # load dataset and logits
        _dataset = get_data(dataset)
        x_train, y_train, x_test, y_test = _dataset.get_data()
        train_features = np.load('{}/models/teacher/npy/train_logits.npy'.format(dataset))
        test_features = np.load('{}/models/teacher/npy/test_logits.npy'.format(dataset))
        # normalized output with temperature shape=(num_samples,num_classes)
        y_train_soft = softmax(train_features/temperature)
        y_test_soft = softmax(test_features/temperature)
        # concatenated output labels=(num_samples,2*num_classes)
        y_train_new = np.concatenate([y_train, y_train_soft], axis=1)
        y_test_new =  np.concatenate([y_test, y_test_soft], axis =1)
        # build student model
        student = get_model(_dataset, 'distillation', is_dropout=True)
        # remove softmax
        student.layers.pop()
        # get features
        logits = student.layers[-1].output 
        # normal softmax output
        probs = Activation('softmax')(logits) 
        # softmax output with temperature
        logits_T = Lambda(lambda x: x / temperature)(logits) 
        probs_T = Activation('softmax')(logits_T) 
        # concatanete
        output = concatenate([probs, probs_T])
        # This is our new student model
        student = Model(student.input, output)
        compile_model(student, loss=distillation_loss(_dataset.num_classes,alpha),metrics=[acc_distillation(_dataset.num_classes)])
        # create a new dataset with generated data
        dataset_s = DatasetCls(x_train,y_train_new,x_test,y_test_new, dataset_name=dataset)
        # train student
        student = train(dataset_s, student, 
                        PARAMS[dataset]['epochs']*2, PARAMS[dataset]['batch_size'],
                        log_dir=model_path, 
                        callbacks=[early_stop(patience=PARAMS[dataset]['patience'], monitor='val_loss', verbose=verbose)],
                        verbose=verbose)
        # save output files
        save_model_outputs(student, _dataset, model_path)

        K.clear_session()

def prep_noisylabels(dataset, folders, noise_type, noise_ratio, verbose, alpha,temperature, is_dropout):   
    # prepare required models first
    prep_teacher_model(dataset, verbose)
    prep_student(dataset, verbose,alpha,temperature)

    # generate and save corrupted labels for each noise type for given noise ratio
    _dataset = get_data(dataset)

    # copy xy model as none for baseline
    path = str(Path(folders['logdir']).parent)
    if not os.path.isdir(path+'/none/'):
        shutil.copytree('{}/models/teacher'.format(dataset), path+'/none/')

    # generate noisy labels
    y_train_noisy, y_test_noisy, probs = get_noisy_labels(_dataset, noise_type, noise_ratio)
    if PLOT_NOISE:
        y_train_clean, y_test_clean = _dataset.y_train_int(), _dataset.y_test_int()
        create_folders(folders['noisedir'])
        if not isfile(folders['noisedir']+'cmtest.png'):
            print('Noise for {} doesnt exist, creating it...'.format(folders['noisedir']))
            # plot confused samples
            if probs is not None:
                create_folders(folders['noisedir']+'/plots')
                np.save(folders['noisedir']+'probs.npy', probs)
                _dataset_noisy = get_data(dataset, y_noisy=y_train_noisy, y_noisy_test=y_test_noisy)
                plot_confused_samples(probs, _dataset_noisy, path=folders['noisedir']+'plots/')
                plot_confused_samples2(probs, _dataset_noisy, path=folders['noisedir']+'plots/')
            # save confusion matrix
            cm = confusion_matrix(y_train_clean,y_train_noisy)
            plot_matrix(cm, _dataset.class_names, title='Noise ratio: {}'.format(noise_ratio))
            plt.savefig(folders['noisedir']+'cmtrain.png')
            cm = confusion_matrix(y_test_clean,y_test_noisy)
            plot_matrix(cm, _dataset.class_names, title='Noise ratio: {}'.format(noise_ratio))
            plt.savefig(folders['noisedir']+'cmtest.png')

    return y_train_noisy, y_test_noisy

def postprocess(dataset, model, noise_type, noise_ratio, folders, y_test_noisy):
    log_dir = folders['logdir']
    loss, acc = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
    print('loss:', loss, '- acc:', acc)

    # calculate similarity ofgiven confusion matrix and output confusion matrix
    pred = model.predict(dataset.x_test)
    pred_int = np.argmax(pred, axis=1)
    sim = 1 - distance.cosine(pred_int, y_test_noisy)
    print('Similarity is',sim)    
    # plot confusion matrix
    plot_cm(model, dataset.x_test,dataset.y_test_int(),dataset.class_names, log_dir+'/cm.png', title='acc({}), similarity({})'.format(round(acc,3),round(sim,2)))
    # plot accuracies and losses for all models
    base_folder = folders['logbase_nr']
    plot_overall(base_folder)
    # save variables
    np.save(log_dir+'preds.npy', pred_int)
    save_model(model, log_dir+'model/model')

def main(dataset_name, model_name, epochs, batch_size, noise_type, noise_ratio, verbose=1, alpha=util.ALPHA, temperature=16, is_dropout=False, percentage=1):   
    K.clear_session()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    seed_everything()  
    # folders to be used
    noise_base_path = 'nr_{}'.format(noise_ratio) if not is_dropout else 'nr_{}_do'.format(noise_ratio)
    folders = {'logbase': '{}/logs_{}/'.format(dataset_name,percentage),
               'logbase_nr': '{}/logs_{}/{}/{}/'.format(dataset_name,percentage,noise_base_path,model_name),
               'logdir': '{}/logs_{}/{}/{}/{}/'.format(dataset_name,percentage,noise_base_path,model_name, noise_type),
               'modelbase' : '{}/models/'.format(dataset_name),
               'noisebase': '{}/noisylabels/'.format(dataset_name),
               'noisedir': '{}/noisylabels/{}/'.format(dataset_name,noise_type),
               'dataset': '{}/dataset'.format(dataset_name)
               }

    # if log file already exis"ts dont run it again
    if isfile(folders['logdir']+'model/model.h5') and isfile(folders['logdir']+'model/model.json'):
        print('Logs exists, skipping run...')
        return
    
    # clean empty logs if there is any
    clean_empty_logs()
    # create necessary folders
    create_folders(folders['dataset'], folders['logdir'])
    # generate noisy labels
    y_train_noisy, y_test_noisy = prep_noisylabels(dataset_name, folders, noise_type, noise_ratio, verbose, alpha, temperature, is_dropout)

    # load dataset with noisy labels
    dataset = get_data(dataset_name, y_noisy=y_train_noisy, y_noisy_test=y_test_noisy)
    dataset.get_percentage(percentage)

    # stats before training
    print('Dataset: {}, model: {}, noise_type: {}, noise_ratio: {}, epochs: {}, batch: {} , dropout: {}'.format(
        dataset.name, model_name, noise_type, noise_ratio, epochs, batch_size, is_dropout))
    dataset.get_stats()
    dataset.save_cm_train(folders['logdir']+'corrupted_data.png')
    

    # train model
    if model_name == 'coteaching':
        model1 = get_model(dataset, model_name, is_dropout=is_dropout)
        model2 = get_model(dataset, model_name, is_dropout=is_dropout)
        model = train_coteaching(dataset, model1, model2, epochs, batch_size, folders['logdir'])
    else:
        #cm = np.load('{}/models/xy/npy/test_cm.npy'.format(dataset_name))
        cm = dataset.get_cm_train()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        model = get_model(dataset, model_name, cm, is_dropout=is_dropout)
        model = train(dataset, model, epochs, batch_size, folders['logdir'], verbose=verbose)

    # performance analysis
    postprocess(dataset, model, noise_type, noise_ratio, folders, y_test_noisy)
    K.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', help="Dataset to use: 'mnist', cifar10'", required=False, type=str, default='mnist_fashion')
    parser.add_argument('-m', '--model_name', help="Model name: 'ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'.", required=False, type=str, default='ce')
    parser.add_argument('-e', '--epochs', help="The number of epochs to train for.", required=False, type=int, default=30)
    parser.add_argument('-b', '--batch_size', help="The batch size to use for training.", required=False, type=int, default=128)
    parser.add_argument('-n', '--noise_type', help="'none', 'uniform', 'random', 'random_symmetric', 'pairwise', 'model_pred', 'xy'",required=False, type=str, default='feature-dependent')
    parser.add_argument('-r', '--noise_ratio', help="The percentage of noisy labels [0, 100].", required=False, type=int, default=35)
    parser.add_argument('-t', '--temperature', help="Temeperature for student to be trained", required=False, type=int, default=16)
    parser.add_argument('-a', '--alpha', help="Alpha for learning with distillation", required=False, type=float, default=0.1)
    parser.add_argument('-v', '--verbose', help="V,erbose one of the following: 0,1,2", required=False, type=int, default=1)
    parser.add_argument('-p', '--percentage', help="Percentage of dataset to be used. Between 0-1", required=False, type=float, default=1)
    parser.add_argument('--dropout', dest='dropout', action='store_true')
    parser.add_argument('--no-dropout', dest='dropout', action='store_false')

    args = parser.parse_args()
    main(args.dataset_name, args.model_name, args.epochs, args.batch_size, args.noise_type, args.noise_ratio, 
         args.verbose, args.alpha, args.temperature, args.dropout, args.percentage)
