import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adadelta, Adam, SGD
from tensorflow.keras import regularizers

from loss import cross_entropy, forward, backward, boot_hard, boot_soft, lid_paced_loss, distillation_loss, acc_distillation
from noise import cm_uniform
from util import NUM_CLASS, DATASETS

def get_model_architecture(dataset, is_dropout=False):
    assert dataset.name in DATASETS, "dataset must be one of: mnist, mnist_fashion, cifar10, cifar100"

    num_classes = dataset.num_classes
    img_shape = dataset.img_shape
    img_input = Input(shape=img_shape)     

    if dataset.name == 'mnist' or dataset.name == 'mnist_fashion':
        
        # architecture from: https://keras.io/examples/mnist_cnn/
        x = Conv2D(32, kernel_size=(3, 3))(img_input)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
        if is_dropout:
            x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        if is_dropout:
            x = Dropout(0.5)(x)
        x = Dense(num_classes, name='features')(x)
        x = Activation('softmax')(x)
        # Create model
        model = Model(img_input, x)  

    elif dataset.name == 'cifar10':

        x = Conv2D(32, (3, 3), padding='same')(img_input)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        if is_dropout:
            x = Dropout(0.25)(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        if is_dropout:
            x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        if is_dropout:
            x = Dropout(0.5)(x)
        x = Dense(num_classes, name='features')(x)
        x = Activation('softmax')(x)
        # Create model
        model = Model(img_input, x)  

    elif dataset.name == 'cifar100':
        # taken from: https://github.com/geifmany/cifar-vgg/tree/e7d4bd4807d15631177a2fafabb5497d0e4be3ba
        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=img_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if is_dropout:
            model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        if is_dropout:
            model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        if is_dropout:
            model.add(Dropout(0.5))
        model.add(Dense(num_classes, name='features'))
        model.add(Activation('softmax'))

        # architecture from: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
        #model = resnet_v1(img_shape,20,10)

    #model.summary()
    return model

def get_model(dataset, model_name='ce', cm=None, **kwargs):
    # model architecture
    model = get_model_architecture(dataset, **kwargs)
    optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # loss
    if model_name == 'forward':
        assert cm is not None
        model.compile(loss=forward(cm), optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'backward':
        assert cm is not None
        model.compile(loss=backward(cm), optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'boot_hard':
        model.compile(loss=boot_hard, optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'boot_soft':
        model.compile(loss=boot_soft, optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'd2l':
        model.compile(loss=lid_paced_loss(), optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'ce' or model_name == 'coteaching':
        model.compile(loss=cross_entropy, optimizer=optimizer, metrics=['accuracy'])
    elif model_name == 'distillation':
        model.compile(loss=distillation_loss(dataset.num_classes), optimizer=optimizer, metrics=[acc_distillation(dataset.num_classes)])

    model._name = model_name
    return model

def compile_model(model, loss=None, optimizer=None, metrics=['accuracy']):
    if loss is None:
        loss = cross_entropy
    if optimizer is None:
        optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    return model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
