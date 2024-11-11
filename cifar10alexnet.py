from __future__ import print_function

import keras.losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, RandomRotation, RandomFlip, RandomZoom, Input
import numpy as np
from keras import optimizers, callbacks
from myconv2d import MyConv2D
from mydense import MyDense


class cifar10alexnet:
    def __init__(self, load=False, orig=False, flush=0):
        self.num_classes = 10
        self.x_shape = [32, 32, 3]

        self.orig = orig
        self.flush = flush

        self.model = self.build_model()
        if load:
            self.model.load_weights('cifar10alexnet.weights.h5')

    def build_model(self):
        # Build modified AlexNet as seen on https://github.com/Natsu6767/Modified-AlexNet-Tensorflow

        model = Sequential()

        model.add(Input(self.x_shape))

        model.add(RandomZoom(.2))
        model.add(RandomRotation(.1))
        model.add(RandomFlip("horizontal"))

        model.add(MyConv2D(96, (11, 11), strides=(2, 2), padding='same',
                           use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))

        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(MyConv2D(192, (5, 5), padding='same', use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))

        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(MyConv2D(384, (3, 3), padding='same', use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))

        model.add(MyConv2D(256, (3, 3), padding='same', use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #model.add(Dropout(0.1))

        model.add(MyConv2D(256, (3, 3), padding='same', use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.1))

        # model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())

        model.add(Dropout(0.5))
        model.add(MyDense(4096, use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(MyDense(1024, use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MyDense(self.num_classes, use_original=self.orig, denorm_flush_zero=self.flush))
        model.add(Activation('softmax'))

        model.run_eagerly = not self.orig

        return model

    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x, batch_size)

    def train(self, x, y, xt, yt):
        # training parameters
        batch_size = 100
        maxepochs = 100
        learning_rate = 0.1
        weight_decay = 5e-4
        lr_drop = 20

        x, xt = self.normalize(x, xt)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = self.model.fit(x, y, batch_size=batch_size, epochs=maxepochs, validation_data=(xt, yt),
                                     callbacks=[reduce_lr], verbose=1)
        self.model.save_weights('cifar10alexnet.weights.h5')
