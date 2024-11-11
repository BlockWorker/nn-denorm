from __future__ import print_function

from keras.layers import Dense, Dropout, Activation, Flatten, Add, Rescaling
from keras.layers import MaxPooling2D, BatchNormalization, RandomRotation, RandomFlip, RandomZoom, Input
import numpy as np
from keras import optimizers, callbacks, Model
from myconv2d import MyConv2D
from mydense import MyDense


class cifar10resnet:
    def __init__(self, load=False, orig=False, flush=0):
        self.num_classes = 10
        self.x_shape = [32, 32, 3]

        self.orig = orig
        self.flush = flush

        self.model = self.build_model()
        if load:
            self.model.load_weights('cifar10resnet.weights.h5')

    def build_model(self):
        # Build ResNet as described in https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/

        def base_conv(p, channels):
            p_i = MyConv2D(channels, (3, 3), padding='same', use_original=self.orig,
                           denorm_flush_zero=self.flush)(p)
            p_i = BatchNormalization()(p_i)
            p_i = Activation('relu')(p_i)
            return p_i

        def res(p, channels):
            p_i = base_conv(p, channels)
            p_i = base_conv(p_i, channels)
            p_i = Add()([p, p_i])
            return p_i

        inputs = Input(self.x_shape)

        x = RandomZoom(.2)(inputs)
        x = RandomRotation(.1)(x)
        x = RandomFlip("horizontal")(x)

        x = base_conv(x, 64)  # prep

        x = base_conv(x, 128)  # layer 1
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = res(x, 128)

        x = base_conv(x, 256)  # layer 2
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = base_conv(x, 512)  # layer 3
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = res(x, 512)

        x = MaxPooling2D(pool_size=(2, 2))(x)  # classifier
        x = Flatten()(x)
        x = MyDense(self.num_classes, use_original=self.orig, denorm_flush_zero=self.flush)(x)
        x = Rescaling(.125)(x)
        x = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=x)

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
        batch_size = 512
        maxepochs = 20
        lr_max_epoch = 5
        learning_rate = 0.4
        weight_decay = 5e-4

        x, xt = self.normalize(x, xt)

        def lr_scheduler(epoch):
            if epoch < lr_max_epoch:
                return learning_rate * (epoch + 1) / lr_max_epoch
            else:
                return learning_rate * (maxepochs - epoch) / (maxepochs - lr_max_epoch + 1)

        reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = self.model.fit(x, y, batch_size=batch_size, epochs=maxepochs, validation_data=(xt, yt),
                                     callbacks=[reduce_lr], verbose=1)
        self.model.save_weights('cifar10resnet.weights.h5')
