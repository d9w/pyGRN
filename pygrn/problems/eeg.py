from __future__ import print_function
from .base import Problem
import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, SGD
from keras import backend as K
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from pygrn import GRNLayer, RecurrentGRNLayer


class EEG(Problem):
    # currently broken

    def __init__(self, namestr='', learn=True):

        all_dat = np.genfromtxt('/home/d9w/pyGRN/data/eye_eeg.csv', delimiter=',')
        winsize = 5
        data = all_dat[:, :-1]
        labels = all_dat[winsize:, -1]

        data = preprocessing.normalize(data, norm='max', axis=0)
        windowed = data[:-winsize, :]
        # windowed = data
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize-i), :]), axis=1)

        num_train = int(np.floor(windowed.shape[0]/2))
        self.x_train = windowed[:num_train, :]
        self.x_test = windowed[num_train:, :]

        labels = keras.utils.to_categorical(labels, 2)
        self.y_train = labels[:num_train, :]
        self.y_test = labels[num_train:, :]

        self.batch_size = 10
        self.epochs = 20
        self.learn = learn
        self.generation = 0

        self.nin = windowed.shape[1]
        self.nout = 2
        self.cacheable = True
        self.logfile = 'eeg_' + namestr + '.log'
        print(self.learn)

    def generation_function(self, grneat, generation):
        self.generation += 1
        # TODO: deterministic fitness
        for sp in grneat.species:
            for ind in sp.individuals:
                ind.hasBeenEvaluated = False

    def eval(self, grn):
        model = Sequential()
        model.add(Dense(100, input_shape=(self.x_train.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        #model.add(RecurrentGRNLayer(grn, input_shape=(self.x_train.shape[1],)))
        #model.add(GRNLayer(grn, input_shape=(self.x_train.shape[1],)))
        #model.add(Dense(2))
        #model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer=SGD(lr=0.1),
                    metrics=['accuracy'])

        fit = -np.inf
        print(self.learn)

        if self.learn:

            history = model.fit(self.x_train, self.y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=1,
                                shuffle=False,
                                validation_data=(self.x_test, self.y_test))
            fit = history.history['acc'][-1]
            with open(self.logfile, 'a') as f:
                for i in range(len(history.history['acc'])):
                    train_fit = history.history['acc'][i]
                    test_fit = history.history['val_acc'][i]
                    f.write('L,' + str(datetime.now().isoformat()) + ',' +
                            str(self.generation) + ',' + str(i) + ',' + str(train_fit) +
                            ',' + str(test_fit) +'\n')
        else:
            fit = model.evaluate(self.x_train, self.y_train, verbose=0)[1]
            test_fit = model.evaluate(self.x_test, self.y_test, verbose=0)[1]
            with open(self.logfile, 'a') as f:
                f.write('E,' + str(datetime.now().isoformat()) + ',' +
                        str(self.generation) + ',0,' + str(fit) +
                        ',' + str(test_fit) + '\n')
        del model
        K.clear_session()
        return fit

