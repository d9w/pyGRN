from __future__ import print_function
from .base import Problem
import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
import csv
import os
from datetime import datetime
from sklearn import preprocessing
from pygrn import GRNLayer, RecurrentGRNLayer


class AirQuality(Problem):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True, epochs=1,
                 root_dir='/projets/reva/'):

        data_file = os.path.join(root_dir, 'data/normalized_air_quality.csv')
        all_dat = np.genfromtxt(data_file, delimiter=',')

        winsize = 5
        data = all_dat[:, 1:]
        labels = all_dat[winsize:, 0]
        windowed = data[:-winsize, :]
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize-i), :]), axis=1)

        num_train = int(3*np.floor(windowed.shape[0]/4))
        self.x_train = windowed[:num_train, :]
        self.x_test = windowed[num_train:, :]

        self.y_train = labels[:num_train]
        self.y_test = labels[num_train:]

        self.batch_size = 30
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.error = 0.1
        self.error_decrease = 0.9

        self.nin = data.shape[1]
        self.nout = 1
        self.cacheable = False # False for Lamarckian
        self.logfile = os.path.join(root_dir, 'logs/air_' + namestr + '.log')
        # killfile allows stopping the job via command line due to cluster configuration
        self.killfile = os.path.join(root_dir, 'kf/' + namestr)
        open(self.killfile, 'a').close()

    def generation_function(self, grneat, generation):
        self.generation = generation
        self.error *= self.error_decrease
        for sp in grneat.species:
            for ind in sp.individuals:
                ind.hasBeenEvaluated = False

    def eval(self, grn):
        if not os.path.isfile(self.killfile):
            raise FileNotFoundError("Killfile has been deleted, stopping process")

        model = Sequential()
        #model.add(Dense(100, input_shape=(self.x_train.shape[1],)))
        #model.add(Activation('relu'))
        #model.add(Dense(2))
        #model.add(Activation('softmax'))
        model.add(RecurrentGRNLayer(grn, warmup_count=1,
                                    input_shape=(self.x_train.shape[1],)))

        model.compile(loss='mean_squared_error', optimizer=Adam())

        if self.learn:
            history = model.fit(self.x_train, self.y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=0,
                                shuffle=True,
                                validation_data=(self.x_test, self.y_test))
            with open(self.logfile, 'a') as f:
                for i in range(len(history.history['loss'])):
                    train_fit = history.history['loss'][i]
                    test_fit = history.history['val_loss'][i]
                    f.write('L,' + str(datetime.now().isoformat()) + ',' +
                            str(self.generation) + ',' + str(i) + ',' + str(train_fit) +
                            ',' + str(test_fit) + '\n')
            # lamarckian evolution
            layer.set_learned_genes()
        fit = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_fit = model.evaluate(self.x_test, self.y_test, verbose=0)
        with open(self.logfile, 'a') as f:
            f.write('M,' + str(datetime.now().isoformat()) + ',' +
                    str(self.generation) + ',' + str(fit) + ',' + str(test_fit) + '\n')
        del model
        K.clear_session()
        return 1.0-fit
