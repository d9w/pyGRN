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
import csv
import os
from datetime import datetime
from sklearn import preprocessing
from pygrn import GRNLayer, RecurrentGRNLayer


class Energy(Problem):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True,
                 root_dir='/projets/reva/'):

        data_file = os.path.join(root_dir, 'data/energydata_complete.csv')
        rows = []
        with open(data_file, newline='') as csvfile:
             reader = csv.reader(csvfile, delimiter=',')
             for row in reader:
                 rows.append(row);

        all_dat = [];
        for i in range(1,len(rows)):
             all_dat.append([float(j) for j in rows[i][1:]])
        all_dat = np.array(all_dat)
        self.data_min = np.min(all_dat, axis=0)
        self.data_max = np.max(all_dat, axis=0)
        all_dat = (all_dat - self.data_min)/(self.data_max - self.data_min)

        winsize = 5
        data = all_dat[:, 1:-2]
        labels = all_dat[:-winsize, 0]
        windowed = data[:-winsize, :]
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize-i), :]), axis=1)

        num_train = int(3*np.floor(windowed.shape[0]/4))
        self.x_train = windowed[:num_train, :]
        self.x_test = windowed[num_train:, :]

        self.y_train = labels[:num_train]
        self.y_test = labels[num_train:]

        self.batch_size = 30
        self.epochs = 1
        self.learn = learn
        self.generation = 0
        self.error = 0.1
        self.error_decrease = 0.9

        self.nin = data.shape[1]
        self.nout = 1
        self.cacheable = True
        self.logfile = os.path.join(root_dir, 'logs/energy_' + namestr + '.log')
        # killfile allows stopping the job via command line due to cluster configuration
        self.killfile = os.path.join(root_dir, 'kf/' + namestr)
        open(self.killfile, 'a').close()

    def generation_function(self, grneat, generation):
        self.generation = generation
        self.error *= self.error_decrease
        # TODO: deterministic fitness
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
        layer = RecurrentGRNLayer(grn, warmup_count=1,
                                    input_shape=(self.x_train.shape[1],))
        model.add(layer)

        model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.1))

        if self.learn:
            history = model.fit(self.x_train, self.y_train,
                                batch_size=self.batch_size,
                                epochs=self.epochs,
                                verbose=0,
                                shuffle=False,
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
        norm_labels = model.predict(self.x_train, batch_size=self.batch_size)[:, 0]
        acc = np.sum(np.abs(norm_labels - self.y_train) < self.error)/len(self.y_train)
        rmse = np.sqrt(np.mean(((norm_labels - self.y_train) *
                               (self.data_max[0] - self.data_min[0]))**2))
        test_fit = model.evaluate(self.x_test, self.y_test, verbose=0)
        with open(self.logfile, 'a') as f:
            f.write('M,' + str(datetime.now().isoformat()) + ',' +
                    str(self.generation) + ',' + str(rmse) + ',' +
                    str(acc) + ',' + str(test_fit) + '\n')
        del model
        K.clear_session()
        return acc
