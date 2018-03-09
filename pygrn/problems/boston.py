from __future__ import print_function
from .base import Problem
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD, Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing
from pygrn import GRNLayer


class Boston(Problem):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True, epochs = 1,
                 root_dir='.'):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        self.data_mins = np.min(np.vstack((np.min(self.x_train, axis=0),
                                           np.min(self.x_test, axis=0))), axis=0)
        self.data_maxs = np.max(np.vstack((np.max(self.x_train, axis=0),
                                           np.max(self.x_test, axis=0))), axis=0)

        self.label_min = np.minimum(np.min(self.y_train), np.min(self.y_test))
        self.label_max = np.maximum(np.max(self.y_train), np.max(self.y_test))

        self.x_train = (self.x_train - self.data_mins) / (self.data_maxs - self.data_mins)
        self.x_test = (self.x_test - self.data_mins) / (self.data_maxs - self.data_mins)
        self.y_train = (self.y_train - self.label_min) / (self.label_max - self.label_min)
        self.y_test = (self.y_test - self.label_min) / (self.label_max - self.label_min)

        self.batch_size = 10
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.error = 0.1
        self.error_decrease = 0.9

        self.nin = self.x_train.shape[1]
        self.nout = 1
        self.cacheable = False
        self.logfile = os.path.join(root_dir, 'logs/boston_' + namestr + '.log')
        # killfile allows stopping the job via command line due to cluster configuration
        self.killfile = os.path.join(root_dir, 'kf/' + namestr)
        open(self.killfile, 'a').close()

    def generation_function(self, grneat, generation):
        self.generation = generation
        self.error *= self.error_decrease

    def eval(self, grn):
        if not os.path.isfile(self.killfile):
            raise FileNotFoundError("Killfile has been deleted, stopping process")

        model = Sequential()
        layer = GRNLayer(grn, warmup_count=1, input_shape=(self.nin,))
        model.add(layer)

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
            # layer.set_learned_genes()
        fit = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_fit = model.evaluate(self.x_test, self.y_test, verbose=0)
        with open(self.logfile, 'a') as f:
            f.write('M,' + str(datetime.now().isoformat()) + ',' +
                    str(self.generation) + ',' + str(fit) + ',' + str(test_fit) + '\n')
        del model
        K.clear_session()
        return 1.0-fit
