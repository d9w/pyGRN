from __future__ import print_function
from keras.datasets import boston_housing
import numpy as np
import os
from datetime import datetime
from pygrn.problems import Regression


class Boston(Regression):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True,
                 epochs=1, root_dir='./', lamarckian=False):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (
            boston_housing.load_data())
        self.data_mins = np.min(np.vstack((np.min(self.x_train, axis=0),
                                           np.min(self.x_test, axis=0))),
                                axis=0)
        self.data_maxs = np.max(np.vstack((np.max(self.x_train, axis=0),
                                           np.max(self.x_test, axis=0))),
                                axis=0)

        self.label_min = np.minimum(np.min(self.y_train), np.min(self.y_test))
        self.label_max = np.maximum(np.max(self.y_train), np.max(self.y_test))

        self.x_train = ((self.x_train - self.data_mins) /
                        (self.data_maxs - self.data_mins))
        self.x_test = ((self.x_test - self.data_mins) /
                       (self.data_maxs - self.data_mins))
        self.y_train = ((self.y_train - self.label_min) /
                        (self.label_max - self.label_min))
        self.y_test = ((self.y_test - self.label_min) /
                       (self.label_max - self.label_min))

        self.batch_size = 10
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.error = 0.1
        self.error_decrease = 0.9
        self.lamarckian = lamarckian

        self.nin = self.x_train.shape[1]
        self.nout = 1
        self.cacheable = False
        self.logfile = os.path.join(root_dir,
                                    'logs/boston_' + namestr + '.log')
