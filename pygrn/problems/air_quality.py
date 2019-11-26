from __future__ import print_function
import numpy as np
import os
from datetime import datetime
from pygrn.problems import TimeRegression


class AirQuality(TimeRegression):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True,
                 epochs=1, root_dir='./', lamarckian=False):
        data_file = os.path.join(root_dir, 'data/normalized_air_quality.csv')
        all_dat = np.genfromtxt(data_file, delimiter=',')

        winsize = 5
        data = all_dat[:, 1:]
        labels = all_dat[winsize:, 0]
        windowed = data[:-winsize, :]
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize-i), :]),
                                      axis=1)

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
        self.lamarckian = lamarckian

        self.nin = data.shape[1]
        self.nout = 1
        self.cacheable = False
        self.logfile = os.path.join(root_dir, 'logs/air_' + namestr + '.log')
