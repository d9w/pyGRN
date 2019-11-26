from __future__ import print_function
import keras
import numpy as np
from sklearn import preprocessing
from pygrn.problems import TimeRegression


class EEG(TimeRegression):
    # currently broken

    def __init__(self, namestr='', learn=True):

        all_dat = np.genfromtxt('data/eye_eeg.csv', delimiter=',')
        winsize = 5
        data = all_dat[:, :-1]
        labels = all_dat[winsize:, -1]

        data = preprocessing.normalize(data, norm='max', axis=0)
        windowed = data[:-winsize, :]
        # windowed = data
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize-i), :]),
                                      axis=1)

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
        self.cacheable = False
        self.logfile = 'eeg_' + namestr + '.log'
        print(self.learn)
