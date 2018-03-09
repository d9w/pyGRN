from __future__ import print_function
import numpy as np
import csv
import os
from datetime import datetime
from pygrn.problems import TimeRegression


class Energy(TimeRegression):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True,
                 root_dir='./'):

        data_file = os.path.join(root_dir, 'data/energydata_complete.csv')
        rows = []
        with open(data_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                rows.append(row)

        all_dat = []
        for i in range(1, len(rows)):
            all_dat.append([float(j) for j in rows[i][1:]])
        all_dat = np.array(all_dat)
        self.data_min = np.min(all_dat, axis=0)
        self.data_max = np.max(all_dat, axis=0)
        all_dat = (all_dat - self.data_min) / (self.data_max - self.data_min)

        winsize = 5
        data = all_dat[:, 1:-2]
        labels = all_dat[:-winsize, 0]
        windowed = data[:-winsize, :]
        for i in range(1, winsize):
            windowed = np.concatenate((windowed, data[i:-(winsize - i), :]),
                                      axis=1)

        num_train = int(3 * np.floor(windowed.shape[0] / 4))
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
        self.cacheable = False
        self.logfile = os.path.join(
            root_dir, 'logs/energy_' + namestr + '.log')
        # killfile for stopping the job
        self.killfile = os.path.join(root_dir, 'kf/' + namestr)
        open(self.killfile, 'a').close()
