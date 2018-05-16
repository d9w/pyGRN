from __future__ import print_function
from .base import Problem
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import os
from datetime import datetime
from pygrn import RGRN


class Prediction(Problem):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True, batch_size=1,
                 epochs=1, root_dir='./', lamarckian=False, unsupervised=True):
        train_data_file = os.path.join(root_dir, 'data/kliens_train.csv')
        test_data_file = os.path.join(root_dir, 'data/kliens_train.csv')
        train = np.genfromtxt(train_data_file, delimiter=',')
        test = np.genfromtxt(test_data_file, delimiter=',')

        X, y = train[:batch_size*10, 0:-1], train[:batch_size*10, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        Xtest, ytest = test[:batch_size*10, 0:-1], test[:batch_size*10, -1]
        Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])

        self.x_train = X
        self.y_train = y
        self.x_test = Xtest
        self.y_test = ytest

        self.batch_size = batch_size
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.lamarckian = lamarckian
        self.unsupervised = unsupervised
        if self.unsupervised:
            self.y_train = np.random.rand(len(self.y_train))

        self.nin = X.shape[2]
        self.nout = 20
        self.cacheable = False
        self.logfile = os.path.join(root_dir, 'logs/pred_' + namestr + '.log')

    def generation_function(self, grneat, generation):
        self.generation = generation
        if self.unsupervised:
            i = np.random.randint(self.x_train.shape[1])
            self.y_train = 0.5*self.x_train[:,i] + 0.5*np.random.rand(len(self.y_train))
            self.y_test = 0.5*self.x_test[:,i] + 0.5*np.random.rand(len(self.y_test))

    def eval(self, grn):
        model = Sequential()
        layer = RGRN(str(grn), stateful=True,
                     batch_input_shape=(self.batch_size, self.x_train.shape[1],
                                        self.x_train.shape[2]))
        model.add(layer)
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam())

        start_error = model.evaluate(self.x_train, self.y_train,
                                   batch_size=self.batch_size, verbose=0)
        if self.learn:
            for i in range(self.epochs):
                history = model.fit(self.x_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=1, verbose=0, shuffle=False)
                model.reset_states()
                with open(self.logfile, 'a') as f:
                    for i in range(len(history.history['loss'])):
                        train_fit = history.history['loss'][i]
                        f.write('L,%s,%d,%d,%f\n' % (datetime.now().isoformat(),
                                                    self.generation, i,
                                                    train_fit))
            # lamarckian evolution
            if self.lamarckian:
                layer.set_learned_genes()
        end_error = model.evaluate(self.x_train, self.y_train,
                                 batch_size=self.batch_size, verbose=0)
        # end_fit = mean_squared_error(self.y_train, y)
        fit = start_error - end_error
        with open(self.logfile, 'a') as f:
            f.write('M,%s,%d,%f,%f,%f\n' % (datetime.now().isoformat(),
                                            self.generation, start_error, end_error, fit))
        del model
        K.clear_session()
        return fit
