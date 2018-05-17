from __future__ import print_function
from .base import Problem
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import os
import pandas as pd
from datetime import datetime
from pygrn import RGRN


class Prediction(Problem):

    def __init__(self, log_file, seed=0, learn=True, batch_size=1,
                 epochs=1, data_dir='./', lamarckian=False, unsupervised=True,
                 stateful=True, model='RGRN', ntrain=75000):
        train_data_file = os.path.join(data_dir, 'kliens_train.csv')
        test_data_file = os.path.join(data_dir, 'kliens_train.csv')
        train = pd.read_csv(train_data_file).values
        test = pd.read_csv(test_data_file).values

        ntrain = len(train)-ntrain
        ntest = round(ntrain/3.0)
        X, y = train[:-ntrain, 0:-1], train[:-ntrain, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        Xtest, ytest = test[:ntest, 0:-1], test[:ntest, -1]
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
        self.stateful = stateful
        self.model_type = eval(model)
        if self.unsupervised:
            i = np.random.randint(self.x_train.shape[2])
            self.y_train = 0.5*self.x_train[:,0,i] + 0.5*np.random.rand(len(self.y_train))

        self.nin = X.shape[2]
        self.nout = 20
        self.cacheable = False
        self.logfile = log_file

    def generation_function(self, grneat, generation):
        self.generation = generation
        if self.unsupervised:
            i = np.random.randint(self.x_train.shape[2])
            self.y_train = 0.5*self.x_train[:,0,i] + 0.5*np.random.rand(len(self.y_train))

    def eval(self, grn):
        model = Sequential()
        batch_input_shape=(self.batch_size, self.x_train.shape[1], self.x_train.shape[2])
        if self.model_type == LSTM or self.model_type == SimpleRNN:
            layer = self.model_type(self.nout, stateful=True, batch_input_shape=batch_input_shape)
            model.add(layer)
        else:
            layer = self.model_type(str(grn), stateful=True, batch_input_shape=batch_input_shape)
            model.add(layer)
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        error = []
        if self.learn:
            for i in range(self.epochs):
                history = model.fit(self.x_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=1, verbose=0, shuffle=False)
                model.reset_states()
                with open(self.logfile, 'a') as f:
                    for l in range(len(history.history['loss'])):
                        train_fit = history.history['loss'][l]
                        error += [train_fit]
                        f.write('L,%s,%d,%d,%e\n' % (datetime.now().isoformat(),
                                                    self.generation, i,
                                                    train_fit))
            # lamarckian evolution
            if self.lamarckian:
                layer.set_learned_genes()
        start_error = error[0]
        end_error = error[-1]
        fit = start_error - end_error
        if not self.unsupervised:
            fit = -model.evaluate(self.x_test, self.y_test,
                                  batch_size=self.batch_size, verbose=0)

        if np.isnan(fit):
            fit = -1e10
        with open(self.logfile, 'a') as f:
            f.write('M,%s,%d,%e,%e,%e\n' % (datetime.now().isoformat(),
                                            self.generation, start_error, end_error, fit))
        del model
        K.clear_session()
        return fit
