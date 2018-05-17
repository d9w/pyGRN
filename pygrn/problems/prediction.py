from __future__ import print_function
from .base import Problem
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
from datetime import datetime
from pygrn import RGRN


class Prediction(Problem):

    def __init__(self, log_file, seed=0, learn=True, batch_size=1,
                 epochs=1, data_dir='./', lamarckian=False, unsupervised=True,
                 stateful=True, model='RGRN', ntrain=6*24*60, ntest=24*60,
                 shift=1, lag=60):
        rawvals = pd.read_csv(os.path.join(data_dir, 'kliens_raw.csv'))
        self.df = rawvals
        if shift != 0:
            self.df = rawvals - rawvals.shift(-1)
        self.df['target'] = rawvals['close'] - rawvals['close'].shift(-shift-lag)
        self.df.dropna(inplace=True)

        train = self.df.tail(ntrain+ntest).head(ntrain)
        test = self.df.tail(ntest)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(train)
        train = self.scaler.transform(train)
        test = self.scaler.transform(test)

        self.x_train = train[:, 0:-1]
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.x_train.shape[1])
        self.y_train = train[:, -1]

        self.x_test = test[:, 0:-1]
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.x_test.shape[1])
        self.y_test = test[:, -1]

        self.batch_size = batch_size
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.ntrain = ntrain
        self.ntest = ntest
        self.shift = shift
        self.lag = lag
        self.seed = seed
        self.lamarckian = lamarckian
        self.unsupervised = unsupervised
        self.stateful = stateful
        self.model_type = eval(model)
        if self.unsupervised:
            i = np.random.randint(self.x_train.shape[2])
            self.y_train = 0.5*self.x_train[:,0,i] + 0.5*np.random.rand(len(self.y_train))

        self.nin = self.x_train.shape[2]
        self.nout = 10
        self.cacheable = False
        self.logfile = log_file

    def generation_function(self, grneat, generation):
        self.generation = generation
        if self.unsupervised:
            i = np.random.randint(self.x_train.shape[2])
            self.y_train = 0.5*self.x_train[:,0,i] + 0.5*np.random.rand(len(self.y_train))

    def eval(self, grn):
        seed = np.random.randint(1e5)
        model = Sequential()
        batch_input_shape=(self.batch_size, self.x_train.shape[1], self.x_train.shape[2])
        start_time = datetime.now()
        if self.model_type == LSTM or self.model_type == SimpleRNN:
            np.random.seed(int(np.round(grn.identifiers[0]*100)))
            layer = self.model_type(self.nout, stateful=self.stateful,
                                    batch_input_shape=batch_input_shape)
            model.add(layer)
        else:
            np.random.seed(self.seed)
            layer = self.model_type(str(grn), stateful=self.stateful,
                                    batch_input_shape=batch_input_shape)
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
                        f.write('L,%e,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e\n' % (
                            (datetime.now()-start_time).total_seconds(),
                            self.seed, self.model_type.__name__, self.epochs,
                            self.lamarckian, self.unsupervised, self.stateful,
                            self.ntrain, self.ntest, self.shift, self.lag,
                            self.generation, i, train_fit))
            # lamarckian evolution
            if self.lamarckian:
                layer.set_learned_genes(grn)
        start_error = error[0]
        end_error = error[-1]
        fit = start_error - end_error
        total_error = np.sum(np.abs(error))
        if not self.unsupervised:
            # predict and return unscaled difference
            ntest = len(self.y_test)
            yhat = model.predict(self.x_test, batch_size=self.batch_size, verbose=0)
            final = np.concatenate((
                np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[2])),
                np.reshape(yhat, (ntest,1))), axis=1)
            inverted = self.scaler.inverse_transform(final)
            yhatp = inverted[:, -1]
            target = self.df['target'].tail(ntest)
            fit = np.sqrt(mean_squared_error(yhatp, target))
            total_error = np.sum(np.abs(yhatp - target))
        if np.isnan(fit):
            fit = -1e10
        with open(self.logfile, 'a') as f:
            f.write('M,%e,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e,%e,%e,%e\n' % (
                (datetime.now()-start_time).total_seconds(),
                self.seed, self.model_type.__name__, self.epochs,
                self.lamarckian, self.unsupervised, self.stateful,
                self.ntrain, self.ntest, self.shift, self.lag,
                self.generation, start_error, end_error, total_error, fit))
        del model
        K.clear_session()
        np.random.seed(seed)
        return fit
