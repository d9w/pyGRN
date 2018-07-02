from __future__ import print_function
from .base import Problem
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LambdaCallback
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
import random
import sys
from datetime import datetime
from pygrn import RGRN

class TextGen(Problem):

    def __init__(self, log_file, seed=0, learn=True, batch_size=1,
                 epochs=1, data_dir='./', lamarckian=False,
                 stateful=True, model='RGRN'):
        with open(os.path.join(data_dir, 'nietzsche.txt'), encoding='utf-8') as f:
            text = f.read().lower()
        print('corpus length:', len(text))

        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = 40
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))
        self.maxlen = maxlen
        self.chars = chars

        print('Vectorization...')
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn = learn
        self.generation = 0
        self.seed = seed
        self.lamarckian = lamarckian
        self.stateful = stateful
        self.model_type = eval(model)

        self.nin = len(chars)
        self.nout = 128
        self.cacheable = False
        self.logfile = log_file

    def generation_function(self, grneat, generation):
        self.generation = generation

    def eval(self, grn):
        seed = np.random.randint(1e5)
        np.random.seed(self.seed)
        model = Sequential()
        batch_input_shape = (self.batch_size, self.maxlen, len(self.chars))
        start_time = datetime.now()
        if self.model_type == LSTM or self.model_type == SimpleRNN:
            layer = self.model_type(self.nout, stateful=self.stateful,
                                    batch_input_shape=batch_input_shape)
            model.add(layer)
        else:
            layer = self.model_type(str(grn), stateful=self.stateful,
                                    batch_input_shape=batch_input_shape)
            model.add(layer)
        model.add(Dense(len(self.chars)))
        model.add(Activation('softmax'))
        model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.01))

        if self.learn:
            for i in range(self.epochs): # to reset states between each epoch
                history = model.fit(self.x, self.y,
                                    batch_size=self.batch_size,
                                    epochs=1, verbose=1, shuffle=False)
                model.reset_states()
                with open(self.logfile, 'a') as f:
                    for l in range(len(history.history['loss'])):
                        train_fit = history.history['loss'][l]
                        f.write('L,%e,%d,%s,%d,%d,%d,%d,%d,%e\n' % (
                            (datetime.now()-start_time).total_seconds(),
                            self.seed, self.model_type.__name__, self.epochs,
                            self.lamarckian, self.stateful,
                            self.generation, i, train_fit))
            # lamarckian evolution
            if self.lamarckian:
                layer.set_learned_genes(grn)
        with open(self.logfile, 'a') as f:
            f.write('M,%e,%d,%s,%d,%d,%d,%d,%e,%e,%e,%e\n' % (
                (datetime.now()-start_time).total_seconds(),
                self.seed, self.model_type.__name__, self.epochs,
                self.lamarckian, self.stateful,
                self.generation, start_error, end_error, total_error, fit))
        del model
        K.clear_session()
        np.random.seed(seed)
        return fit


