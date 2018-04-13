from __future__ import print_function
from .base import Problem
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
import os
from datetime import datetime
from pygrn import RecurrentGRNLayer


class TimeRegression(Problem):

    def __init__(self, namestr=datetime.now().isoformat(), learn=True,
                 epochs=1, root_dir='./', lamarckian=False):
        pass

    def generation_function(self, grneat, generation):
        self.generation = generation
        self.error *= self.error_decrease

    def eval(self, grn):
        model = Sequential()
        layer = RecurrentGRNLayer(grn, warmup_count=1,
                                  input_shape=(self.x_train.shape[1],))
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
                            str(self.generation) + ',' + str(i) + ',' +
                            str(train_fit) + ',' + str(test_fit) + '\n')
            if self.lamarckian:
                layer.set_learned_genes()
        fit = model.evaluate(self.x_train, self.y_train, verbose=0)
        test_fit = model.evaluate(self.x_test, self.y_test, verbose=0)
        with open(self.logfile, 'a') as f:
            f.write('M,' + str(datetime.now().isoformat()) + ',' +
                    str(self.generation) + ',' + str(fit) + ',' +
                    str(test_fit) + '\n')
        del model
        K.clear_session()
        return 1.0-fit
