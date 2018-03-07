import abc
import numpy as np
import tensorflow as tf
from datetime import datetime


class Problem(abc.ABC):
    nin = 0
    nout = 0
    cacheable = True
    fit_range = [0.0, 1.0]

    def __init__(self, namestr=''):
        self.logfile = 'fits_' + namestr + '.log'

    @abc.abstractmethod
    def eval(self, grn):
        pass

    def generation_function(self, grneat, generation):
        pass

    def grn_init(self, grn):
        grn.setup()
        grn.warmup(25)


class Random(Problem):
    nin = 1
    nout = 1
    cacheable = True
    fit_range = [-0.5, 0.5]

    def generation_function(self, grneat, generation):
        for sp in grneat.species:
            for ind in sp.individuals:
                ind.fitness = ind.getFitness(sp.problem) + 1.0

    def eval(self, grn):
        return np.random.rand()

class TFRandom(Problem):
    nin = 10
    nout = 5
    cacheable = False
    fit_range = [-0.5, 0.5]

    def eval(self, grn):
        grn.setup()
        grn.warmup(25)
        fit = 0.0
        for i in range(100):
            t1 = datetime.now()
            grn.set_input(np.random.rand(10))
            grn.step()
            with tf.Session() as sess:
                fit += sess.run(tf.reduce_sum(grn.get_output_tensor()))
                print(len(sess.graph.get_operations()))
            t2 = datetime.now()
            print(t2 - t1)
        print(len(tf.get_default_graph().get_operations()))
        return fit


class Static(Problem):
    nin = 1
    nout = 1
    cacheable = False
    fit_range = [-0.5, 0.5]

    def eval(self, grn):
        return 0.0


class Counter(Problem):
    nin = 11
    nout = 4
    cacheable = True
    count = 0
    fit_range = [-0.5, 0.5]

    def __init__(self, namestr=''):
        self.logfile = 'fits_' + namestr + '.log'
        self.count = 0

    def eval(self, grn):
        self.count += 1
        return np.random.rand()


