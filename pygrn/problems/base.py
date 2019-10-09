import abc
import numpy as np
from datetime import datetime


class Problem(abc.ABC):
    nin = 0
    nout = 0
    cacheable = True

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

    def generation_function(self, grneat, generation):
        for sp in grneat.species:
            for ind in sp.individuals:
                ind.fitness = ind.getFitness(sp.problem) + 1.0

    def eval(self, grn):
        return np.random.rand()


class Static(Problem):
    nin = 1
    nout = 1
    cacheable = False

    def eval(self, grn):
        return 0.0


class Counter(Problem):
    nin = 11
    nout = 4
    cacheable = True
    count = 0

    def __init__(self, namestr=''):
        self.logfile = 'fits_' + namestr + '.log'
        self.count = 0

    def eval(self, grn):
        self.count += 1
        return np.random.rand()
