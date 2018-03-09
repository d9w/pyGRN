from .base import Problem
import numpy as np


class DoubleFrequency(Problem):

    def __init__(self):
        self.nin = 1
        self.nout = 1
        self.cacheable = False
        self.time = np.arange(0, 1000)
        self.signals = [self.gen_signal(p) for p in [250, 1000]]
        self.targets = [self.gen_signal(p) for p in [125, 500]]
        self.ed = [self.events(s) for s in self.targets]

    def gen_signal(self, period):
        return 0.5*np.sin((2*np.pi*self.time/period)-(np.pi/2))+0.5

    def events(self, signal):
        return np.sum(np.abs(np.diff(np.array(signal >= 0.5, dtype=float))))

    def eval(self, grn):
        self.grn_init(grn)
        fitness = 0.0
        for s in range(2):
            outsignal = np.zeros(1000)
            for t in self.time:
                grn.set_input(self.signals[s][t:t+1])
                grn.step()
                outsignal[t] = grn.get_output()[0]
            ediff = 1/(1+np.abs(self.events(outsignal)-self.ed[s])/self.ed[s])
            signaldiff = np.sum(np.abs(self.signals[s]-outsignal) *
                                (1+self.signals[s]))
            fitness -= ediff*signaldiff
        return fitness
