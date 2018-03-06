import numpy as np
from copy import deepcopy
from .base import GRN


class MatrixGRN(GRN):
    """Matrix CPU-based GRN"""
    concentration = []
    enhance_match = []
    inhibit_match = []

    def __init__(self):
        pass

    def reset(self):
        self.concentration = np.ones(
            len(self.identifiers)) * (1.0/len(self.identifiers))
        return self

    def warmup(self, nsteps):
        self.set_input(np.zeros(self.num_input))
        for i in range(nsteps):
            self.step()

    def setup(self):
        ids = np.tile(self.identifiers, [len(self.identifiers), 1])
        enh = np.transpose(np.tile(self.enhancers, [len(self.identifiers), 1]))
        inh = np.transpose(np.tile(self.inhibitors, [len(self.identifiers), 1]))
        self.enhance_match = np.exp(-self.beta * np.abs(enh - ids))
        self.inhibit_match = np.exp(-self.beta * np.abs(inh - ids))

        self.reset()

    def get_signatures(self):
        return self.enhance_match - self.inhibit_match

    def get_concentrations(self):
        return self.concentration

    def set_input(self, inputs):
        self.concentration[0:self.num_input] = inputs
        return self

    def get_output(self):
        return self.concentration[self.num_input:(
            self.num_output + self.num_input)]

    def step(self):
        regulation = deepcopy(self.concentration)
        regulation[self.num_input:(self.num_input+self.num_output)] = (
            np.zeros(self.num_output))
        c_diff = self.delta / len(self.identifiers) * np.dot(
            np.transpose(regulation), self.enhance_match - self.inhibit_match)
        print("Diff: ", c_diff.tolist())
        self.concentration = np.maximum(0.0, self.concentration + c_diff)
        sumconc = sum(self.concentration[self.num_input:])
        if sumconc > 0:
            self.concentration /= sumconc
        self.concentration[0:self.num_input] = regulation[0:self.num_input]
        return self

    def clone(self):
        return deepcopy(self)
