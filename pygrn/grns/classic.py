import numpy as np
from copy import deepcopy
from .base import GRN


class ClassicGRN(GRN):
    """Classic CPU-based GRN

    Dynamics equations are written mostly in loop form
    """
    concentration = []
    next_concentration = []
    enhance_match = []
    inhibit_match = []

    def __init__(self):
        pass

    def reset(self):
        self.concentration = np.ones(
            len(self.identifiers)) * (1.0/len(self.identifiers))
        self.next_concentration = np.zeros(len(self.identifiers))
        return self

    def warmup(self, nsteps):
        self.set_input(np.zeros(self.num_input))
        for i in range(nsteps):
            self.step()

    def setup(self):
        self.inhibit_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        self.enhance_match = np.zeros(
            [len(self.identifiers), len(self.identifiers)])
        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = np.abs(
                    self.enhancers[k] - self.identifiers[j])
                self.inhibit_match[k, j] = np.abs(
                    self.inhibitors[k] - self.identifiers[j])

        for k in range(len(self.identifiers)):
            for j in range(len(self.identifiers)):
                self.enhance_match[k, j] = np.exp(
                    - self.beta * self.enhance_match[k, j])
                self.inhibit_match[k, j] = np.exp(
                    - self.beta * self.inhibit_match[k, j])

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
        if len(self.next_concentration) != len(self.concentration):
            self.next_concentration = np.zeros(len(self.concentration))

        sum_concentration = 0.0
        for k in range(len(self.identifiers)):
            if k < self.num_input:
                self.next_concentration[k] = self.concentration[k]
            else:
                enhance = 0.0
                inhibit = 0.0
                for j in range(len(self.identifiers)):
                    if not (j >= self.num_input and
                            j < (self.num_output + self.num_input)):
                        enhance += self.concentration[j] * self.enhance_match[j,k]
                        inhibit += self.concentration[j] * self.inhibit_match[j,k]
                diff = self.delta / len(self.identifiers) * (enhance - inhibit)
                self.next_concentration[k] = max(0.0, self.concentration[k] + diff)
                sum_concentration += self.next_concentration[k]
        if sum_concentration > 0:
            for k in range(len(self.identifiers)):
                if k >= self.num_input:
                    self.next_concentration[k] = min(
                        1.0, self.next_concentration[k] / sum_concentration)

        self.concentration = self.next_concentration
        return self

    def clone(self):
        return deepcopy(self)
