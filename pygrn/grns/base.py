import numpy as np
import abc
import json


class GRN(abc.ABC):
    """Abstract class for GRNs

    Implements methods related to the protein encoding of a GRN, but requires
    subclasses to implement all methods related to dynamics.
    """
    inhibitors = []
    enhancers = []
    identifiers = []

    num_input = 0
    num_output = 0
    num_regulatory = 0

    beta = 0.5
    delta = 0.5

    id_coef = 0.75
    enh_coef = 0.125
    inh_coef = 0.125

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets concentrations and any other state variables"""
        pass

    @abc.abstractmethod
    def setup(self):
        """Calculates signatures and sets initial concentrations

        Requires that all genome variables have been set (inhibitors,
        enhancers, identifiers,num_input, num_output, num_regulatory)
        """
        pass

    @abc.abstractmethod
    def set_input(self, inputs):
        """Set the concentration of input GRNs to the provided array

        Requires that the input numpy array is bounded in [0, 1]
        """
        pass

    @abc.abstractmethod
    def step(self):
        """Runs the GRN concentration update"""
        pass

    @abc.abstractmethod
    def get_output(self):
        """Outputs a numpy array of concentrations of the output proteins"""
        pass

    @abc.abstractmethod
    def clone(self):
        """Creates a new GRN of the same type as the subclass

        Must safely pass all necessary genomic data and setup the new GRN
        """
        pass

    def __str__(self):
        return json.dumps({'ids': self.identifiers.tolist(),
                           'enh': self.enhancers.tolist(),
                           'inh': self.inhibitors.tolist(),
                           'beta': self.beta,
                           'delta': self.delta})

    def from_str(self, grn_str):
        g_dict = json.loads(grn_str)
        self.identifiers = np.array(g_dict['ids'])
        self.enhancers = np.array(g_dict['enh'])
        self.inhibitors = np.array(g_dict['inh'])
        self.num_regulatory = len(self.identifiers) - self.num_input - self.num_output
        self.beta = g_dict['beta']
        self.delta = g_dict['delta']

    def random(self, num_input, num_output, num_regulatory):
        """Sets the GRNs genome data randomly based on input sizes"""
        grn_size = num_input + num_output + num_regulatory

        self.num_input = num_input
        self.num_output = num_output
        self.num_regulatory = num_regulatory

        self.inhibitors = np.random.random([grn_size])
        self.enhancers = np.random.random([grn_size])
        self.identifiers = np.random.random([grn_size])
        self.beta = (np.random.random() * (self.beta_max - self.beta_min) +
                     self.beta_min)
        self.delta = (np.random.random() * (self.delta_max - self.delta_min) +
                      self.delta_min)
        return self

    def size(self):
        return len(self.identifiers)

    def protein_distance(self, other, k, j):
        return (abs(self.identifiers[k] - other.identifiers[j]) * self.id_coef +
                abs(self.inhibitors[k] - other.inhibitors[j]) * self.inh_coef +
                abs(self.enhancers[k] - other.enhancers[j]) * self.enh_coef)

    def distance_to(self, other):
        """Returns the distance """
        distance = 0.0
        if self.size() > other.size():
            gsmall = other
            glarge = self
        else:
            gsmall = self
            glarge = other

        # First compare inputs and outputs
        for k in range(glarge.num_input + glarge.num_output):
            gDist = glarge.protein_distance(gsmall, k, k)
            distance += gDist

        # Compare regulatory TODO: aligned
        for k in range(glarge.num_input + glarge.num_output, glarge.size()):
            if gsmall.num_regulatory == 0:
                distance += self.id_coef + self.inh_coef + self.enh_coef
            else:
                minDist = np.inf
                for j in range(gsmall.num_input + gsmall.num_output, gsmall.size()):
                    gDist = glarge.protein_distance(gsmall, k, j)
                    if minDist > gDist:
                        minDist = gDist
                distance += minDist

        # Compare dynamics
        distance += (abs(glarge.beta - gsmall.beta) /
                     (glarge.beta_max - glarge.beta_min))
        distance += (abs(glarge.delta - gsmall.delta) /
                     (gsmall.delta_max - glarge.delta_min))

        distance /= glarge.size() + 2
        return distance
