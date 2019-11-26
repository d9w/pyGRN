from copy import deepcopy
from .classic import ClassicGRN
import numpy as np
import tensorflow as tf


class GPUGRN(ClassicGRN):

    def __init__(self):
        pass

    def reset(self):
        self.concentration = np.ones(
            len(self.identifiers)) * (1.0/len(self.identifiers))
        self.tf_input_conc = tf.convert_to_tensor(
            self.concentration[0:self.num_input], dtype=tf.float32)
        self.tf_output_conc = tf.convert_to_tensor(
            self.concentration[self.num_input:(self.num_input +
                                               self.num_output)],
            dtype=tf.float32)
        self.tf_regulatory_conc = tf.convert_to_tensor(
            self.concentration[self.num_input+self.num_output:],
            dtype=tf.float32)
        return self

    def warmup(self, nsteps):
        self.concentration[0:self.num_input] = np.zeros(self.num_input)
        for i in range(nsteps):
            super(GPUGRN, self).step()
        self.tf_input_conc = tf.convert_to_tensor(
            self.concentration[0:self.num_input], dtype=tf.float32)
        self.tf_output_conc = tf.convert_to_tensor(
            self.concentration[self.num_input:(self.num_input +
                                               self.num_output)],
            dtype=tf.float32)
        self.tf_regulatory_conc = tf.convert_to_tensor(
            self.concentration[self.num_input+self.num_output:],
            dtype=tf.float32)

    def setup(self):
        super(GPUGRN, self).setup()
        self.length = self.num_input + self.num_output + self.num_regulatory
        self.tf_input_conc = tf.convert_to_tensor(
            self.concentration[0:self.num_input], dtype=tf.float32)
        self.tf_output_conc = tf.convert_to_tensor(
            self.concentration[self.num_input:(self.num_input +
                                               self.num_output)],
            dtype=tf.float32)
        self.tf_regulatory_conc = tf.convert_to_tensor(
            self.concentration[self.num_input+self.num_output:],
            dtype=tf.float32)
        self.tf_sigs = tf.convert_to_tensor(self.enhance_match -
                                            self.inhibit_match,
                                            dtype=tf.float32)
        self.tf_beta = tf.convert_to_tensor(self.beta, dtype=tf.float32)
        self.tf_delta_n = tf.convert_to_tensor(self.delta/self.length,
                                               dtype=tf.float32)
        self.tf_output_mask = tf.convert_to_tensor(
            np.concatenate((np.ones(self.num_input),
                            np.zeros(self.num_output),
                            np.ones(self.num_regulatory))),
            dtype=tf.float32)

    def get_signatures(self):
        with tf.Session() as s:
            return s.run(self.tf_sigs)

    def get_concentrations(self):
        with tf.Session() as s:
            return s.run(tf.concat([self.tf_input_conc,
                                    self.tf_output_conc,
                                    self.tf_regulatory_conc], 0))

    def set_input(self, input_t):
        inp_concs = tf.convert_to_tensor(input_t, dtype=tf.float32)
        self.tf_input_conc = inp_concs

    def step(self):
        concs = tf.concat([self.tf_input_conc, self.tf_output_conc,
                           self.tf_regulatory_conc], 0)
        conc_diff = tf.multiply(concs, self.tf_output_mask)
        conc_diff = tf.reshape(conc_diff, [1, self.length])
        conc_diff = tf.matmul(conc_diff, self.tf_sigs)
        conc_diff = tf.multiply(self.tf_delta_n, conc_diff)
        concs = tf.add(concs, conc_diff)
        concs = tf.maximum(0.0, concs)
        concs = tf.reshape(concs, [self.length])
        _, regs = tf.split(concs, [self.num_input,
                                   self.num_regulatory+self.num_output])
        sumconcs = tf.reduce_sum(regs)
        concs = tf.cond(tf.greater(sumconcs, 0),
                        lambda: tf.div(concs, sumconcs), lambda: concs)
        _, self.tf_output_conc, self.tf_regulatory_conc = tf.split(
            concs, [self.num_input, self.num_output, self.num_regulatory])

    def get_output_tensor(self):
        return self.tf_output_conc

    def get_output(self):
        with tf.Session() as s:
            return s.run(self.get_output_tensor())

    def clone(self):
        g = GPUGRN()
        g.identifiers = deepcopy(self.identifiers)
        g.enhancers = deepcopy(self.enhancers)
        g.inhibitors = deepcopy(self.inhibitors)
        g.beta = deepcopy(self.beta)
        g.delta = deepcopy(self.delta)

        g.num_input = deepcopy(self.num_input)
        g.num_output = deepcopy(self.num_output)
        g.num_regulatory = deepcopy(self.num_regulatory)
        return g
