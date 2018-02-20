from copy import deepcopy
from .base import GRN
import numpy as np
import tensorflow as tf


class DiffGRN(GRN):

    def __init__(self):
        pass

    def reset(self):
        self.tf_input_conc = tf.multiply(1.0/self.size(), tf.ones([self.num_input]))
        self.tf_output_conc =  tf.multiply(1.0/self.size(), tf.ones([self.num_output]))
        self.tf_regulatory_conc =  tf.multiply(1.0/self.size(), tf.ones([self.num_regulatory]))
        return self


    def warmup(self, nsteps):
        self.set_input(np.zeros(self.num_input))
        for i in range(nsteps):
            self.step()

    def convert_to_tensor(self):
        self.tf_identifiers = tf.convert_to_tensor(self.identifiers, dtype=tf.float32)
        self.tf_enhancers = tf.convert_to_tensor(self.enhancers, dtype=tf.float32)
        self.tf_inhibitors = tf.convert_to_tensor(self.inhibitors, dtype=tf.float32)
        self.tf_beta = tf.convert_to_tensor(self.beta, dtype=tf.float32)
        self.tf_delta = tf.convert_to_tensor(self.delta, dtype=tf.float32)

    def setup(self):
        self.reset()
        ids = tf.maximum(0.0, tf.minimum(1.0, self.tf_identifiers))
        enh = tf.maximum(0.0, tf.minimum(1.0, self.tf_enhancers))
        inh = tf.maximum(0.0, tf.minimum(1.0, self.tf_inhibitors))
        beta = tf.maximum(self.beta_min, tf.minimum(self.beta_max, self.tf_beta))
        delta = tf.maximum(self.delta_min, tf.minimum(self.delta_max, self.tf_delta))

        ids = tf.reshape(
            tf.tile(ids, [self.size()]), [self.size(), self.size()])
        enh = tf.transpose(tf.reshape(
            tf.tile(enh, [self.size()]), [self.size(), self.size()]))
        inh = tf.transpose(tf.reshape(
            tf.tile(inh, [self.size()]), [self.size(), self.size()]))
        self.tf_enhance_match = tf.exp(-beta * tf.abs(enh - ids))
        self.tf_inhibit_match = tf.exp(-beta * tf.abs(inh - ids))
        self.tf_sigs = self.tf_enhance_match - self.tf_inhibit_match
        self.tf_delta_n = tf.divide(delta, tf.to_float(self.size()))
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
        concs = tf.concat([self.tf_input_conc, self.tf_output_conc, self.tf_regulatory_conc],
                          0)
        conc_diff = tf.multiply(concs, self.tf_output_mask)
        conc_diff = tf.reshape(conc_diff, [1, self.size()])
        conc_diff = tf.matmul(conc_diff, self.tf_sigs)
        conc_diff = tf.multiply(self.tf_delta_n, conc_diff)
        concs = tf.add(concs, conc_diff)
        concs = tf.maximum(0.0, concs)
        concs = tf.reshape(concs, [self.size()])
        _, regs = tf.split(concs, [self.num_input, self.num_regulatory+self.num_output])
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

    def input_step(self, inputs):
        self.set_input(inputs)
        self.step()
        return self.get_output_tensor()

    def clone(self):
        g = DiffGRN()

        g.identifiers = deepcopy(self.identifiers)
        g.enhancers = deepcopy(self.enhancers)
        g.inhibitors = deepcopy(self.inhibitors)
        g.beta = deepcopy(self.beta)
        g.delta = deepcopy(self.delta)

        g.num_input = deepcopy(self.num_input)
        g.num_output = deepcopy(self.num_output)
        g.num_regulatory = deepcopy(self.num_regulatory)

        # TODO: move globals somewhere else
        g.beta_min = deepcopy(self.beta_min)
        g.beta_max = deepcopy(self.beta_max)
        g.delta_min = deepcopy(self.delta_min)
        g.delta_max = deepcopy(self.delta_max)
        g.id_coef = deepcopy(self.id_coef)
        g.enh_coef = deepcopy(self.enh_coef)
        g.inh_coef = deepcopy(self.inh_coef)
        return g
