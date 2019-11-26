from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers.recurrent import RNN
import numpy as np
from pygrn.grns import DiffGRN
from pygrn.layer import GRNInit, GRNLayer


class GRNCell(Layer):

    def __init__(self, grn_str, **kwargs):
        self.grn = DiffGRN()
        self.grn.from_str(grn_str)
        self.units = self.grn.num_output
        self.state_size = (self.grn.num_output, self.grn.num_regulatory)
        super(GRNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.identifiers = self.add_weight(
            shape=(self.grn.size(),),
            initializer=GRNInit(np.copy(self.grn.identifiers)),
            name='identifiers')
        self.enhancers = self.add_weight(
            shape=(self.grn.size(),),
            initializer=GRNInit(np.copy(self.grn.enhancers)),
            name='enhancers')
        self.inhibitors = self.add_weight(
            shape=(self.grn.size(),),
            initializer=GRNInit(np.copy(self.grn.inhibitors)),
            name='inhibitors')
        self.beta = self.add_weight(
            shape=(1,),
            initializer=initializers.Constant(value=self.grn.beta),
            name='beta')
        self.delta = self.add_weight(
            shape=(1,),
            initializer=initializers.Constant(value=self.grn.delta),
            name='delta')

        self.grn.tf_identifiers = self.identifiers
        self.grn.tf_enhancers = self.enhancers
        self.grn.tf_inhibitors = self.inhibitors
        self.grn.tf_beta = self.beta
        self.grn.tf_delta = self.delta
        self.built = True

    def set_learned_genes(self, g):
        weights = self.get_weights()
        g.identifiers = weights[0]
        g.enhancers = weights[1]
        g.inhibitors = weights[2]
        g.beta = float(weights[3][0])
        g.delta = float(weights[4][0])

    def call(self, inputs, states):
        # TODO: only works for batch size of 1
        self.grn.setup()
        start_state = K.tf.reshape(states[1][0,:], (self.grn.num_regulatory,))
        self.grn.tf_regulatory_conc = start_state
        batch_size = K.tf.shape(inputs)[0]

        self.tf_input_conc = inputs[0,:]
        self.grn.step()
        output = K.tf.stack([self.grn.tf_output_conc], axis=0)
        state = K.tf.stack([self.grn.tf_regulatory_conc], axis=0)
        # state = K.tf.tile(state, (1,batch_size,))
        # state = K.tf.reshape(state, (batch_size, self.grn.num_regulatory))
        return output, [output, state]


class RGRN(RNN):

    def __init__(self, grn_str, **kwargs):
        cell = GRNCell(grn_str)
        super(RGRN, self).__init__(cell, **kwargs)

    def reset_states(self):
        self.cell.grn.reset()
        super(RGRN, self).reset_states()

    def set_learned_genes(self, g):
        self.cell.set_learned_genes(g)

    def get_config(self):
        config = {'grn_str': str(self.grn)}
        base_config = super(RGRN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
