from keras import backend as K
from keras import initializers
from keras.engine import Layer
from keras.engine import InputSpec
import numpy as np


class GRNInit(initializers.Initializer):
    def __init__(self, proteins):
        self.proteins = proteins

    def __call__(self, shape, dtype=None):
        vals = K.tf.convert_to_tensor(self.proteins, dtype=K.tf.float32)
        return vals

    def get_config(self):
        return {}


class GRNLayer(Layer):

    def __init__(self, grn, warmup_count=25, **kwargs):
        super(GRNLayer, self).__init__(**kwargs)
        self.grn = grn
        self.warmup_count = warmup_count

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

        self.input_spec = InputSpec(min_ndim=2)
        self.built = True

    def set_learned_genes(self):
        weights = self.get_weights()
        self.grn.identifiers = weights[0]
        self.grn.enhancers = weights[1]
        self.grn.inhibitors = weights[2]
        self.grn.beta = float(weights[3][0])
        self.grn.delta = float(weights[4][0])

    def call(self, inputs):
        self.grn.setup()
        self.grn.warmup(self.warmup_count)
        out_conc = self.grn.tf_output_conc
        reg_conc = self.grn.tf_regulatory_conc

        def grn_func(inp):
            self.grn.tf_input_conc = inp
            self.grn.tf_output_conc = out_conc
            self.grn.tf_regulatory_conc = reg_conc
            self.grn.step()
            return self.grn.tf_output_conc

        return K.tf.map_fn(grn_func, inputs)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.grn.num_output
        return tuple(output_shape)

    def get_config(self):
        # TODO: useful config
        config = {'florp': False}
        base_config = super(GRNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RecurrentGRNLayer(GRNLayer):

    def call(self, inputs):
        self.grn.setup()
        self.grn.warmup(self.warmup_count)
        out_conc = self.grn.tf_output_conc
        reg_conc = self.grn.tf_regulatory_conc

        def grn_split(inp):
            inp = K.tf.reshape(inp, [-1, self.grn.num_input])
            self.grn.tf_output_conc = out_conc
            self.grn.tf_regulatory_conc = reg_conc
            out = K.tf.map_fn(grn_run, inp)
            return out[-1]

        def grn_run(inp):
            self.grn.tf_input_conc = inp
            self.grn.step()
            return self.grn.tf_output_conc

        return K.tf.map_fn(grn_split, inputs)
