import tensorflow as tf
import numpy as np
from copy import deepcopy
from pygrn import grns


def test_simple():
    x = tf.Variable(np.random.rand(), name='x', dtype=tf.float32)
    log_x = tf.log(x)
    log_x_squared = tf.square(log_x)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(log_x_squared)

    init = tf.variables_initializer(tf.global_variables())

    with tf.Session() as session:
        session.run(init)
        print("starting at", "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
        for step in range(10):
            session.run(train)
            print("step", step, "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))


def diff_square(nin, nout, grn):
    inps = tf.divide(tf.range(0, nin, 1, dtype=tf.float32), nin)
    # expected, _ = tf.split(tf.sin(inps), [nout, nin-nout])
    error = tf.subtract(tf.sin(inps), grn.input_step(inps))
    outs = tf.reduce_sum(tf.multiply(error, error))
    outs = tf.Print(outs, [outs], "Loss: ")
    outs = tf.cond(tf.greater(1, 0), lambda: outs, lambda: outs)
    return outs


def test_diff_grn():
    nin = 10; nout = 10; nreg = 100;
    glength = nin + nout + nreg
    grn = grns.DiffGRN()
    grn.random(nin, nout, nreg)
    grn.tf_identifiers = tf.Variable(np.random.rand(glength), dtype = tf.float32)
    grn.tf_enhancers = tf.Variable(np.random.rand(glength), dtype = tf.float32)
    grn.tf_inhibitors = tf.Variable(np.random.rand(glength), dtype = tf.float32)
    grn.tf_beta = tf.Variable(grn.beta_min + grn.beta_max * np.random.rand(),
                           dtype = tf.float32)
    grn.tf_delta = tf.Variable(grn.delta_min + grn.delta_max * np.random.rand(),
                           dtype = tf.float32)

    grn.setup()

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(diff_square(nin, nout, grn))

    init = tf.variables_initializer(tf.global_variables())

    with tf.Session() as session:
        session.run(init)
        start_ids = deepcopy(session.run(grn.tf_identifiers))
        start_enh = deepcopy(session.run(grn.tf_enhancers))
        start_inh = deepcopy(session.run(grn.tf_inhibitors))
        start_beta = deepcopy(session.run(grn.tf_beta))
        start_delta = deepcopy(session.run(grn.tf_delta))
        for step in range(1000):
            session.run(train)
        end_ids = session.run(grn.tf_identifiers)
        end_enh = session.run(grn.tf_enhancers)
        end_inh = session.run(grn.tf_inhibitors)
        end_beta = session.run(grn.tf_beta)
        end_delta = session.run(grn.tf_delta)
        assert np.any(start_ids != end_ids)
        assert np.any(start_enh != end_enh)
        assert np.any(start_inh != end_inh)
        assert (np.abs(start_beta - end_beta) + np.abs(start_delta - end_delta)) > 0
