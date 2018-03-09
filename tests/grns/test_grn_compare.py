from copy import deepcopy
import numpy as np
from pygrn import grns
import tensorflow as tf

r_error = 1e-2


def compare_grns(grn1, grn2):
    grn1.random(5, 5, 5)
    grn2.random(5, 5, 5)

    grn2.identifiers = deepcopy(grn1.identifiers)
    grn2.enhancers = deepcopy(grn1.enhancers)
    grn2.inhibitors = deepcopy(grn1.inhibitors)
    grn2.beta = deepcopy(grn1.beta)
    grn2.delta = deepcopy(grn1.delta)

    grn1.setup()
    grn2.setup()
    assert np.sum(np.abs(grn1.get_signatures() - grn2.get_signatures())) < r_error
    assert np.sum(np.abs(grn1.get_concentrations() - grn2.get_concentrations())) < r_error

    grn1.warmup(5)
    grn2.warmup(5)
    assert np.sum(np.abs(grn1.get_output() - grn2.get_output())) < r_error
    assert np.sum(np.abs(grn1.get_concentrations() - grn2.get_concentrations())) < r_error

    inputs = np.random.rand(5)
    grn1.set_input(inputs)
    grn1.step()
    grn2.set_input(inputs)
    grn2.step()
    assert np.sum(np.abs(grn1.get_output() - grn2.get_output())) < r_error
    assert np.sum(np.abs(grn1.get_concentrations() - grn2.get_concentrations())) < r_error

#TODO: grn equality on load from string

def test_sanity():
    grn1 = grns.ClassicGRN()
    grn2 = grns.ClassicGRN()
    compare_grns(grn1, grn2)


def test_classic_matrix():
    grn1 = grns.ClassicGRN()
    grn2 = grns.MatrixGRN()
    compare_grns(grn1, grn2)


def test_classic_gpu():
    grn1 = grns.ClassicGRN()
    grn2 = grns.GPUGRN()
    compare_grns(grn1, grn2)


def test_classic_diff():
    grn1 = grns.ClassicGRN()
    grn2 = grns.DiffGRN()
    compare_grns(grn1, grn2)
