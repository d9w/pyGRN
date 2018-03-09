from pygrn.grns import ClassicGRN
from pygrn import evolution, config
import numpy as np


def test_mutate_add():
    parent = ClassicGRN()
    parent.random(5, 5, 5)
    child = evolution.mutate_add(parent)
    assert np.all(parent.inhibitors == child.inhibitors[0:-1])
    assert np.all(parent.enhancers == child.enhancers[0:-1])
    assert np.all(parent.identifiers == child.identifiers[0:-1])
    assert len(parent.inhibitors) == len(child.inhibitors) - 1
    assert len(parent.enhancers) == len(child.enhancers) - 1
    assert len(parent.identifiers) == len(child.identifiers) - 1
    assert parent.size() == child.size() - 1
    assert parent.distance_to(child) < 1 / (child.size() + 2)


def test_mutate_remove():
    parent = ClassicGRN()
    parent.random(5, 5, 5)
    child = evolution.mutate_remove(parent)
    assert len(parent.inhibitors) == len(child.inhibitors) + 1
    assert len(parent.enhancers) == len(child.enhancers) + 1
    assert len(parent.identifiers) == len(child.identifiers) + 1
    assert parent.size() == child.size() + 1
    assert parent.distance_to(child) < 1 / (parent.size() + 2)


def test_mutate_modify():
    parent = ClassicGRN()
    parent.random(5, 5, 5)
    child = evolution.mutate_modify(parent)
    if parent.beta == child.beta and parent.delta == child.delta:
        assert (np.sum(parent.inhibitors != child.inhibitors) +
                np.sum(parent.enhancers != child.enhancers) +
                np.sum(parent.identifiers != child.identifiers)) == 1
    assert len(parent.inhibitors) == len(child.inhibitors)
    assert len(parent.enhancers) == len(child.enhancers)
    assert len(parent.identifiers) == len(child.identifiers)
    assert parent.size() == child.size()
    assert parent.distance_to(child) < 1 / (parent.size() + 2)


def test_mutate():
    parent = ClassicGRN()
    parent.random(5, 5, 5)
    config.MUTATION_ADD_RATE = 1.0
    child = evolution.mutate(parent)
    assert parent.size() == child.size() - 1
    assert parent.distance_to(child) < 1 / (child.size() + 2)
    config.MUTATION_ADD_RATE = 0.0
    config.MUTATION_DEL_RATE = 1.0
    child = evolution.mutate(parent)
    assert parent.size() == child.size() + 1
    assert parent.distance_to(child) < 1 / (child.size() + 2)
    config.MUTATION_DEL_RATE = 0.0
    child = evolution.mutate(parent)
    assert parent.size() == child.size()
    assert parent.distance_to(child) < 1 / (child.size() + 2)
