from pygrn.grns import ClassicGRN
from pygrn import evolution


def test_eq_size_crossover():
    parent1 = ClassicGRN()
    parent2 = ClassicGRN()
    parent1.random(5, 5, 5)
    parent2.random(5, 5, 5)
    child = evolution.crossover(parent1, parent2)
    assert (child.beta == parent1.beta) or (child.beta == parent2.beta)
    assert (child.delta == parent1.delta) or (child.delta == parent2.delta)
    assert parent1.size() == child.size()
    assert parent2.size() == child.size()
    assert parent1.distance_to(child) < parent1.distance_to(parent2)
    assert parent2.distance_to(child) < parent2.distance_to(parent1)
    assert parent1.distance_to(child) > 0
    assert parent2.distance_to(child) > 0


def test_unequal_size_crossover():
    parent1 = ClassicGRN()
    parent2 = ClassicGRN()
    parent1.random(5, 5, 1)
    parent2.random(5, 5, 10)
    child = evolution.crossover(parent1, parent2)
    assert (child.beta == parent1.beta) or (child.beta == parent2.beta)
    assert (child.delta == parent1.delta) or (child.delta == parent2.delta)
    assert parent1.size() <= child.size()
    assert parent2.size() >= child.size()
    assert parent1.distance_to(child) < parent1.distance_to(parent2)
    assert parent2.distance_to(child) < parent2.distance_to(parent1)
    assert parent1.distance_to(child) > 0
    assert parent2.distance_to(child) > 0
