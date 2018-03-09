from pygrn.grns import ClassicGRN
from pygrn.problems import Random
from pygrn import evolution
import numpy as np


def test_evaluate():
    ind = evolution.Individual(ClassicGRN())
    problem = Random()
    ind.grn.random(problem.nin, problem.nout, 1)
    assert ind.fitness == 0.0
    fitness = ind.get_fitness(problem)
    assert ind.fitness == fitness
    assert fitness != 0


def test_sum_adjusted_fitness():
    species = evolution.Species()
    assert species.sum_adjusted_fitness == None
    problem = Random()
    for i in range(10):
        ind = evolution.Individual(ClassicGRN())
        ind.grn.random(problem.nin, problem.nout, 1)
        species.individuals += [ind]
    assert species.get_adjusted_fitness(0.0, 1.0) == 0.0
    assert species.sum_adjusted_fitness == 0.0
    for ind in species.individuals:
        ind.get_fitness(problem)
    assert species.get_adjusted_fitness(0.0, 1.0) > 0.0
    assert species.sum_adjusted_fitness > 0.0


def test_representative_distances():
    species = evolution.Species()
    assert species.sum_adjusted_fitness == None
    problem = Random()
    rep = evolution.Individual(ClassicGRN())
    rep.grn.random(problem.nin, problem.nout, 1)
    species.representative = rep
    for i in range(10):
        ind = evolution.Individual(evolution.mutate(rep.grn))
        species.individuals += [ind]
    dists0 = species.get_representative_distances()
    assert np.sum(dists0) > 0
    rep.grn.random(problem.nin, problem.nout, 1)
    dists1 = species.get_representative_distances()
    assert np.sum(dists1) > 0
    assert np.sum(dists1) > np.sum(dists0)
