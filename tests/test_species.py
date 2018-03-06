from pygrn.species import Species, Individual
from pygrn.grns import ClassicGRN
from pygrn.problems import Random


def test_evaluate():
    ind = Individual(ClassicGRN())
    problem = Random()
    ind.grn.random(problem.nin, problem.nout, 1)
    assert ind.fitness == 0.0
    fitness = ind.get_fitness(problem)
    assert ind.fitness == fitness
    assert fitness != 0


def test_sum_adjusted_fitness():
    species = Species()
    assert species.sum_adjusted_fitness == None
    problem = Random()
    for i in range(10):
        ind = Individual(ClassicGRN())
        ind.grn.random(problem.nin, problem.nout, 1)
        species.individuals += [ind]
    assert species.get_adjusted_fitness() == 0.0
    assert species.sum_adjusted_fitness == 0.0
    for ind in species.individuals:
        ind.get_fitness(problem)
    assert species.get_adjusted_fitness() > 0.0
    assert species.sum_adjusted_fitness > 0.0
