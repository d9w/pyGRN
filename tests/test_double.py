from pygrn import grns, problems, evolution, config


def test_double_fit():
    p = problems.DoubleFrequency()
    g = grns.ClassicGRN()
    g.random(p.nin, p.nout, 10)
    fitness = p.eval(g)
    print(fitness)
    assert fitness <= 0.0


def test_double_evolution():
    p = problems.DoubleFrequency()
    g = grns.ClassicGRN()
    config.POPULATION_SIZE = 30
    g.random(p.nin, p.nout, 5)
    random_fitness = p.eval(g)
    evo = evolution.Evolution(p)
    best_fit, best_ind = evo.run(5)
    print(random_fitness, ", ", best_fit)
    assert best_fit > random_fitness
