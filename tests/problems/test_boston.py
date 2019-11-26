from pygrn import grns, problems, evolution, config


def test_evolution_boston():
    config.POPULATION_SIZE = 5
    config.INITIALIZATION_DUPLICATION = 2
    config.START_SPECIES_THRESHOLD = 0.05
    config.MIN_SPECIES_SIZE = 1
    p = problems.Boston()
    newgrn = lambda: grns.DiffGRN()
    grneat = evolution.Evolution(p, newgrn)
    best_fit, _ = grneat.run(2)
    assert best_fit >= 0.0
    assert best_fit <= 1.0
