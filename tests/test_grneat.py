from pygrn import grns, GRNEAT, problems
from pygrn.GRNEAT import Individual, Species


def test_randoms():
    rep = grns.ClassicGRN()
    nin = 11; nout = 4; nreg = 10
    rep.random(nin, nout, nreg)
    randoms = [None] * 20
    distances = [None] * 20
    for i in range(len(randoms)):
        randoms[i] = grns.ClassicGRN()
        randoms[i].random(nin, nout, nreg)
        distances[i] = randoms[i].distance_to(rep)
    print(distances)
    assert True


def test_representative():
    rep = grns.ClassicGRN()
    nin = 11; nout = 4; nreg = 10
    rep.random(nin, nout, nreg)
    irep = Individual(rep, True, 0.0)
    newgrn = lambda: grns.ClassicGRN()
    grneat = GRNEAT(newgrn)
    grneat.setup(nin, nout, nreg)
    children = [grneat.mutate(irep) for i in range(20)]
    print([c.grn.distance_to(irep.grn) for c in children])
    assert True


def test_grneat_classic():
    p = problems.Random()
    g = grns.ClassicGRN()
    g.random(p.nin, p.nout, 10)
    random_fitness = p.eval(g)

    newgrn = lambda: grns.ClassicGRN()
    grneat = GRNEAT(newgrn)
    grneat.setup(p.nin, p.nout, 100)
    grneat.run(5, p)
    bestFitness, _ = grneat.get_best()
    print(bestFitness)
    assert True
    # assert bestFitness > random_fitness
