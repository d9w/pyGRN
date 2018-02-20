from pygrn import grns, problems


def test_double():
    p = problems.DoubleFrequency()
    g = grns.ClassicGRN()
    g.random(p.nin, p.nout, 10)
    fitness = p.eval(g)
    print(fitness)
    assert fitness <= 0.0
