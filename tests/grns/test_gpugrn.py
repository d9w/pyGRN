from pygrn import grns, problems
from datetime import datetime


def test_gpugrn_simple():
    p = problems.TFRandom()
    g = grns.GPUGRN()
    g.random(p.nin, p.nout, 10)
    t1 = datetime.now()
    random_fitness = p.eval(g)
    t2 = datetime.now()
    print(t2 - t1)
    assert random_fitness > 0
    assert random_fitness < 10 * p.nout
