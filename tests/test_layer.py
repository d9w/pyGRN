from pygrn import grns, problems
from pygrn import GRNEAT
from datetime import datetime
from copy import deepcopy
import numpy as np
import random


#def test_random_boston():
#    # TODO: test that non-learning GRNs run
#    p = problems.Boston()
#    g = grns.DiffGRN()
#    np.random.seed(1234)
#    g.random(p.nin, p.nout, 10)
#    print(g)
#    ids = deepcopy(g.identifiers)
#    random_fitness = p.eval(g)
#    print(random_fitness)
#    assert np.all(g.identifiers == ids)
#    assert True
#

#def test_random_eeg():
#    p = problems.EEG()
#    g = grns.DiffGRN()
#    g.random(p.nin, p.nout, 10)
#    ids = deepcopy(g.identifiers)
#    random_fitness = p.eval(g)
#    print(random_fitness)
#    assert np.all(g.identifiers == ids)
#    assert True

def test_random_energy():
    p = problems.Energy()
    g = grns.DiffGRN()
    g.random(p.nin, p.nout, 10)
    ids = deepcopy(g.identifiers)
    random_fitness = p.eval(g)
    print(random_fitness)
    assert np.all(g.identifiers == ids)
    assert True

#def test_grneat_boston():
#    p = problems.Boston()
#    newgrn = lambda: grns.DiffGRN()
#    grneat = GRNEAT(newgrn)
#    grneat.setup(p.nin, p.nout, 40)
#    grneat.run(2, p)
#    bestFitness, _ = grneat.get_best()
#    print(bestFitness)
#    assert True
