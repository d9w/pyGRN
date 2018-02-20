import numpy as np
from pygrn import grns, GRNEAT, problems
import tensorflow as tf
from datetime import datetime


def test_gpugrn_simple():
    p = problems.TFRandom()
    g = grns.GPUGRN()
    g.random(p.nin, p.nout, 10)
    t1 = datetime.now()
    random_fitness = p.eval(g)
    t2 = datetime.now()
    print(t2 - t1)
    assert True

# def test_gpugrn_grneat():
#     p = problems.TFRandom()
#     g = grns.GPUGRN()
#     g.random(p.nin, p.nout, 10)
#     random_fitness = p.eval(g)

#     newgrn = lambda: grns.GPUGRN()
#     grneat = GRNEAT(newgrn)
#     grneat.setup(p.nin, p.nout, 50)
#     grneat.run(2, p)
#     bestFitness = -np.inf
#     for sp in grneat.species:
#         for ind in sp.individuals:
#             if ind.getFitness(sp.problem) > bestFitness:
#                 bestFitness = ind.getFitness(sp.problem)
#     assert bestFitness > random_fitness
