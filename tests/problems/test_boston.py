from pygrn import grns, problems, evolution, config
from datetime import datetime
from copy import deepcopy
import numpy as np
import random


def test_evolution_boston():
   config.POPULATION_SIZE = 20
   p = problems.Boston()
   newgrn = lambda: grns.DiffGRN()
   grneat = evolution.Evolution(p, newgrn)
   best_fit, _ = grneat.run(2)
   assert best_fit >= 0.0
   assert best_fit <= 1.0
