from pygrn.evolution import Population
from pygrn.grns import ClassicGRN
import numpy as np
import os
import pathlib
from datetime import datetime
from uuid import uuid4


class Evolution:

    def __init__(self, problem, new_grn_function=lambda: ClassicGRN(),
                 run_id=str(uuid4()), grn_dir='grns', log_dir='logs'):
        pathlib.Path(grn_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.grn_file = os.path.join(grn_dir, 'grns_' + run_id + '.log')
        self.log_file = os.path.join(log_dir, 'gen_' + run_id + '.log')
        self.problem = problem
        self.population = Population(new_grn_function, problem.nin,
                                     problem.nout)
        self.generation = 0

    def step(self):
        self.population.evaluate(self.problem)
        self.population.speciation()
        self.population.adjust_thresholds()
        self.population.set_offspring_count()
        self.population.make_offspring()
        self.report()
        self.problem.generation_function(self, self.generation)
        self.generation += 1

    def run(self, generations):
        for gen in range(generations):
            self.step()
        best_fit, best_ind = self.population.get_best()
        return best_fit, best_ind

    def report(self):
        for species_id in range(len(self.population.species)):
            sp = self.population.species[species_id]
            sp_best = sp.get_best_individual()
            with open(self.log_file, 'a') as f:
                f.write('%s,Species,%d,%d,%d,%f,%f,%d,%f,%f,%f\n' % (
                    datetime.now().isoformat(),
                    self.generation, species_id,
                    len(sp.individuals),
                    sp.sum_adjusted_fitness,
                    sp_best.fitness,
                    sp_best.grn.size(),
                    sp.species_threshold,
                    np.mean(sp.get_representative_distances()),
                    np.mean([i.grn.size() for i in sp.individuals])))
        best_fitness, best_ind = self.population.get_best()
        fit_mean, fit_std = self.population.get_stats()
        with open(self.log_file, 'a') as f:
            f.write('%s,Generation,%d,%d,%f,%d,%f,%f\n' % (
                datetime.now().isoformat(),
                self.generation, self.population.size(),
                best_fitness, best_ind.grn.size(),
                fit_mean, fit_std))
        with open(self.grn_file, 'a') as f:
            f.write(str(best_ind.grn) + '\n')
