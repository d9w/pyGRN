from pygrn.evolution import Species, Individual, Population
from pygrn.grns import ClassicGRN
import numpy as np
import random
import os
import pathlib
from copy import deepcopy
from datetime import datetime


class Evolution:

    def __init__(self, problem, new_grn_function=lambda: ClassicGRN(),
                 grn_file=datetime.now().isoformat(),
                 grn_dir='grns'):
        pathlib.Path(grn_dir).mkdir(parents=True, exist_ok=True)
        self.grn_file = os.path.join(grn_dir, 'grns_' + grn_file + '.log')
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

    def run(self, generations):
        for gen in range(generations):
            self.step()
            self.generation += 1
        best_fit, best_ind = self.population.get_best()
        fit = (best_fit*(self.problem.fit_range[1] - self.problem.fit_range[0])
               + self.problem.fit_range[0])
        return fit, best_ind

    def report(self):
        return
        idx = 0
        numIndividuals = 0
        sumFitness = 0
        bestFitness = -float('inf')
        bestGRN = None
        for sp in self.species:
            with open(sp.problem.logfile, 'a') as f:
                f.write( 'Generation\t%d\t%d\t%d\t%f\t%f\t%d\t%f\t%f\t%f\n' %
                         (generation,
                          idx,
                          len(sp.individuals),
                          sp.sumAdjustedFitness / float(len(sp.individuals)),
                          sp.getBestGenome().getFitness(sp.problem),
                          sp.getBestGenome().grn.size(),
                          sp.speciesThreshold,
                          np.mean(sp.representative_distances()),
                          np.mean([i.grn.size() for i in sp.individuals])))
            # if len(sp.individuals) < self.minSpeciesSize:
                # print(sorted(sp.representative_distances()))
            for ind in sp.individuals:
                sumFitness += ind.getFitness(sp.problem)
                if ind.getFitness(sp.problem) > bestFitness:
                    bestFitness = ind.getFitness(sp.problem)
                    bestGRN = ind
                numIndividuals += 1
            idx += 1
        print( 'Best fitness: %f\tAverage fitness: %f\tPopulation size: %d\tTime: %s' % ( bestFitness, sumFitness/numIndividuals, numIndividuals, str(datetime.now().isoformat())) )
        print( 'Best individual\n' + str(bestGRN.grn))
        with open(self.logfile, 'a') as f:
            f.write(str(bestGRN.grn) + '\n')

