import numpy as np
import random


class Species:
    sum_adjusted_fitness = None
    species_threshold = 0.15
    num_offspring = 0
    best_genome = None

    def __init__(self):
        self.individuals = []
        self.representative = None

    def reset(self):
        self.individuals = []
        self.sum_adjusted_fitness = 0
        self.num_offspring = 0
        best_genome = None

    def tournament_select(self, tournament_size=3, with_replacement=True):
        tournament = []
        if with_replacement:
            tournament = [random.choice(self.individuals) for
                          k in range(tournamentSize)]
        else:
            tournament = random.shuffle(self.individuals)[:tournamentSize]
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

    def get_adjusted_fitness(self):
        self.sum_adjusted_fitness = np.sum(
            [ind.fitness for ind in self.individuals])
        return self.sum_adjusted_fitness

    def get_best_genome(self):
        if self.best_genome == None:
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)
            self.best_genome = self.individuals[0]
        return self.bestGenome

    def get_representative_distances(self):
        return [ind.grn.distance_to(self.representative.grn)
                for ind in self.individuals]


class Individual:

    def __init__(self, grn, evaluated=False, fitness=0.0):
        self.grn = grn
        self.evaluated = evaluated
        self.fitness = fitness

    def get_fitness(self, problem):
        if not problem.cacheable or not self.evaluated:
            self.fitness = (problem.eval(self.grn) - problem.fit_range[0]) / (
                problem.fit_range[1] - problem.fit_range[0])
            self.evaluated = True
        return self.fitness

    def clone(self):
        return Individual(self.grn.clone(), self.evaluated, self.fitness)

