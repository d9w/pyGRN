from pygrn import config
import numpy as np
import random


class Species:
    sum_adjusted_fitness = None
    species_threshold = config.START_SPECIES_THRESHOLD
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

    def tournament_select(self):
        tournament = []
        if config.TOURNAMENT_WITH_REPLACEMENT:
            tournament = [random.choice(self.individuals) for
                          k in range(config.TOURNAMENT_SIZE)]
        else:
            tournament = random.shuffle(
                self.individuals)[:config.TOURNAMENT_SIZE]
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
