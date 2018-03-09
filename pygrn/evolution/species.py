from pygrn import config
import numpy as np
import random


class Species:
    sum_adjusted_fitness = None
    species_threshold = config.START_SPECIES_THRESHOLD
    num_offspring = 0

    def __init__(self):
        self.individuals = []
        self.representative = None

    def reset(self):
        self.individuals = []
        self.sum_adjusted_fitness = 0
        self.num_offspring = 0

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

    def get_adjusted_fitness(self, fit_min, fit_max):
        sum_fit = 0.0
        if fit_max > fit_min:
            for ind in self.individuals:
                sum_fit += (ind.fitness - fit_min)/(fit_max - fit_min)
        self.sum_adjusted_fitness = sum_fit/len(self.individuals)
        return self.sum_adjusted_fitness

    def get_best_individual(self):
        best_fitness = -np.inf
        best_ind = None
        for ind in self.individuals:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_ind = ind
        return best_ind

    def get_representative_distances(self):
        return [ind.grn.distance_to(self.representative.grn)
                for ind in self.individuals]
