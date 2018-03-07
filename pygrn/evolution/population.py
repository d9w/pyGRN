from pygrn.evolution import Species, Individual, mutate_modify, mutate
from pygrn import config
import numpy as np
import random

class Population:

    def __init__(self, new_grn_function, nin, nout):
        self.new_grn_function = new_grn_function
        self.species = []
        self.offspring = []

        while len(self.offspring) < config.POPULATION_SIZE:
            g = self.new_grn_function()
            g.random(nin, nout, 1)

            # Create a small species if there isn't enough space
            num_create = min(config.INITIALIZATION_DUPLICATION,
                             config.POPULATION_SIZE - len(self.offspring)) - 1

            self.offspring += [Individual(mutate_modify(g))
                               for k in range(num_create)]
            self.offspring += [Individual(g)]

    def evaluate(self, problem):
        for ind in self.offspring:
            ind.get_fitness(problem)

    def speciation(self):
        for sp in self.species:
            sp.representative = random.choice(sp.individuals).clone()
            sp.reset()

        for ind in self.offspring:
            species_match = None
            min_dist = np.inf
            for spi in range(len(self.species)):
                sp_dist = self.species[spi].representative.grn.distance_to(
                    ind.grn)
                if (sp_dist < min_dist and
                    sp_dist < self.species[spi].species_threshold):
                    species_match = spi
                    min_dist = sp_dist
            if species_match != None:
                self.species[species_match].individuals += [ind]
            else:
                new_sp = Species()
                new_sp.individuals += [ind]
                new_sp.representative = ind.clone()
                self.species += [new_sp]

        self.offspring = []

        num_removed = 0
        species_to_remove = []
        for sp in self.species:
            if len(sp.individuals) < config.MIN_SPECIES_SIZE:
                num_removed += len(sp.individuals)
                species_to_remove += [sp]
                del sp.representative
                for ind in sp.individuals:
                    del ind

        self.species = list(set(self.species) - set(species_to_remove))

        if len(self.species) == 0:
            raise Exception(('The entire population was removed. '
                             'Try again with a smaller MIN_SPECIES_SIZE '
                             'or a larger INITIALIZATION_DUPLICATION'))

        while num_removed > 0:
            sp = random.choice(self.species)
            ind = sp.tournament_select()

            tries = 0
            has_added = False
            while not has_added:
                child_grn = mutate(ind.grn)

                if (child_grn.distance_to(sp.representative.grn) <
                    sp.species_threshold):
                    sp.individuals += [Individual(child_grn)]
                    num_removed -= 1
                    has_added = True
                tries += 1
                if tries == config.MAX_SELECTION_TRIES:
                    ind = Individual(child_grn)
                    new_sp = Species()
                    new_sp.individuals += [ind]
                    new_sp.representative = ind.clone()
                    self.species += [new_sp]
                    num_removed -= 1
                    has_added = True

    def get_best(self):
        best_fitness = -np.inf
        best_ind = None
        for sp in self.species:
            for ind in sp.individuals:
                if ind.fitness > best_fitness:
                    best_fitness = ind.fitness
                    best_ind = ind
        return (best_fitness, best_ind)
