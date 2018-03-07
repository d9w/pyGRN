from pygrn.evolution import Species, Individual
from pygrn.evolution import mutate_modify, mutate, crossover
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

    def size(self):
        return np.sum([len(sp.individuals) for sp in self.species])

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

    def adjust_thresholds(self):
        avg_species_size = config.POPULATION_SIZE / len(self.species)
        for sp in self.species:
            if len(sp.individuals) < avg_species_size:
                sp.species_threshold += config.SPECIES_THRESHOLD_UPDATE
            else:
                sp.species_threshold -= config.SPECIES_THRESHOLD_UPDATE
            sp.species_threshold = min(config.MAX_SPECIES_THRESHOLD,
                                       max(config.MIN_SPECIES_THRESHOLD,
                                           sp.species_threshold))

    def set_offspring_count(self):
        total_adjusted_fitness = 0
        for sp in self.species:
            total_adjusted_fitness += sp.get_adjusted_fitness()
        total_num_offspring = 0
        for sp in self.species:
            if (total_adjusted_fitness == 0):
                sp.num_offspring = int(
                    config.POPULATION_SIZE/len(self.species))
            else:
                sp.num_offspring = int(
                    (sp.sum_adjusted_fitness/total_adjusted_fitness)*
                    config.POPULATION_SIZE)
            total_num_offspring += sp.num_offspring

        # Correcting approximation error
        # (sometimes species have fewer offspring because of division)
        while total_num_offspring < config.POPULATION_SIZE:
            sp = random.choice(self.species)
            sp.num_offspring += 1
            total_num_offspring += 1

    def make_offspring(self):
        self.offspring = []
        for sp in self.species:
            if sp.num_offspring == 0:
                continue
            species_offspring = []
            # Create children with crossover
            for k in range(int(sp.num_offspring * config.CROSSOVER_RATE)):
                parent1 = sp.tournament_select()
                parent2 = sp.tournament_select()
                # Don't use the same parent 2x
                num_tries = 0
                while ((parent1 == parent2) and
                       (num_tries < config.MAX_SELECTION_TRIES)):
                    parent2 = sp.tournament_select()
                    num_tries += 1
                if parent1 != parent2:
                    species_offspring += [Individual(
                        crossover(parent1.grn, parent2.grn))]
            # Create children with mutation
            for k in range(int(sp.num_offspring * config.MUTATION_RATE)):
                parent = sp.tournament_select()
                species_offspring += [Individual(mutate(parent.grn))]
            # Add elite
            if len(species_offspring) == sp.num_offspring:
                species_offspring[np.random.randint(
                    len(species_offspring))] = sp.get_best_individual()
            else:
                species_offspring += [sp.get_best_individual()]
            # Fill with tournament champions if extra space
            while len(species_offspring) < sp.num_offspring:
                species_offspring += [sp.tournament_select().clone()]

            self.offspring += species_offspring

    def get_best(self):
        best_fitness = -np.inf
        best_ind = None
        for sp in self.species:
            for ind in sp.individuals:
                if ind.fitness > best_fitness:
                    best_fitness = ind.fitness
                    best_ind = ind
        return (best_fitness, best_ind)
