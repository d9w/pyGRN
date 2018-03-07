from pygrn.evolution import Species, Individual, Population
from pygrn.grns import ClassicGRN
import numpy as np
import random
import os
import pathlib
from copy import deepcopy
from datetime import datetime


class Evolution:

    def __init__(self, problem, new_grn_function=ClassicGRN(),
                 grn_file=datetime.now().isoformat(),
                 grn_dir='grns'):
        pathlib.Path(grn_dir).mkdir(parents=True, exist_ok=True)
        self.grn_file = os.path.join(grn_dir, 'grns_' + grn_file + '.log')
        self.problem = problem
        self.population = Population(new_grn_function, problem.nin,
                                     problem.nout)

    def setup(self, nInputs, nOutputs, popSize):
        currentGeneration = 0

        self.offspring = []

        self.populationSize = popSize

        while len( self.offspring ) < popSize:
            g = self.newgrn()
            g.random(nInputs,nOutputs,1)

            parent = Individual(g,False,None)

            # Create a small species if there isnt enough space in the population
            numToCreate = min( ( self.initializationDuplication - 1 ), ( popSize - len( self.offspring ) - 1 ) )
            
            self.offspring += [ self.mutateModify(parent) for k in range(numToCreate) ]
            self.offspring += [ parent ]
        
        return self

    # Problem should have: 
    # - evaluate(grn): float
    # - cacheable: bool
    # TODO: make grneat accept lambda functions for problem creation
    def run( self, maxGenerations, problem ):

        # problem = problemClass(self.start_time.isoformat())

        for generation in range( maxGenerations ):

            if not problem.cacheable:
                for ind in self.offspring:
                    ind.hasBeenEvaluated = False

            self.speciation( problem )

            problem.generation_function(self, generation)

            sumAdjustedFitness = 0
            speciesAvgSize = self.populationSize / len( self.species )
            for sp in self.species:
                sumAdjustedFitness += sp.calculateAdjustedFitness()
                if len(sp.individuals) < speciesAvgSize:
                    sp.speciesThreshold = sp.speciesThreshold + 0.01
                else:
                    sp.speciesThreshold = sp.speciesThreshold - 0.01
                sp.speciesThreshold = min(0.5, max(0.03, sp.speciesThreshold))

            totalNumOffspring = 0
            for sp in self.species:
                if (sumAdjustedFitness == 0):
                    sp.numOffspring = int(self.populationSize / len(self.species))
                else:
                    sp.numOffspring = sp.sumAdjustedFitness / sumAdjustedFitness * self.populationSize
                totalNumOffspring += sp.numOffspring

            # Correcting approximation error (sometimes species have fewer offspring because of division)
            while totalNumOffspring < self.populationSize:
                sp = random.choice( self.species )
                sp.numOffspring += 1
                totalNumOffspring += 1

            self.report( generation )

            # Make babies
            self.offspring = []
            for sp in self.species:
                # Create children with crossover
                if sp.numOffspring == 0:
                    continue
                speciesOffspring = []
                for k in range( int( sp.numOffspring * self.crossoverRate ) ):
                    parent1 = sp.tournamentSelect( self.tournamentSize )
                    parent2 = sp.tournamentSelect( self.tournamentSize )
                    # Don't use the same parent 2x
                    numTries = 0
                    while ( parent1 == parent2 ) and ( numTries < self.maxSelectionTries ):
                        parent2 = sp.tournamentSelect( self.tournamentSize )
                        numTries += 1
                    if parent1 != parent2:
                        speciesOffspring += [ self.crossover( parent1, parent2 ) ]
                # Create children with mutation
                for k in range( int( sp.numOffspring * self.mutationRate ) ):
                    parent = sp.tournamentSelect( self.tournamentSize )
                    child = self.mutate( parent )
                    speciesOffspring += [ child ]
                # Add elite
                if len( speciesOffspring ) == sp.numOffspring:
                    speciesOffspring[np.random.randint(len(speciesOffspring))] = sp.getBestGenome()
                else:
                    speciesOffspring += [sp.getBestGenome()]
                # Fill with tournament champions if extra space
                while len(speciesOffspring) < sp.numOffspring:
                    speciesOffspring += [sp.tournamentSelect( self.tournamentSize ).clone()]
                
                self.offspring += speciesOffspring
                
        return self
    

    def speciation( self, problem ):
        for sp in self.species:
            sp.representative = random.choice( sp.individuals ).clone()
            sp.reset()

        for ind in self.offspring:
            bestMatch = None
            minDist = float('inf')
            ind.getFitness(problem)
            for spi in range(len(self.species)):
                spDist = self.species[spi].representative.grn.distance_to(ind.grn)
                if spDist < minDist and spDist < self.species[spi].speciesThreshold:
                    bestMatch = spi
                    minDist = spDist
            if bestMatch != None:
                self.species[bestMatch].individuals += [ind]
            else:
                newSp = Species()
                newSp.problem = problem
                newSp.individuals += [ind]
                newSp.representative = ind
                self.species += [newSp]

        numRemoved = 0
        spToRemove = []
        for sp in self.species:
            if len(sp.individuals) < self.minSpeciesSize:
                numRemoved += len(sp.individuals)
                spToRemove += [sp]

        self.species = list( set( self.species ) - set( spToRemove ) )

        while numRemoved > 0:
            sp = random.choice( self.species )
            ind = sp.tournamentSelect( self.tournamentSize )

            tries = 0
            hasAdded = False
            while not hasAdded:
                child = self.mutate(ind)

                if child.grn.distance_to( sp.representative.grn ) < sp.speciesThreshold:
                    sp.individuals += [child]
                    numRemoved -= 1
                    hasAdded = True
                tries += 1
                if tries == self.maxSelectionTries:
                    newSp = Species()
                    newSp.problem = problem
                    newSp.individuals += [child]
                    newSp.representative = child
                    self.species += [newSp]
                    numRemoved -= 1
                    hasAdded = True
                    break

        return self

    def crossover( self, parent1, parent2 ):
        g = self.newgrn()
        g.num_input = parent1.grn.num_input
        g.num_output = parent1.grn.num_output
        child = Individual(g,False,None)
        
        for k in range( parent1.grn.num_input + parent1.grn.num_output ):
            chosenParent = None
            if np.random.randint(2) == 0:
                chosenParent = parent1
            else:
                chosenParent = parent2
            child.grn.identifiers = np.append( child.grn.identifiers, chosenParent.grn.identifiers[k] )
            child.grn.inhibitors = np.append( child.grn.inhibitors, chosenParent.grn.inhibitors[k] )
            child.grn.enhancers = np.append( child.grn.enhancers, chosenParent.grn.enhancers[k] )

            
        p1range = list( range( parent1.grn.num_input + parent1.grn.num_output, parent1.grn.size() ) )
        random.shuffle( p1range )
        p2range = list( range( parent2.grn.num_input + parent2.grn.num_output, parent2.grn.size() ) )
        random.shuffle( p2range )        
        
        p1remaining = deepcopy(p1range)
        
        # Crossing regulatory
        numFromP1 = 0
        numFromP2 = 0
        for p1idx in p1range:
            minDist = float('inf')
            pairedIdx = None
            for p2idx in p2range:
                gDist = parent1.grn.protein_distance( parent2.grn, p1idx, p2idx )
                if gDist < self.crossoverThreshold and gDist < minDist:
                    minDist = gDist
                    pairedIdx = p2idx
            if pairedIdx != None:
                if np.random.randint(2) == 0:
                    chosenParent = parent1
                    chosenIdx = p1idx
                    numFromP1 += 1
                else:
                    chosenParent = parent2
                    chosenIdx = p2idx
                    numFromP2 += 1
                child.grn.identifiers = np.append( child.grn.identifiers, chosenParent.grn.identifiers[chosenIdx] )
                child.grn.inhibitors = np.append( child.grn.inhibitors, chosenParent.grn.inhibitors[chosenIdx] )
                child.grn.enhancers = np.append( child.grn.enhancers, chosenParent.grn.enhancers[chosenIdx] )
                # Remove from consideration again
                p2range = list( set( p2range ) - set( [p2idx] ) )
                p1remaining = list( set( p1remaining ) - set( [p1idx] ) )
        
        # Add remaining material
        if child.grn.size() == ( child.grn.num_input + child.grn.num_output ):
            prob = 0.5
        else:
            prob = numFromP1 / ( numFromP1 + numFromP2 )
            
        if np.random.random() < prob:
            chosenParent = parent1
            chosenRange = p1remaining
        else:
            chosenParent = parent2
            chosenRange = p2range
            
        for chosenIdx in chosenRange:
            child.grn.identifiers = np.append( child.grn.identifiers, chosenParent.grn.identifiers[chosenIdx] )
            child.grn.inhibitors = np.append( child.grn.inhibitors, chosenParent.grn.inhibitors[chosenIdx] )
            child.grn.enhancers = np.append( child.grn.enhancers, chosenParent.grn.enhancers[chosenIdx] )
            
        child.grn.num_regulatory = child.grn.size() - (child.grn.num_input +
                                                       child.grn.num_output)
        # Cross dynamics
        if np.random.random() < 0.5:
            child.grn.beta = parent1.grn.beta
        else:
            child.grn.beta = parent2.grn.beta
            
        if np.random.random() < 0.5:
            child.grn.delta = parent1.grn.delta
        else:
            child.grn.delta = parent2.grn.delta

        return child

    def get_best(self):
        best_fitness = -np.inf
        best_ind = None
        for sp in self.species:
            for ind in sp.individuals:
                if ind.getFitness(sp.problem) > best_fitness:
                    best_fitness = ind.getFitness(sp.problem)
                    best_ind = ind
        return (best_fitness, best_ind)

    def report( self, generation ):
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
