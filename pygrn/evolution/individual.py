class Individual:

    def __init__(self, grn, evaluated=False, fitness=0.0):
        self.grn = grn
        self.evaluated = evaluated
        self.fitness = fitness

    def get_fitness(self, problem):
        if not problem.cacheable or not self.evaluated:
            self.fitness = problem.eval(self.grn)
            self.evaluated = True
        return self.fitness

    def clone(self):
        return Individual(self.grn.clone(), self.evaluated, self.fitness)

