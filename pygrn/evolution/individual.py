class Individual:

    def __init__(self, grn, evaluated=False, fitness=0.0):
        self.grn = grn
        self.evaluated = evaluated
        self.fitness = fitness

    def get_fitness(self, problem):
        if not problem.cacheable or not self.evaluated:
            fit = problem.eval(self.grn)
            if (fit < problem.fit_range[0]) or (fit > problem.fit_range[1]):
                raise Exception("The returned fitness " + str(fit) +
                                " is outside of the problem fitness range " +
                                str(problem.fit_range))
            self.fitness = (fit - problem.fit_range[0]) / (
                problem.fit_range[1] - problem.fit_range[0])
            self.evaluated = True
        return self.fitness

    def clone(self):
        return Individual(self.grn.clone(), self.evaluated, self.fitness)

