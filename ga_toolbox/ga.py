from dataclasses import dataclass
import random
from ga_toolbox.config import GAConfig


@dataclass
class GAResult:
    best: object
    best_history: list
    avg_history: list
    generations_run: int



class GeneticAlgorithm:

    def __init__(self, chromosome_factory, fitness_fn, feasibility_fn,
                 selection, crossover, mutation, replacement,
                 config=None, repair_fn=None):
        self.chromosome_factory = chromosome_factory
        self.fitness_fn = fitness_fn
        self.feasibility_fn = feasibility_fn
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement

        if config is None:
            config = GAConfig()
        self.config = config
        self.repair_fn = repair_fn

        if config.infeasible_strategy not in ("penalty", "repair"):
            raise ValueError("infeasible_strategy must be 'penalty' or 'repair'")

        if config.infeasible_strategy == "repair" and repair_fn is None:
            raise ValueError("infeasible_strategy is 'repair' but no repair_fn was given")


    def evaluate(self, c):
        feasible = self.feasibility_fn(c)

        if not feasible and self.config.infeasible_strategy == "repair":
            repaired = self.repair_fn(c)
            c.genes = list(repaired.genes)
            feasible = self.feasibility_fn(c)

        fit = self.fitness_fn(c)
        if not feasible:
            if self.config.maximize:
                fit = fit - self.config.infeasible_penalty
            else:
                fit = fit + self.config.infeasible_penalty

        c.fitness = fit
        c.feasible = feasible

    def is_better(self, a, b):
        if self.config.maximize:
            return a > b
        return a < b


    def find_best(self, population):
        best = population[0]
        for c in population:
            if self.is_better(c.fitness, best.fitness):
                best = c
        return best

    def average_fitness(self, population):
        total = 0
        for c in population:
            total = total + c.fitness
        return total / len(population)


    def check_operators(self, sample):
        operators = [("crossover", self.crossover), ("mutation", self.mutation)]
        for name, op in operators:
            if not isinstance(sample, op.compatible):
                allowed = []
                for t in op.compatible:
                    allowed.append(t.__name__)
                raise TypeError(type(op).__name__ + " cannot be used as " + name +
                                " with " + type(sample).__name__ +
                                " (it works with: " + ", ".join(allowed) + ")")


    def run(self):
        cfg = self.config
        if cfg.seed is not None:
            random.seed(cfg.seed)

        population = []
        for i in range(cfg.population_size):
            c = self.chromosome_factory()
            self.evaluate(c)
            population.append(c)
        self.check_operators(population[0])

        best = self.find_best(population).copy()
        best_history = [best.fitness]
        avg_history = [self.average_fitness(population)]

        no_improvement = 0
        generations_run = 0

        for gen in range(cfg.max_generations):

            selected = self.selection.select(population, cfg.population_size, cfg.maximize)
            parents = []
            for p in selected:
                parents.append(p.copy())


            offspring = []
            i = 0
            while len(offspring) < cfg.population_size:
                p1 = parents[i % len(parents)]
                p2 = parents[(i+1) % len(parents)]
                i = i + 2

                if random.random() < cfg.crossover_rate:
                    child1, child2 = self.crossover.crossover(p1, p2)
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()

                offspring.append(child1)
                offspring.append(child2)

            offspring = offspring[:cfg.population_size]

            for k in range(len(offspring)):
                if random.random() < cfg.mutation_rate:
                    offspring[k] = self.mutation.mutate(offspring[k])
                self.evaluate(offspring[k])


            population = self.replacement.replace(population, offspring, cfg.maximize)

            gen_best = self.find_best(population)
            if self.is_better(gen_best.fitness, best.fitness):
                best = gen_best.copy()
                no_improvement = 0
            else:
                no_improvement = no_improvement + 1

            best_history.append(best.fitness)
            avg_history.append(self.average_fitness(population))
            generations_run = generations_run + 1

            if cfg.patience is not None and no_improvement >= cfg.patience:
                break

        return GAResult(best, best_history, avg_history, generations_run)