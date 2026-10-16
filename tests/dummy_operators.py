# simple test operators for unit tests (member 2 will implement real ones)
import random

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome
from ga_toolbox.selection import Selection
from ga_toolbox.crossover import Crossover
from ga_toolbox.replacement import Replacement


class DummySelection(Selection):
    # small tournament selection for testing
    def select(self, population, n, maximize):
        chosen = []
        for i in range(n):
            a = random.choice(population)
            b = random.choice(population)

            if maximize:
                if a.fitness >= b.fitness:
                    chosen.append(a)
                else:
                    chosen.append(b)
            else:
                if a.fitness <= b.fitness:
                    chosen.append(a)
                else:
                    chosen.append(b)

        return chosen



class DummyCrossover(Crossover):
    compatible = (BinaryChromosome, FloatChromosome)

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 1)

        child1 = parent1.copy()
        child2 = parent2.copy()

        child1.genes = parent1.genes[:point] + parent2.genes[point:]
        child2.genes = parent2.genes[:point] + parent1.genes[point:]

        child1.fitness = None
        child2.fitness = None
        return child1, child2



class DummyReplacement(Replacement):
    # takes top n individuals from combined population
    def replace(self, population, offspring, maximize):
        everyone = population + offspring
        everyone.sort(key=lambda c: c.fitness, reverse=maximize)
        return everyone[:len(population)]

