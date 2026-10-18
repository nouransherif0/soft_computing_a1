from abc import ABC, abstractmethod

from ga_toolbox.config import GAConfig


class Replacement(ABC):

    @abstractmethod
    def replace(self, population, offspring, maximize):
        pass


def best_first(individuals, maximize):
    # sorted copy, the best individual first (biggest if maximize, smallest if not)
    return sorted(individuals, key=lambda c: c.fitness, reverse=maximize)


class GenerationalReplacement(Replacement):
    # the offspring become the new generation, except that the best elitism_count
    # individuals of the old generation always survive (so the best is never lost).

    def __init__(self, elitism_count=None):
        if elitism_count is None:
            elitism_count = GAConfig().elitism_count
        if elitism_count < 0:
            raise ValueError("elitism_count cannot be negative")
        self.elitism_count = elitism_count

    def replace(self, population, offspring, maximize):
        size = len(population)
        old = best_first(population, maximize)
        elites = old[:min(self.elitism_count, size)]

        need = size - len(elites)
        new = best_first(offspring, maximize)[:need]

        if len(new) < need:
            # not enough offspring: fill the gap with the next best old individuals
            new = new + old[len(elites):len(elites) + need - len(new)]

        return elites + new


class MuPlusLambdaReplacement(Replacement):
    # (mu + lambda): put the parents (mu) and the offspring (lambda) together and keep
    # the best mu. Stronger selection pressure than generational replacement, because
    # old good individuals compete with the children.

    def replace(self, population, offspring, maximize):
        everyone = best_first(list(population) + list(offspring), maximize)
        return everyone[:len(population)]
