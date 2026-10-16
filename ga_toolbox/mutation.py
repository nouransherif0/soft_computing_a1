from abc import ABC, abstractmethod
import random

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome


class Mutation(ABC):

    compatible = ()

    @abstractmethod
    def mutate(self, chromosome):
        pass


class BitFlipMutation(Mutation):
    compatible = (BinaryChromosome,)

    def __init__(self, gene_rate=0.05):
        self.gene_rate = gene_rate

    def mutate(self, chromosome):
        for i in range(len(chromosome.genes)):
            r = random.random()
            if r <= self.gene_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]

        chromosome.fitness = None
        chromosome.feasible = None
        return chromosome



class SwapMutation(Mutation):
    compatible = (IntegerChromosome,)

    def mutate(self, chromosome):
        if len(chromosome.genes) >= 2:
            pos1, pos2 = random.sample(range(len(chromosome.genes)), 2)
            temp = chromosome.genes[pos1]

            chromosome.genes[pos1] = chromosome.genes[pos2]
            chromosome.genes[pos2] = temp
        chromosome.fitness = None
        chromosome.feasible = None

        return chromosome


class InversionMutation(Mutation):
    compatible = (IntegerChromosome,)

    def mutate(self, chromosome):
        if len(chromosome.genes) >= 2:
            pos1, pos2 = random.sample(range(len(chromosome.genes)), 2)
            start = min(pos1, pos2)
            end = max(pos1, pos2)
            chromosome.genes[start:end+1] = chromosome.genes[start:end + 1][::-1]

        chromosome.fitness = None
        chromosome.feasible = None
        return chromosome




class GaussianMutation(Mutation):
    compatible = (FloatChromosome,)

    def __init__(self, gene_rate=0.2, sigma_fraction=0.1):
        self.gene_rate = gene_rate
        self.sigma_fraction = sigma_fraction

    def mutate(self, chromosome):
        sigma = (chromosome.high - chromosome.low) * self.sigma_fraction

        for i in range(len(chromosome.genes)):
            r = random.random()
            if r <= self.gene_rate:
                val = chromosome.genes[i] + random.gauss(0, sigma)

                if val < chromosome.low:
                    val = chromosome.low
                if val > chromosome.high:
                    val = chromosome.high
                chromosome.genes[i] = val

        chromosome.fitness = None
        chromosome.feasible = None
        return chromosome