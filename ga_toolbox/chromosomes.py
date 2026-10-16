from abc import ABC, abstractmethod
import random


class Chromosome(ABC):

    def __init__(self, genes):
        self.genes = list(genes)
        self.fitness = None
        self.feasible = None

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        return str(self.genes)

    @abstractmethod
    def copy(self):
        pass

    @classmethod
    @abstractmethod
    def random(cls):
        pass


class BinaryChromosome(Chromosome):

    def copy(self):
        new_c = BinaryChromosome(self.genes[:])
        new_c.fitness = self.fitness
        new_c.feasible = self.feasible
        return new_c

    @classmethod
    def random(cls, length):
        genes = []
        for i in range(length):
            genes.append(random.randint(0, 1))
        return cls(genes)



class IntegerChromosome(Chromosome):

    def __init__(self, genes, low, high, permutation=False):
        super().__init__(genes)
        self.low = low
        self.high = high
        self.permutation = permutation

    def copy(self):
        new_c = IntegerChromosome(self.genes[:], self.low, self.high, self.permutation)
        new_c.fitness = self.fitness
        new_c.feasible = self.feasible

        return new_c

    @classmethod
    def random(cls, length, low, high, permutation=False):
        if permutation:
            if length != high - low + 1:
                raise ValueError("for a permutation, length must be high - low + 1")
            genes = list(range(low, high+1))
            random.shuffle(genes)
        else:
            genes = []
            for i in range(length):
                genes.append(random.randint(low, high))

        return cls(genes, low, high, permutation)


class FloatChromosome(Chromosome):

    def __init__(self, genes, low, high):
        super().__init__(genes)
        self.low = low
        self.high = high

        for i in range(len(self.genes)):
            if self.genes[i] < low:
                self.genes[i] = low
            if self.genes[i] > high:
                self.genes[i] = high


    def copy(self):
        new_c = FloatChromosome(self.genes[:], self.low, self.high)
        new_c.fitness = self.fitness
        new_c.feasible = self.feasible
        return new_c

    @classmethod
    def random(cls, length, low, high):
        genes = []
        for i in range(length):
            genes.append(random.uniform(low, high))
        return cls(genes, low, high)