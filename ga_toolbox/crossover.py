from abc import ABC, abstractmethod
import random

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome
from ga_toolbox.config import GAConfig


class Crossover(ABC):

    compatible = ()

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass


def make_child(template, genes):
    # copy() keeps the type and the settings (low, high, permutation) of the parent
    child = template.copy()
    child.genes = genes
    child.fitness = None
    child.feasible = None
    return child


class MultiPointCrossover(Crossover):
    # cut both parents at k random points and swap every second segment.
    # works for binary, float and (non permutation) integer chromosomes.
    # it is NOT valid for permutations (it would create duplicates), so those are rejected.
    compatible = (BinaryChromosome, IntegerChromosome, FloatChromosome)

    def __init__(self, points=None):
        if points is None:
            points = GAConfig().crossover_points
        if points < 1:
            raise ValueError("points must be at least 1")
        self.points = points

    def crossover(self, parent1, parent2):
        if getattr(parent1, "permutation", False) or getattr(parent2, "permutation", False):
            raise ValueError("MultiPointCrossover would break a permutation, use OrderCrossover")
        if len(parent1) != len(parent2):
            raise ValueError("both parents must have the same length")

        n = len(parent1)
        if n < 2:
            return make_child(parent1, parent1.genes[:]), make_child(parent2, parent2.genes[:])

        # a cut point is a position between two genes (1 .. n-1), so at most n-1 points
        k = min(self.points, n - 1)
        cuts = sorted(random.sample(range(1, n), k))
        cuts.append(n)

        genes1 = []
        genes2 = []
        start = 0
        swap = False
        for cut in cuts:
            if swap:
                genes1 = genes1 + parent2.genes[start:cut]
                genes2 = genes2 + parent1.genes[start:cut]
            else:
                genes1 = genes1 + parent1.genes[start:cut]
                genes2 = genes2 + parent2.genes[start:cut]
            start = cut
            swap = not swap

        return make_child(parent1, genes1), make_child(parent2, genes2)


class OrderCrossover(Crossover):
    # order crossover (OX) for permutations:
    # 1. keep a random slice of parent A in the same positions
    # 2. fill the other positions with the genes that are not in the slice,
    #    in the order they appear in parent B (starting after the slice, wrapping around)
    # no gene is repeated and none is lost, so the child is always a valid permutation.
    compatible = (IntegerChromosome,)

    def build_child(self, parent_a, parent_b, start, end):
        n = len(parent_a)
        child = [None] * n
        child[start:end + 1] = parent_a[start:end + 1]
        in_slice = set(parent_a[start:end + 1])

        remaining = []
        for i in range(n):
            gene = parent_b[(end + 1 + i) % n]
            if gene not in in_slice:
                remaining.append(gene)

        pos = (end + 1) % n
        for gene in remaining:
            child[pos] = gene
            pos = (pos + 1) % n
        return child

    def crossover(self, parent1, parent2):
        if not (parent1.permutation and parent2.permutation):
            raise ValueError("OrderCrossover needs IntegerChromosome(permutation=True)")
        if len(parent1) != len(parent2):
            raise ValueError("both parents must have the same length")

        n = len(parent1)
        if n < 2:
            return make_child(parent1, parent1.genes[:]), make_child(parent2, parent2.genes[:])

        start, end = sorted(random.sample(range(n), 2))
        genes1 = self.build_child(parent1.genes, parent2.genes, start, end)
        genes2 = self.build_child(parent2.genes, parent1.genes, start, end)
        return make_child(parent1, genes1), make_child(parent2, genes2)
