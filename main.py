# Member 1 demo script for GA toolbox
# (Member 3 will replace/extend this for CVRP integration)

from ga_toolbox.chromosomes import BinaryChromosome, FloatChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.ga import GeneticAlgorithm
from ga_toolbox.mutation import BitFlipMutation, GaussianMutation
from tests.dummy_operators import DummySelection, DummyCrossover, DummyReplacement


def run_onemax_demo():
    print("=== Demo 1: OneMax Problem (Binary Chromosome) ===")
    chrom_len = 30

    def make_chrom():
        return BinaryChromosome.random(chrom_len)

    def fitness_fn(c):
        return sum(c.genes)

    def feasibility_fn(c):
        return True

    cfg = GAConfig(population_size=40, max_generations=50, seed=42)

    ga = GeneticAlgorithm(
        chromosome_factory=make_chrom,
        fitness_fn=fitness_fn,
        feasibility_fn=feasibility_fn,
        selection=DummySelection(),
        crossover=DummyCrossover(),
        mutation=BitFlipMutation(gene_rate=0.05),
        replacement=DummyReplacement(),
        config=cfg
    )

    result = ga.run()
    print(f"Generations run: {result.generations_run}")
    print(f"Best fitness: {result.best.fitness} / {chrom_len}")
    print(f"Best genes: {result.best.genes}")
    print()


def run_sphere_demo():
    print("=== Demo 2: Sphere Function Minimization (Float Chromosome) ===")
    dim = 5
    low = -5.0
    high = 5.0

    def make_chrom():
        return FloatChromosome.random(dim, low, high)

    def fitness_fn(c):
        # f(x) = sum(x_i ^ 2), minimum is 0 at (0, 0, ...)
        total = 0.0
        for x in c.genes:
            total = total + (x * x)
        return total

    def feasibility_fn(c):
        return True

    cfg = GAConfig(population_size=50, max_generations=80, maximize=False, seed=42)

    ga = GeneticAlgorithm(
        chromosome_factory=make_chrom,
        fitness_fn=fitness_fn,
        feasibility_fn=feasibility_fn,
        selection=DummySelection(),
        crossover=DummyCrossover(),
        mutation=GaussianMutation(gene_rate=0.3, sigma_fraction=0.1),
        replacement=DummyReplacement(),
        config=cfg
    )

    result = ga.run()
    print(f"Generations run: {result.generations_run}")
    print(f"Best fitness (min distance to 0): {result.best.fitness:.4f}")
    rounded_genes = []
    for g in result.best.genes:
        rounded_genes.append(round(g, 4))
    print(f"Best genes: {rounded_genes}")
    print()


if __name__ == "__main__":
    print("Soft Computing Assignment 1 - Genetic Algorithm Toolbox")
    print("Testing toolbox with Member 1 foundation components...\n")
    run_onemax_demo()
    run_sphere_demo()
    print("Toolbox foundation working properly!")
