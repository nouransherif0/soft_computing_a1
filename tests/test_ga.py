import unittest

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.ga import GeneticAlgorithm
from ga_toolbox.mutation import BitFlipMutation, SwapMutation, GaussianMutation
from tests.dummy_operators import DummySelection, DummyCrossover, DummyReplacement

chrom_len = 20


def one_max(c):
    return sum(c.genes)


def always_feasible(c):
    return True


def make_binary():
    return BinaryChromosome.random(chrom_len)


def make_engine(config, feasibility_fn=always_feasible, repair_fn=None):
    return GeneticAlgorithm(make_binary, one_max, feasibility_fn,
                            DummySelection(), DummyCrossover(), BitFlipMutation(),
                            DummyReplacement(), config, repair_fn)


class TestEngine(unittest.TestCase):

    def test_onemax_maximize(self):
        config = GAConfig(population_size=30, max_generations=60, seed=1)
        result = make_engine(config).run()

        self.assertGreaterEqual(result.best.fitness, 18)
        self.assertEqual(result.generations_run, 60)
        self.assertEqual(len(result.best_history), 61)
        self.assertEqual(len(result.avg_history), 61)

    def test_onemax_minimize(self):
        config = GAConfig(population_size=30, max_generations=60, maximize=False, seed=1)
        result = make_engine(config).run()

        self.assertLessEqual(result.best.fitness, 2)

    def test_best_history_never_gets_worse(self):
        config = GAConfig(population_size=20, max_generations=30, seed=2)
        result = make_engine(config).run()

        for i in range(1, len(result.best_history)):
            self.assertGreaterEqual(result.best_history[i], result.best_history[i - 1])

    def test_same_seed_same_result(self):
        config1 = GAConfig(population_size=20, max_generations=20, seed=5)
        config2 = GAConfig(population_size=20, max_generations=20, seed=5)

        r1 = make_engine(config1).run()
        r2 = make_engine(config2).run()

        self.assertEqual(r1.best_history, r2.best_history)
        self.assertEqual(r1.best.genes, r2.best.genes)

    def test_patience_stops_early(self):
        config = GAConfig(population_size=30, max_generations=500, patience=10, seed=3)
        result = make_engine(config).run()

        self.assertLess(result.generations_run, 500)

    def test_penalty_strategy(self):
        # only solutions with at most 12 ones are allowed
        def feasible_fn(c):
            return sum(c.genes) <= 12

        config = GAConfig(population_size=40, max_generations=80, seed=4)
        result = make_engine(config, feasible_fn).run()

        self.assertTrue(result.best.feasible)
        self.assertEqual(result.best.fitness, 12)

    def test_penalty_is_applied(self):
        def never_feasible(c):
            return False

        config = GAConfig(population_size=5, max_generations=1, infeasible_penalty=1000)
        engine = make_engine(config, never_feasible)
        c = BinaryChromosome([1, 1, 1, 0])
        engine.evaluate(c)

        self.assertEqual(c.fitness, 3 - 1000)
        self.assertFalse(c.feasible)

    def test_penalty_is_added_when_minimizing(self):
        def never_feasible(c):
            return False

        config = GAConfig(maximize=False, infeasible_penalty=1000)
        engine = make_engine(config, never_feasible)
        c = BinaryChromosome([1, 1, 1, 0])
        engine.evaluate(c)

        self.assertEqual(c.fitness, 3 + 1000)

    def test_repair_strategy(self):
        def feasible_fn(c):
            return sum(c.genes) <= 12

        def repair_fn(c):
            fixed = c.copy()
            while sum(fixed.genes) > 12:
                fixed.genes[fixed.genes.index(1)] = 0
            return fixed

        config = GAConfig(population_size=40, max_generations=60,
                          infeasible_strategy="repair", seed=4)
        result = make_engine(config, feasible_fn, repair_fn).run()

        self.assertTrue(result.best.feasible)
        self.assertEqual(result.best.fitness, 12)

    def test_repair_without_function(self):
        config = GAConfig(infeasible_strategy="repair")
        with self.assertRaises(ValueError):
            make_engine(config)

    def test_bad_strategy_name(self):
        config = GAConfig(infeasible_strategy="ignore")
        with self.assertRaises(ValueError):
            make_engine(config)

    def test_incompatible_mutation(self):
        # binary chromosomes with a float mutation
        engine = GeneticAlgorithm(make_binary, one_max, always_feasible,
                                  DummySelection(), DummyCrossover(), GaussianMutation(),
                                  DummyReplacement(), GAConfig(population_size=5, max_generations=2))
        with self.assertRaises(TypeError):
            engine.run()

    def test_incompatible_crossover(self):
        def make_int():
            return IntegerChromosome.random(5, 1, 5, True)

        engine = GeneticAlgorithm(make_int, one_max, always_feasible,
                                  DummySelection(), DummyCrossover(), SwapMutation(),
                                  DummyReplacement(), GAConfig(population_size=5, max_generations=2))
        with self.assertRaises(TypeError):
            engine.run()

    def test_float_chromosomes_work(self):
        # minimize sum of x^2, target is all zeros
        def make_float():
            return FloatChromosome.random(4, -5, 5)

        def neg_squares(c):
            total = 0
            for g in c.genes:
                total = total - g * g
            return total

        engine = GeneticAlgorithm(make_float, neg_squares, always_feasible,
                                  DummySelection(), DummyCrossover(), GaussianMutation(),
                                  DummyReplacement(),
                                  GAConfig(population_size=40, max_generations=100, seed=7))
        result = engine.run()
        self.assertGreater(result.best.fitness, -0.5)


if __name__ == "__main__":
    unittest.main()

