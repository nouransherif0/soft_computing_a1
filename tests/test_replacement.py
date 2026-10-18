import unittest

from ga_toolbox.chromosomes import BinaryChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.replacement import GenerationalReplacement, MuPlusLambdaReplacement


def make_population(fitnesses):
    population = []
    for f in fitnesses:
        c = BinaryChromosome([0])
        c.fitness = f
        population.append(c)
    return population


def fits(population):
    return sorted(c.fitness for c in population)


class TestGenerational(unittest.TestCase):

    def test_default_elitism_from_config(self):
        self.assertEqual(GenerationalReplacement().elitism_count, GAConfig().elitism_count)

    def test_keeps_best_old_maximize(self):
        old = make_population([10, 9, 1, 1, 1])
        new = make_population([2, 3, 4, 5, 6])
        result = GenerationalReplacement(2).replace(old, new, True)
        self.assertEqual(fits(result), [4, 5, 6, 9, 10])

    def test_keeps_best_old_minimize(self):
        old = make_population([1, 2, 50, 60])
        new = make_population([30, 40, 10, 20])
        result = GenerationalReplacement(1).replace(old, new, False)
        self.assertEqual(fits(result), [1, 10, 20, 30])

    def test_size_stays_fixed(self):
        old = make_population([5, 4, 3, 2, 1])
        for count in (0, 1, 2, 5, 9):
            result = GenerationalReplacement(count).replace(old, make_population([9] * 5), True)
            self.assertEqual(len(result), 5)

    def test_zero_elitism_takes_offspring_only(self):
        old = make_population([100, 100, 100])
        new = make_population([1, 2, 3])
        result = GenerationalReplacement(0).replace(old, new, True)
        self.assertEqual(fits(result), [1, 2, 3])

    def test_too_few_offspring(self):
        old = make_population([5, 4, 3, 2])
        result = GenerationalReplacement(1).replace(old, make_population([9]), True)
        self.assertEqual(fits(result), [3, 4, 5, 9])

    def test_best_is_never_lost(self):
        old = make_population([7, 3, 2])
        result = GenerationalReplacement(1).replace(old, make_population([1, 1, 1]), True)
        self.assertIn(7, fits(result))

    def test_inputs_not_modified(self):
        old = make_population([3, 1, 2])
        new = make_population([9, 8, 7])
        GenerationalReplacement(1).replace(old, new, True)
        self.assertEqual([c.fitness for c in old], [3, 1, 2])
        self.assertEqual([c.fitness for c in new], [9, 8, 7])

    def test_negative_elitism(self):
        with self.assertRaises(ValueError):
            GenerationalReplacement(-1)


class TestMuPlusLambda(unittest.TestCase):

    def test_best_of_both_maximize(self):
        result = MuPlusLambdaReplacement().replace(make_population([1, 2, 3]),
                                                   make_population([10, 0, 7]), True)
        self.assertEqual(fits(result), [3, 7, 10])

    def test_best_of_both_minimize(self):
        result = MuPlusLambdaReplacement().replace(make_population([1, 2, 3]),
                                                   make_population([10, 0, 7]), False)
        self.assertEqual(fits(result), [0, 1, 2])

    def test_size_stays_fixed(self):
        result = MuPlusLambdaReplacement().replace(make_population([1, 2, 3]),
                                                   make_population([4] * 8), True)
        self.assertEqual(len(result), 3)

    def test_inputs_not_modified(self):
        old = make_population([3, 1, 2])
        new = make_population([9, 8, 7])
        MuPlusLambdaReplacement().replace(old, new, True)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(new), 3)


if __name__ == "__main__":
    unittest.main()
