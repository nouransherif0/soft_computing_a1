import random
import unittest

from ga_toolbox.chromosomes import BinaryChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.selection import TournamentSelection, RouletteWheelSelection


def make_population(fitnesses):
    population = []
    for i in range(len(fitnesses)):
        c = BinaryChromosome([i])
        c.fitness = fitnesses[i]
        population.append(c)
    return population


def count_picks(chosen):
    counts = {}
    for c in chosen:
        counts[c.fitness] = counts.get(c.fitness, 0) + 1
    return counts


class TestTournament(unittest.TestCase):

    def setUp(self):
        random.seed(1)

    def test_returns_n_individuals(self):
        pop = make_population([1, 2, 3, 4, 5])
        self.assertEqual(len(TournamentSelection().select(pop, 12, True)), 12)

    def test_default_size_comes_from_config(self):
        self.assertEqual(TournamentSelection().tournament_size, GAConfig().tournament_size)

    def test_maximize_prefers_big(self):
        pop = make_population(list(range(1, 11)))
        counts = count_picks(TournamentSelection(3).select(pop, 3000, True))
        self.assertGreater(counts[10], counts.get(1, 0))

    def test_minimize_prefers_small(self):
        pop = make_population(list(range(1, 11)))
        counts = count_picks(TournamentSelection(3).select(pop, 3000, False))
        self.assertGreater(counts[1], counts.get(10, 0))

    def test_size_one_is_random(self):
        pop = make_population([1, 2, 3])
        counts = count_picks(TournamentSelection(1).select(pop, 3000, True))
        self.assertEqual(len(counts), 3)

    def test_size_bigger_than_population(self):
        pop = make_population([1, 2])
        chosen = TournamentSelection(10).select(pop, 5, True)
        self.assertEqual(len(chosen), 5)

    def test_bad_size(self):
        with self.assertRaises(ValueError):
            TournamentSelection(0)


class TestRoulette(unittest.TestCase):

    def setUp(self):
        random.seed(2)

    def test_returns_n_individuals(self):
        pop = make_population([1, 2, 3])
        self.assertEqual(len(RouletteWheelSelection().select(pop, 7, True)), 7)

    def test_maximize_prefers_big(self):
        pop = make_population([1, 2, 3, 4, 10])
        counts = count_picks(RouletteWheelSelection().select(pop, 3000, True))
        self.assertGreater(counts[10], counts.get(1, 0))

    def test_minimize_prefers_small(self):
        pop = make_population([1, 2, 3, 4, 10])
        counts = count_picks(RouletteWheelSelection().select(pop, 3000, False))
        self.assertGreater(counts[1], counts.get(10, 0))

    def test_negative_fitness_maximize(self):
        pop = make_population([-50, -20, -10, 0, 5])
        counts = count_picks(RouletteWheelSelection().select(pop, 3000, True))
        self.assertGreater(counts[5], counts.get(-50, 0))

    def test_negative_fitness_minimize(self):
        pop = make_population([-50, -20, -10, 0, 5])
        counts = count_picks(RouletteWheelSelection().select(pop, 3000, False))
        self.assertGreater(counts[-50], counts.get(5, 0))

    def test_all_equal_fitness(self):
        pop = make_population([4, 4, 4])
        self.assertEqual(len(RouletteWheelSelection().select(pop, 9, True)), 9)

    def test_weights_are_never_negative(self):
        pop = make_population([-5, 0, 7])
        for maximize in (True, False):
            for w in RouletteWheelSelection().get_weights(pop, maximize):
                self.assertGreater(w, 0)


if __name__ == "__main__":
    unittest.main()
