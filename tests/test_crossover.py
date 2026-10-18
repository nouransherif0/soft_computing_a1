import random
import unittest

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.crossover import MultiPointCrossover, OrderCrossover


class TestMultiPoint(unittest.TestCase):

    def setUp(self):
        random.seed(3)

    def test_default_points_from_config(self):
        self.assertEqual(MultiPointCrossover().points, GAConfig().crossover_points)

    def test_binary_children_complement_each_other(self):
        p1 = BinaryChromosome([0] * 12)
        p2 = BinaryChromosome([1] * 12)
        for points in (1, 2, 3, 5):
            for i in range(30):
                c1, c2 = MultiPointCrossover(points).crossover(p1, p2)
                self.assertEqual(len(c1), 12)
                for k in range(12):
                    self.assertEqual(c1.genes[k] + c2.genes[k], 1)

    def test_number_of_cuts(self):
        # genes are 0 and 1, so the number of switches between them is the number of cuts
        p1 = BinaryChromosome([0] * 20)
        p2 = BinaryChromosome([1] * 20)
        for points in (1, 2, 3, 4):
            c1, c2 = MultiPointCrossover(points).crossover(p1, p2)
            switches = 0
            for k in range(19):
                if c1.genes[k] != c1.genes[k + 1]:
                    switches = switches + 1
            self.assertEqual(switches, points)

    def test_parents_are_not_changed(self):
        p1 = BinaryChromosome([0] * 8)
        p2 = BinaryChromosome([1] * 8)
        c1, c2 = MultiPointCrossover().crossover(p1, p2)
        self.assertEqual(p1.genes, [0] * 8)
        self.assertEqual(p2.genes, [1] * 8)
        self.assertIsNot(c1, p1)
        self.assertIsNot(c2, p2)

    def test_children_need_evaluation(self):
        p1 = BinaryChromosome([0] * 6)
        p2 = BinaryChromosome([1] * 6)
        p1.fitness = 5
        p2.fitness = 5
        c1, c2 = MultiPointCrossover().crossover(p1, p2)
        self.assertIsNone(c1.fitness)
        self.assertIsNone(c2.fitness)

    def test_float_keeps_type_and_bounds(self):
        p1 = FloatChromosome([0.1, 0.2, 0.3, 0.4], -1, 1)
        p2 = FloatChromosome([0.5, 0.6, 0.7, 0.8], -1, 1)
        c1, c2 = MultiPointCrossover().crossover(p1, p2)
        self.assertIsInstance(c1, FloatChromosome)
        self.assertEqual((c1.low, c1.high), (-1, 1))
        self.assertEqual(sorted(c1.genes + c2.genes), sorted(p1.genes + p2.genes))

    def test_integer_non_permutation(self):
        p1 = IntegerChromosome([1, 2, 3, 4], 0, 9)
        p2 = IntegerChromosome([5, 6, 7, 8], 0, 9)
        c1, c2 = MultiPointCrossover().crossover(p1, p2)
        self.assertIsInstance(c1, IntegerChromosome)
        self.assertEqual(sorted(c1.genes + c2.genes), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_too_many_points_for_short_chromosome(self):
        p1 = BinaryChromosome([0, 0])
        p2 = BinaryChromosome([1, 1])
        c1, c2 = MultiPointCrossover(5).crossover(p1, p2)
        self.assertEqual(c1.genes, [0, 1])
        self.assertEqual(c2.genes, [1, 0])

    def test_rejects_permutation(self):
        p1 = IntegerChromosome([1, 2, 3], 1, 3, True)
        p2 = IntegerChromosome([3, 2, 1], 1, 3, True)
        with self.assertRaises(ValueError):
            MultiPointCrossover().crossover(p1, p2)

    def test_different_lengths(self):
        with self.assertRaises(ValueError):
            MultiPointCrossover().crossover(BinaryChromosome([0, 1]), BinaryChromosome([0, 1, 1]))

    def test_bad_points(self):
        with self.assertRaises(ValueError):
            MultiPointCrossover(0)

    def test_compatible_types(self):
        self.assertEqual(set(MultiPointCrossover.compatible),
                         {BinaryChromosome, IntegerChromosome, FloatChromosome})


class TestOrder(unittest.TestCase):

    def setUp(self):
        random.seed(4)

    def test_known_example(self):
        # slice of p1 at positions 3..6 stays, the rest follows the order of p2
        p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]
        child = OrderCrossover().build_child(p1, p2, 3, 6)
        self.assertEqual(child, [3, 8, 2, 4, 5, 6, 7, 1, 9])

    def test_children_are_always_valid_permutations(self):
        for i in range(500):
            n = random.randint(2, 15)
            g1 = list(range(1, n + 1))
            g2 = list(range(1, n + 1))
            random.shuffle(g1)
            random.shuffle(g2)
            p1 = IntegerChromosome(g1, 1, n, True)
            p2 = IntegerChromosome(g2, 1, n, True)

            c1, c2 = OrderCrossover().crossover(p1, p2)

            self.assertEqual(sorted(c1.genes), list(range(1, n + 1)))
            self.assertEqual(sorted(c2.genes), list(range(1, n + 1)))
            self.assertEqual(p1.genes, g1)
            self.assertEqual(p2.genes, g2)

    def test_child_keeps_permutation_settings(self):
        p1 = IntegerChromosome.random(6, 1, 6, True)
        p2 = IntegerChromosome.random(6, 1, 6, True)
        c1, c2 = OrderCrossover().crossover(p1, p2)
        self.assertTrue(c1.permutation)
        self.assertEqual((c1.low, c1.high), (1, 6))
        self.assertIsNone(c1.fitness)

    def test_identical_parents_give_same_child(self):
        p = IntegerChromosome.random(8, 1, 8, True)
        c1, c2 = OrderCrossover().crossover(p, p.copy())
        self.assertEqual(c1.genes, p.genes)
        self.assertEqual(c2.genes, p.genes)

    def test_rejects_non_permutation(self):
        p1 = IntegerChromosome([1, 2, 3], 0, 9)
        p2 = IntegerChromosome([3, 2, 1], 0, 9)
        with self.assertRaises(ValueError):
            OrderCrossover().crossover(p1, p2)

    def test_compatible_types(self):
        self.assertEqual(OrderCrossover.compatible, (IntegerChromosome,))
        self.assertNotIn(FloatChromosome, OrderCrossover.compatible)


if __name__ == "__main__":
    unittest.main()
