import unittest

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome


class TestChromosomes(unittest.TestCase):

    def test_binary_random(self):
        c = BinaryChromosome.random(10)
        self.assertEqual(len(c), 10)

        for g in c.genes:
            self.assertIn(g, (0, 1))

    def test_binary_copy_is_independent(self):
        c = BinaryChromosome.random(10)
        c.fitness = 5
        d = c.copy()

        d.genes[0] = 1 - d.genes[0]
        self.assertNotEqual(c.genes[0], d.genes[0])
        self.assertEqual(d.fitness, 5)

    def test_integer_range(self):
        c = IntegerChromosome.random(20, 1, 10)

        for g in c.genes:
            self.assertTrue(1 <= g <= 10)

    def test_integer_permutation(self):
        c = IntegerChromosome.random(6, 1, 6, True)
        self.assertEqual(sorted(c.genes), [1, 2, 3, 4, 5, 6])

        d = c.copy()
        self.assertTrue(d.permutation)
        self.assertEqual(c.genes, d.genes)

    def test_integer_permutation_wrong_length(self):
        with self.assertRaises(ValueError):
            IntegerChromosome.random(5, 1, 10, True)

    def test_integer_copy_independent(self):
        c = IntegerChromosome([1, 2, 3], 1, 3, True)
        d = c.copy()
        d.genes[0] = 99
        self.assertNotEqual(c.genes[0], d.genes[0])

    def test_float_random_in_bounds(self):
        c = FloatChromosome.random(30, -10, 10)

        for g in c.genes:
            self.assertTrue(-10 <= g <= 10)

    def test_float_clipping(self):
        c = FloatChromosome([15.5, -20.0, 3.2], -10, 10)
        self.assertEqual(c.genes, [10, -10, 3.2])

    def test_float_copy_independent(self):
        c = FloatChromosome([1.0, 2.0], -5, 5)
        d = c.copy()
        d.genes[0] = 4.0
        self.assertNotEqual(c.genes[0], d.genes[0])

    def test_chromosome_len_and_repr(self):
        c = BinaryChromosome([0, 1, 1])
        self.assertEqual(len(c), 3)
        self.assertEqual(str(c), "[0, 1, 1]")


if __name__ == "__main__":
    unittest.main()

