import unittest

from ga_toolbox.chromosomes import BinaryChromosome, IntegerChromosome, FloatChromosome
from ga_toolbox.mutation import BitFlipMutation, SwapMutation, InversionMutation, GaussianMutation


class TestMutation(unittest.TestCase):

    def test_bitflip_all(self):
        c = BinaryChromosome([0, 1, 0, 1, 1])
        BitFlipMutation(gene_rate=1.0).mutate(c)
        self.assertEqual(c.genes, [1, 0, 1, 0, 0])

    def test_bitflip_none(self):
        c = BinaryChromosome([0, 1, 0, 1, 1])
        BitFlipMutation(gene_rate=-1).mutate(c)
        self.assertEqual(c.genes, [0, 1, 0, 1, 1])

    def test_bitflip_resets_fitness(self):
        c = BinaryChromosome([0, 1])
        c.fitness = 3
        BitFlipMutation().mutate(c)
        self.assertIsNone(c.fitness)

    def test_swap_changes_two_positions(self):
        for i in range(50):
            c = IntegerChromosome.random(8, 1, 8, True)
            old = c.genes[:]
            SwapMutation().mutate(c)

            self.assertEqual(sorted(c.genes), [1, 2, 3, 4, 5, 6, 7, 8])

            diff_count = 0
            for k in range(8):
                if old[k] != c.genes[k]:
                    diff_count = diff_count + 1

            self.assertEqual(diff_count, 2)

    def test_swap_resets_fitness(self):
        c = IntegerChromosome([1, 2, 3], 1, 3, True)
        c.fitness = 10
        SwapMutation().mutate(c)
        self.assertIsNone(c.fitness)

    def test_inversion_keeps_permutation(self):
        for i in range(50):
            c = IntegerChromosome.random(8, 1, 8, True)
            InversionMutation().mutate(c)
            self.assertEqual(sorted(c.genes), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_inversion_reverses_part(self):
        c = IntegerChromosome([1, 2, 3, 4, 5], 1, 5, True)
        InversionMutation().mutate(c)

        # permutation elements remain same but order changes
        self.assertEqual(sorted(c.genes), [1, 2, 3, 4, 5])
        self.assertNotEqual(c.genes, [1, 2, 3, 4, 5])

    def test_inversion_resets_fitness(self):
        c = IntegerChromosome([1, 2, 3], 1, 3, True)
        c.fitness = 55
        InversionMutation().mutate(c)
        self.assertIsNone(c.fitness)

    def test_gaussian_stays_in_bounds(self):
        for i in range(100):
            c = FloatChromosome.random(10, -10, 10)
            GaussianMutation(gene_rate=1.0, sigma_fraction=0.5).mutate(c)

            for g in c.genes:
                self.assertTrue(-10 <= g <= 10)

    def test_gaussian_changes_genes(self):
        c = FloatChromosome([0.0, 0.0, 0.0], -10, 10)
        GaussianMutation(gene_rate=1.0).mutate(c)
        self.assertNotEqual(c.genes, [0.0, 0.0, 0.0])

    def test_returns_same_object(self):
        c = FloatChromosome([1.0, 2.0], -10, 10)
        self.assertIs(GaussianMutation().mutate(c), c)


if __name__ == "__main__":
    unittest.main()

