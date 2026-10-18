import unittest

from ga_toolbox.chromosomes import IntegerChromosome
from cvrp.instance import CVRPInstance
from cvrp.decoder import CVRPDecoder, CVRPSolution, cvrp_chromosome_factory


def small_instance(vehicles=2):
    # depot (0,0), four customers with demand 3, vehicle capacity 6
    coords = [(0, 0), (3, 0), (3, 4), (0, 4), (6, 0)]
    demands = [0, 3, 3, 3, 3]
    return CVRPInstance(coords, demands, 6, vehicles)


class TestInstance(unittest.TestCase):

    def test_basic_properties(self):
        inst = small_instance()
        self.assertEqual(inst.num_customers, 4)
        self.assertEqual(inst.customer_ids, [1, 2, 3, 4])

    def test_distance(self):
        inst = small_instance()
        self.assertAlmostEqual(inst.distance(0, 2), 5.0)      # 3-4-5 triangle
        self.assertAlmostEqual(inst.distance(1, 4), 3.0)
        self.assertAlmostEqual(inst.distance(2, 2), 0.0)
        self.assertAlmostEqual(inst.distance(1, 3), inst.distance(3, 1))

    def test_route_load(self):
        inst = small_instance()
        self.assertEqual(inst.route_load([1, 2]), 6)
        self.assertEqual(inst.route_load([]), 0)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            CVRPInstance([(0, 0), (1, 1)], [0], 5, 1)


class TestDecoder(unittest.TestCase):

    def test_split_by_capacity(self):
        solution = CVRPDecoder(small_instance()).decode([1, 2, 3, 4])
        self.assertEqual(solution.routes, [[1, 2], [3, 4]])
        self.assertEqual(solution.unassigned, [])

    def test_order_is_kept(self):
        solution = CVRPDecoder(small_instance()).decode([4, 2, 1, 3])
        self.assertEqual(solution.routes, [[4, 2], [1, 3]])

    def test_not_enough_vehicles(self):
        solution = CVRPDecoder(small_instance(1)).decode([1, 2, 3, 4])
        self.assertEqual(solution.routes, [[1, 2]])
        self.assertEqual(solution.unassigned, [3, 4])

    def test_one_vehicle_enough(self):
        inst = CVRPInstance([(0, 0), (1, 0), (2, 0)], [0, 2, 2], 10, 1)
        solution = CVRPDecoder(inst).decode([2, 1])
        self.assertEqual(solution.routes, [[2, 1]])
        self.assertEqual(solution.unassigned, [])

    def test_customer_bigger_than_capacity(self):
        inst = CVRPInstance([(0, 0), (1, 0), (2, 0)], [0, 10, 2], 6, 2)
        solution = CVRPDecoder(inst).decode([1, 2])
        self.assertEqual(solution.routes, [[2]])
        self.assertEqual(solution.unassigned, [1])

    def test_every_customer_appears_once(self):
        inst = small_instance(2)
        solution = CVRPDecoder(inst).decode([3, 1, 4, 2])
        seen = []
        for route in solution.routes:
            seen = seen + route
        seen = seen + solution.unassigned
        self.assertEqual(sorted(seen), [1, 2, 3, 4])

    def test_loads_never_exceed_capacity(self):
        inst = small_instance(2)
        solution = CVRPDecoder(inst).decode([1, 2, 3, 4])
        for route in solution.routes:
            self.assertLessEqual(inst.route_load(route), inst.capacity)

    def test_accepts_chromosome(self):
        chromosome = IntegerChromosome([1, 2, 3, 4], 1, 4, True)
        solution = CVRPDecoder(small_instance()).decode(chromosome)
        self.assertEqual(solution.routes, [[1, 2], [3, 4]])

    def test_solution_type(self):
        self.assertIsInstance(CVRPDecoder(small_instance()).decode([1, 2, 3, 4]), CVRPSolution)


class TestRepresentation(unittest.TestCase):

    def test_factory_makes_permutations(self):
        inst = small_instance()
        make = cvrp_chromosome_factory(inst)
        for i in range(20):
            c = make()
            self.assertIsInstance(c, IntegerChromosome)
            self.assertTrue(c.permutation)
            self.assertEqual(sorted(c.genes), inst.customer_ids)


if __name__ == "__main__":
    unittest.main()
