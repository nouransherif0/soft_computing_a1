import unittest
from unittest.mock import MagicMock
from cvrp.fitness import calculate_route_distance, calculate_total_distance, fitness_evaluation


class TestCalculateRouteDistance(unittest.TestCase):
    
    

    def setUp(self):
        
        
        self.distance_matrix = [
            [0, 10, 20, 30],  # Distances from Depot (0)
            [10, 0, 15, 25],  # Distances from Customer 1
            [20, 15, 0, 10],  # Distances from Customer 2
            [30, 25, 10, 0]   # Distances from Customer 3
        ]

    def test_empty_route(self):  #Test that an empty route returns a distance of 0.0.
        
        
        self.assertEqual(calculate_route_distance([], self.distance_matrix), 0.0)

    def test_single_customer_route(self):
        
        
        route = [1]
        expected_distance = 20.0
        self.assertEqual(calculate_route_distance(route, self.distance_matrix), expected_distance)

    def test_multiple_customers_route(self):
        
        
        route = [1, 2]
        expected_distance = 45.0
        self.assertEqual(calculate_route_distance(route, self.distance_matrix), expected_distance)

    def test_longer_route(self):
       
       
        route = [1, 3, 2]
        expected_distance = 65.0
        self.assertEqual(calculate_route_distance(route, self.distance_matrix), expected_distance)


class TestCalculateTotalDistance(unittest.TestCase):
    """
    Unit tests for the calculate_total_distance function.
    """

    def setUp(self):
        self.distance_matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]

    def test_empty_routes_list(self): #Test that an empty list of routes returns 0.0.
        
        self.assertEqual(calculate_total_distance([], self.distance_matrix), 0.0)

    def test_multiple_routes(self):
        """
        Test total distance for multiple vehicle routes.
        Route 1: [1] -> 20.0
        Route 2: [2] -> dist(0,2) + dist(2,0) = 20 + 20 = 40.0
        Total expected: 60.0
        """
        routes = [[1], [2]]
        expected_total = 60.0
        self.assertEqual(calculate_total_distance(routes, self.distance_matrix), expected_total)


class TestFitnessEvaluation(unittest.TestCase):
    
    

    def setUp(self):
        self.distance_matrix = [
            [0, 10, 20],
            [10, 0, 15],
            [20, 15, 0]
        ]
        
        # Mock instance
        self.mock_instance = MagicMock()
        self.mock_instance.dist = self.distance_matrix  # Changed from distance_matrix to dist
        self.mock_instance.demands = [0, 5, 5]
        self.mock_instance.capacity = 10
        self.mock_instance.num_vehicles = 2

    def test_feasible_solution(self):
        
        
        mock_decoder = MagicMock()
        mock_solution = MagicMock()
        mock_solution.routes = [[1], [2]]
        mock_solution.unassigned = []
        mock_decoder.decode.return_value = mock_solution

        # Expected distance: Route [1] = 20, Route [2] = 40. Total = 60.
        expected_fitness = 60.0
        
        fitness = fitness_evaluation(None, self.mock_instance, mock_decoder)
        self.assertEqual(fitness, expected_fitness)

    def test_infeasible_unassigned_customers(self):
        
        
        mock_decoder = MagicMock()
        mock_solution = MagicMock()
        mock_solution.routes = [[1]]
        mock_solution.unassigned = [2]  # Customer 2 is unassigned
        mock_decoder.decode.return_value = mock_solution

        fitness = fitness_evaluation(None, self.mock_instance, mock_decoder)
        self.assertEqual(fitness, 1000000.0)

    def test_infeasible_capacity_violation(self):
       
       
        # Mock demands to force a capacity violation (demand 15 > capacity 10)
        self.mock_instance.demands = [0, 15, 5] 
        
        mock_decoder = MagicMock()
        mock_solution = MagicMock()
        mock_solution.routes = [[1, 2]]
        mock_solution.unassigned = []
        mock_decoder.decode.return_value = mock_solution

        fitness = fitness_evaluation(None, self.mock_instance, mock_decoder)
        # Should be base distance + penalty
        self.assertGreater(fitness, 1000000.0)


if __name__ == '__main__':
    unittest.main()