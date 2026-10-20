import unittest
from cvrp.constraints import is_feasible # Import  isfeasibil function

class TestCVRPConstraints(unittest.TestCase): # defining  test class


    def setUp(self):  # Set up>> runs automatically before eachtest method , self>> current instance of the class    
        self.demands = [0, 5, 5, 5, 5] # Set customer demands (4 customers)
        self.capacities = [15, 15] # Set vehicle capacities (2 vehicles)


    def test_feasible_solution(self):  # Test a valid solution where all constraints are met  
        routes = [[1, 2], [3, 4]] # Assign customers to vehicles (2 routes)
        self.assertTrue(is_feasible(routes, self.demands, self.capacities)) # Check that the solution is feasible


    def test_infeasible_capacity_exceeded(self): # Test capacity violation where a vehicle's load exceeds its maximum capacity.
        routes = [[1, 2, 3, 4], []] # vehicle 0 >> has a route, vehicle 1>> has no route and will not visit any customer
        self.assertFalse(is_feasible(routes, self.demands, self.capacities)) # capacity= 20, so infeasible


    def test_infeasible_duplicate_customer(self): # Test if a customer is visited more than once
        routes = [[1, 2, 1], [3, 4]] # Customer 1 is duplicated
        self.assertFalse(is_feasible(routes, self.demands, self.capacities)) # Check that the solution is infeasible


    def test_infeasible_missing_customer(self):
        routes = [[1, 2], [3]] # Customer 4 is missing
        self.assertFalse(is_feasible(routes, self.demands, self.capacities))  # Check that the solution is infeasible


    def test_empty_routes(self):
        routes = [[], []]
        self.assertFalse(is_feasible(routes, self.demands, self.capacities))


if __name__ == '__main__':
    unittest.main()  # Run all unit tests