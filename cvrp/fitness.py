from cvrp.decoder import CVRPDecoder, CVRPSolution  # Import the decoder and solution classes from decoder.py
from cvrp.constraints import is_feasible 


def calculate_route_distance(route, distance_matrix):  # Calculate the distance of one vehicle route
    """
    The route starts and ends at the depot (index 0).
    """

    if not route:  # Check if the route is empty
        return 0.0  # Return zero because no distance is traveled
        
    total_distance = 0.0 
    current_node = 0  # Start at the depot
    
    for customer in route: # Visit each customer in the route
        total_distance += distance_matrix[current_node][customer] #Calculate total distance from each node to customer
        current_node = customer   # Updating the current location
        
    total_distance += distance_matrix[current_node][0] #after the car finishes all the routes, it will return back to the depot so we calculate this distance
    
    return total_distance  


def calculate_total_distance(routes, distance_matrix):  #Calculate the total distance for all vehicles
    """
    it will return: Sum of distances of all routes.
    """
    total_distance = 0.0   
    for route in routes:  # Loop through all vehicle routes
        total_distance += calculate_route_distance(route, distance_matrix)  # Add each route's distance
    return total_distance


def fitness_evaluation(chromosome, instance, decoder):  #fitness function
    
    # Decode the chromosome into actual routes using the provided decoder
    solution = decoder.decode(chromosome)   # Convert the chromosome into vehicle routes
    routes = solution.routes   # Get the routes from the decoded solution
    
    #Check if there are any unassigned customers (infeasible solution)
    if len(solution.unassigned) > 0:  # Check if any customers were not assigned
        return 1000000.0  # Large penalty for infeasible solutions
    

    total_distance = calculate_total_distance(routes, instance.dist) 
    
    if not is_feasible(routes, instance.demands, [instance.capacity] * instance.num_vehicles): # Check capacity and route constraints
        total_distance += 1000000.0  # Penalty if constraints are violated
        
    return total_distance


def verify_fitness_by_an_example(): 
    """
    verification:
    Scenario:
    - Depot: Node 0
    - Customers: Node 1, Node 2
    - Distance Matrix:
      0 -> 1 = 10
      1 -> 2 = 15
      2 -> 0 = 20
    - Route: [1, 2]
    
    Calculation:
    Distance = dist(0,1) + dist(1,2) + dist(2,0)
    Distance = 10 + 15 + 20 = 45
    """
    #
    
    distance_matrix = [
        [0, 10, 25],  # Distances from the depot (node 0)
        [10, 0, 15],  # Distances from customer 1
        [25, 15, 0]   # Distances from customer 2
    ]
    
    # Define a single route visiting customer 1 then customer 2
    route = [1, 2]
    
    # Calculate using the function
    calculated_distance = calculate_route_distance(route, distance_matrix)
    
    # Expected result from manual calculation
    expected_distance = 45
    
    print(f"Manual Calculation: {expected_distance}")
    print(f"Function Output: {calculated_distance}")
    
    assert calculated_distance == expected_distance, "Fitness function calculation is incorrect!"
    print("Test Passed: Fitness function matches manual calculation.")


if __name__ == '__main__':
    verify_fitness_by_an_example()