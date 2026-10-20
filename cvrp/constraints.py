def is_feasible(routes, demands, vehicle_capacities):
   
 #  Check capacity constraints for each vehicle
    for vehicle_id, route in enumerate(routes): #loop through each vehicle and its route
        total_load = 0
        for customer in route:
            total_load = total_load + demands[customer] # Calculate total load of the route, demand[customer]>>the weight of the packages
         
        if total_load > vehicle_capacities[vehicle_id]: # Check if load exceeds max capacity
            return False # Return False if capacity is exceeded

    
    #check customer visits
    all_customers = [] # an empty list that will contain customers on the route
    for route in routes: # Loop through each route
        all_customers.extend(route) # Add all customers to the list
    

    #  Check if any customer is visited more than once
    if len(all_customers) != len(set(all_customers)): # Check for duplicate customers, set>> collection of unique values onlyyyy
        return False # Return False if duplicates exist
    
    # Check if all customers are visited
    # The number of customers is the total number of locations minus the depot (index 0)
    num_customers = len(demands) - 1 
    if len(all_customers) != num_customers:  # Check if any customers are missing
        return False  # Return False if the number of customers is incorrect
    
    return True # Return True if all constraints are satisfied