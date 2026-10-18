import math


class CVRPInstance:

    def __init__(self, coords, demands, capacity, num_vehicles, name="cvrp"):
        if len(coords) != len(demands):
            raise ValueError("coords and demands must have the same length")
        if len(coords) < 2:
            raise ValueError("need a depot and at least one customer")
        self.coords = list(coords)
        self.demands = list(demands)
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.name = name

        # distance matrix, computed once because fitness is called thousands of times
        n = len(self.coords)
        self.dist = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(math.dist(self.coords[i], self.coords[j]))
            self.dist.append(row)

    @property
    def num_customers(self):
        return len(self.coords) - 1

    @property
    def customer_ids(self):
        return list(range(1, len(self.coords)))

    def distance(self, i, j):
        return self.dist[i][j]

    def route_load(self, route):
        total = 0
        for c in route:
            total = total + self.demands[c]
        return total
