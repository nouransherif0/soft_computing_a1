from dataclasses import dataclass, field

from ga_toolbox.chromosomes import IntegerChromosome


@dataclass
class CVRPSolution:

    routes: list = field(default_factory=list)
    unassigned: list = field(default_factory=list)


class CVRPDecoder:

    def __init__(self, instance):
        self.instance = instance

    def decode(self, tour):
        # tour can be a chromosome or a plain list
        genes = getattr(tour, "genes", tour)
        inst = self.instance

        routes = []
        unassigned = []
        current = []
        load = 0
        out_of_vehicles = False

        for customer in genes:
            demand = inst.demands[customer]

            if out_of_vehicles or demand > inst.capacity:
                unassigned.append(customer)
                continue

            if load + demand > inst.capacity:
                # current vehicle is full, close it and try to open the next one
                routes.append(current)
                current = []
                load = 0
                if len(routes) == inst.num_vehicles:
                    out_of_vehicles = True
                    unassigned.append(customer)
                    continue

            current.append(customer)
            load = load + demand

        if len(current) > 0:
            routes.append(current)

        return CVRPSolution(routes, unassigned)


def cvrp_chromosome_factory(instance):
    # representation: one IntegerChromosome(permutation=True) holding all customer ids
    # (the "giant tour"). Use it as chromosome_factory of GeneticAlgorithm.
    n = instance.num_customers

    def factory():
        return IntegerChromosome.random(n, 1, n, True)

    return factory
