# Member 1 demo script for GA toolbox
# (Member 3 will replace/extend this for CVRP integration)

from ga_toolbox.chromosomes import BinaryChromosome, FloatChromosome
from ga_toolbox.config import GAConfig
from ga_toolbox.ga import GeneticAlgorithm
from ga_toolbox.mutation import BitFlipMutation, GaussianMutation
from tests.dummy_operators import DummySelection, DummyCrossover, DummyReplacement

from cvrp.instance import CVRPInstance
from cvrp.decoder import CVRPDecoder, cvrp_chromosome_factory
from cvrp.fitness import fitness_evaluation
from cvrp.constraints import is_feasible
import time


def run_onemax_demo():
    print(" Demo 1: OneMax Problem (Binary Chromosome) ")
    chrom_len = 30

    def make_chrom():
        return BinaryChromosome.random(chrom_len)

    def fitness_fn(c):
        return sum(c.genes)

    def feasibility_fn(c):
        return True

    cfg = GAConfig(population_size=40, max_generations=50, seed=42)

    ga = GeneticAlgorithm(
        chromosome_factory=make_chrom,
        fitness_fn=fitness_fn,
        feasibility_fn=feasibility_fn,
        selection=DummySelection(),
        crossover=DummyCrossover(),
        mutation=BitFlipMutation(gene_rate=0.05),
        replacement=DummyReplacement(),
        config=cfg
    )

    result = ga.run()
    print(f"Generations run: {result.generations_run}")
    print(f"Best fitness: {result.best.fitness} / {chrom_len}")
    print(f"Best genes: {result.best.genes}")
    print()


def run_sphere_demo():
    print(" Demo 2: Sphere Function Minimization (Float Chromosome) ")
    dim = 5
    low = -5.0
    high = 5.0

    def make_chrom():
        return FloatChromosome.random(dim, low, high)

    def fitness_fn(c):
        # f(x) = sum(x_i ^ 2), minimum is 0 at (0, 0, ...)
        total = 0.0
        for x in c.genes:
            total = total + (x * x)
        return total

    def feasibility_fn(c):
        return True

    cfg = GAConfig(population_size=50, max_generations=80, maximize=False, seed=42)

    ga = GeneticAlgorithm(
        chromosome_factory=make_chrom,
        fitness_fn=fitness_fn,
        feasibility_fn=feasibility_fn,
        selection=DummySelection(),
        crossover=DummyCrossover(),
        mutation=GaussianMutation(gene_rate=0.3, sigma_fraction=0.1),
        replacement=DummyReplacement(),
        config=cfg
    )

    result = ga.run()
    print(f"Generations run: {result.generations_run}")
    print(f"Best fitness (min distance to 0): {result.best.fitness:.4f}")
    rounded_genes = []
    for g in result.best.genes:
        rounded_genes.append(round(g, 4))
    print(f"Best genes: {rounded_genes}")
    print()


if __name__ == "__main__":
    print("Soft Computing Assignment 1 - Genetic Algorithm Toolbox")
    print("Testing toolbox with Member 1 foundation components...\n")
    run_onemax_demo()
    run_sphere_demo()

    print("Toolbox foundation working properly")


# Hanin

def run_cvrp_demo():
    print("Demo 3: CVRP ")
    
    # Create a  CVRP instance for testing
    # Node 0 is the depot, Nodes 1-4 are customers
    coords = [
        (0, 0),    # Depot (Node 0)
        (10, 5),   # Customer 1
        (5, 15),   # Customer 2
        (20, 10),  # Customer 3
        (15, 20),  # Customer 4
    ]
    demands = [0, 5, 5, 5, 5]  # Depot demand is 0
    capacity = 10
    num_vehicles = 2
    
    instance = CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
        num_vehicles=num_vehicles,
        name="small_cvrp_test"
    )
    
    print(f"Instance: {instance.num_customers} customers, {instance.num_vehicles} vehicles")
    print(f"Vehicle capacity: {instance.capacity}")
    
    # Initialize the decoder
    decoder = CVRPDecoder(instance)
    
    #  Define fitness and feasibility wrappers (using Public API only)
    def cvrp_fitness(chromosome):
        return fitness_evaluation(chromosome, instance, decoder)
    
    def cvrp_feasibility(chromosome):
        solution = decoder.decode(chromosome)
        if len(solution.unassigned) > 0:
            return False
        return is_feasible(
            solution.routes,
            instance.demands,
            [instance.capacity] * instance.num_vehicles
        )
    
    #  Configure GA
    cfg = GAConfig(
        population_size=50,
        max_generations=100,
        maximize=False,  # We want to MINIMIZE distance
        seed=42
    )
    
    #  Create chromosome factory using the decoder's factory
    chrom_factory = cvrp_chromosome_factory(instance)
    
    
    # Note: Replace these with the real operator classes your team implemented
    from ga_toolbox.selection import TournamentSelection
    from ga_toolbox.crossover import OrderCrossover
    from ga_toolbox.mutation import SwapMutation
    from ga_toolbox.replacement import GenerationalReplacement
    
    selection = TournamentSelection()
    crossover = OrderCrossover()
    mutation = SwapMutation()
    replacement = GenerationalReplacement()
    
    # Initialize and run the GA 
    ga = GeneticAlgorithm(
        chromosome_factory=chrom_factory,
        fitness_fn=cvrp_fitness,
        feasibility_fn=cvrp_feasibility,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        replacement=replacement,
        config=cfg
    )
    
    result = ga.run()
    
    print(f"Generations run: {result.generations_run}")
    print(f"Best fitness (total distance): {result.best.fitness:.2f}")
    
    #  Decode and print the final solution
    best_solution = decoder.decode(result.best)
    
    print("\n--- Best Routes ---")
    for i, route in enumerate(best_solution.routes):
        print(f"Vehicle {i}: {route} | Load: {instance.route_load(route)}")
    
    if best_solution.unassigned:
        print(f"WARNING: Unassigned customers: {best_solution.unassigned}")
    else:
        print("All customers assigned successfully")
    print()



#Last Part

    

def run_tuning_experiments(instance):
    
    print("\n Performance Tuning and Results Table \n")
    
    # Define different configurations to test (Population Size, Mutation Rate, Generations)
    configurations = [
        {"pop_size": 50, "mut_rate": 0.1, "generations": 100},
        {"pop_size": 100, "mut_rate": 0.1, "generations": 200},
        {"pop_size": 100, "mut_rate": 0.05, "generations": 300},
        {"pop_size": 150, "mut_rate": 0.15, "generations": 500},
    ]

    # Print table header
    header = f"{'Run':<5} | {'Pop Size':<10} | {'Mut Rate':<10} | {'Gens':<6} | {'Distance':<12} | {'Feasible':<10} | {'Time(s)':<8}"
    print(header)
    print("-" * len(header))

    decoder = CVRPDecoder(instance)

    for i, params in enumerate(configurations, 1):
        #  Define fitness and feasibility wrappers
        def cvrp_fitness(chromosome):
            return fitness_evaluation(chromosome, instance, decoder)
        
        def cvrp_feasibility(chromosome):
            solution = decoder.decode(chromosome)
            if len(solution.unassigned) > 0:
                return False
            return is_feasible(solution.routes, instance.demands, [instance.capacity] * instance.num_vehicles)

        # Configure GA with current parameters
        cfg = GAConfig(
            population_size=params["pop_size"],
            max_generations=params["generations"],
            maximize=False,
            seed=42
        )

       
        from ga_toolbox.selection import TournamentSelection
        from ga_toolbox.crossover import OrderCrossover
        from ga_toolbox.mutation import SwapMutation
        from ga_toolbox.replacement import GenerationalReplacement

        ga = GeneticAlgorithm(
            chromosome_factory=cvrp_chromosome_factory(instance),
            fitness_fn=cvrp_fitness,
            feasibility_fn=cvrp_feasibility,
            selection=TournamentSelection(),
            crossover=OrderCrossover(),
            mutation=SwapMutation(),
            replacement=GenerationalReplacement(),
            config=cfg
        )

        #  Run and measure time
        start_time = time.time()
        result = ga.run()
        elapsed_time = time.time() - start_time

        #  Check feasibility of the best solution
        best_solution = decoder.decode(result.best)
        is_feasible_sol = len(best_solution.unassigned) == 0 and is_feasible(
            best_solution.routes, instance.demands, [instance.capacity] * instance.num_vehicles
        )

        #  Print row in the table
        row = f"{i:<5} | {params['pop_size']:<10} | {params['mut_rate']:<10} | {params['generations']:<6} | {result.best.fitness:<12.2f} | {'Yes' if is_feasible_sol else 'No':<10} | {elapsed_time:<8.2f}"
        print(row)

    print("-" * len(header))
    print("Tuning completed. The configuration with the lowest feasible distance is recommended.")



if __name__ == "__main__":
    print("Soft Computing Assignment 1 - Genetic Algorithm Toolbox")
    print("Testing toolbox with all components...\n")
    
    # Member 1 demos
    run_onemax_demo()
    run_sphere_demo()
    
    # Member 3: 
    run_cvrp_demo()
    
    
    # instance for tuning 
    coords = [(0, 0), (10, 5), (5, 15), (20, 10), (15, 20)]
    demands = [0, 5, 5, 5, 5]
    instance_for_tuning = CVRPInstance(coords=coords, demands=demands, capacity=10, num_vehicles=2)
    
    run_tuning_experiments(instance_for_tuning)
    
    print("\nAll toolbox components and tuning working properly")



