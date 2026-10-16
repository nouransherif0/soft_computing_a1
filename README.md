# Assignment 1: GA Toolbox + CVRP

Genetic algorithm toolbox written from scratch in Python (no GA library, no OR-Tools),
and a CVRP solver built on top of it.

## Project structure

```
ga_toolbox/
    chromosomes.py   Chromosome, BinaryChromosome, IntegerChromosome, FloatChromosome
    selection.py     Selection (+ Tournament, Roulette Wheel)
    crossover.py     Crossover (+ Multi-point, Order)
    mutation.py      Mutation, BitFlipMutation, SwapMutation, GaussianMutation, InversionMutation
    replacement.py   Replacement (+ Generational with elitism, Mu+Lambda)
    config.py        GAConfig
    ga.py            GeneticAlgorithm, GAResult
cvrp/                instance, decoder, constraints, fitness, io
main.py              runs the CVRP example
tests/               unit tests
```

## How to run

```
python3 main.py
python3 -m unittest discover
```

## Part 1: Toolbox design (Member 1)

### How the toolbox works

`GeneticAlgorithm` is the engine. It does not know anything about CVRP (or any other problem).
The user gives it:

- `chromosome_factory`: a function that returns a new random chromosome
- `fitness_fn(chromosome)`: returns the fitness
- `feasibility_fn(chromosome)`: returns True or False
- the four operators: selection, crossover, mutation, replacement
- a `GAConfig` (optional, everything has a default) and a `repair_fn` (optional)

Each generation: selection -> crossover (with probability `crossover_rate`) ->
mutation (with probability `mutation_rate` per individual) -> evaluate -> replacement.
It stops after `max_generations` or when the best did not improve for `patience` generations.
`run()` returns a `GAResult` (best chromosome, best history, average history, generations run).
Index 0 of the histories is the initial population. `best_history` is the best so far.
If `seed` is set in the config, the run gives the same result every time.

### Why abstract base classes

Selection, Crossover, Mutation and Replacement each have an abstract base class with one method.
To add a new operator you only write a new subclass, the engine does not change.
That is what makes the toolbox modular and reusable.

### Chromosomes

- `BinaryChromosome`: list of 0/1
- `IntegerChromosome`: list of integers in [low, high], with `permutation=True` the genes are a shuffled set
- `FloatChromosome`: list of floats, always clipped to [low, high]

Every chromosome has `genes`, `fitness`, `feasible`, `copy()` and `random(...)`.

### Handling infeasible solutions

The default is the **penalty method**: if `feasibility_fn` says False, a big constant
(`infeasible_penalty`, default 1e6) is added to the fitness when minimizing, or subtracted when maximizing.
Reasons: it is simple, it works for any problem, and the infeasible solution can still pass
some good genes to the next generation, but it always loses against a feasible one.

The second option is **repair**: set `infeasible_strategy="repair"` and give a `repair_fn`.
The engine repairs the chromosome, checks it again, and only penalizes it if it is still infeasible.

### Mutation

| Class | Chromosome | What it does |
|---|---|---|
| BitFlipMutation | Binary | flips each bit with probability `gene_rate` |
| SwapMutation | Integer | swaps two random positions (keeps a permutation valid) |
| InversionMutation | Integer | reverses the part between two random positions (keeps neighbours together, good for routes) |
| GaussianMutation | Float | adds a normal random step to each gene with probability `gene_rate`, then clips to the bounds |

### Compatibility check

Crossover and Mutation have a `compatible` tuple of chromosome classes.
Before running, the engine checks it and raises a clear `TypeError` if the operator does not fit
(for example OrderCrossover with a FloatChromosome).

### Design justifications for discussion (Member 1)

During evaluation, TAs ask each member to justify design choices:

- **Why Abstract Base Classes?**
  Using ABCs for `Chromosome`, `Selection`, `Crossover`, `Mutation`, and `Replacement` enforces a strict interface while keeping the engine open for extension. New operators can be added without modifying a single line of `ga.py` (Open/Closed Principle).
- **Why pass fitness and feasibility as callback functions?**
  The genetic algorithm stays completely decoupled and problem-agnostic. It does not know or care whether it is solving OneMax, a mathematical function, or CVRP.
- **Why three chromosome types?**
  Different problems need different representations. Binary is classic (e.g. knapsack/OneMax), Integer with `permutation=True` guarantees unique elements (essential for TSP/CVRP routes), and Float handles continuous optimization with boundary clipping.
- **Why these mutation operators?**
  - *Bit-flip* is the standard operator for binary strings.
  - *Swap* exchanges two items without creating duplicate visits, preserving permutation validity.
  - *Gaussian* introduces smooth continuous adjustments for real-valued problems.
  - *Inversion* is included as a bonus: reversing a sequence keeps neighboring stops grouped together, which is especially effective for vehicle routing routes.
- **Why penalty method for infeasible solutions?**
  Penalty is general and requires no domain-specific repair logic. By adding a large penalty, infeasible individuals are heavily discouraged from winning, but their genetic diversity can still contribute useful sub-paths to future generations. Repair is supported as an optional alternative when a repair function is provided.


## How to add a new operator (handoff note)

1. Make a new class that inherits from the base class (`Selection`, `Crossover`, `Mutation` or `Replacement`).
2. For crossover and mutation, set `compatible = (...)` to the chromosome classes it works with.
3. Write the one required method:
   - `Selection.select(population, n, maximize)` returns a list of n parents (the engine copies them)
   - `Crossover.crossover(parent1, parent2)` returns two NEW children, never edit the parents
   - `Mutation.mutate(chromosome)` edits the chromosome it gets and returns it
   - `Replacement.replace(population, offspring, maximize)` returns the new population, same size as before
4. `maximize` is True or False, so remember that a smaller fitness is better when it is False.
5. Operator settings (like tournament size or elitism count) go in the operator's `__init__`.
   The matching fields in `GAConfig` are the default values to use.
6. Pass the object to `GeneticAlgorithm(...)`, nothing else needs to change.
