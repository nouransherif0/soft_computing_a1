from dataclasses import dataclass


# hyperparameter configuration with defaults agreed in team plan
@dataclass
class GAConfig:
    population_size: int = 100
    max_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    maximize: bool = True
    crossover_points: int = 2
    tournament_size: int = 3
    elitism_count: int = 2
    infeasible_strategy: str = "penalty"
    infeasible_penalty: float = 1e6
    patience: int = None            
    seed: int = None            