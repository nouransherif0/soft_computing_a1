from abc import ABC, abstractmethod


class Crossover(ABC):

    compatible = ()

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass
