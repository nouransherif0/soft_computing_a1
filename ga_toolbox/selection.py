from abc import ABC, abstractmethod


class Selection(ABC):

    @abstractmethod
    def select(self, population, n, maximize):
        pass