from abc import ABC, abstractmethod


class Replacement(ABC):

    @abstractmethod
    def replace(self, population, offspring, maximize):
        pass