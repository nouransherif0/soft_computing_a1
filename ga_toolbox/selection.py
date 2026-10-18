from abc import ABC, abstractmethod
import random

from ga_toolbox.config import GAConfig


class Selection(ABC):

    @abstractmethod
    def select(self, population, n, maximize):
        pass


class TournamentSelection(Selection):
    # pick tournament_size random individuals, the best of them wins, repeat n times.
    # works for maximize and minimize and for any fitness scale (only compares values)

    def __init__(self, tournament_size=None):
        if tournament_size is None:
            tournament_size = GAConfig().tournament_size
        if tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        self.tournament_size = tournament_size

    def select(self, population, n, maximize):
        size = min(self.tournament_size, len(population))
        chosen = []
        for i in range(n):
            contenders = random.sample(population, size)
            winner = contenders[0]
            for c in contenders:
                if maximize and c.fitness > winner.fitness:
                    winner = c
                if not maximize and c.fitness < winner.fitness:
                    winner = c
            chosen.append(winner)
        return chosen


class RouletteWheelSelection(Selection):
    # fitness proportionate selection.
    # raw fitness cannot be used as a probability when it is negative or when smaller is
    # better, so it is first turned into non negative weights:
    #   maximize: weight = fitness - worst fitness
    #   minimize: weight = worst fitness - fitness   (worst = biggest value)
    # the worst individual would get weight 0, so a tiny epsilon is added to keep
    # every individual selectable.

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def get_weights(self, population, maximize):
        fits = []
        for c in population:
            fits.append(c.fitness)
        lowest = min(fits)
        highest = max(fits)

        if highest == lowest:
            # everybody is equal, so everybody has the same chance
            return [1.0] * len(population)

        weights = []
        for f in fits:
            if maximize:
                w = f - lowest
            else:
                w = highest - f
            weights.append(w + self.epsilon * (highest - lowest))
        return weights

    def select(self, population, n, maximize):
        weights = self.get_weights(population, maximize)
        return random.choices(population, weights=weights, k=n)
