from axelrod import Player
from math import inf
import random  # Mueve el import aquí para evitar issues en métodos

class TitForTat(Player):
    name = "TitForTat"
    classifier = {
        'memory_depth': 1,
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return 'C'
        return opponent.history[-1]

class GrimTrigger(Player):
    name = "GrimTrigger"
    classifier = {
        'memory_depth': inf,
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        if 'D' in opponent.history:
            return 'D'
        return 'C'

class AlwaysDefect(Player):
    name = "AlwaysDefect"
    classifier = {
        'memory_depth': 0,
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        return 'D'

class AdaptiveGrok(Player):
    name = "AdaptiveGrok"
    classifier = {
        'memory_depth': inf,  # Usa toda la historia para defect_rate
        'stochastic': True,   # Por el random
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return 'C'
        defect_rate = opponent.defections / len(opponent.history)  # Float ok, no se usa como index
        return 'D' if random.random() < defect_rate else 'C'
