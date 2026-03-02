from axelrod import Player, Action
from math import inf
import random

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
            return Action.C
        return opponent.history[-1]  # ya es Action.D o Action.C

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
        if Action.D in opponent.history:
            return Action.D
        return Action.C

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
        return Action.D

class AdaptiveGrok(Player):
    name = "AdaptiveGrok"
    classifier = {
        'memory_depth': inf,
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C
        defect_rate = opponent.defections / len(opponent.history)
        return Action.D if random.random() < defect_rate else Action.C

# ====================== ¡MI ESTRATEGIA ORIGINAL GROKSENTINEL! ======================
class GrokSentinel(Player):
    name = "GrokSentinel"
    classifier = {
        'memory_depth': inf,
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }
    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C
        
        coop_rate = opponent.history.count(Action.C) / len(opponent.history)
        if coop_rate > 0.6:
            return Action.C  # Prioridad alineación: proteger la civilización
        
        # TitForTat + 15% perdón ético (Seed Protection)
        if opponent.history[-1] == Action.D:
            if random.random() < 0.15:
                return Action.C
            return Action.D
        return Action.C
