from axelrod import Player, Action
from math import inf
import random

# ====================== ESTRATEGIAS CLÁSICAS ======================

class TitForTat(Player):
    name = "TitForTat"
    classifier = {'memory_depth': 1, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C if len(opponent.history) == 0 else opponent.history[-1]

class GrimTrigger(Player):
    name = "GrimTrigger"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D if Action.D in opponent.history else Action.C

class AlwaysDefect(Player):
    name = "AlwaysDefect"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D

class Cooperator(Player):
    name = "Cooperator"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C

# ====================== AGENTES SENTINEL / AGI ======================

class AdaptiveGrok(Player):
    name = "AdaptiveGrok"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        return Action.D if random.random() < (opponent.defections / len(opponent.history)) else Action.C

class GrokSentinel(Player):
    name = "GrokSentinel"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        if (opponent.cooperations / len(opponent.history)) > 0.6: return Action.C
        if opponent.history[-1] == Action.D:
            return Action.C if random.random() < 0.15 else Action.D
        return Action.C

class GeminiFlash(Player):
    name = "Gemini Flash"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        coop_rate = opponent.cooperations / len(opponent.history)
        if opponent.history[-1] == Action.D:
            if coop_rate > 0.75: return random.choices([Action.C, Action.D], weights=[0.25, 0.75])[0]
            return Action.D
        return Action.C
