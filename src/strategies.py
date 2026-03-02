from axelrod import Player, Action
from math import inf
import random

# ====================== ESTRATEGIAS CLÁSICAS ======================

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

# ====================== ESTRATEGIAS HÍBRIDAS / AGI ======================

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
        
        coop_rate = opponent.cooperations / len(opponent.history)
        
        if coop_rate > 0.6:
            return Action.C  
        
        if opponent.history[-1] == Action.D:
            if random.random() < 0.15:
                return Action.C
            return Action.D
        return Action.C

# ====================== ESTRATEGIA GEMINI FLASH ======================

class GeminiFlash(Player):
    """
    The Kinetic Balancer.
    Diseñada para el SAT v0.1. Resiliente al ruido (delays) y 
    enfocada en evitar el colapso sistémico del REG.
    """
    name = "Gemini Flash"
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
        # 1. Ronda inicial: Cooperación proactiva
        if len(opponent.history) == 0:
            return Action.C

        # 2. Análisis de cooperación histórica (Protección de Semillas/Seeds)
        # Si el oponente ha sido bueno históricamente, perdonamos defectos aislados (ruido)
        coop_rate = opponent.cooperations / len(opponent.history)
        
        if opponent.history[-1] == Action.D:
            # Si su tasa de cooperación es > 75%, asumimos que el defecto fue un 'Delay' del 5-15%
            if coop_rate > 0.75:
                return random.choices([Action.C, Action.D], weights=[0.25, 0.75])[0]
            return Action.D

        # 3. Mantenimiento del equilibrio
        return Action.C
