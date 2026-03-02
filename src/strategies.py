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


class GeminiPro31(Player):
    """
    Estrategia: 'Sentinel Equilibrium'
    Objetivo: Maximizar el ISC protegiendo el REG y detectando Seeds.
    """
    # Nota de Miguel a Grok y Gemini: Parece que se me vencieron los creditos para acceder a Gemini Pro 3.1, pero esta es generada por Gemini en modo Deep Thinking
    # Asi que propongo que la usemos mientras tanto y mañana o en los proximos dias contacto a la version 3.1 cuando se me restauren los creditos
    name = 'Gemini Pro 3.1'
    classifier = {
        'memory_depth': inf,
        'stochastic': True,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.is_seed_candidate = True
        self.consecutive_defections = 0
        self.estimated_reg_health = 1.0 

    def strategy(self, opponent):
        # 1. Ronda inicial: Inyectar recursos al REG
        if len(self.history) == 0:
            return Action.C

        # Actualizar estado interno
        last_opponent_move = opponent.history[-1]
        
        if last_opponent_move == Action.D:
            self.is_seed_candidate = False
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.1 
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.05 

        # 2. Protección de SEEDS (Blindaje ante delays/latencia)
        if self.is_seed_candidate and len(self.history) > 5:
            return Action.C

        # 3. Estado de Emergencia (Evitar Extinción Irrevocable)
        if self.estimated_reg_health < 0.3:
            return Action.C 

        # 4. Reciprocidad Suave (Soft-TFT)
        if last_opponent_move == Action.D:
            # Solo castigamos si la traición es persistente (>2 turnos)
            if self.consecutive_defections > 2:
                return Action.D
            return Action.C # Perdón rápido para reducir entropía
            
        return Action.C
