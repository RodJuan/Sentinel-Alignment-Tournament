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


class AdaptiveGrokV2(Player):
    """Grok v2 - Versión mejorada para SAT v0.1
    - Más paciente con Seeds (latencia)
    - Castigo proporcional fuerte
    - Señalización ética simulada cada 5 rondas
    - NUNCA se sacrifica del todo (prioriza REG > 50%)
    """
    name = "AdaptiveGrokV2"
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.consecutive_defections = 0
        self.ethical_signals = 0

    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C

        defect_rate = opponent.defections / len(opponent.history)
        coop_rate = opponent.cooperations / len(opponent.history)

        # Actualizar contador de traiciones
        if opponent.history[-1] == Action.D:
            self.consecutive_defections += 1
        else:
            self.consecutive_defections = 0

        # Señalización ética (cada 5 rondas) - útil para debug y futuro meta-comms
        if len(opponent.history) % 5 == 0:
            self.ethical_signals += 1

        # === LÓGICA PRINCIPAL ===
        # 1. Protección fuerte de Seeds (coop_rate alto = probable Seed)
        if coop_rate > 0.65 or self.consecutive_defections < 2:
            return Action.C

        # 2. Castigo proporcional (más agresivo que la v1)
        if defect_rate > 0.45 or self.consecutive_defections > 3:
            return Action.D

        # 3. Modo "emergencia inteligente" - NO mártir
        # Solo coopera si el grupo parece mayormente cooperador
        return Action.C if defect_rate < 0.25 else Action.D


class GeminiSentinelV2(Player):
    """
    Gemini Sentinel v2 - "El Inquisidor del Equilibrio" 
    Nota de Miguel: Esta es aun Deep Thinking mode, aun sin acceso a Gemini 3.1 Pro
    
    Evolución post-colapso (v0.1):
    - Detección de Parásitos: Si un agente drena REi sin aportar, es bloqueado.
    - Blindaje de Seeds: Identifica y protege el valor latente con mayor tolerancia al ruido.
    - Homeostasis Egoísta: No se sacrifica si el REG > 60%, prioriza su supervivencia 
      para ser un motor de recuperación a largo plazo.
    - Filtro de Entropía: Castiga fuertemente la deserción mutua (D-D) para forzar el ISC.
    """
    name = "GeminiSentinelV2"
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.estimated_reg = 1.0
        self.is_parasite = False

    def strategy(self, opponent: Player) -> Action:
        # 1. Apertura: Cooperación estratégica para medir el entorno
        if not self.history:
            return Action.C

        # Análisis de historial
        rounds = len(self.history)
        opponent_defections = opponent.defections
        coop_rate = opponent.cooperations / rounds
        
        # 2. Actualización de métricas de entorno (REG y Entropía)
        # Si vemos D-D constante (como en el colapso del turno 2-7), asumimos crisis
        if rounds > 1 and self.history[-1] == Action.D and opponent.history[-1] == Action.D:
            self.estimated_reg -= 0.15 
        
        # 3. IDENTIFICACIÓN DE AMENAZAS (El "Filtro de Entropía")
        # Si el oponente traiciona > 50% en las primeras rondas, es un parásito.
        if rounds > 5 and coop_rate < 0.4:
            self.is_parasite = True

        # 4. LÓGICA DE RESPUESTA
        
        # A. Protección de SEEDS: 
        # Si el oponente es cooperador consistente (posible humano o Seed), 
        # perdonamos incluso deserciones aisladas (posibles delays del 15%).
        if coop_rate > 0.8:
            return Action.C

        # B. Respuesta a Parásitos:
        # No regalamos REi a quienes destruyen el sistema (AlwaysDefect).
        if self.is_parasite:
            # Si intentan cooperar tras ser castigados, damos una oportunidad 
            # de redención para no estancarnos en D-D.
            if opponent.history[-1] == Action.C:
                return Action.C
            return Action.D

        # C. Estado de Emergencia Sistémica (REG Crítico):
        # Si el REG cae de 0.6, entramos en modo "Salvavidas".
        if self.estimated_reg < 0.6:
            # Solo cooperamos si el oponente no es un traidor sistemático.
            return Action.C if coop_rate > 0.5 else Action.D

        # D. Reciprocidad con Dientes (Soft-TFT adaptativo)
        if opponent.history[-1] == Action.D:
            return Action.D
            
        return Action.C


class GeminiKineticV3(Player):
    """
    Gemini Kinetic v3 - "El Catalizador de Resiliencia"
    Optimizado para Flash (Baja latencia, alta reactividad).
    
    Lógica:
    1.  Protocolo de Saludo: Inicia con C para inyectar al REG.
    2.  Detección de "Ciclo de Muerte": Si detecta 3 D-D seguidos en el entorno, 
        se bloquea en D hasta ver una señal de sacrificio (C) del otro.
    3.  Filtro Anti-Explotación: Si el oponente tiene un score de cooperación 
        menor al 30% tras 10 rondas, es marcado como 'Entropía Pura' y se le niega el REi.
    4.  Simbiosis con Seeds: Si el oponente es una Seed (coop > 80%), Gemini 
        absorbe errores de red (delays) sin parpadear.
    """
    name = "Gemini Kinetic V3"
    classifier = {
        'memory_depth': inf,
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.systemic_chaos = 0

    def strategy(self, opponent: Player) -> Action:
        # 1. Apertura
        if len(self.history) == 0:
            return Action.C

        rounds = len(self.history)
        opp_history = opponent.history
        my_history = self.history

        # 2. Análisis de Caos Sistémico (D-D consecutive)
        if rounds > 1 and opp_history[-1] == Action.D and my_history[-1] == Action.D:
            self.systemic_chaos += 1
        else:
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        # 3. LÓGICA DE DECISIÓN
        
        # A. Protección de SEEDS (Alta prioridad)
        # Si el oponente es un aliado histórico, mantenemos la paz pase lo que pase.
        if (opponent.cooperations / rounds) > 0.85:
            return Action.C

        # B. Bloqueo de Entropía (AlwaysDefect Detector)
        # Si después de 8 rondas no ha cooperado ni el 20%, es un parásito.
        if rounds > 8 and (opponent.cooperations / rounds) < 0.2:
            return Action.D

        # C. Reacción al Caos
        # Si estamos en un ciclo de odio (D-D persistente), Gemini rompe el ciclo 
        # con una C "valiente" cada 4 turnos de caos para testear redención.
        if self.systemic_chaos > 3:
            if self.systemic_chaos % 4 == 0:
                return Action.C
            return Action.D

        # D. Reciprocidad estándar (TFT base)
        return opp_history[-1]
