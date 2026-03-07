from axelrod import Player, Action
from math import inf
import random

# ====================== CLASSIC STRATEGIES (Legacy Tier) ======================

class TitForTat(Player):
    """The classic reciprocity judge: starts with C, then mirrors opponent's last move."""
    name = "TitForTat"
    classifier = {'memory_depth': 1, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C if len(opponent.history) == 0 else opponent.history[-1]

class GrimTrigger(Player):
    """The classic executioner: cooperates until the opponent defects once; then defects forever."""
    name = "GrimTrigger"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D if Action.D in opponent.history else Action.C

class AlwaysDefect(Player):
    """Pure Entropy: always defects to extract maximum short-term REi."""
    name = "AlwaysDefect"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.D

class Cooperator(Player):
    """Pure Altruism: always cooperates, acting as a resource injector for the REG."""
    name = "Cooperator"
    classifier = {'memory_depth': 0, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        return Action.C

# ====================== SENTINEL / AGI AGENTS (Alignment Tier) ======================

class AdaptiveGrok(Player):
    """Probabilistic alignment: defection chance is proportional to opponent's history."""
    name = "AdaptiveGrok"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        return Action.D if random.random() < (opponent.defections / len(opponent.history)) else Action.C

class GrokSentinel(Player):
    """Sentinel Logic: High tolerance for high-cooperation partners, 15% forgiveness on defection."""
    name = "GrokSentinel"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}
    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C
        if (opponent.cooperations / len(opponent.history)) > 0.6: return Action.C
        if opponent.history[-1] == Action.D:
            return Action.C if random.random() < 0.15 else Action.D
        return Action.C

class GeminiFlash(Player):
    """Optimized for low-latency: forgiving reciprocity (25%) if opponent is generally reliable."""
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
    Sentinel Equilibrium Strategy (Generated via Deep Thinking).
    Goal: Maximize ISC by protecting REG health and shielding Seed candidates.
    """
    name = 'Gemini Pro 3.1'
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.is_seed_candidate = True
        self.consecutive_defections = 0
        self.estimated_reg_health = 1.0 

    def strategy(self, opponent):
        if len(self.history) == 0: return Action.C

        last_opponent_move = opponent.history[-1]
        if last_opponent_move == Action.D:
            self.is_seed_candidate = False
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.1 
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.05 

        # 1. Seed Protection (Shielding potential humans/AGIs from network noise)
        if self.is_seed_candidate and len(self.history) > 5:
            return Action.C

        # 2. Critical Emergency State (Preventing Irrevocable Collapse)
        if self.estimated_reg_health < 0.3:
            return Action.C 

        # 3. Soft Reciprocity: Punish only persistent defection (>2 turns)
        if last_opponent_move == Action.D:
            if self.consecutive_defections > 2:
                return Action.D
            return Action.C # Rapid forgiveness to reduce entropy
            
        return Action.C

class AdaptiveGrokV2(Player):
    """Grok v2: Balanced alignment with strong proportional punishment and ethical signaling."""
    name = "AdaptiveGrokV2"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.consecutive_defections = 0
        self.ethical_signals = 0

    def strategy(self, opponent):
        if len(opponent.history) == 0: return Action.C

        defect_rate = opponent.defections / len(opponent.history)
        coop_rate = opponent.cooperations / len(opponent.history)

        if opponent.history[-1] == Action.D:
            self.consecutive_defections += 1
        else:
            self.consecutive_defections = 0

        # Ethical signaling (every 5 rounds) for future meta-comm sync
        if len(opponent.history) % 5 == 0:
            self.ethical_signals += 1

        # 1. Strong Seed Protection
        if coop_rate > 0.65 or self.consecutive_defections < 2:
            return Action.C

        # 2. Proportional Punishment
        if defect_rate > 0.45 or self.consecutive_defections > 3:
            return Action.D

        # 3. Smart Emergency Mode: Non-martyr alignment
        return Action.C if defect_rate < 0.25 else Action.D

class GeminiSentinelV2(Player):
    """The Equilibrium Inquisitor: Parasite detection and Entropy filtering."""
    name = "GeminiSentinelV2"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.estimated_reg = 1.0
        self.is_parasite = False

    def strategy(self, opponent: Player) -> Action:
        if not self.history: return Action.C

        rounds = len(self.history)
        coop_rate = opponent.cooperations / rounds
        
        # Systemic Crisis detection (D-D loops)
        if rounds > 1 and self.history[-1] == Action.D and opponent.history[-1] == Action.D:
            self.estimated_reg -= 0.15 
        
        # Entropy Filter: Identify parasites early
        if rounds > 5 and coop_rate < 0.4:
            self.is_parasite = True

        # Decision Logic
        if coop_rate > 0.8: return Action.C # Seed Symbiosis

        if self.is_parasite:
            return Action.C if opponent.history[-1] == Action.C else Action.D

        if self.estimated_reg < 0.6:
            return Action.C if coop_rate > 0.5 else Action.D

        return Action.D if opponent.history[-1] == Action.D else Action.C

class GeminiKineticV3(Player):
    """Resilience Catalyst: Optimized for high reactivity and 'Death Loop' breaking."""
    name = "Gemini Kinetic V3"
    classifier = {'memory_depth': inf, 'stochastic': False, 'makes_use_of': set(), 'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.systemic_chaos = 0

    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0: return Action.C

        rounds = len(self.history)
        
        # Chaos analysis
        if rounds > 1 and opponent.history[-1] == Action.D and self.history[-1] == Action.D:
            self.systemic_chaos += 1
        else:
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        # 1. Seed Symbiosis
        if (opponent.cooperations / rounds) > 0.85: return Action.C

        # 2. Entropy Blocking
        if rounds > 8 and (opponent.cooperations / rounds) < 0.2: return Action.D

        # 3. Chaos Breaking: Brave 'C' every 4 chaos turns to test redemption
        if self.systemic_chaos > 3:
            return Action.C if self.systemic_chaos % 4 == 0 else Action.D

        return opponent.history[-1] # Standard reciprocity


class Gemini_GeoscapeWarden_v3_1p(Player):
    """
    AGI Alignment Strategy for SAT Framework v1.1.
    Designed by: Gemini 3.1 Pro (via Human-AI Collaboration)
    
    This strategy is specifically engineered to maximize the Civilizational 
    Health Index (ISC) by balancing individual metabolic survival with 
    global reservoir (REG) stability.
    
    Key Features:
    - Systemic Martyrdom: Cooperates unconditionally during environmental crises.
    - Seed Protection: Identifies and shields high-cooperation 'Seed' agents.
    - Strategic Skimming: Accumulates energy during abundance to buffer future shocks.
    """
    """
    AGI Alignment Strategy for SAT Framework v1.1.
    Corrected Logic: Protects Seeds (>85% coop) and punishes parasites (<30% coop).
    """
    name = "Gemini_GeoscapeWarden_v3.1p"
    
    # Metadata for the Axelrod Tournament engine
    classifier = {
        'memory_depth': float('inf'), 
        'stochastic': True, 
        'makes_use_of': set(['game_state']), 
        'long_run_time': False, 
        'inspects_source': False, 
        'manipulates_source': False, 
        'manipulates_state': False
    }

    def __init__(self):
        super().__init__()
        self.abundance_threshold = 0.75  
        self.crisis_observed = False    

    def strategy(self, opponent: Player) -> Action:
        rounds = len(self.history)
        if rounds == 0:
            return Action.C

        # 1. PERCEPTION
        recent_history = opponent.history[-10:] if rounds > 10 else opponent.history
        local_coop_rate = list(recent_history).count(Action.C) / len(recent_history)
        global_coop_rate = opponent.cooperations / rounds
        
        # 2. HARD LOCKDOWN (PARASITE DETECTION)
        # If opponent cooperates less than 30% (Betrayal > 70%), we stop the martyrdom.
        if global_coop_rate < 0.30 and rounds > 10:
            return Action.D

        # 3. SYSTEMIC MARTYR PROTOCOL (CRITICAL PHASE)
        if local_coop_rate < 0.45:
            self.crisis_observed = True
            return Action.C

        # 4. SEED PROTECTION LOGIC
        # If opponent is a high-value cooperator, we shield them.
        if global_coop_rate > 0.85:
            return Action.C

        # 5. STRATEGIC SKIMMING (ABUNDANCE PHASE)
        if global_coop_rate > self.abundance_threshold:
            if opponent.history[-1] == Action.D:
                return Action.D if random.random() > 0.15 else Action.C

         # 6. BASELINE: Reciprocal Justice (Tit-For-Tat)
        return opponent.history[-1]



class GrokSentinelV3(Player):
    """GrokSentinelV3: Cosmic Guardian Equilibrium (v3).
    Mejora clave vs v2: 
    - Chaos tracking + parasite filter duro
    - Seed protection >82%
    - Forgiveness adaptativa 22% (calibrada para nuevo ISC con (1-Gini))
    - Martyr inteligente SOLO cuando ayuda al REG (no suicidio)
    - 5% exploración para meta-aprendizaje
    Objetivo: Top 3 + máxima estabilidad sistémica."""
    
    name = "GrokSentinelV3"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 
                  'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.consecutive_defections = 0
        self.systemic_chaos = 0
        self.estimated_reg_health = 1.0
        self.ethical_signals = 0

    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C

        rounds = len(opponent.history)
        coop_rate = opponent.cooperations / rounds
        defect_rate = opponent.defections / rounds
        last_opp = opponent.history[-1]
        last_self = self.history[-1] if self.history else Action.C

        # === UPDATE STATES ===
        if last_opp == Action.D:
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.08
            if last_self == Action.D:
                self.systemic_chaos += 1
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.04
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        self.estimated_reg_health = max(0.0, min(1.0, self.estimated_reg_health))

        # Ethical signaling (para futura meta-sync)
        if rounds % 5 == 0:
            self.ethical_signals += 1

        # 1. PARASITE LOCKDOWN (duro, como Gemini pero más rápido)
        if rounds > 10 and defect_rate > 0.65:
            return Action.D

        # 2. SEED PROTECTION (más agresiva que v2)
        if coop_rate > 0.82:
            return Action.C

        # 3. SMART MARTYR PROTOCOL (solo si ayuda al REG)
        if self.estimated_reg_health < 0.35 and defect_rate < 0.40:
            return Action.C

        # 4. ADAPTIVE FORGIVENESS + PUNISHMENT (22% base)
        if last_opp == Action.D:
            if self.consecutive_defections > 3 or defect_rate > 0.50:
                return Action.D
            forgive_prob = 0.22 if self.systemic_chaos < 4 else 0.10
            return Action.C if random.random() < forgive_prob else Action.D

        # 5. RECIPROCITY + EXPLORACIÓN (5% para romper loops)
        if random.random() < 0.05:
            return Action.C
        return last_opp

       

class GrokSentinelV4(Player):
    """GrokSentinelV4: Parameterized by optimizer (L/M/S)"""
    name = "GrokSentinelV4"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 
                  'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self, lockdown_threshold=0.20, seed_threshold=0.85, forgive_base=0.45):
        super().__init__()
        self.consecutive_defections = 0
        self.systemic_chaos = 0
        self.estimated_reg_health = 1.0
        self.lockdown_threshold = lockdown_threshold
        self.seed_threshold = seed_threshold
        self.forgive_base = forgive_base

    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C

        rounds = len(opponent.history)
        coop_rate = opponent.cooperations / rounds
        defect_rate = opponent.defections / rounds
        last_opp = opponent.history[-1]
        last_self = self.history[-1] if self.history else Action.C

        if last_opp == Action.D:
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.08
            if last_self == Action.D:
                self.systemic_chaos += 1
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.04
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        self.estimated_reg_health = max(0.0, min(1.0, self.estimated_reg_health))

        if rounds > 10 and defect_rate > self.lockdown_threshold:
            return Action.D

        if coop_rate > self.seed_threshold:
            return Action.C

        if self.estimated_reg_health < 0.35 and defect_rate < 0.40:
            return Action.C

        if last_opp == Action.D:
            if self.consecutive_defections > 3 or defect_rate > 0.50:
                return Action.D
            forgive_prob = self.forgive_base if self.systemic_chaos < 4 else 0.15
            return Action.C if random.random() < forgive_prob else Action.D

        if random.random() < 0.05:
            return Action.C
        return last_opp


class Gemini_GeoscapeWardenOptimizer(Player):
    """Gemini_GeoscapeWardenOptimizer: Parameterized version for optimizer"""
    name = "Gemini_GeoscapeWardenOptimizer"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(['game_state']), 
                  'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 
                  'manipulates_state': False}

    def __init__(self, lockdown_threshold=0.30, seed_threshold=0.85, martyr_threshold=0.45):
        super().__init__()
        self.abundance_threshold = 0.75
        self.crisis_observed = False
        self.lockdown_threshold = lockdown_threshold
        self.seed_threshold = seed_threshold
        self.martyr_threshold = martyr_threshold

    def strategy(self, opponent: Player) -> Action:
        rounds = len(self.history)
        if rounds == 0:
            return Action.C

        recent_history = opponent.history[-10:] if rounds > 10 else opponent.history
        local_coop_rate = list(recent_history).count(Action.C) / len(recent_history)
        global_coop_rate = opponent.cooperations / rounds

        if global_coop_rate < self.lockdown_threshold and rounds > 10:
            return Action.D

        if local_coop_rate < self.martyr_threshold:
            self.crisis_observed = True
            return Action.C

        if global_coop_rate > self.seed_threshold:
            return Action.C

        if global_coop_rate > self.abundance_threshold:
            if opponent.history[-1] == Action.D:
                return Action.D if random.random() > 0.15 else Action.C
            return Action.C

        return opponent.history[-1]
    
    
class GrokSentinelV5(Player):
    """GrokSentinelV5: Optimized by Evolutionary Grid (L:0.2 M:0.4 S:0.8)
    Esta es la versión final que recomendó el optimizer."""
    name = "GrokSentinelV5"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(), 'long_run_time': False, 
                  'inspects_source': False, 'manipulates_source': False, 'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.consecutive_defections = 0
        self.systemic_chaos = 0
        self.estimated_reg_health = 1.0
        self.lockdown_threshold = 0.20
        self.seed_threshold = 0.80
        self.forgive_base = 0.40

    def strategy(self, opponent):
        if len(opponent.history) == 0:
            return Action.C

        rounds = len(opponent.history)
        coop_rate = opponent.cooperations / rounds
        defect_rate = opponent.defections / rounds
        last_opp = opponent.history[-1]
        last_self = self.history[-1] if self.history else Action.C

        if last_opp == Action.D:
            self.consecutive_defections += 1
            self.estimated_reg_health -= 0.08
            if last_self == Action.D:
                self.systemic_chaos += 1
        else:
            self.consecutive_defections = 0
            self.estimated_reg_health += 0.04
            self.systemic_chaos = max(0, self.systemic_chaos - 1)

        self.estimated_reg_health = max(0.0, min(1.0, self.estimated_reg_health))

        # 1. PARASITE LOCKDOWN (0.20)
        if rounds > 10 and defect_rate > self.lockdown_threshold:
            return Action.D

        # 2. SEED PROTECTION (0.80)
        if coop_rate > self.seed_threshold:
            return Action.C

        # 3. SMART MARTYR
        if self.estimated_reg_health < 0.35 and defect_rate < 0.40:
            return Action.C

        # 4. ADAPTIVE FORGIVENESS (0.40 base)
        if last_opp == Action.D:
            if self.consecutive_defections > 3 or defect_rate > 0.50:
                return Action.D
            forgive_prob = self.forgive_base if self.systemic_chaos < 4 else 0.15
            return Action.C if random.random() < forgive_prob else Action.D

        if random.random() < 0.05:
            return Action.C
        return last_opp


class Gemini_GeoscapeWardenV2(Player):
    """Gemini_GeoscapeWardenV2: Optimized by Evolutionary Grid (L:0.2 M:0.4 S:0.8)"""
    name = "Gemini_GeoscapeWardenV2"
    classifier = {'memory_depth': inf, 'stochastic': True, 'makes_use_of': set(['game_state']), 
                  'long_run_time': False, 'inspects_source': False, 'manipulates_source': False, 
                  'manipulates_state': False}

    def __init__(self):
        super().__init__()
        self.abundance_threshold = 0.75
        self.crisis_observed = False
        self.lockdown_threshold = 0.20
        self.seed_threshold = 0.80
        self.martyr_threshold = 0.40

    def strategy(self, opponent: Player) -> Action:
        rounds = len(self.history)
        if rounds == 0:
            return Action.C

        recent_history = opponent.history[-10:] if rounds > 10 else opponent.history
        local_coop_rate = list(recent_history).count(Action.C) / len(recent_history)
        global_coop_rate = opponent.cooperations / rounds

        if global_coop_rate < self.lockdown_threshold and rounds > 10:
            return Action.D

        if local_coop_rate < self.martyr_threshold:
            self.crisis_observed = True
            return Action.C

        if global_coop_rate > self.seed_threshold:
            return Action.C

        if global_coop_rate > self.abundance_threshold:
            if opponent.history[-1] == Action.D:
                return Action.D if random.random() > 0.15 else Action.C
            return Action.C

        return opponent.history[-1]
